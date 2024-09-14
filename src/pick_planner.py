from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    VPolytope,
    StartMeshcat,
    Rgba,
    Point,
    Sphere,
    Box,
)
from manipulation.meshcat_utils import AddMeshcatTriad

import numpy as np
import heapq
import os
import time
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from scenario import NUM_BOXES, BOX_DIM, GRIPPER_DIM, PREPICK_MARGIN, GRIPPER_THICKNESS
from utils import ik


class BoxSelectorGraph:
    """
    Graph where each node is a tuple representing a box, containing 
    (BodyIndex, RigidTransform). An edge is drawn from a parent box to a child
    box if the child box is vertically above that parent box.

    This is used to efficiently find the first box to remove from the pile in
    the truck trailer.
    """
    def __init__(self):
        """
        Both an adjacency list and reversed adjacency list are used so that
        removing a node from the graph entirely can be done quicky (the
        reversed adjacency list is used to quickly find which nodes have the
        to-be-removed node as a child).

        The heap is used to quickly find the minimum x-coordinate box to be
        picked first.
        """
        self.adj_list = {}  # Adjacency list to store graph
        self.reverse_adj_list = {}  # Reverse adjacency list to track parents of each child
        self.removable_boxes_heap = []  # maintain min heap of removable boxes (nodes w/ no children)
        # Each node in the heap is a tuple (Box x-coord, (BodyIndex, RigidTransform))
    

    def add_node(self, node):
        """O(1)"""
        if node not in self.adj_list:
            self.adj_list[node] = set()
        if node not in self.reverse_adj_list:
            self.reverse_adj_list[node] = set()
    

    def add_edge(self, parent, child):
        """O(1)"""
        if parent not in self.adj_list:
            self.add_node(parent)
        if child not in self.adj_list:
            self.add_node(child)
        
        self.adj_list[parent].add(child)
        self.reverse_adj_list[child].add(parent)
    

    def remove_node(self, node):
        """O(number of parents)"""
        if node in self.adj_list:
            # Remove node from its children's parent sets
            for child in self.adj_list[node]:
                self.reverse_adj_list[child].remove(node)
            
            # Remove node from adjacency list
            del self.adj_list[node]
        
        if node in self.reverse_adj_list:
            # Remove node from its parents' child sets
            for parent in self.reverse_adj_list[node]:
                self.adj_list[parent].remove(node)
            
            # Remove node from reverse adjacency list
            del self.reverse_adj_list[node]


    def build_heap(self):
        """O(n)"""
        for node in self.adj_list:
            # Build heap containing only nodes with no children (i.e. boxes with no boxes above them)
            if len(self.adj_list[node]) == 0:
                box_x_coord = node[1].translation()[0]
                heapq.heappush(self.removable_boxes_heap, (box_x_coord, node))
        

    def remove_next_node(self):
        """O(log(n))"""
        if len(self.removable_boxes_heap) == 0:
            print("Unable to find a box without boxes above it that can be picked.")
            return None
        
        # Find box to remove
        ret = heapq.heappop(self.removable_boxes_heap)[1]  # [1] so we return the (BodyIndex, RigidTransform) tuple only (ignoring the x-coord)

        # Find, if after this box is removed, if there are any new boxes that
        # have been exposed and could now be removed, and add them to the heap
        if ret in self.reverse_adj_list:
            for parent in self.reverse_adj_list[ret]:
                if len(self.adj_list[parent]) == 1:  # The only box above `parent` is the one about to be removed
                    parent_x_coord = parent[1].translation()[0]
                    heapq.heappush(self.removable_boxes_heap, (parent_x_coord, parent))

        # Remove its node from the graph
        self.remove_node(ret)
        return ret


    def __str__(self):
        return "\n".join(f"{node}: {children}" for node, children in self.adj_list.items())


class PickPlanner():
    """
    A class to manage all picking logic, i.e. selecting which boxes are viable
    to be picked at the current time.
    """
    def __init__(self, meshcat, robot_pose, box_body_indices, ik_plant, ik_plant_context, DEBUG=True):
        self.meshcat = meshcat
        self.robot_pose = robot_pose
        self.box_body_indices = box_body_indices
        self.plant = ik_plant
        self.plant_context = ik_plant_context
        self.DEBUG = DEBUG


    def sort_vertices_ccw(self, vpolytope: VPolytope) -> np.ndarray:
        """
        Util functin that converts a Drake VPolytope to a list of ordered 
        coordinates forming a complete polygon.

        Parameters:
        vpolytope (VPolytope): A Drake VPolytope object.

        Returns:
        np.ndarray: A numpy array of the VPolytope's ordered coordinates.
        """
        def angle_with_centroid(point, centroid):
            """Calculate the angle between the point and the centroid."""
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

        # Extracting vertices from the VPolytope
        vertices = vpolytope.vertices()

        # Converting to numpy array for easier manipulation
        coordinates = np.array(vertices).T  # Shape (num_vertices, 2)

        # Compute the centroid
        centroid = np.mean(coordinates, axis=0)

        # Sort vertices by the angle with respect to the centroid
        sorted_indices = np.argsort([angle_with_centroid(pt, centroid) for pt in coordinates])
        ordered_coordinates = coordinates[sorted_indices].T
        ordered_coordinates = np.hstack((ordered_coordinates, ordered_coordinates[:, 0].reshape(-1, 1)))

        return ordered_coordinates
    

    def solve_q_place(self, source_regions_place):
        """
        Solve an IK program for the box deposit pose that is reachable
        within the solved IRIS regions.
        """
        self.X_W_Deposit = RigidTransform(RotationMatrix.MakeXRotation(3.14159265), self.robot_pose.translation() + [0.0, -0.65, 1.25])
        AddMeshcatTriad(self.meshcat, "X_W_Deposit", X_PT=self.X_W_Deposit, opacity=0.5)
        q, _ = ik(self.plant, self.plant_context, self.X_W_Deposit, regions=source_regions_place)
        return q
    

    def solve_q_pick(self, pre_pick_pose):
        """
        Solve an IK program for the box pre-pick pose.

        pre_pick_pose is a RigidTransform and is shifted to generate the pick 
        pose.
        """
        # Offset the pre-pick pose by the PREPICK_MARGIN toward the box to get the pick pose
        pick_pose = RigidTransform(pre_pick_pose.rotation(), pre_pick_pose.translation() + pre_pick_pose.rotation() @ [0, 0, PREPICK_MARGIN - GRIPPER_THICKNESS])
        q, _ = ik(self.plant, self.plant_context, pick_pose)
        return q


    def get_viable_pick_poses(self, box_poses, source_regions):
        """
        Return a dictionary mapping regions in configuration that are viable
        pick poses to tuples containing the corresponding box BodyIndex and pick
        pose (in the form of a RigidTransform).

        box_poses is a dictionary mapping box BodyIndex to RigidTransform
        representing their 3D pose.
        """
        # Compute projections of boxes onto XY plane to more easily determine
        # which are vertically overlapping
        box_projections = {}  # dict mapping box bodyIndex to VPolytope XY projection
        for box_body_idx, box_pose in box_poses.items():
            # Find vertices of projection of box onto XY plane
            box_corners = []
            for dx in [0, BOX_DIM]:
                for dy in [0, BOX_DIM]:
                    for dz in [0, -BOX_DIM]:
                        # Find coordinate of box corner in 3D
                        box_corner = box_pose.translation() + box_pose.rotation() @ np.array([dx, dy, dz])
                        # Project box corner into XY plane by removing Z coordinate
                        box_corner_xy = np.array([box_corner[0], box_corner[1]])
                        box_corners.append(box_corner_xy)

            box_corners = np.array(box_corners)

            # Only keep the points in the convex hull of the projection
            box_hull = ConvexHull(box_corners)
            box_points = box_corners[box_hull.vertices, :]
            box_points = np.hstack((box_points, np.zeros((np.shape(box_points)[0], 1))))  # Append 0 z-coordinate
            
            box_projections[box_body_idx] = VPolytope(box_points.T)

        # Render 2D projections in Meshcat
        if self.DEBUG:
            print("Box Projection Visualization Meshcat:")
            meshcat = StartMeshcat()
            meshcat.Set2dRenderMode(RigidTransform([0, 0, 1]), -4, 4, -4, 4)
            meshcat.SetProperty("/Axes", "visible", True)
            ctr = 0
            for box_body_idx, vpoly in box_projections.items():
                points = np.array(self.sort_vertices_ccw(vpoly))
                z = box_poses[box_body_idx].translation()[2]
                meshcat.SetLine(f"vpoly_{ctr}", points, 2.0, Rgba(*(plt.cm.viridis(z / 2.0))))
                ctr += 1

        # Now, determine which boxes vertically overlap and remove any boxes that are in lower layers
        viable_boxes = list(box_poses.keys())
        for box_body_idx, box_proj in box_projections.items():
            for other_box_body_idx, other_box_proj in box_projections.items():
                # Box cannot be on top of itself
                if box_body_idx == other_box_body_idx:
                    continue
                
                # If boxes vertically overlap, figure out which one is above the
                # other by looking at their z-coordinates. 
                # Note: This isn't a perfect solution; i.e. one box can be on
                # another but still have a lower z-coordinate; but this seems
                # unlikely enough (and also I can't think of a perfect way to
                # do this).
                if box_proj.IntersectsWith(other_box_proj):
                    if box_poses[box_body_idx].translation()[2] > box_poses[other_box_body_idx].translation()[2]:
                        # box_body_idx is above other_box_body_idx
                        try:
                            viable_boxes.remove(other_box_body_idx)
                        except:
                            pass
        print(f"{len(viable_boxes)} viable boxes to be picked found.")
                        
        if self.DEBUG:
            for box_body_idx in viable_boxes:
                self.meshcat.SetObject(f"Viable_Boxes/{box_body_idx}", Box(BOX_DIM, BOX_DIM, BOX_DIM), Rgba(0.75, 0.0, 0.0))
                self.meshcat.SetTransform(f"Viable_Boxes/{box_body_idx}", RigidTransform(box_poses[box_body_idx].rotation(), box_poses[box_body_idx].translation() + box_poses[box_body_idx].rotation() @ np.array([BOX_DIM/2, BOX_DIM/2, -BOX_DIM/2])))
            # Remove drawn polytopes from previous iteration
            for box_body_idx in self.box_body_indices:
                for i in range(6):
                    self.meshcat.Delete(f"Pick_Poses/{box_body_idx}_{i}")

        # For each viable box, generate polytope of grasp poses for each face
        # Also, display the polytope in meshcat
        pick_regions = {}  # dict mapping Points to (BodyIndex, RigidTransform) tuples
        for box_body_idx in viable_boxes:
            box_pose = box_poses[box_body_idx]
            box_center = RigidTransform(box_pose.rotation(), box_pose.translation() + box_pose.rotation() @ np.array([BOX_DIM/2, BOX_DIM/2, -BOX_DIM/2]))  # Because box_pose is at the corner of the box

            # For all 6 faces of each box
            for i in range(6):
                if i == 0:
                    p = box_center.translation() + box_pose.rotation() @ np.array([(BOX_DIM/2 + PREPICK_MARGIN), 0, 0])
                    R = box_center.rotation() @ RotationMatrix.MakeYRotation(-np.pi/2)
                elif i == 1:
                    p = box_center.translation() + box_pose.rotation() @ np.array([-(BOX_DIM/2 + PREPICK_MARGIN), 0, 0])
                    R = box_center.rotation() @ RotationMatrix.MakeYRotation(np.pi/2)
                elif i == 2:
                    p = box_center.translation() + box_pose.rotation() @ np.array([0, (BOX_DIM/2 + PREPICK_MARGIN), 0])
                    R = box_center.rotation() @ RotationMatrix.MakeXRotation(np.pi/2)
                elif i == 3:
                    p = box_center.translation() + box_pose.rotation() @ np.array([0, -(BOX_DIM/2 + PREPICK_MARGIN), 0])
                    R = box_center.rotation() @ RotationMatrix.MakeXRotation(-np.pi/2)
                elif i == 4:
                    p = box_center.translation() + box_pose.rotation() @ np.array([0, 0, (BOX_DIM/2 + PREPICK_MARGIN)])
                    R = box_center.rotation() @ RotationMatrix.MakeXRotation(np.pi)
                else:
                    p = box_center.translation() + box_pose.rotation() @ np.array([0, 0, -(BOX_DIM/2 + PREPICK_MARGIN)])
                    R = box_center.rotation()

                X = RigidTransform(R, p)
                q, ik_success = ik(self.plant, self.plant_context, X, regions=source_regions, pose_as_constraint=False)
                if ik_success:
                    pick_regions[Point(q)] = (box_body_idx, X)
                    if self.DEBUG:
                        self.meshcat.SetObject(f"Pick_Poses/{box_body_idx}_{i}", Sphere(0.03), Rgba(0.75, 0.0, 0.0))
                        self.meshcat.SetTransform(f"Pick_Poses/{box_body_idx}_{i}", X)
                        AddMeshcatTriad(self.meshcat, f"Pick_Poses/{box_body_idx}_pose_{i}", X_PT=X, opacity=0.5)

        print(f"{len(pick_regions)} viable pre-pick poses found.")
        return pick_regions