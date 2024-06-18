from pydrake.all import (
    RigidTransform,
    VPolytope,
    StartMeshcat,
    Rgba
)
from manipulation.meshcat_utils import AddMeshcatTriad

import numpy as np
import heapq
from scipy.spatial import ConvexHull

from utils import NUM_BOXES, BOX_DIM


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


class PickPlanner:
    """
    A class to manage all picking logic, i.e. selecting which box to pick,
    which face to pick it up from.
    """
    def __init__(self, box_poses, DEBUG=True):
        """
        Initialize the BoxSelectorGraph data structure. This is done by checking
        which boxes vertically overlap and which boxes are closer to the robot
        in the world x-axis.

        box_poses: dictionary mapping BodyIndex: RigidTransform for each box.
        """
        self.box_selector_graph = BoxSelectorGraph()

        # Compute projections of boxes onto XY plane to more easily determine
        # which are vertically overlapping
        box_projections = {}  # dict mapping box bodyIndex to VPolytope XY projection
        for box_body_idx, box_pose in box_poses.items():
            # Find vertices of projection of box onto XY plane
            box_corners = []
            for dx in [-BOX_DIM/2, BOX_DIM/2]:
                for dy in [-BOX_DIM/2, BOX_DIM/2]:
                    for dz in [-BOX_DIM/2, BOX_DIM/2]:
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

        if DEBUG:
            # Render projections in Meshcat
            meshcat = StartMeshcat()
            meshcat.Set2dRenderMode(RigidTransform([0, 0, 1]), -4, 4, -4, 4)
            meshcat.SetProperty("/Axes", "visible", True)
            
            ctr = 0
            for vpoly in box_projections.values():
                points = np.array(self.sort_vertices_ccw(vpoly))
                print(points)

                meshcat.SetLine(f"vpoly_{ctr}", points, 2.0, Rgba(np.random.random(), np.random.random(), np.random.random()))
                ctr += 1


        # Now, determine which boxes vertically overlap and add corresponding nodes and edges to graph
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
                        self.box_selector_graph.add_edge((other_box_body_idx, box_poses[other_box_body_idx]),
                                                         (box_body_idx, box_poses[box_body_idx]))
                    else:
                        # other_box_body_idx is above box_body_idx
                        self.box_selector_graph.add_edge((box_body_idx, box_poses[box_body_idx]),
                                                         (other_box_body_idx, box_poses[other_box_body_idx]))
                        
        self.box_selector_graph.build_heap()


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

    def get_box_idx_to_pick(self):
        """
        Pick the first box to grab.
        """
        return self.box_selector_graph.remove_next_node()
