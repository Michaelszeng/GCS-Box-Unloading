from pydrake.all import (
    RigidTransform,
    VPolytope,
)
from manipulation.meshcat_utils import AddMeshcatTriad

import numpy as np
import heapq

from utils import NUM_BOXES, BOX_DIM


class BoxSelectorGraph:
    """
    Graph where each node is a tuple representing a box, containing 
    (BodyIndex, RigidTransform).

    This is used to efficiently find the first box to remove from the pile in
    the truck trailer.
    """
    def __init__(self):
        self.adj_list = {}  # Adjacency list to store graph
        self.reverse_adj_list = {}  # Reverse adjacency list to track parents of each child
        self.removable_boxes_heap = []  # maintain min heap of removable boxes (nodes w/ no children), with value being x-coordinate
    

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
            print("Unable to find a box without boxes above it.")
            return None
        # Find box to remove
        ret = heapq.heappop(self.removable_boxes_heap)[1]
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
    def __init__(self, box_poses):
        self.box_selector_graph = BoxSelectorGraph()
        for box_body_idx, box_pose in box_poses.items():
            node = (box_body_idx, box_pose)
            self.box_selector_graph.add_node(node)


    def get_box_idx_to_pick(self, box_poses):
        """
        Simulate how a perception system might order the boxes to be picked.

        Firstly, only boxes without other boxes above it can be picked. Then, the
        boxes with the minimum x-pose (closest to the robot) will be selected.
        """
        box_projections = {}  # dict mapping box bodyIndex to VPolytope XY projection
        for box_body_idx, box_pose in box_poses.items():
            # Find vertices of projection of box onto XY plane
            box_corners = []
            for dx in [-BOX_DIM/2, BOX_DIM/2]:
                for dy in [-BOX_DIM/2, BOX_DIM/2]:
                    for dz in [-BOX_DIM/2, BOX_DIM/2]:
                        # Find coordinate of box corner in 3D
                        box_corner = box_pose.translation() + RigidTransform(box_pose.rotation(), [dx, dy,dz])
                        # Project box corner into XY plane by removing Z coordinate
                        box_corner_xy = np.array([box_corner[0], box_corner[1]])
                        box_corners.append(box_corner_xy)
            
            box_projections[box_body_idx] = VPolytope(np.array(box_corner))

        for box_body_idx, box_proj in box_projections:
            for other_box_body_idx, other_box_proj in box_projections:
                # box cannot be on top of itself
                if box_body_idx == other_box_body_idx:
                    continue



            
            
