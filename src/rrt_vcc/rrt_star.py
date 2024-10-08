from pydrake.all import (
    Sphere,
    Rgba,
    RigidTransform,
)

import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import scipy
import time


class RRTOptions:
    def __init__(
        self,
        step_size=1e-1,
        check_size=1e-2,
        min_vertices=1e3,
        max_vertices=1e3,
        max_iters=1e4,
        goal_sample_frequency=0.05,
        neighbor_radius=0.2,  # Added for RRT*
        timeout=np.inf,
        index=0,
        draw_rrt=True,
    ):
        self.step_size = step_size
        self.check_size = check_size
        self.min_vertices = int(min_vertices)
        self.max_vertices = int(max_vertices)
        self.max_iters = int(max_iters)
        self.goal_sample_frequency = goal_sample_frequency
        self.neighbor_radius = neighbor_radius  # Added for RRT*
        self.timeout = timeout
        self.idx = index
        self.draw_rrt = draw_rrt
        assert self.goal_sample_frequency >= 0
        assert self.goal_sample_frequency <= 1


class RRTStar:
    def __init__(self, RandomConfig, ValidityChecker, Distance=None, ForwardKinematis=None, meshcat=None):
        self.RandomConfig = RandomConfig
        self.ValidityChecker = ValidityChecker
        if Distance is None:
            self.Distance = lambda x, y: np.linalg.norm(x - y)
        else:
            self.Distance = Distance

        self.options = None
        self.tree = None
        
        self.ForwardKinematics = ForwardKinematis
        self.meshcat = meshcat

    def plan(self, start, goal, options):
        t0 = time.time()
        self.options = options
        self.tree = nx.DiGraph()  # Changed to directed graph
        self.tree.add_node(0, q=start, cost=0.0)  # Initialize cost
        success = False
        goal_nodes = []  # To store nodes close to the goal

        visualize = False
        ambient_dim = len(start)
        if self.meshcat:
            visualize = True
            self.meshcat.SetObject(
                f"rrt_{self.options.idx}/points/_start",
                Sphere(radius=0.02 if ambient_dim==3 else 0.01),
                rgba=Rgba(0, 1, 0, 1),
            )
            self.meshcat.SetTransform(
                f"rrt_{self.options.idx}/points/_start", RigidTransform(start if ambient_dim==3 else self.ForwardKinematics(start))
            )

            self.meshcat.SetObject(
                f"rrt_{self.options.idx}/points/_goal",
                Sphere(radius=0.02 if ambient_dim==3 else 0.01),
                rgba=Rgba(0, 1, 0, 1),
            )
            self.meshcat.SetTransform(
                f"rrt_{self.options.idx}/points/_goal", RigidTransform(goal if ambient_dim==3 else self.ForwardKinematics(goal))
            )

        vertices = tqdm(total=self.options.max_vertices, position=1, desc="Vertices")
        while True:
            if time.time() - t0 > self.options.timeout:
                break
            old_tree_size = len(self.tree)
            if len(self.tree) >= self.options.max_vertices:
                break

            sample_goal = np.random.random() < self.options.goal_sample_frequency
            q_rand = goal.copy() if sample_goal else self.RandomConfig()

            q_nearest_idx = self._nearest_idx(q_rand)
            q_nearest = self.tree.nodes[q_nearest_idx]['q']

            q_new = self._steer(q_nearest, q_rand, self.options.step_size)

            if not self.ValidityChecker(q_new):
                continue

            if not self._check_path(q_nearest, q_new):
                continue

            # Find nearby nodes within neighbor_radius
            Near = self._near_indices(q_new, self.options.neighbor_radius)

            # Choose the best parent
            q_min_idx = q_nearest_idx
            c_min = self.tree.nodes[q_nearest_idx]['cost'] + self.Distance(q_nearest, q_new)

            for idx in Near:
                q_near = self.tree.nodes[idx]['q']
                c = self.tree.nodes[idx]['cost'] + self.Distance(q_near, q_new)
                if c < c_min and self._check_path(q_near, q_new):
                    q_min_idx = idx
                    c_min = c

            # Add q_new to the tree
            q_new_idx = len(self.tree)
            self.tree.add_node(q_new_idx, q=q_new, cost=c_min)
            self.tree.add_edge(q_min_idx, q_new_idx)

            if visualize and self.options.draw_rrt:
                q_parent = self.tree.nodes[q_min_idx]['q']
                self.meshcat.SetLine(
                    f"rrt_{self.options.idx}/edges/({q_min_idx:03d},{q_new_idx:03d})",
                    np.hstack((q_parent.reshape(3, 1), q_new.reshape(3, 1))) if ambient_dim==3 else 
                    np.hstack((self.ForwardKinematics(q_parent).reshape(3, 1), self.ForwardKinematics(q_new).reshape(3, 1))),
                    rgba=Rgba(0, 0, 1, 1),
                )
                if self.options.max_vertices <= 1000:
                    self.meshcat.SetObject(
                        f"rrt_{self.options.idx}/points/{q_new_idx:03d}",
                        Sphere(radius=0.01 if ambient_dim==3 else 0.005),
                        rgba=Rgba(0, 0, 1, 1),
                    )
                    self.meshcat.SetTransform(
                        f"rrt_{self.options.idx}/points/{q_new_idx:03d}", RigidTransform(q_new if ambient_dim==3 else self.ForwardKinematics(q_new))
                    )

            # Rewire the tree
            for idx in Near:
                if idx == q_min_idx:
                    continue
                q_near = self.tree.nodes[idx]['q']
                c = c_min + self.Distance(q_new, q_near)
                if c < self.tree.nodes[idx]['cost'] and self._check_path(q_new, q_near):
                    old_parent_idx = list(self.tree.predecessors(idx))[0]
                    self.tree.remove_edge(old_parent_idx, idx)
                    self.tree.add_edge(q_new_idx, idx)
                    self.tree.nodes[idx]['cost'] = c
                    self._update_costs(idx)

                    if visualize and self.options.draw_rrt:
                        # Remove old edge visualization
                        self.meshcat.Delete(f"rrt_{self.options.idx}/edges/({old_parent_idx:03d},{idx:03d})")
                        # Add new edge visualization
                        self.meshcat.SetLine(
                            f"rrt_{self.options.idx}/edges/({q_new_idx:03d},{idx:03d})",
                            np.hstack((q_new.reshape(3, 1), q_near.reshape(3, 1))) if ambient_dim==3 else
                            np.hstack((self.ForwardKinematics(q_new).reshape(3, 1), self.ForwardKinematics(q_near).reshape(3, 1))),
                            rgba=Rgba(0, 0, 1, 1),
                        )

            # Check if q_new is close to the goal
            if self.Distance(q_new, goal) <= self.options.step_size:
                goal_nodes.append(q_new_idx)
                
            # If we've reached goal and min_vertices has been reached, then just return
            if goal_nodes and len(self.tree) >= self.options.min_vertices:
                break
            
            vertices.update(len(self.tree) - old_tree_size)

        # Check if any goal nodes were found
        if goal_nodes:
            costs = [self.tree.nodes[idx]['cost'] + self.Distance(self.tree.nodes[idx]['q'], goal) for idx in goal_nodes]
            min_cost_idx = goal_nodes[np.argmin(costs)]
            # Add goal node
            goal_idx = len(self.tree)
            min_cost = self.tree.nodes[min_cost_idx]['cost'] + self.Distance(self.tree.nodes[min_cost_idx]['q'], goal)
            self.tree.add_node(goal_idx, q=goal, cost=min_cost)
            self.tree.add_edge(min_cost_idx, goal_idx)

            return self._path(0, goal_idx, visualize, ambient_dim)
        else:
            return []

    def _steer(self, q_from, q_to, step_size):
        dist = self.Distance(q_from, q_to)
        if dist <= step_size:
            return q_to.copy()
        else:
            return q_from + step_size * (q_to - q_from) / dist

    def _check_path(self, q_from, q_to):
        dist = self.Distance(q_from, q_to)
        steps = int(dist / self.options.check_size)
        if steps == 0:
            steps = 1
        for i in range(1, steps + 1):
            q = q_from + (i / steps) * (q_to - q_from)
            if not self.ValidityChecker(q):
                return False
        return True

    def _nearest_idx(self, q_rand):
        dists = [
            self.Distance(q_rand, self.tree.nodes[i]["q"])
            for i in range(len(self.tree))
        ]
        return np.argmin(dists)

    def _near_indices(self, q_new, radius):
        indices = []
        for idx in range(len(self.tree)):
            q = self.tree.nodes[idx]['q']
            if self.Distance(q_new, q) <= radius:
                indices.append(idx)
        return indices

    def _update_costs(self, idx):
        # Recursively update costs of descendants
        for child_idx in self.tree.successors(idx):
            parent_q = self.tree.nodes[idx]['q']
            child_q = self.tree.nodes[child_idx]['q']
            self.tree.nodes[child_idx]['cost'] = self.tree.nodes[idx]['cost'] + self.Distance(parent_q, child_q)
            self._update_costs(child_idx)

    def _path(self, start_idx, goal_idx, visualize, ambient_dim):
        path_idx = nx.shortest_path(self.tree, source=start_idx, target=goal_idx)
        path = [self.tree.nodes[idx]["q"] for idx in path_idx]

        if visualize:
            for idx in range(len(path) - 1):
                q_current = path[idx].reshape(ambient_dim, 1)
                q_next = path[idx + 1].reshape(ambient_dim, 1)

                self.meshcat.SetLine(
                    f"path_{self.options.idx}/edges/({path_idx[idx]:03d},{path_idx[idx+1]:03d})",
                    np.hstack((q_current, q_next)) if ambient_dim==3 else
                    np.hstack((self.ForwardKinematics(q_current).reshape(3,1), self.ForwardKinematics(q_next).reshape(3,1))),
                    line_width=2.0,
                    rgba=Rgba(1, 0, 0, 1),  # Red lines and points
                )
                self.meshcat.SetObject(
                    f"path_{self.options.idx}/points/{path_idx[idx]:03d}",
                    Sphere(radius=0.011 if ambient_dim==3 else 0.006),
                    rgba=Rgba(1, 0, 0, 1),
                )
                self.meshcat.SetTransform(
                    f"path_{self.options.idx}/points/{path_idx[idx]:03d}", RigidTransform(q_current if ambient_dim==3 else self.ForwardKinematics(q_current))
                )

            # Also plot the last node
            q_last = path[-1].reshape(ambient_dim, 1)
            self.meshcat.SetObject(
                f"path_{self.options.idx}/points/{path_idx[-1]}",
                Sphere(radius=0.01 if ambient_dim==3 else 0.005),
                rgba=Rgba(1, 0, 0, 1),
            )
            self.meshcat.SetTransform(f"path_{self.options.idx}/points/{path_idx[-1]}", RigidTransform(q_last if ambient_dim==3 else self.ForwardKinematics(q_last)))

        return path

    def furthest_path(self):
        return self._path(0, self._furthest_idx(self.tree.nodes[0]["q"]))

    def nodes(self):
        return np.array([self.tree.nodes[i]["q"] for i in range(len(self.tree))])

    def adj_mat(self):
        return nx.adjacency_matrix(self.tree).toarray()
