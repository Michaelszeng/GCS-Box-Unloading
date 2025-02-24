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
        max_vertices=1e3,
        goal_sample_frequency=0.05,
        always_swap=False,
        timeout=np.inf,
        index=0,
        draw_rrt=True,
    ):
        self.step_size = step_size
        self.check_size = check_size
        self.max_vertices = int(max_vertices)
        self.goal_sample_frequency = goal_sample_frequency
        self.always_swap = always_swap
        self.timeout = timeout
        self.idx = index
        self.draw_rrt = draw_rrt
        assert self.goal_sample_frequency >= 0
        assert self.goal_sample_frequency <= 1

class RRT:
    def __init__(self, RandomConfig, ValidityChecker, Distance=None, ForwardKinematics=None, meshcat=None):
        self.RandomConfig = RandomConfig
        self.ValidityChecker = ValidityChecker
        if Distance is None:
            self.Distance = lambda x, y: np.linalg.norm(x - y)
        else:
            self.Distance = Distance

        self.options = None
        self.tree = None
        self.ForwardKinematics = ForwardKinematics
        self.meshcat = meshcat

    def plan(self, start, goal, options):
        t0 = time.time()
        self.options = options
        self.tree = nx.Graph()
        self.tree.add_node(0, q=start)
        success = False
        
        ambient_dim = len(start)

        if self.meshcat:
            visualize = True
   
            self.meshcat.SetObject(
                f"rrt_{self.options.idx}/points/_start",
                Sphere(radius=0.01 if ambient_dim==3 else 0.005),
                rgba=Rgba(0, 1, 0, 1),
            )
            self.meshcat.SetTransform(
                f"rrt_{self.options.idx}/points/_start", RigidTransform(start if ambient_dim==3 else self.ForwardKinematics(start))
            )

            self.meshcat.SetObject(
                f"rrt_{self.options.idx}/points/_goal",
                Sphere(radius=0.01 if ambient_dim==3 else 0.005),
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
            if len(self.tree) >= self.options.max_vertices or success == True:
                break
            sample_goal = np.random.random() < self.options.goal_sample_frequency
            q_subgoal = goal.copy() if sample_goal else self.RandomConfig()
            q_near_idx = self._nearest_idx(q_subgoal)
            while len(self.tree) < self.options.max_vertices:
                status = self._extend(q_near_idx, q_subgoal, visualize, ambient_dim)
                q_new = self.tree.nodes[len(self.tree) - 1]["q"]
                if self.Distance(q_new, goal) <= self.options.step_size:
                    success = True
                    break
                if status == "stopped" or status == "reached":
                    break
                q_near_idx = len(self.tree) - 1
            vertices.update(len(self.tree) - old_tree_size)

        if success:
            goal_idx = len(self.tree)
            self.tree.add_node(goal_idx, q=goal)
            self.tree.add_edge(goal_idx - 1, goal_idx)
            start_idx = 0
            return self._path(start_idx, goal_idx, visualize, ambient_dim)
        else:
            return []

    def _nearest_idx(self, q_subgoal):
        dists = [
            self.Distance(q_subgoal, self.tree.nodes[i]["q"])
            for i in range(len(self.tree))
        ]
        return np.argmin(dists)

    def _furthest_idx(self, q_subgoal):
        dists = [
            self.Distance(q_subgoal, self.tree.nodes[i]["q"])
            for i in range(len(self.tree))
        ]
        return np.argmax(dists)

    def _extend(self, q_near_idx, q_subgoal, visualize, ambient_dim):
        q_near = self.tree.nodes[q_near_idx]["q"]
        step = q_subgoal - q_near
        unit_step = step / self.Distance(q_near, q_subgoal)
        q_new = q_near + self.options.step_size * unit_step
        validity_step = self.options.check_size * unit_step
        if self.ValidityChecker(q_new):
            for i in range(1, int(self.options.step_size / self.options.check_size)):
                if not self.ValidityChecker(q_near + i * validity_step):
                    return "stopped"
            q_new_idx = len(self.tree)
            self.tree.add_node(q_new_idx, q=q_new)
            self.tree.add_edge(q_near_idx, q_new_idx)

            if visualize and self.options.draw_rrt:
                self.meshcat.SetLine(
                    f"rrt_{self.options.idx}/edges/({q_near_idx:03d},{q_new_idx:03d})",
                    np.hstack((q_near.reshape(3, 1), q_new.reshape(3, 1))) if ambient_dim==3 else
                    np.hstack((self.ForwardKinematics(q_near).reshape(3, 1), self.ForwardKinematics(q_new).reshape(3, 1))),
                    rgba=Rgba(0, 0, 1, 1),
                )
                self.meshcat.SetObject(
                    f"rrt_{self.options.idx}/points/{q_new_idx:03d}",
                    Sphere(radius=0.01 if ambient_dim==3 else 0.005),
                    rgba=Rgba(0, 0, 1, 1),
                )
                self.meshcat.SetTransform(
                    f"rrt_{self.options.idx}/points/{q_new_idx:03d}", RigidTransform(q_new if ambient_dim==3 else self.ForwardKinematics(q_new))
                )
            
            dist_to_subgoal = self.Distance(q_new, q_subgoal)
            if dist_to_subgoal <= self.options.step_size:
                step = q_subgoal - q_new
                unit_step = step / dist_to_subgoal
                validity_step = self.options.check_size * unit_step
                for i in range(1, int(dist_to_subgoal / self.options.check_size)):
                    if not self.ValidityChecker(q_new + i * validity_step):
                        return "stopped"
                return "reached"
            else:
                return "extended"
        else:
            return "stopped"

    def _path(self, i, j, visualize, ambient_dim):
        path_idx = nx.shortest_path(self.tree, source=i, target=j)
        path = [self.tree.nodes[idx]["q"] for idx in path_idx]
        
        if visualize:
            for idx in range(len(path) - 1):
                q_current = path[idx]
                q_next = path[idx + 1]

                self.meshcat.SetLine(
                    f"path_{self.options.idx}/edges/({path_idx[idx]:03d},{path_idx[idx+1]:03d})",
                    np.hstack((q_current.reshape(3, 1), q_next.reshape(3, 1))) if ambient_dim==3 else
                    np.hstack((self.ForwardKinematics(q_current).reshape(3, 1), self.ForwardKinematics(q_next).reshape(3, 1))),
                    line_width=2.0,
                    rgba=Rgba(1, 0, 0, 1),  # Red lines
                )
                self.meshcat.SetObject(
                    f"path_{self.options.idx}/points/{path_idx[idx]:03d}",
                    Sphere(radius=0.011 if ambient_dim==3 else 0.006),
                    rgba=Rgba(1, 0, 0, 1),
                )
                self.meshcat.SetTransform(
                    f"path_{self.options.idx}/points/{path_idx[idx]:03d}", RigidTransform(q_current)
                )

            # Also plot the last node
            q_last = path[-1]
            self.meshcat.SetObject(
                f"path_{self.options.idx}/points/{path_idx[-1]}", Sphere(radius=0.011 if ambient_dim==3 else 0.006), rgba=Rgba(1, 0, 0, 1)
            )
            self.meshcat.SetTransform(f"path_{self.options.idx}/points/{path_idx[-1]}", RigidTransform(q_last.reshape(3, 1) if ambient_dim==3 else self.ForwardKinematics(q_last).reshape(3, 1)))
            
        return path

    def furthest_path(self):
        return self._path(0, self._furthest_idx(self.tree.nodes[0]["q"]))

    def nodes(self):
        return np.array([self.tree.nodes[i]["q"] for i in range(len(self.tree))])

    def adj_mat(self):
        return nx.adjacency_matrix(self.tree).toarray()


class BiRRT:
    def __init__(self, RandomConfig, ValidityChecker, Distance=None, ForwardKinematics=None, meshcat=None):
        """
        RandomConfig is a function to generate a random q

        ValidityChecker is a function to check th collision-free-ness of a q

        Distance is a lambda x,y function (Euclidean distance if left as None)
        """
        self.RandomConfig = RandomConfig
        self.ValidityChecker = ValidityChecker
        if Distance is None:
            self.Distance = lambda x, y: np.linalg.norm(x - y)
        else:
            self.Distance = Distance

        self.options = None
        self.tree_a = None
        self.tree_b = None
        self.ForwardKinematics = ForwardKinematics
        self.meshcat = meshcat

    def plan(self, start, goal, options):
        t0 = time.time()

        self.options = options
        self.tree_a = nx.Graph()
        self.tree_a.add_node(0, q=start)
        self.tree_b = nx.Graph()
        self.tree_b.add_node(0, q=goal)

        ambient_dim = len(start)

        if self.meshcat:
            visualize = True

            if self.options.draw_rrt:
                self.meshcat.SetObject(
                    f"rrt_{self.options.idx}/points/_start",
                    Sphere(radius=0.01 if ambient_dim==3 else 0.005),
                    rgba=Rgba(0, 1, 0, 1),
                )
                self.meshcat.SetTransform(
                    f"rrt_{self.options.idx}/points/_start", RigidTransform(start if ambient_dim==3 else self.ForwardKinematics(start))
                )

                self.meshcat.SetObject(
                    f"rrt_{self.options.idx}/points/_goal",
                    Sphere(radius=0.01 if ambient_dim==3 else 0.005),
                    rgba=Rgba(0, 1, 0, 1),
                )
                self.meshcat.SetTransform(
                    f"rrt_{self.options.idx}/points/_goal", RigidTransform(goal if ambient_dim==3 else self.ForwardKinematics(goal))
                )

        success = False
        vertices = tqdm(total=self.options.max_vertices, position=1, desc="Vertices")
        while True:
            if time.time() - t0 > self.options.timeout:
                break

            old_tree_size = len(self.tree_a) + len(self.tree_b)
            if old_tree_size >= self.options.max_vertices or success == True:
                break

            q_subgoal = self.RandomConfig()
            q_near_idx = self._nearest_idx(self.tree_a, q_subgoal)
            nodes_added = 0
            while len(self.tree_a) + len(self.tree_b) < self.options.max_vertices:
                status = self._extend(self.tree_a, q_near_idx, q_subgoal, visualize, ambient_dim)
                if status == "stopped":
                    break
                else:
                    nodes_added += 1
                if status == "reached":
                    break
                q_near_idx = len(self.tree_a) - 1
            if nodes_added == 0:
                if self.options.always_swap:
                    self.tree_a, self.tree_b = self.tree_b, self.tree_a
                continue
            vertices.update(len(self.tree_a) + len(self.tree_b) - old_tree_size)
            old_tree_size = len(self.tree_a) + len(self.tree_b)

            selected = np.random.randint(1, nodes_added + 1)
            q_subgoal_idx = len(self.tree_a) - selected
            q_subgoal = self.tree_a.nodes[q_subgoal_idx]["q"]
            q_near_idx = self._nearest_idx(self.tree_b, q_subgoal)

            while len(self.tree_a) + len(self.tree_b) < self.options.max_vertices:
                status = self._extend(self.tree_b, q_near_idx, q_subgoal, visualize, ambient_dim)
                if status == "stopped":
                    break
                elif status == "reached":
                    success = True
                    break
                q_near_idx = len(self.tree_b) - 1
            vertices.update(len(self.tree_a) + len(self.tree_b) - old_tree_size)
            self.tree_a, self.tree_b = self.tree_b, self.tree_a

        if success:
            path_a = self._path(self.tree_a, 0, len(self.tree_a) - 1)
            path_b = self._path(self.tree_b, q_subgoal_idx, 0)
            path = path_a + path_b
            self._visualize_path(path, visualize, ambient_dim)
            if np.linalg.norm(path[0] - start) > 1e-15:
                path.reverse()
            return path
        else:
            return []

    def _nearest_idx(self, tree, q_subgoal):
        dists = [self.Distance(q_subgoal, tree.nodes[i]["q"]) for i in range(len(tree))]
        return np.argmin(dists)

    def _furthest_idx(self, tree, q_subgoal):
        dists = [self.Distance(q_subgoal, tree.nodes[i]["q"]) for i in range(len(tree))]
        return np.argmax(dists)

    def _extend(self, tree, q_near_idx, q_subgoal, visualize, ambient_dim):
        q_near = tree.nodes[q_near_idx]["q"]
        step = q_subgoal - q_near
        unit_step = step / self.Distance(q_near, q_subgoal)
        q_new = q_near + self.options.step_size * unit_step
        validity_step = self.options.check_size * unit_step
        if self.ValidityChecker(q_new):
            for i in range(1, int(self.options.step_size / self.options.check_size)):
                if not self.ValidityChecker(q_near + i * validity_step):
                    return "stopped"
            q_new_idx = len(tree)
            tree.add_node(q_new_idx, q=q_new)
            tree.add_edge(q_near_idx, q_new_idx)

            if visualize and self.options.draw_rrt:
                if tree == self.tree_a:
                    which_tree = "a"
                else:
                    which_tree = "b"
                
                self.meshcat.SetLine(
                    f"rrt_{self.options.idx}/{which_tree}/edges/({q_near_idx:03d},{q_new_idx:03d})",
                    np.hstack((q_near.reshape(3, 1), q_new.reshape(3, 1))) if ambient_dim==3 else
                    np.hstack((self.ForwardKinematics(q_near).reshape(3, 1), self.ForwardKinematics(q_new).reshape(3, 1))),
                    rgba=Rgba(0, 0, 1, 1),
                )
                self.meshcat.SetObject(
                    f"rrt_{self.options.idx}/{which_tree}/points/{q_new_idx:03d}",
                    Sphere(radius=0.01 if ambient_dim==3 else 0.005),
                    rgba=Rgba(0, 0, 1, 1),
                )
                self.meshcat.SetTransform(
                    f"rrt_{self.options.idx}/{which_tree}/points/{q_new_idx:03d}", RigidTransform(q_new if ambient_dim==3 else self.ForwardKinematics(q_new))
                )

            dist_to_subgoal = self.Distance(q_new, q_subgoal)
            if dist_to_subgoal <= self.options.step_size:
                step = q_subgoal - q_new
                unit_step = step / dist_to_subgoal
                validity_step = self.options.check_size * unit_step
                for i in range(1, int(dist_to_subgoal / self.options.check_size)):
                    if not self.ValidityChecker(q_new + i * validity_step):
                        return "stopped"
                return "reached"
            else:
                return "extended"
        else:
            return "stopped"

    def _visualize_path(self, path, visualize, ambient_dim):
        if visualize:
            for idx in range(len(path) - 1):
                q_current = path[idx]
                q_next = path[idx + 1]

                self.meshcat.SetLine(
                    f"path_{self.options.idx}/edges/({idx:03d},{idx+1:03d})",
                    np.hstack((q_current.reshape(3, 1), q_next.reshape(3, 1))) if ambient_dim==3 else
                    np.hstack((self.ForwardKinematics(q_current).reshape(3, 1), self.ForwardKinematics(q_next).reshape(3, 1))),
                    line_width=2.0,
                    rgba=Rgba(1, 0, 0, 1),  # Red lines
                )
                self.meshcat.SetObject(
                    f"path_{self.options.idx}/points/{idx:03d}",
                    Sphere(radius=0.011 if ambient_dim==3 else 0.006),
                    rgba=Rgba(1, 0, 0, 1),
                )
                self.meshcat.SetTransform(
                    f"path_{self.options.idx}/points/{idx:03d}", RigidTransform(q_current if ambient_dim==3 else self.ForwardKinematics(q_current))
                )

            # Also plot the last node
            q_last = path[-1]
            self.meshcat.SetObject(
                f"path_{self.options.idx}/points/-1", Sphere(radius=0.011 if ambient_dim==3 else 0.006), rgba=Rgba(1, 0, 0, 1)
            )
            self.meshcat.SetTransform(f"path_{self.options.idx}/points/-1", RigidTransform(q_last if ambient_dim==3 else self.ForwardKinematics(q_last)))

    def _path(self, tree, i, j):
        path_idx = nx.shortest_path(tree, source=i, target=j)
        path = [tree.nodes[idx]["q"] for idx in path_idx]
        return path

    def furthest_path(self):
        path_a = self._path(
            self.tree_a, 0, self._furthest_idx(self.tree_a, self.tree_a.nodes[0]["q"])
        )
        path_b = self._path(
            self.tree_b, 0, self._furthest_idx(self.tree_b, self.tree_b.nodes[0]["q"])
        )
        return path_a + path_b

    def nodes(self):
        nodes_a = np.array([self.tree_a.nodes[i]["q"] for i in range(len(self.tree_a))])
        nodes_b = np.array([self.tree_b.nodes[i]["q"] for i in range(len(self.tree_b))])
        return np.vstack((nodes_a, nodes_b))

    def adj_mat(self):
        adj_mat_a = nx.adjacency_matrix(self.tree_a).toarray()
        adj_mat_b = nx.adjacency_matrix(self.tree_b).toarray()
        full_adj_mat = scipy.linalg.block_diag(adj_mat_a, adj_mat_b)
        idx = len(self.tree_a) - 1
        full_adj_mat[idx, -1] = full_adj_mat[-1, idx] = 1
        return full_adj_mat
