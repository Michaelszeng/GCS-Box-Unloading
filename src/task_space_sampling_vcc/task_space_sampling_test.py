"""
Main file to run and test VisibilityGraph and Clique Covers on a simulation
environment with a seeded sampling distribution.
"""

from pydrake.all import (
    StartMeshcat,
    AddDefaultVisualization,
    Simulator,
    VisibilityGraph,
    RobotDiagramBuilder,
    VPolytope,
    HPolyhedron,
    SceneGraphCollisionChecker,
    ConfigurationSpaceObstacleCollisionChecker,
    RandomGenerator,
    PointCloud,
    Rgba,
    Quaternion,
    RigidTransform,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import robot_yaml
from utils import ik

import numpy as np
import importlib
from scipy.spatial.transform import Rotation
from scipy.sparse import find

# TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
# TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
TEST_SCENE = "BOXUNLOADING"

NUM_SAMPLES = 1000

task_space_sampling_region = importlib.import_module(f"task_space_sampling_regions.{TEST_SCENE}")

scene_yaml_file = os.path.dirname(os.path.abspath(__file__)) + "/../../data/iris_benchmarks_scenes_urdf/yamls/" + TEST_SCENE + ".dmd.yaml"


meshcat = StartMeshcat()

hpoly = HPolyhedron(VPolytope(np.array(task_space_sampling_region.sampling_bounds).T))


robot_diagram_builder = RobotDiagramBuilder()
parser = robot_diagram_builder.parser()
parser.package_map().Add("iris_environments", os.path.dirname(os.path.abspath(__file__)) + "/../../data/iris_benchmarks_scenes_urdf/iris_environments/assets")
if TEST_SCENE == "BOXUNLOADING":
    robot_model_instances = parser.AddModelsFromString(robot_yaml, ".dmd.yaml")
else:
    robot_model_instances = parser.AddModels(scene_yaml_file)
plant = robot_diagram_builder.plant()
plant.Finalize()
AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
diagram = robot_diagram_builder.Build()

simulator = Simulator(diagram)
simulator.AdvanceTo(0.001)

plant_context = plant.CreateDefaultContext()

num_robot_positions = plant.num_positions()

collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)
config_obstacle_collision_checker = ConfigurationSpaceObstacleCollisionChecker(collision_checker, [])

rng = RandomGenerator(1234)

points_3d = np.zeros((3, NUM_SAMPLES))
points = np.zeros((num_robot_positions, NUM_SAMPLES))
last_polytope_sample = hpoly.UniformSample(rng, hpoly.ChebyshevCenter())
for i in range(np.shape(points)[1]):
    last_polytope_sample = hpoly.UniformSample(rng, last_polytope_sample)
    quaternion_sample = Rotation.random().as_quat()  # [x, y, z, w] order
    quaternion_sample = Quaternion(quaternion_sample[3], quaternion_sample[0], quaternion_sample[1], quaternion_sample[2])
    q, ik_success = ik(plant, plant_context, RigidTransform(quaternion_sample, last_polytope_sample), translation_error=0, rotation_error=0.05)

    while not ik_success or not config_obstacle_collision_checker.CheckConfigCollisionFree(q):
        last_polytope_sample = hpoly.UniformSample(rng, last_polytope_sample)
        quaternion_sample = Rotation.random().as_quat()  # [x, y, z, w] order
        quaternion_sample = Quaternion(quaternion_sample[3], quaternion_sample[0], quaternion_sample[1], quaternion_sample[2])
        q, ik_success = ik(plant, plant_context, RigidTransform(quaternion_sample, last_polytope_sample), translation_error=0, rotation_error=0.05)

    points_3d[:,i] = last_polytope_sample
    points[:,i] = q


# Display convex hull of sampling region
from scipy.spatial import ConvexHull
hull = ConvexHull(np.array(task_space_sampling_region.sampling_bounds))
vertices = np.array(task_space_sampling_region.sampling_bounds).T  # Transpose to get 3xN matrix
faces = hull.simplices.T  # Transpose to get 3xM matrix (each column is a triangle)
meshcat.SetTriangleMesh(
    path="/convex_polygon",
    vertices=vertices,
    faces=faces,
    rgba=Rgba(r=0.5, g=0.1, b=0.1, a=0.1),
)

# Visualize Samples with PointCloud
pc = PointCloud(NUM_SAMPLES)
pc.mutable_xyzs()[:] = points_3d
meshcat.SetObject(f"samples", pc, point_size=0.05, rgba=Rgba(r=0.5, g=0.1, b=0.1, a=0.5))

visibility_graph = VisibilityGraph(config_obstacle_collision_checker, points)

row_indices, col_indices, _ = find(visibility_graph)
    
# Iterate over each pair of connected points
for i, j in zip(row_indices, col_indices):
    # Extract coordinates of the two points
    p1 = points_3d[:, i]
    p2 = points_3d[:, j]
    
    # Draw the line in Meshcat between the two points
    meshcat.SetLine(f"visibility/line_{i}_{j}", np.column_stack((p1, p2)), 0.01, Rgba(r=0.5, g=0.1, b=0.1, a=0.5))
    if i > 20: 
        break