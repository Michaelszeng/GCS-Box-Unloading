"""
Template file that shows how to build a generic MultibodyPlant containing one of
the 9 test scenes.
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
    RandomGenerator,
    PointCloud,
    Rgba,
    Quaternion,
    RigidTransform,
    RationalForwardKinematics,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import scenario_yaml_for_iris
from utils import ik
from rrt import *

import numpy as np
import importlib
from scipy.spatial.transform import Rotation
from scipy.sparse import find
import pickle
import time

# TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
# TEST_SCENE = "BOXUNLOADING"

rng = RandomGenerator(1234)

src_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_directory = os.path.dirname(src_directory)
data_directory = os.path.join(parent_directory)
scene_yaml_file = os.path.join(data_directory, "data", "iris_benchmarks_scenes_urdf", "yamls", TEST_SCENE + ".dmd.yaml")

meshcat = StartMeshcat()


robot_diagram_builder = RobotDiagramBuilder()
parser = robot_diagram_builder.parser()
iris_environement_assets = os.path.join(data_directory, "data", "iris_benchmarks_scenes_urdf", "iris_environments", "assets")
parser.package_map().Add("iris_environments",iris_environement_assets)
if TEST_SCENE == "BOXUNLOADING":
    robot_model_instances = parser.AddModelsFromString(scenario_yaml_for_iris, ".dmd.yaml")
else:
    robot_model_instances = parser.AddModels(scene_yaml_file)
plant = robot_diagram_builder.plant()
plant.Finalize()
num_robot_positions = plant.num_positions()
AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
diagram = robot_diagram_builder.Build()
context = diagram.CreateDefaultContext()

# Roll forward sim a bit to show the visualization
simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
plant_context = diagram.GetMutableSubsystemContext(plant, context)

simulator.AdvanceTo(0.001)

collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)

if TEST_SCENE == "3DOFFLIPPER":
    joint_control = True
if TEST_SCENE == "5DOFUR3":
    joint_control = True
    ee_frame = plant.GetFrameByName("ur_ee_link")
    ee_body = plant.GetBodyByName("ur_ee_link")
if TEST_SCENE == "6DOFUR3":
    ee_frame = plant.GetFrameByName("ur_ee_link")
    ee_body = plant.GetBodyByName("ur_ee_link")
if TEST_SCENE == "7DOFIIWA":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
    ee_body = plant.GetBodyByName("iiwa_link_7")
if TEST_SCENE == "7DOFBINS":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
    ee_body = plant.GetBodyByName("iiwa_link_7")
if TEST_SCENE == "7DOF4SHELVES":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
    ee_body = plant.GetBodyByName("iiwa_link_7")
if TEST_SCENE == "14DOFIIWAS":
    print("Teleop does not yet work for 14DOFIIWAS.")
if TEST_SCENE == "15DOFALLEGRO":
    joint_control = True
if TEST_SCENE == "BOXUNLOADING":
    ee_frame = plant.GetFrameByName("arm_eef")
    ee_body = plant.GetBodyByName("arm_eef")

pickle_file = f'{TEST_SCENE}_endpts.pkl'
with open(pickle_file, 'rb') as f:
    endpts = pickle.load(f)

def check_collision(q):
    return not collision_checker.CheckConfigCollisionFree(q)

print(f"Start: {endpts['start_pts'][0]}")
print(f"End: {endpts['end_pts'][0]}")

tree = BiRRT(tuple(endpts["start_pts"][0]), 
            tuple(endpts["end_pts"][0]),
            plant.GetPositionLowerLimits(),
            plant.GetPositionUpperLimits(),
            StraightLineCollisionChecker(check_collision))

tree_connected = tree.build_tree(max_iter = int(1e3))
print(tree_connected)

# construct the RationalForwardKinematics of this plant. This object handles the
# computations for the forward kinematics in the tangent-configuration space
Ratfk = RationalForwardKinematics(plant)
q_star = np.zeros(num_robot_positions)

vis_bundle = vis_utils.VisualizationBundle(
    diagram, context, plant, plant_context,
    Ratfk, meshcat, q_star
)
# if tree_connected:
tree.draw_tree(vis_bundle, ee_body, prefix = f"bi_rrt")
time.sleep(10)