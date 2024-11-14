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
    FastIris,
    FastIrisOptions,
    SaveIrisRegionsYamlFile,
    LoadIrisRegionsYamlFile,
    Hyperellipsoid,
    RotationMatrix,
    QuaternionFloatingJoint,
    ApplyMultibodyPlantConfig,
    MultibodyPlantConfig,
    MultibodyPlant,
    Parser,
    GeometrySet,
    Role,
    CollisionFilterDeclaration,
)

from manipulation.meshcat_utils import AddMeshcatTriad

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import NUM_BOXES, get_fast_box_poses, scenario_yaml_with_boxes, BOX_DIM, q_nominal, q_place_nominal, scenario_yaml, robot_yaml, scenario_yaml_for_iris, robot_pose, set_hydroelastic, set_up_scene, get_W_X_eef
from utils import ik
from iris import IrisRegionGenerator
from poses import get_grasp_poses, get_deposit_poses, grasp_q, deposit_q

import numpy as np
import importlib
from scipy.spatial.transform import Rotation
from scipy.sparse import find
import time


box_names = ["Boxes/Box_4", "Boxes/Box_16", "Boxes/Box_17", "Boxes/Box_12"]

meshcat = StartMeshcat()

rng = RandomGenerator(1234)
np.random.seed(1234)


robot_diagram_builder = RobotDiagramBuilder()
parser = robot_diagram_builder.parser()
scene_graph = robot_diagram_builder.scene_graph()
robot_model_instances = parser.AddModelsFromString(scenario_yaml_with_boxes, ".dmd.yaml")
plant = robot_diagram_builder.plant()
plant_config = MultibodyPlantConfig(
        discrete_contact_solver="sap",
        time_step=0.001
    )
ApplyMultibodyPlantConfig(plant_config, plant)

# Set poses for all boxes
fast_box_poses = get_fast_box_poses()  # Get pre-computed box poses
for j in range(NUM_BOXES):
    box_name = f"Boxes/Box_{j}"
    box_model_idx = plant.GetModelInstanceByName(box_name)  # ModelInstanceIndex
    box_frame = plant.GetFrameByName("Box_0_5_0_5_0_5", box_model_idx)
    plant.WeldFrames(plant.world_frame(), box_frame, fast_box_poses[j])


plant.Finalize()
AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
diagram = robot_diagram_builder.Build()

simulator = Simulator(diagram)

simulator_context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)


num_robot_positions = plant.num_positions()

robot_model_instances = robot_model_instances
collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)

# Roll forward sim a bit to show the visualization
simulator.AdvanceTo(0.001)

# RUN IRIS
seed = [ 0.31076813, -0.93517756,  1.876914,    1.18583468, 1.10499359, -0.00917231]

options = FastIrisOptions()
options.random_seed = 0
options.verbose = True
options.require_sample_point_is_contained = True
domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(),
                            plant.GetPositionUpperLimits())
kEpsilonEllipsoid = 1e-5
AddMeshcatTriad(meshcat, f"{0}", X_PT=seed)
q = ik(plant, plant_context, seed, translation_error=0, rotation_error=0.05, regions=None, pose_as_constraint=True)[0]
print(f"seed: {q.flatten()}")
clique_ellipse = Hyperellipsoid.MakeHypersphere(kEpsilonEllipsoid, q)
region = FastIris(collision_checker, clique_ellipse, domain, options)

print(region.PointInSet(q))
print(region.Projection(q))