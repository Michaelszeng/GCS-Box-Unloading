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
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import NUM_BOXES, get_fast_box_poses, scenario_yaml_with_boxes, BOX_DIM, q_nominal, q_place_nominal, scenario_yaml, robot_yaml, scenario_yaml_for_iris, robot_pose, set_hydroelastic, set_up_scene, get_W_X_eef
from utils import ik
from iris import IrisRegionGenerator
from poses import get_grasp_poses, get_deposit_poses

import numpy as np
import importlib
from scipy.spatial.transform import Rotation
from scipy.sparse import find
import time

TEST_SCENE = "BOXUNLOADING"

rng = RandomGenerator(1234)
np.random.seed(1234)

meshcat = StartMeshcat()


robot_diagram_builder = RobotDiagramBuilder()
parser = robot_diagram_builder.parser()
robot_model_instances = parser.AddModelsFromString(scenario_yaml_with_boxes, ".dmd.yaml")

    
def add_suction_joints(parser):
    """
    Add joints between each box and eef to be able lock these later to simulate
    the gripper's suction. This called as part of the Hardware Station
    initialization routine.
    """
    plant = parser.plant()
    eef_model_idx = plant.GetModelInstanceByName("kuka")  # ModelInstanceIndex
    eef_body_idx = plant.GetBodyIndices(eef_model_idx)[-1]  # BodyIndex
    frame_parent = plant.get_body(eef_body_idx).body_frame()
    for i in range(NUM_BOXES):
        box_model_idx = plant.GetModelInstanceByName(f"Boxes/Box_{i}")  # ModelInstanceIndex
        box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex
        frame_child = plant.get_body(box_body_idx).body_frame()

        joint = QuaternionFloatingJoint(f"{eef_body_idx}-{box_body_idx}", frame_parent, frame_child)
        plant.AddJoint(joint)

add_suction_joints(parser)


plant = robot_diagram_builder.plant()

plant_config = MultibodyPlantConfig(
        discrete_contact_solver="sap",
        time_step=0.001
        # time_step=0.005
    )
ApplyMultibodyPlantConfig(plant_config, plant)


plant.Finalize()
AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
diagram = robot_diagram_builder.Build()


def set_up_scene_modified(plant, plant_context):
    fast_box_poses = get_fast_box_poses()  # Get pre-computed box poses

    # 'Remove' Top of truck trailer
    trailer_roof_model_idx = plant.GetModelInstanceByName("Truck_Trailer_Roof")  # ModelInstanceIndex
    trailer_roof_body_idx = plant.GetBodyIndices(trailer_roof_model_idx)[0]  # BodyIndex

    # Set poses for all boxes
    # Because of added floating joints between boxes and eef, we must express free body pose relative to eef
    W_X_eef = get_W_X_eef(plant, plant_context)
    for i in range(NUM_BOXES):
        box_model_idx = plant.GetModelInstanceByName(f"Boxes/Box_{i}")  # ModelInstanceIndex
        box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex

        plant.SetFreeBodyPose(plant_context, plant.get_body(box_body_idx), W_X_eef @ fast_box_poses[i])

    # Put Top of truck trailer back and lock it
    plant.SetFreeBodyPose(plant_context, plant.get_body(trailer_roof_body_idx), RigidTransform([0,0,0]))
    trailer_roof_joint_idx = plant.GetJointIndices(trailer_roof_model_idx)[0]  # JointIndex object
    trailer_roof_joint = plant.get_joint(trailer_roof_joint_idx)  # Joint object
    trailer_roof_joint.Lock(plant_context)
    

simulator = Simulator(diagram)

simulator_context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)


set_up_scene_modified(plant, plant_context)

num_robot_positions = plant.num_positions()

robot_model_instances = robot_model_instances[:2]
for x in robot_model_instances:
    print(plant.GetModelInstanceName(x))
print(robot_model_instances)
collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)

plant_no_boxes = MultibodyPlant(time_step=0.001)
Parser(plant_no_boxes).AddModelsFromString(robot_yaml, ".dmd.yaml")[0]  # ModelInstance object
plant_no_boxes.Finalize()
plant_no_boxes_context = plant_no_boxes.CreateDefaultContext()

# Roll forward sim a bit to show the visualization
simulator.AdvanceTo(0.001)



# RUN IRIS
seeds = get_grasp_poses() + [get_deposit_poses()[0]] + [RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.69, 0, 1.74]),
                                                    RigidTransform(RotationMatrix.MakeXRotation(np.pi), [1.5, 0, 1.0])]

# seeds = [RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.69, 0, 1.74]),
#          RigidTransform(RotationMatrix.MakeXRotation(np.pi), [1.5, 0, 1.0])]

# print(seeds)

regions_dict = {}
box_names = ["Boxes/Box_4", "Boxes/Box_16", "Boxes/Box_17", "Boxes/Box_12"]
for i, seed in enumerate(seeds):
    options = FastIrisOptions()
    options.random_seed = 0
    options.verbose = True
    options.require_sample_point_is_contained = True
    domain = HPolyhedron.MakeBox(plant_no_boxes.GetPositionLowerLimits(),
                                plant_no_boxes.GetPositionUpperLimits())
    kEpsilonEllipsoid = 1e-5
    q = ik(plant_no_boxes, plant_no_boxes_context, seed, translation_error=0, rotation_error=0.05, regions=None, pose_as_constraint=True)[0]
    clique_ellipse = Hyperellipsoid.MakeHypersphere(kEpsilonEllipsoid, q)
    region = FastIris(collision_checker, clique_ellipse, domain, options)

    regions_dict[f"set{i}"] = region
    
    box_model_idx = plant.GetModelInstanceByName(box_names[i])  # ModelInstanceIndex
    box_body_idx = plant.GetBodyIndices(box_names)[0]  # BodyIndex
    plant.SetFreeBodyPose(plant_context, plant.get_body(box_body_idx), RigidTransform([-100,0,0]))
    
    simulator.AdvanceTo(0.001*(i+2))
    
# Save regions
SaveIrisRegionsYamlFile("IRIS_REGIONS.yaml", regions_dict)


regions_dict = LoadIrisRegionsYamlFile("IRIS_REGIONS.yaml")

# Visualize IRIS regions
IrisRegionGenerator.visualize_connectivity(regions_dict, 0.0, output_file='IRIS_REGIONS.svg', skip_svg=False)
IrisRegionGenerator.visualize_iris_region(plant_no_boxes, plant_no_boxes_context, meshcat, regions_dict)

time.sleep(100)