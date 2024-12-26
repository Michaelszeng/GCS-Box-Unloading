"""
Template file that allows teleoperation of robots in all 9 scenes.
"""

from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    MeshcatVisualizer,
    LeafSystem,
    Simulator,
    RigidTransform,
    MultibodyPlant,
    RobotDiagramBuilder,
    Parser,
    InverseDynamicsController,
    ConstantVectorSource,
    BasicVector,
    Rgba,
    RotationMatrix,
    RollPitchYaw,
    InverseKinematics,
    Solve,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import scenario_yaml_welded_trailer
from utils import diagram_visualize_connections

import numpy as np
import time
import importlib
import argparse

# TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
# TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
TEST_SCENE = "BOXUNLOADING"

src_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(src_directory)
data_directory = os.path.join(parent_directory)
scene_yaml_file = os.path.join(data_directory, "data", "iris_benchmarks_scenes_urdf", "yamls", TEST_SCENE + ".dmd.yaml")


class VectorSplitter(LeafSystem):
    """
    Simple LeafSystem that takes a vector input of size out1 + out2 and splits 
    it into 2 output vectors of size out1 and out2. This is used for multi-robot
    plants so the output of one controller can control both robots.
    """
    def __init__(self, out1, out2=0):
        super().__init__()
        self.out1 = out1
        self.out2 = out2
        self.DeclareVectorInputPort("input", BasicVector(out1 + out2))
        self.DeclareVectorOutputPort("output_1", BasicVector(out1), self.Output1)
        if out2 > 0:
            self.DeclareVectorOutputPort("output_2", BasicVector(out2), self.Output2)

    def Output1(self, context, output):
        input_vector = self.get_input_port(0).Eval(context)
        output.SetFromVector(input_vector[:self.out1])  # return first `out1` elements

    def Output2(self, context, output):
        input_vector = self.get_input_port(0).Eval(context)
        output.SetFromVector(input_vector[self.out1:])  # return latter `out2` elements


parser = argparse.ArgumentParser()
parser.add_argument('--joint_control', default='F', help="T/F; whether to control joint positions (instead of xyz rpy)")
args = parser.parse_args()
joint_control = (args.joint_control == 'T')


meshcat = StartMeshcat()

builder = DiagramBuilder()

if TEST_SCENE == "BOXUNLOADING":
    scenario = load_scenario(data=scenario_yaml_welded_trailer)
else:
    scenario = load_scenario(filename=scene_yaml_file)

# Hardware station setup
station = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,

    # This is to be able to load our own models from a local path
    # we can refer to this using the "package://" URI directive
    parser_preload_callback=lambda parser: parser.package_map().Add("iris_environments", os.path.join(data_directory, "data", "iris_benchmarks_scenes_urdf", "iris_environments", "assets")),
))
scene_graph = station.GetSubsystemByName("scene_graph")
plant = station.GetSubsystemByName("plant")

num_robot_positions = plant.num_positions()
default_joint_positions = plant.GetPositions(plant.CreateDefaultContext())

if TEST_SCENE == "3DOFFLIPPER":
    joint_control = True
if TEST_SCENE == "5DOFUR3":
    joint_control = True
    ee_frame = plant.GetFrameByName("ur_ee_link")
if TEST_SCENE == "6DOFUR3":
    ee_frame = plant.GetFrameByName("ur_ee_link")
if TEST_SCENE == "7DOFIIWA":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
if TEST_SCENE == "7DOFBINS":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
if TEST_SCENE == "7DOF4SHELVES":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
if TEST_SCENE == "14DOFIIWAS":
    print("Teleop does not yet work for 14DOFIIWAS.")
if TEST_SCENE == "15DOFALLEGRO":
    joint_control = True
if TEST_SCENE == "BOXUNLOADING":
    ee_frame = plant.GetFrameByName("arm_eef")

if joint_control:
    # Add slider for each joint
    for i in range(num_robot_positions):
        meshcat.AddSlider(f'q{i}', -np.pi, np.pi, 0.01, default_joint_positions[i])
else:
    default_pose = plant.CalcRelativeTransform(plant.CreateDefaultContext(), plant.world_frame(), ee_frame)
    meshcat.AddSlider('x', -1.0, 2.0, 0.01, default_pose.translation()[0])
    meshcat.AddSlider('y', -1.0, 1.0, 0.01, default_pose.translation()[1])
    meshcat.AddSlider('z', 0.0, 3.0, 0.01, default_pose.translation()[2])
    meshcat.AddSlider('roll', -np.pi, np.pi, 0.01, default_pose.rotation().ToRollPitchYaw().vector()[0])
    meshcat.AddSlider('pitch', -np.pi, np.pi, 0.01, default_pose.rotation().ToRollPitchYaw().vector()[1])
    meshcat.AddSlider('yaw', -np.pi, np.pi, 0.01, default_pose.rotation().ToRollPitchYaw().vector()[2])
meshcat.AddButton("Close")

# Figure out how many robots there are and how many joints each has
model_instances_indices_with_actuators = {}
for actuator_idx in plant.GetJointActuatorIndices():
    robot_model_instance_idx = plant.get_joint_actuator(actuator_idx).model_instance()
    if robot_model_instance_idx not in model_instances_indices_with_actuators.keys():
        model_instances_indices_with_actuators[robot_model_instance_idx] = 1
    else:
        model_instances_indices_with_actuators[robot_model_instance_idx] += 1

# Add controller and splitter (for when there are multiple robots)
controller = builder.AddSystem(InverseDynamicsController(plant, [100]*num_robot_positions, [0]*num_robot_positions, [50]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
control_splitter = builder.AddSystem(VectorSplitter(*model_instances_indices_with_actuators.values()))

# Set controller desired state
builder.Connect(station.GetOutputPort("state"), controller.GetInputPort("estimated_state"))
builder.Connect(controller.GetOutputPort("generalized_force"), control_splitter.GetInputPort("input"))
builder.Connect(TODO: FILL IN HERE, controller.GetInputPort("desired_state"))

# Set controller desired accel
builder.Connect(TODO: FILL IN HERE, controller.GetInputPort("desired_acceleration"))

# Connect each output of the splitter to the actuation input for each robot
for i, (robot_model_instance_idx, num_joints) in enumerate(model_instances_indices_with_actuators.items()):
    builder.Connect(control_splitter.GetOutputPort(f"output_{i+1}"), station.GetInputPort(f"{plant.GetModelInstanceName(robot_model_instance_idx)}_actuation"))

diagram = builder.Build()
diagram_visualize_connections(diagram, "diagram_.svg")
context = diagram.CreateDefaultContext()

simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(simulator_context)
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)

# Main simulation loop
ctr = 0
while not meshcat.GetButtonClicks("Close"):
    simulator.AdvanceTo(simulator_context.get_time() + 0.01)