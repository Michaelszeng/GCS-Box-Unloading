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
from scenario import scenario_yaml_for_iris
from utils import diagram_visualize_connections

import numpy as np
import time
import importlib
import argparse
import pickle

TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
# TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
# TEST_SCENE = "BOXUNLOADING"

scene_yaml_file = os.path.dirname(os.path.abspath(__file__)) + "/../../data/iris_benchmarks_scenes_urdf/yamls/" + TEST_SCENE + ".dmd.yaml"


class MeshcatSliderSource(LeafSystem):
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        self.meshcat = meshcat

        if joint_control:
            self.DeclareVectorOutputPort("slider_values", BasicVector(2*num_robot_positions), self.DoCalcOutput)
        else:
            self.DeclareVectorOutputPort("slider_values", BasicVector(6), self.DoCalcOutput)

    def DoCalcOutput(self, context, output):
        if joint_control:
            out = []
            for i in range(num_robot_positions):
                out.append(self.meshcat.GetSliderValue(f'q{i}'))
            out += [0]*num_robot_positions
            output.SetFromVector(out)
        else:
            x = self.meshcat.GetSliderValue('x')
            y = self.meshcat.GetSliderValue('y')
            z = self.meshcat.GetSliderValue('z')
            roll = self.meshcat.GetSliderValue('roll')
            pitch = self.meshcat.GetSliderValue('pitch')
            yaw = self.meshcat.GetSliderValue('yaw')
            output.SetFromVector([x, y, z, roll, pitch, yaw])


class InverseKinematicsSystem(LeafSystem):
    def __init__(self, plant, meshcat):
        LeafSystem.__init__(self)
        self.plant = plant
        self.meshcat = meshcat
        self.DeclareVectorInputPort("slider_values", BasicVector(6))
        self.DeclareVectorOutputPort("desired_state", BasicVector(num_robot_positions*2), self.CalculateDesiredState)
        
    def CalculateDesiredState(self, context, output):
        slider_values = self.get_input_port(0).Eval(context)
        x, y, z, roll, pitch, yaw = slider_values
        desired_pose = RigidTransform(RotationMatrix(RollPitchYaw(roll, pitch, yaw)), [x, y, z])
        
        ik = InverseKinematics(self.plant)
        ik.AddPositionConstraint(
            ee_frame,
            [0, 0, 0],
            self.plant.world_frame(),
            desired_pose.translation(),
            desired_pose.translation()
        )
        ik.AddOrientationConstraint(
            ee_frame,
            RotationMatrix(),
            self.plant.world_frame(),
            desired_pose.rotation(),
            0.05
        )
        
        prog = ik.prog()
        prog.SetInitialGuess(ik.q(), default_joint_positions)
        result = Solve(prog)
        
        if result.is_success():
            q_solution = result.GetSolution(ik.q())
            v_solution = np.zeros_like(q_solution)  # Assuming zero velocity for simplicity
            desired_state = np.concatenate([q_solution, v_solution])
        else:
            print("ik fail; defaulting to zero state.")
            desired_state = np.zeros(num_robot_positions*2)
            
        output.SetFromVector(desired_state)


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
    scenario = load_scenario(data=scenario_yaml_for_iris)
else:
    scenario = load_scenario(filename=scene_yaml_file)

# Hardware station setup
station = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,

    # This is to be able to load our own models from a local path
    # we can refer to this using the "package://" URI directive
    parser_preload_callback=lambda parser: parser.package_map().Add("iris_environments", os.path.dirname(os.path.abspath(__file__)) + "/../../data/iris_benchmarks_scenes_urdf/iris_environments/assets"),
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
start_pt_button = "Capture Start Point"
end_pt_button = "Capture End Point"
meshcat.AddButton(start_pt_button)
meshcat.AddButton(end_pt_button)
meshcat.AddButton("Close")

# Figure out how many robots there are and how many joints each has
model_instances_indices_with_actuators = {}
for actuator_idx in plant.GetJointActuatorIndices():
    robot_model_instance_idx = plant.get_joint_actuator(actuator_idx).model_instance()
    if robot_model_instance_idx not in model_instances_indices_with_actuators.keys():
        model_instances_indices_with_actuators[robot_model_instance_idx] = 1
    else:
        model_instances_indices_with_actuators[robot_model_instance_idx] += 1

# Add Meshcat Slider Source System
slider_source = builder.AddSystem(MeshcatSliderSource(meshcat))

# Add controller and splitter (for when there are multiple robots)
controller = builder.AddSystem(InverseDynamicsController(plant, [100]*num_robot_positions, [0]*num_robot_positions, [50]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
control_splitter = builder.AddSystem(VectorSplitter(*model_instances_indices_with_actuators.values()))

# Set controller desired state
builder.Connect(station.GetOutputPort("state"), controller.GetInputPort("estimated_state"))
builder.Connect(controller.GetOutputPort("generalized_force"), control_splitter.GetInputPort("input"))

if joint_control:
    builder.Connect(slider_source.get_output_port(0), controller.GetInputPort("desired_state"))
else:
    # Add IK System
    ik_system = builder.AddSystem(InverseKinematicsSystem(plant, meshcat))
    # Connect sliders to IK system
    builder.Connect(slider_source.get_output_port(0), ik_system.get_input_port(0))
    builder.Connect(ik_system.get_output_port(0), controller.GetInputPort("desired_state"))

# Set controller desired accel
zero_accel_source = builder.AddSystem(ConstantVectorSource([0]*num_robot_positions))
builder.Connect(zero_accel_source.get_output_port(), controller.GetInputPort("desired_acceleration"))

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

slider_source_context = slider_source.GetMyMutableContextFromRoot(simulator_context)
if not joint_control:
    ik_system_context = ik_system.GetMyMutableContextFromRoot(simulator_context)

# Main simulation loop
print(f"Default joint positions: {default_joint_positions}")

ctr = 0
end_pt_button_clicks = 0
start_pt_button_clicks = 0
start_pts = []
end_pts = []

while not meshcat.GetButtonClicks("Close"):
    if joint_control:
        plant.SetPositions(plant_context, slider_source.get_output_port(0).Eval(slider_source_context)[:num_robot_positions])
    else:
        plant.SetPositions(plant_context, ik_system.get_output_port(0).Eval(ik_system_context)[:num_robot_positions])
    simulator.AdvanceTo(simulator_context.get_time() + 0.01)
   
    if meshcat.GetButtonClicks(end_pt_button) != end_pt_button_clicks:
        end_pt_button_clicks = meshcat.GetButtonClicks(end_pt_button)
        print("Placing End Point")
        end_pts.append(plant.GetPositions(plant_context))
    if meshcat.GetButtonClicks(start_pt_button) != start_pt_button_clicks:
        start_pt_button_clicks = meshcat.GetButtonClicks(start_pt_button)
        print("Placing Start Point")
        start_pts.append(plant.GetPositions(plant_context))

pickle_file = f'{TEST_SCENE}_endpts.pkl'
with open(pickle_file, 'wb') as file:
    pickle.dump({"start_pts": start_pts, "end_pts": end_pts}, file)

print(f"Start and end points saved to {pickle_file}")