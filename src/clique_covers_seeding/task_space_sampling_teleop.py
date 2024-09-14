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
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import scenario_yaml_for_iris
from utils import diagram_visualize_connections

import numpy as np
import time

TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
# TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
# TEST_SCENE = "BOX-UNLOADING"


yaml_file = os.path.dirname(os.path.abspath(__file__)) + "/../../data/iris_benchmarks_scenes_urdf/yamls/" + TEST_SCENE + ".dmd.yaml"


class VectorSplitter(LeafSystem):
    """
    Simple LeafSystem that takes a vector input of size out1 + out2 and splits 
    it into 2 output vectors of size out1 and out2. This is used for multi-robot
    plants so the output of one controller can control both robots.
    """
    def __init__(self, out1, out2):
        super().__init__()
        self.out1 = out1
        self.out2 = out2
        self.DeclareVectorInputPort("input", BasicVector(out1 + out2))
        self.DeclareVectorOutputPort("output_1", BasicVector(out1), self.Output1)
        self.DeclareVectorOutputPort("output_2", BasicVector(out2), self.Output2)

    def Output1(self, context, output):
        input_vector = self.get_input_port(0).Eval(context)
        output.SetFromVector([input_vector[:self.out1]])  # return first `out1` elements

    def Output2(self, context, output):
        input_vector = self.get_input_port(0).Eval(context)
        output.SetFromVector(input_vector[self.out1:])  # return latter `out2` elements


meshcat = StartMeshcat()

builder = DiagramBuilder()

if TEST_SCENE == "BOX-UNLOADING":
    scenario = load_scenario(data=scenario_yaml_for_iris)
else:
    scenario = load_scenario(filename=yaml_file)

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

# Figure out how many robots there are and how many joints each has
model_instances_indices_with_actuators = {}
for actuator_idx in plant.GetJointActuatorIndices():
    robot_model_instance_idx = plant.get_joint_actuator(actuator_idx).model_instance()
    if robot_model_instance_idx not in model_instances_indices_with_actuators.keys():
        model_instances_indices_with_actuators[robot_model_instance_idx] = 1
    else:
        model_instances_indices_with_actuators[robot_model_instance_idx] += 1

# Add controller and splitter (for when there are multiple robots)
controller = builder.AddSystem(InverseDynamicsController(plant, [150]*num_robot_positions, [10]*num_robot_positions, [50]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
control_splitter = builder.AddSystem(VectorSplitter(*model_instances_indices_with_actuators.values()))
builder.Connect(station.GetOutputPort("state"), controller.GetInputPort("estimated_state"))
builder.Connect(controller.GetOutputPort("generalized_force"), control_splitter.GetInputPort("input"))

# Connect each output of the splitter to the actuation input for each robot
for i, (robot_model_instance_idx, num_joints) in enumerate(model_instances_indices_with_actuators.items()):
    builder.Connect(control_splitter.GetOutputPort(f"output_{i+1}"), station.GetInputPort(f"{plant.GetModelInstanceName(robot_model_instance_idx)}_actuation"))

# Set controller desired state
default_joint_positions = plant.GetPositions(plant.CreateDefaultContext())
state_controller = builder.AddSystem(ConstantVectorSource(np.concatenate([default_joint_positions, np.zeros(num_robot_positions)])))
builder.Connect(state_controller.get_output_port(), controller.GetInputPort("desired_state"))

# Set zero controller desired accel
acceleration_controller = builder.AddSystem(ConstantVectorSource(num_robot_positions*[0]))
builder.Connect(acceleration_controller.get_output_port(), controller.GetInputPort("desired_acceleration"))

diagram = builder.Build()
diagram_visualize_connections(diagram, "diagram_.svg")
context = diagram.CreateDefaultContext()

simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(simulator_context)
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)

simulator.set_publish_every_time_step(True)
meshcat.StartRecording()
simulator.AdvanceTo(1.5)
meshcat.PublishRecording()
time.sleep(5)
print(f"View the scene simulation at: {meshcat.web_url()}/download")