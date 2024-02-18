from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    Box,
    ModelInstanceIndex,
    InverseDynamicsController,
    PidController,
    RigidTransform,
    MultibodyPlant,
    RotationMatrix,
    RollPitchYaw,
    SpatialVelocity,
    SpatialForce,
    ExternallyAppliedSpatialForce,
    ConstantVectorSource,
    AbstractValue,
    ContactModel,
    Parser,
    PdControllerGains,
)

# from manipulation.station import MakeHardwareStation, load_scenario
from station import MakeHardwareStation, load_scenario, add_directives  # local version allows ForceDriver
from manipulation.scenarios import AddMultibodyTriad, AddShape
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.utils import ConfigureParser

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("tkagg")
import os
import time
import argparse
import yaml

from utils import NUM_BOXES, diagram_visualize_connections
from scenario import scenario_yaml, robot_yaml
from iris import generate_source_iris_regions
from gcs import MotionPlanner
from debug import Debugger


parser = argparse.ArgumentParser()
parser.add_argument('--randomization', default=0, help="integer randomization seed.")
args = parser.parse_args()

seed = int(args.randomization)

    
##### Settings #####
close_button_str = "Close"
this_drake_module_name = "cwd"
sim_runtime = 3

robot_pose = RigidTransform([0.0,0.0,0.58])

np.random.seed(seed)

#####################
### Meshcat Setup ###
#####################
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

#####################
### Diagram Setup ###
#####################
builder = DiagramBuilder()
scenario = load_scenario(data=robot_yaml)

### Hardware station setup
station = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,

    # This is to be able to load our own models from a local path
    # we can refer to this using the "package://" URI directive
    parser_preload_callback=lambda parser: parser.package_map().Add(this_drake_module_name, os.getcwd())
))
scene_graph = station.GetSubsystemByName("scene_graph")
plant = station.GetSubsystemByName("plant")

# Plot Triad at end effector
AddMultibodyTriad(plant.GetFrameByName("arm_eef"), scene_graph)

### GCS Motion Planer
motion_planner = builder.AddSystem(MotionPlanner(plant, meshcat, robot_pose, 0))
builder.Connect(station.GetOutputPort("body_poses"), motion_planner.GetInputPort("kuka_current_pose"))
builder.Connect(station.GetOutputPort("kuka_state"), motion_planner.GetInputPort("kuka_state"))

### Controller
controller_plant = MultibodyPlant(time_step=0.001)
parser = Parser(plant)
ConfigureParser(parser)
Parser(controller_plant).AddModelsFromString(robot_yaml, ".dmd.yaml")[0]  # ModelInstance object
controller_plant.Finalize()
num_robot_positions = controller_plant.num_positions()
controller = builder.AddSystem(InverseDynamicsController(controller_plant, [1000]*num_robot_positions, [0]*num_robot_positions, [0]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
builder.Connect(station.GetOutputPort("kuka_state"), controller.GetInputPort("estimated_state"))
builder.Connect(motion_planner.GetOutputPort("kuka_desired_state"), controller.GetInputPort("desired_state"))
builder.Connect(motion_planner.GetOutputPort("kuka_acceleration"), controller.GetInputPort("desired_acceleration"))
builder.Connect(controller.GetOutputPort("generalized_force"), station.GetInputPort("kuka_actuation"))

### Print Debugger
if True:
    debugger = builder.AddSystem(Debugger())
    builder.Connect(station.GetOutputPort("kuka_state"), debugger.GetInputPort("kuka_state"))
    builder.Connect(station.GetOutputPort("body_poses"), debugger.GetInputPort("kuka_current_pose"))
    builder.Connect(controller.GetOutputPort("generalized_force"), debugger.GetInputPort("kuka_actuation"))

### Finalizing diagram setup
diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.set_name("Box Unloader")
diagram_visualize_connections(diagram, "diagram.svg")


########################
### Simulation Setup ###
########################
simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(simulator_context)
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)
controller_context = controller.GetMyMutableContextFromRoot(simulator_context)


### TESTING
# controller.GetInputPort("estimated_state").FixValue(controller_context, np.append(
#     [0.0, -1.8, 1.5, 0.0, 0.0, 0.0],
#     # np.zeros((6,)),
#     np.zeros((6,)),
#     # [0.0, -1.8, 1.5, 0.0, 0.0, 0.0],
# )) # TESTING
controller.GetInputPort("desired_state").FixValue(controller_context, np.append(
    [0.0, -1.8, 1.5, 0.0, 0.0, 0.0],
    # np.zeros((6,)),
    np.zeros((6,)),
    # [0.0, -1.8, 1.5, 0.0, 0.0, 0.0],
)) # TESTING
controller.GetInputPort("desired_acceleration").FixValue(controller_context, np.zeros(6)) # TESTING

# station.GetInputPort("kuka_actuation").FixValue(station_context, -1000*np.ones(6))

####################################
### Running Simulation & Meshcat ###
####################################
simulator.set_target_realtime_rate(1)
simulator.set_publish_every_time_step(True)

meshcat.StartRecording()
simulator.AdvanceTo(sim_runtime)

meshcat.PublishRecording()
print(f"{meshcat.web_url()}/download")

while not meshcat.GetButtonClicks(close_button_str):
    pass