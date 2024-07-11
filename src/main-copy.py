from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    InverseDynamicsController,
    RigidTransform,
    MultibodyPlant,
    ContactModel,
    RobotDiagramBuilder,
    Parser,
    configure_logging,
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
import logging
import datetime

from utils import diagram_visualize_connections
from scenario import NUM_BOXES, scenario_yaml, robot_yaml, scenario_yaml_for_iris, robot_pose, set_up_scene, get_W_X_eef
from iris import IrisRegionGenerator
from gcs import MotionPlanner
from gripper_sim import GripperSimulator
from debug import Debugger


# Set logging level in drake to DEBUG
configure_logging()
log = logging.getLogger("drake")
# log.setLevel("DEBUG")
log.setLevel("INFO")

parser = argparse.ArgumentParser()
parser.add_argument('--fast', default='T', help="T/F; whether or not to use a pre-saved box configuration or randomize box positions from scratch.")
parser.add_argument('--randomization', default=0, help="integer randomization seed.")
args = parser.parse_args()

seed = int(args.randomization)
randomize_boxes = (args.fast == 'F')

    
#####################
###    Settings   ###
#####################
close_button_str = "Close"
this_drake_module_name = "cwd"

if randomize_boxes:
    box_fall_runtime = 0.95
    box_randomization_runtime = box_fall_runtime + 17
    sim_runtime = box_randomization_runtime + 10
else:
    sim_runtime = 10

np.random.seed(seed)


#####################
### Meshcat Setup ###
#####################
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)
# meshcat.SetProperty("/drake/contact_forces", "visible", False)  # Doesn't work for some reason


#####################
### Diagram Setup ###
#####################
builder = DiagramBuilder()
scenario = load_scenario(data=scenario_yaml)

### Add Boxes
box_directives = f"""
directives:
"""
for i in range(NUM_BOXES):
    relative_path_to_box = '../data/Box_0_5_0_5_0_5.sdf'
    absolute_path_to_box = os.path.abspath(relative_path_to_box)

    box_directives += f"""
- add_model: 
    name: Boxes/Box_{i}
    file: file://{absolute_path_to_box}
"""
box_directives += f"""
- add_model: 
    name: Boxes/Box_eef
    file: file://{absolute_path_to_box}
"""
scenario = add_directives(scenario, data=box_directives)


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
motion_planner = builder.AddSystem(MotionPlanner(plant, meshcat, robot_pose, box_randomization_runtime if randomize_boxes else 0))
builder.Connect(station.GetOutputPort("body_poses"), motion_planner.GetInputPort("body_poses"))
builder.Connect(station.GetOutputPort("kuka_state"), motion_planner.GetInputPort("kuka_state"))

### Controller
controller_plant = MultibodyPlant(time_step=0.001)
parser = Parser(plant)
ConfigureParser(parser)
Parser(controller_plant).AddModelsFromString(robot_yaml, ".dmd.yaml")[0]  # ModelInstance object
controller_plant.Finalize()
num_robot_positions = controller_plant.num_positions()
controller = builder.AddSystem(InverseDynamicsController(controller_plant, [100]*num_robot_positions, [1]*num_robot_positions, [20]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
builder.Connect(station.GetOutputPort("kuka_state"), controller.GetInputPort("estimated_state"))
builder.Connect(motion_planner.GetOutputPort("kuka_desired_state"), controller.GetInputPort("desired_state"))
builder.Connect(motion_planner.GetOutputPort("kuka_acceleration"), controller.GetInputPort("desired_acceleration"))
builder.Connect(controller.GetOutputPort("generalized_force"), station.GetInputPort("kuka_actuation"))

### Print Debugger
if True:
    debugger = builder.AddSystem(Debugger())
    builder.Connect(station.GetOutputPort("kuka_state"), debugger.GetInputPort("kuka_state"))
    builder.Connect(station.GetOutputPort("body_poses"), debugger.GetInputPort("body_poses"))
    builder.Connect(controller.GetOutputPort("generalized_force"), debugger.GetInputPort("kuka_actuation"))

### Finalizing diagram setup
diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.set_name("Box Unloader")
diagram_visualize_connections(diagram, "../diagram.svg")


########################
### Simulation Setup ###
########################
simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(simulator_context)
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)
controller_context = controller.GetMyMutableContextFromRoot(simulator_context)

motion_planner.set_context(plant_context)

### TESTING
# controller.GetInputPort("estimated_state").FixValue(controller_context, np.append(
#     [0.0, -2.5, 2.8, 0.0, 1.2, 0.0],
#     np.zeros((6,)),
# )) # TESTING
# controller.GetInputPort("desired_state").FixValue(controller_context, np.append(
#     [0.0, -2.5, 2.8, 0.0, 1.2, 0.0],
#     np.zeros((6,)),
# )) # TESTING
# controller.GetInputPort("desired_acceleration").FixValue(controller_context, np.zeros(6)) # TESTING
# station.GetInputPort("kuka_actuation").FixValue(station_context, -1000*np.ones(6))


####################################
### Running Simulation & Meshcat ###
####################################
simulator.set_target_realtime_rate(1)
simulator.set_publish_every_time_step(True)

meshcat.StartRecording()

set_up_scene(station, station_context, plant, plant_context, simulator, randomize_boxes, box_fall_runtime if randomize_boxes else 0, box_randomization_runtime if randomize_boxes else 0)


box_model_idx = plant.GetModelInstanceByName("Boxes/Box_eef")  # ModelInstanceIndex
box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex
W_X_eef = get_W_X_eef(plant, plant_context)
print(W_X_eef.translation())
box_eef_pose = W_X_eef
plant.SetFreeBodyPose(plant_context, plant.get_body(box_body_idx), box_eef_pose)
simulator.AdvanceTo(0.002)
print('moved box')


# # Generate regions with no obstacles at all
# robot_diagram_builder = RobotDiagramBuilder()
# iris_diagram = robot_diagram_builder.parser().AddModelsFromString(scenario_yaml_for_iris, ".dmd.yaml")
# region_generator = IrisRegionGenerator(meshcat, iris_diagram, robot_pose, regions_file="../data/iris_source_regions.yaml")
# region_generator.load_and_test_regions()

# region_generator.generate_source_region_at_q_nominal()
# region_generator.generate_source_iris_regions(minimum_clique_size=20, 
#                                               coverage_threshold=0.2, 
#                                               use_previous_saved_regions=False)  # False --> regenerate regions from scratch
# region_generator.generate_source_iris_regions(minimum_clique_size=15, 
#                                               coverage_threshold=0.45, 
#                                               use_previous_saved_regions=True)
# region_generator.generate_source_iris_regions(minimum_clique_size=8,
#                                               coverage_threshold=0.7, 
#                                               use_previous_saved_regions=True)


# # Generate regions with box in hand
# robot_diagram_builder = RobotDiagramBuilder()
# scenario_yaml_for_iris_eef_box = scenario_yaml_for_iris + f"""
# - add_model: 
#     name: Boxes/Box_eef
#     file: file://{absolute_path_to_box}
# """
# iris_diagram = robot_diagram_builder.parser().AddModelsFromString(scenario_yaml_for_iris_eef_box, ".dmd.yaml")
# iris_plant = robot_diagram_builder.plant()
# iris_plant_context = iris_plant.GetMyContextFromRoot(context)

# # Set pose of box to be in "grabbed" position relative to eef and weld it there
# box_model_idx = iris_plant.GetModelInstanceByName("Boxes/Box_eef")  # ModelInstanceIndex
# box_body_idx = iris_plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex
# W_X_eef= get_W_X_eef(plant, plant_context)
# box_eef_pose = RigidTransform(W_X_eef.rotation(), W_X_eef.translation() + W_X_eef.rotation() @ [0.0, 0.0, 0.5])
# iris_plant.SetFreeBodyPose(iris_plant_context, iris_plant.get_body(box_body_idx), box_eef_pose)
# region_generator = IrisRegionGenerator(meshcat, iris_diagram, robot_pose, regions_file="../data/iris_source_regions_place.yaml")
# region_generator.generate_source_region_at_q_nominal()
# region_generator.generate_source_iris_regions(minimum_clique_size=20, 
#                                               coverage_threshold=0.2, 
#                                               use_previous_saved_regions=False)  # False --> regenerate regions from scratch
# region_generator.generate_source_iris_regions(minimum_clique_size=15, 
#                                               coverage_threshold=0.45, 
#                                               use_previous_saved_regions=True)
# region_generator.generate_source_iris_regions(minimum_clique_size=8,
#                                               coverage_threshold=0.7, 
#                                               use_previous_saved_regions=True)

# Get box poses to pass to pick planner to select a box to pick first
box_poses = {}
for i in range(NUM_BOXES):
    box_model_idx = plant.GetModelInstanceByName(f"Boxes/Box_{i}")  # ModelInstanceIndex
    box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex
    box_poses[box_body_idx] = plant.GetFreeBodyPose(plant_context, plant.get_body(box_body_idx))

simulator.AdvanceTo(sim_runtime)

meshcat.PublishRecording()
date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"{date}: {meshcat.web_url()}/download")

while not meshcat.GetButtonClicks(close_button_str):
    pass