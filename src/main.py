from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    MeshcatVisualizer,
    AddDefaultVisualization,
    Simulator,
    InverseDynamicsController,
    RigidTransform,
    MultibodyPlant,
    ContactModel,
    RobotDiagramBuilder,
    Parser,
    configure_logging,
    SceneGraphCollisionChecker,
    ConfigurationSpaceObstacleCollisionChecker,
    WeldJoint,
    QuaternionFloatingJoint,
)

# from manipulation.station import MakeHardwareStation, load_scenario
from station import MakeHardwareStation, load_scenario, add_directives  # local version allows ForceDriver
from manipulation.scenarios import AddMultibodyTriad, AddShape
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.utils import ConfigureParser

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")
import numpy as np
import os
import time
import argparse
import yaml
import logging
import datetime

from utils import diagram_visualize_connections
from scenario import NUM_BOXES, BOX_DIM, q_nominal, q_place_nominal, scenario_yaml, robot_yaml, scenario_yaml_for_iris, robot_pose, set_hydroelastic, set_up_scene, get_W_X_eef
from iris import IrisRegionGenerator
from planner import MotionPlanner
from debug import Debugger


configure_logging()
log = logging.getLogger("drake")
# log.setLevel("DEBUG")
log.setLevel("INFO")

parser = argparse.ArgumentParser()
parser.add_argument('--fast', default='T', help="T/F; whether or not to use a pre-saved box configuration or randomize box positions from scratch.")
parser.add_argument('--randomization', default=0, help="integer randomization seed.")
parser.add_argument('--enable_hydroelastic', default='F', help="T/F; whether or not to enable hydroelastic contact in the SDF file.")
args = parser.parse_args()

seed = int(args.randomization)
randomize_boxes = (args.fast == 'F')
set_hydroelastic(args.enable_hydroelastic == 'T')

    
#####################
###    Settings   ###
#####################
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
meshcat.AddButton("Close")
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
scenario = add_directives(scenario, data=box_directives)


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


### Hardware station setup
station, _ = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,

    # This is to be able to load our own models from a local path
    # we can refer to this using the "package://" URI directive
    parser_preload_callback=lambda parser: parser.package_map().Add(this_drake_module_name, os.getcwd()),
    parser_prefinalize_callback=add_suction_joints,
))
scene_graph = station.GetSubsystemByName("scene_graph")
plant = station.GetSubsystemByName("plant")

# Plot Triad at end effector
AddMultibodyTriad(plant.GetFrameByName("arm_eef"), scene_graph)

### Controller
controller_plant = MultibodyPlant(time_step=0.001)
Parser(controller_plant).AddModelsFromString(robot_yaml, ".dmd.yaml")[0]  # ModelInstance object
controller_plant.Finalize()
num_robot_positions = controller_plant.num_positions()
controller = builder.AddSystem(InverseDynamicsController(controller_plant, [150]*num_robot_positions, [50]*num_robot_positions, [50]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
builder.Connect(station.GetOutputPort("kuka_state"), controller.GetInputPort("estimated_state"))
builder.Connect(controller.GetOutputPort("generalized_force"), station.GetInputPort("kuka_actuation"))

### GCS Motion Planer
motion_planner = builder.AddSystem(MotionPlanner(meshcat, scene_graph, plant, controller_plant, box_randomization_runtime if randomize_boxes else 0, "../data/iris_source_regions.yaml", "../data/iris_source_regions_place.yaml"))
builder.Connect(station.GetOutputPort("body_poses"), motion_planner.GetInputPort("body_poses"))
builder.Connect(station.GetOutputPort("kuka_state"), motion_planner.GetInputPort("kuka_state"))
builder.Connect(motion_planner.GetOutputPort("kuka_desired_state"), controller.GetInputPort("desired_state"))
builder.Connect(motion_planner.GetOutputPort("kuka_acceleration"), controller.GetInputPort("desired_acceleration"))

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


####################################
### Running Simulation & Meshcat ###
####################################
simulator.set_publish_every_time_step(True)

meshcat.StartRecording()

set_up_scene(station, station_context, plant, plant_context, simulator, randomize_boxes, box_fall_runtime if randomize_boxes else 0, box_randomization_runtime if randomize_boxes else 0)

simulator.AdvanceTo(sim_runtime)

meshcat.PublishRecording()
date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"{date}: {meshcat.web_url()}/download")

while not meshcat.GetButtonClicks("Close"):
    pass