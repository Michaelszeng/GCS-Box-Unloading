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
    ConstantVectorSource,
    GeometrySet,
    Role,
    CollisionFilterDeclaration,
    Box,
    Rgba,
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
from scenario import NUM_BOXES, BOX_DIM, q_nominal, q_place_nominal, scenario_yaml, scenario_yaml_with_boxes, robot_yaml, scenario_yaml_for_iris, robot_pose, set_hydroelastic, set_up_scene, get_W_X_eef
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
scenario = load_scenario(data=scenario_yaml_with_boxes)

### Hardware station setup
station = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,

    # This is to be able to load our own models from a local path
    # we can refer to this using the "package://" URI directive
    parser_preload_callback=lambda parser: parser.package_map().Add(this_drake_module_name, os.getcwd()),
))
scene_graph = station.GetSubsystemByName("scene_graph")
plant = station.GetSubsystemByName("plant")

# Add collision filters between robot and boxes
filter_manager = scene_graph.collision_filter_manager()
inspector = scene_graph.model_inspector()
robot_gids = []
box_gids = []
for gid in inspector.GetGeometryIds(
    GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
):
    gid_name = inspector.GetName(inspector.GetFrameId(gid))
    # print(f"{gid_name}, {gid}")
    if "kuka" in gid_name or "robot_base" in gid_name:
        robot_gids.append(gid)
    if "Boxes" in gid_name:
        box_gids.append(gid)

def add_exclusion(set1, set2=None):
    if set2 is None:
        filter_manager.Apply(
            CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
        )
    else:
        filter_manager.Apply(
            CollisionFilterDeclaration().ExcludeBetween(
                GeometrySet(set1), GeometrySet(set2)
            )
        )

for robot_gid in robot_gids:
    for box_gid in box_gids:
        add_exclusion(robot_gid, box_gid)

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
motion_planner = builder.AddSystem(MotionPlanner(meshcat, scene_graph, plant, controller_plant, box_randomization_runtime if randomize_boxes else 0, "IRIS_REGIONS.yaml", "IRIS_REGIONS.yaml"))
builder.Connect(station.GetOutputPort("body_poses"), motion_planner.GetInputPort("body_poses"))
builder.Connect(station.GetOutputPort("kuka_state"), motion_planner.GetInputPort("kuka_state"))

# temp = builder.AddSystem(ConstantVectorSource([ 0.31076813, -0.93517756,  1.876914,    1.18583468, 1.10499359, -0.00917231,0,0,0,0,0,0]))
# builder.Connect(temp.get_output_port(), controller.GetInputPort("desired_state"))
# temp2 = builder.AddSystem(ConstantVectorSource([0,0,0,0,0,0]))
# builder.Connect(temp2.get_output_port(), controller.GetInputPort("desired_acceleration"))

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
controller_plant_context = controller_plant.GetMyMutableContextFromRoot(simulator_context)
controller_context = controller.GetMyMutableContextFromRoot(simulator_context)
scene_graph_context = scene_graph.GetMyContextFromRoot(simulator_context)

motion_planner.set_context(scene_graph_context, plant_context, controller_plant_context)


####################################
### Running Simulation & Meshcat ###
####################################
simulator.set_publish_every_time_step(True)

meshcat.StartRecording()

try:
    simulator.AdvanceTo(sim_runtime)
except:
    pass

meshcat.PublishRecording()
date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"{date}: {meshcat.web_url()}/download")

while not meshcat.GetButtonClicks("Close"):
    pass