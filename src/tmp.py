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
    WeldJoint,
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
from scenario import NUM_BOXES, BOX_DIM, q_nominal, q_place_nominal, scenario_yaml, robot_yaml, scenario_yaml_for_iris, \
    robot_pose, set_hydroelastic, set_up_scene, get_W_X_eef
# from iris import IrisRegionGenerator
from gcs import MotionPlanner
from debug import Debugger

# Set logging level in drake to DEBUG
configure_logging()
log = logging.getLogger("drake")
# log.setLevel("DEBUG")
log.setLevel("INFO")

parser = argparse.ArgumentParser()
parser.add_argument('--fast', default='T',
                    help="T/F; whether or not to use a pre-saved box configuration or randomize box positions from scratch.")
parser.add_argument('--randomization', default=0, help="integer randomization seed.")
parser.add_argument('--enable_hydroelastic', default='F',
                    help="T/F; whether or not to enable hydroelastic contact in the SDF file.")
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
motion_planner = builder.AddSystem(
    MotionPlanner(plant, meshcat, robot_pose, box_randomization_runtime if randomize_boxes else 0,
                  "../data/iris_source_regions.yaml", "../data/iris_source_regions_place.yaml"))
builder.Connect(station.GetOutputPort("body_poses"), motion_planner.GetInputPort("body_poses"))
builder.Connect(station.GetOutputPort("kuka_state"), motion_planner.GetInputPort("kuka_state"))

### Controller
controller_plant = MultibodyPlant(time_step=0.001)
parser = Parser(plant)
ConfigureParser(parser)
Parser(controller_plant).AddModelsFromString(robot_yaml, ".dmd.yaml")[0]  # ModelInstance object
controller_plant.Finalize()
num_robot_positions = controller_plant.num_positions()
controller = builder.AddSystem(
    InverseDynamicsController(controller_plant, [150] * num_robot_positions, [50] * num_robot_positions,
                              [50] * num_robot_positions, True))  # True = exposes "desired_acceleration" port
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
simulator.set_publish_every_time_step(True)

meshcat.StartRecording()

set_up_scene(station, station_context, plant, plant_context, simulator, randomize_boxes,
             box_fall_runtime if randomize_boxes else 0, box_randomization_runtime if randomize_boxes else 0)



# # Generate regions with no obstacles at all
robot_diagram_builder = RobotDiagramBuilder()
robot_model_instances = robot_diagram_builder.parser().AddModelsFromString(scenario_yaml_for_iris, ".dmd.yaml")
robot_diagram_builder_plant = robot_diagram_builder.plant()
robot_diagram_builder_plant.WeldFrames(robot_diagram_builder_plant.world_frame(), robot_diagram_builder_plant.GetFrameByName("base_link", robot_diagram_builder_plant.GetModelInstanceByName("robot_base")), robot_pose)
robot_diagram_builder_diagram = robot_diagram_builder.Build()

collision_checker_params = dict(edge_step_size=0.01)
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = robot_diagram_builder_diagram
# collision_checker_params["edge_step_size"] = 0.25
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)


from pydrake.all import HPolyhedron, RandomGenerator, VisibilityGraph, IrisFromCliqueCoverOptions, IrisInConfigurationSpaceFromCliqueCover
from time import time
generator = RandomGenerator(0)
domain = HPolyhedron.MakeBox(collision_checker.plant().GetPositionLowerLimits(),
                           collision_checker.plant().GetPositionUpperLimits())

# options = IrisFromCliqueCoverOptions()
# options.num_points_per_coverage_check = 1000
# options.num_points_per_visibility_round = 1000
# options.coverage_termination_threshold = 0.1
# options.minimum_clique_size = 100  # minimum of 7 points needed to create a shape with volume in 6D
# options.iteration_limit = 1  # Only build 1 visibility graph --> cliques --> region in order not to have too much region overlap
#
# regions = IrisInConfigurationSpaceFromCliqueCover(
#             checker=collision_checker, options=options, generator=RandomGenerator(0), sets = []
#         )

for n in [int(1e2), int(1e3), int(1e4), int(1e5), int(1e6)]:
    def get_points():
        points = []
        last_point = domain.ChebyshevCenter()
        while len(points) < n:
            point = domain.UniformSample(generator, last_point)
            if collision_checker.CheckConfigCollisionFree(point):
                points.append(point)
            last_point = point
        return np.array(points).T
    t0 = time()
    points = get_points()
    t1 = time()
    print(f"{n = }")
    print(f"time to get points = {t1 - t0}")
    G = VisibilityGraph(collision_checker, points)
    t2 = time()

    print(f"time for visibility_graph = {t2-t1}")
    print()











