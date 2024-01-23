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

from utils import diagram_visualize_connections
from scenario import scenario_yaml, robot_yaml
from iris import generate_source_iris_regions

parser = argparse.ArgumentParser()
parser.add_argument('--randomization', default=0, help="integer randomization seed.")
args = parser.parse_args()

seed = int(args.randomization)

    
##### Settings #####
close_button_str = "Close"
this_drake_module_name = "cwd"
box_randomization_runtime = 1.15
sim_runtime = box_randomization_runtime + 4.0
NUM_BOXES = 40

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
    name: Box_{i}
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


controller_plant = MultibodyPlant(time_step=0.001)
parser = Parser(plant)
ConfigureParser(parser)
Parser(controller_plant).AddModelsFromString(robot_yaml, ".dmd.yaml")[0]  # ModelInstance object
controller_plant.Finalize()
num_robot_positions = controller_plant.num_positions()
print(f"num_robot_positions: {num_robot_positions}")
controller = builder.AddSystem(InverseDynamicsController(controller_plant, [300]*num_robot_positions, [1]*num_robot_positions, [20]*num_robot_positions, True))
builder.Connect(controller.GetOutputPort("generalized_force"), station.GetInputPort("kuka.actuation"))
builder.Connect(station.GetOutputPort("kuka_state"), controller.GetInputPort("estimated_state"))

# TEMPORARY
builder.Connect(station.GetOutputPort("kuka_state"), controller.GetInputPort("desired_state"))
# builder.Connect(motion_planner.GetOutputPort("SOMETHING"), controller.GetInputPort("desired_acceleration"))


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


####################################
### Running Simulation & Meshcat ###
####################################
simulator.set_target_realtime_rate(1)
simulator.set_publish_every_time_step(True)
plt.show()

# TEMPORARY
controller.GetInputPort("desired_acceleration").FixValue(controller_context, np.zeros(num_robot_positions))


meshcat.StartRecording()

# 'Remove' Top of truck trailer
trailer_roof_model_idx = plant.GetModelInstanceByName("Truck_Trailer_Roof")  # ModelInstanceIndex
trailer_roof_body_idx = plant.GetBodyIndices(trailer_roof_model_idx)[0]  # BodyIndex
plant.SetFreeBodyPose(plant_context, plant.get_body(trailer_roof_body_idx), RigidTransform([0,0,100]))

# Move Robot to start position
robot_model_idx = plant.GetModelInstanceByName("robot_base")  # ModelInstanceIndex
robot_body_idx = plant.GetBodyIndices(robot_model_idx)[0]  # BodyIndex
robot_pose = RigidTransform([-1.0,0.0,0.58])
plant.SetFreeBodyPose(plant_context, plant.get_body(robot_body_idx), robot_pose)
for joint_idx in plant.GetJointIndices(robot_model_idx):
    robot_joint = plant.get_joint(joint_idx)  # Joint object
    robot_joint.Lock(plant_context)


generate_source_iris_regions(meshcat, robot_pose)


# Set poses for all boxes
for i in range(NUM_BOXES):
    box_model_idx = plant.GetModelInstanceByName(f"Box_{i}")  # ModelInstanceIndex
    box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex

    box_pos_x = np.random.uniform(-1, 1.3, 1)
    box_pos_y = np.random.uniform(-0.95, 0.95, 1)
    box_pos_z = np.random.uniform(0, 8, 1)

    plant.SetFreeBodyPose(plant_context, plant.get_body(box_body_idx), RigidTransform([box_pos_x[0], box_pos_y[0], box_pos_z[0]]))

simulator.AdvanceTo(box_randomization_runtime)

# Put Top of truck trailer back and lock it
plant.SetFreeBodyPose(plant_context, plant.get_body(trailer_roof_body_idx), RigidTransform([0,0,0]))
trailer_roof_joint_idx = plant.GetJointIndices(trailer_roof_model_idx)[0]  # JointIndex object
trailer_roof_joint = plant.get_joint(trailer_roof_joint_idx)  # Joint object
trailer_roof_joint.Lock(plant_context)

# Applied external forces on the box to shove them to the back of the truck trailer
box_forces = []
zero_box_forces = []
for i in range(NUM_BOXES):
    force = ExternallyAppliedSpatialForce()
    zero_force = ExternallyAppliedSpatialForce()

    box_model_idx = plant.GetModelInstanceByName(f"Box_{i}")  # ModelInstanceIndex
    box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex

    force.body_index = box_body_idx
    force.p_BoBq_B = [0,0,0]
    force.F_Bq_W = SpatialForce(tau=[0,0,0], f=[1000,0,0])
    box_forces.append(force)

    zero_force.body_index = box_body_idx
    zero_force.p_BoBq_B = [0,0,0]
    zero_force.F_Bq_W = SpatialForce(tau=[0,0,0], f=[0,0,0])
    zero_box_forces.append(zero_force)

# Apply pushing force to back of truck trailer
station.GetInputPort("applied_spatial_force").FixValue(station_context, box_forces)
simulator.AdvanceTo(box_randomization_runtime+1.5)

# Remove pushing force to back of truck trailer
station.GetInputPort("applied_spatial_force").FixValue(station_context, zero_box_forces)

simulator.AdvanceTo(sim_runtime)

meshcat.PublishRecording()
print(f"{meshcat.web_url()}/download")

while not meshcat.GetButtonClicks(close_button_str):
    pass