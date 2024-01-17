from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    Box,
    ModelInstanceIndex,
    InverseDynamicsController,
    RigidTransform,
    MultibodyPlant,
    RotationMatrix,
    RollPitchYaw,
    SpatialVelocity,
    ConstantVectorSource,
    AbstractValue,
    ContactModel
)

# from manipulation.station import MakeHardwareStation, load_scenario
from station import MakeHardwareStation, load_scenario, add_directives  # local version allows ForceDriver
from manipulation.scenarios import AddMultibodyTriad, AddShape
from manipulation.meshcat_utils import AddMeshcatTriad

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("tkagg")
import os
import time
import argparse
import yaml

from utils import diagram_visualize_connections
from scenario import scenario_yaml

parser = argparse.ArgumentParser()
parser.add_argument('--randomization', default=0, help="integer randomization seed.")
args = parser.parse_args()

seed = int(args.randomization)

    
##### Settings #####
close_button_str = "Close"
this_drake_module_name = "cwd"
box_randomization_runtime = 2
sim_runtime = box_randomization_runtime + 4
NUM_BOXES = 20

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
    
    """
        default_free_body_pose:
        Box_0_5_0_5_0_5:
            translation: [{box_pos_x[0]}, {box_pos_y[0]}, {box_pos_z[0]}]
            rotation: !Rpy {{ deg: [0, 0, 0] }}"""
    
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


### Testing hardware
# station.GetInputPort("iiwa.actuation").FixValue(station_context, np.zeros(7)) # TESTING
# station.GetInputPort("wsg.position").FixValue(station_context, [1]) # TESTING


####################################
### Running Simulation & Meshcat ###
####################################
simulator.set_target_realtime_rate(1)
simulator.set_publish_every_time_step(True)
plt.show()

meshcat.StartRecording()

# 'Remove' Top of truck trailer
trailer_roof_model_idx = plant.GetModelInstanceByName("Truck_Trailer_Roof")  # ModelInstanceIndex
trailer_roof_body_idx = plant.GetBodyIndices(trailer_roof_model_idx)[0]  # BodyIndex
plant.SetFreeBodyPose(plant_context, plant.get_body(trailer_roof_body_idx), RigidTransform([0,0,100]))

# Move Robot back
robot_model_idx = plant.GetModelInstanceByName("kuka")  # ModelInstanceIndex
robot_body_idx = plant.GetBodyIndices(robot_model_idx)[0]  # BodyIndex
plant.SetFreeBodyPose(plant_context, plant.get_body(robot_body_idx), RigidTransform([-2,0,0.59]))
robot_joint_idx = plant.GetJointIndices(robot_model_idx)[0]  # JointIndex object
robot_joint = plant.get_joint(robot_joint_idx)  # Joint object
robot_joint.Lock(plant_context)

# Set poses for all boxes
for i in range(NUM_BOXES):
    box_model_idx = plant.GetModelInstanceByName(f"Box_{i}")  # ModelInstanceIndex
    box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex

    box_pos_x = np.random.uniform(-1.5, 1.5, 1)
    box_pos_y = np.random.uniform(-0.95, 0.95, 1)
    box_pos_z = np.random.uniform(2, 5, 1)

    plant.SetFreeBodyPose(plant_context, plant.get_body(box_body_idx), RigidTransform([box_pos_x[0], box_pos_y[0], box_pos_z[0]]))


simulator.AdvanceTo(box_randomization_runtime)

# Put Top of truck trailer back and lock it
plant.SetFreeBodyPose(plant_context, plant.get_body(trailer_roof_body_idx), RigidTransform([0,0,0]))
trailer_roof_joint_idx = plant.GetJointIndices(trailer_roof_model_idx)[0]  # JointIndex object
trailer_roof_joint = plant.get_joint(trailer_roof_joint_idx)  # Joint object
trailer_roof_joint.Lock(plant_context)

# Set a pos-x velocity for all boxes to shove them against the back of the truck trailer
for i in range(NUM_BOXES):
    box_model_idx = plant.GetModelInstanceByName(f"Box_{i}")  # ModelInstanceIndex
    box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex
    box_rigid_body = plant.GetRigidBodyByName("Box_0_5_0_5_0_5", box_model_idx)  # RigidBody

    W_X_O = plant.GetFreeBodyPose(plant_context, box_rigid_body)  # transform from box frame to world frame

    O_V_B = plant.GetVelocities(plant_context, box_model_idx)  # spatial vel of box relative to box frame
    O_v_B_vel = O_V_B[3:]  # positional velocity is the last 3 elements of O_V_B
    O_v_B_rot = O_V_B[:3]
    W_v_B_vel = W_X_O @ O_v_B_vel
    W_v_B_rot = W_X_O @ O_v_B_rot
    W_v_B_vel_desired = W_v_B_vel + np.array([10,0,0])

    plant.SetFreeBodySpatialVelocity(plant.get_body(box_body_idx), SpatialVelocity([0,0,0], W_v_B_vel_desired), plant_context)

simulator.AdvanceTo(sim_runtime)

meshcat.PublishRecording()

while not meshcat.GetButtonClicks(close_button_str):
    pass