from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    InverseDynamicsController,
    RigidTransform,
    MultibodyPlant,
    Parser,
    configure_logging,
    InverseKinematics,
    RotationMatrix,
    RollPitchYaw,
    LeafSystem,
    BasicVector,
    ConstantVectorSource,
    Solve,
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

from scenario import NUM_BOXES, BOX_DIM, q_nominal, q_place_nominal, scenario_yaml, robot_yaml, scenario_yaml_for_iris, robot_pose, set_up_scene, get_W_X_eef


class MeshcatSliderSource(LeafSystem):
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        self.meshcat = meshcat
        self.DeclareVectorOutputPort("slider_values", BasicVector(6), self.DoCalcOutput)

    def DoCalcOutput(self, context, output):
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
        self.DeclareVectorOutputPort("desired_state", BasicVector(12), self.CalculateDesiredState)
        self.DeclareVectorOutputPort("desired_acceleration", BasicVector(6), self.CalculateDesiredAcceleration)
        
    def CalculateDesiredState(self, context, output):
        slider_values = self.get_input_port(0).Eval(context)
        x, y, z, roll, pitch, yaw = slider_values
        desired_pose = RigidTransform(RotationMatrix(RollPitchYaw(roll, pitch, yaw)), [x, y, z])
        
        ik = InverseKinematics(self.plant)
        ik.AddPositionConstraint(
            self.plant.GetFrameByName("arm_eef"),
            [0, 0, 0],
            self.plant.world_frame(),
            desired_pose.translation(),
            desired_pose.translation()
        )
        ik.AddOrientationConstraint(
            self.plant.GetFrameByName("arm_eef"),
            RotationMatrix(),
            self.plant.world_frame(),
            desired_pose.rotation(),
            0.0
        )
        
        prog = ik.prog()
        initial_guess = np.zeros(6)
        prog.SetInitialGuess(ik.q(), q_nominal)
        result = Solve(prog)
        
        if result.is_success():
            q_solution = result.GetSolution(ik.q())
            v_solution = np.zeros_like(q_solution)  # Assuming zero velocity for simplicity
            desired_state = np.concatenate([q_solution, v_solution])
        else:
            print("ik fail; defaulting to zero state.")
            desired_state = np.zeros(12)
            
        output.SetFromVector(desired_state)

    def CalculateDesiredAcceleration(self, context, output):
        # Assuming zero acceleration for simplicity
        output.SetFromVector(np.zeros(6))



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
meshcat.AddSlider('x', -1.0, 2.0, 0.01, 0.75)
meshcat.AddSlider('y', -1.0, 1.0, 0.01, 0.0)
meshcat.AddSlider('z', 0.0, 3.0, 0.01, 1.0)
meshcat.AddSlider('roll', -np.pi, np.pi, 0.01, np.pi)
meshcat.AddSlider('pitch', -np.pi, np.pi, 0.01, 0.0)
meshcat.AddSlider('yaw', -np.pi, np.pi, 0.01, 0.0)
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


controller_plant = MultibodyPlant(time_step=0.001)
parser = Parser(plant)
ConfigureParser(parser)
Parser(controller_plant).AddModelsFromString(robot_yaml, ".dmd.yaml")[0]  # ModelInstance object
controller_plant.Finalize()
num_robot_positions = controller_plant.num_positions()


# Add Meshcat Slider Source System
slider_source = builder.AddSystem(MeshcatSliderSource(meshcat))

# Add IK System
ik_system = builder.AddSystem(InverseKinematicsSystem(controller_plant, meshcat))

# Connect sliders to IK system
builder.Connect(slider_source.get_output_port(0), ik_system.get_input_port(0))

### Controller
controller = builder.AddSystem(InverseDynamicsController(controller_plant, [150]*num_robot_positions, [50]*num_robot_positions, [50]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
builder.Connect(station.GetOutputPort("kuka_state"), controller.GetInputPort("estimated_state"))
builder.Connect(ik_system.get_output_port(0), controller.GetInputPort("desired_state"))
builder.Connect(ik_system.get_output_port(1), controller.GetInputPort("desired_acceleration"))
builder.Connect(controller.GetOutputPort("generalized_force"), station.GetInputPort("kuka_actuation"))

### Print Debugger
# if True:
#     debugger = builder.AddSystem(Debugger())
#     builder.Connect(station.GetOutputPort("kuka_state"), debugger.GetInputPort("kuka_state"))
#     builder.Connect(station.GetOutputPort("body_poses"), debugger.GetInputPort("body_poses"))
#     builder.Connect(controller.GetOutputPort("generalized_force"), debugger.GetInputPort("kuka_actuation"))

### Finalizing diagram setup
diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.set_name("Box Unloader")
# diagram_visualize_connections(diagram, "../diagram.svg")

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
# simulator.set_publish_every_time_step(False)  # Disable publishing at every time step
# simulator.set_publish_every_time_step(True)

# simulator.get_mutable_integrator().set_maximum_step_size(0.01)  # Increase max step size
# simulator.get_mutable_integrator().set_target_accuracy(1e-3)  # Set target accuracy

set_up_scene(station, station_context, plant, plant_context, simulator, randomize_boxes, box_fall_runtime if randomize_boxes else 0, box_randomization_runtime if randomize_boxes else 0)

# Main simulation loop
ctr = 0
while not meshcat.GetButtonClicks("Close"):
    simulator.AdvanceTo(simulator_context.get_time() + 0.01)
    ctr += 1
    if (ctr == 100):
        ctr = 0
        print(plant.GetPositions(plant_context)[7:13])