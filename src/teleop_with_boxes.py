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
    CollisionFilterManager,
    GeometrySet,
    Role,
    CollisionFilterDeclaration,
    QuaternionFloatingJoint,
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

from scenario import NUM_BOXES, BOX_DIM, q_nominal, q_place_nominal, scenario_yaml, robot_yaml, robot_pose, set_up_scene, get_W_X_eef

class MeshcatSliderSource(LeafSystem):
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        self.meshcat = meshcat

        if joint_control:
            self.DeclareVectorOutputPort("slider_values", BasicVector(12), self.DoCalcOutput)
        else:
            self.DeclareVectorOutputPort("slider_values", BasicVector(6), self.DoCalcOutput)

    def DoCalcOutput(self, context, output):
        if joint_control:
            q1 = self.meshcat.GetSliderValue('q1')
            q2 = self.meshcat.GetSliderValue('q2')
            q3 = self.meshcat.GetSliderValue('q3')
            q4 = self.meshcat.GetSliderValue('q4')
            q5 = self.meshcat.GetSliderValue('q5')
            q6 = self.meshcat.GetSliderValue('q6')
            output.SetFromVector([q1, q2, q3, q4, q5, q6, 0, 0, 0, 0, 0, 0])
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
parser.add_argument('--joint_control', default='F', help="T/F; whether to control joint positions (instead of xyz rpy)")
parser.add_argument('--collision-filter', default='F', help="T/F; whether to filter out collisions between robot and boxes")
args = parser.parse_args()

seed = int(args.randomization)
randomize_boxes = (args.fast == 'F')
joint_control = (args.joint_control == 'T')

    
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
if joint_control:
    meshcat.AddSlider('q1', -np.pi, np.pi, 0.01, 0.0)
    meshcat.AddSlider('q2', -np.pi, np.pi, 0.01, -2.2)
    meshcat.AddSlider('q3', -np.pi, np.pi, 0.01, 2.2)
    meshcat.AddSlider('q4', -np.pi, np.pi, 0.01, 0.0)
    meshcat.AddSlider('q5', -np.pi, np.pi, 0.01, 1.57)
    meshcat.AddSlider('q6', -np.pi, np.pi, 0.01, 0.0)
else:
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
station = builder.AddSystem(MakeHardwareStation(
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

controller_plant = MultibodyPlant(time_step=0.001)
parser = Parser(plant)
ConfigureParser(parser)
Parser(controller_plant).AddModelsFromString(robot_yaml, ".dmd.yaml")[0]  # ModelInstance object
controller_plant.Finalize()
num_robot_positions = controller_plant.num_positions()


# Add Meshcat Slider Source System
slider_source = builder.AddSystem(MeshcatSliderSource(meshcat))

### Controller
if joint_control:
    zero_accel_source = builder.AddSystem(ConstantVectorSource([0,0,0,0,0,0]))

    controller = builder.AddSystem(InverseDynamicsController(controller_plant, [150]*num_robot_positions, [50]*num_robot_positions, [50]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
    builder.Connect(station.GetOutputPort("kuka_state"), controller.GetInputPort("estimated_state"))
    builder.Connect(slider_source.get_output_port(0), controller.GetInputPort("desired_state"))
    builder.Connect(zero_accel_source.get_output_port(), controller.GetInputPort("desired_acceleration"))
    builder.Connect(controller.GetOutputPort("generalized_force"), station.GetInputPort("kuka_actuation"))

else:
    # Add IK System
    ik_system = builder.AddSystem(InverseKinematicsSystem(controller_plant, meshcat))

    # Connect sliders to IK system
    builder.Connect(slider_source.get_output_port(0), ik_system.get_input_port(0))

    controller = builder.AddSystem(InverseDynamicsController(controller_plant, [150]*num_robot_positions, [50]*num_robot_positions, [50]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
    builder.Connect(station.GetOutputPort("kuka_state"), controller.GetInputPort("estimated_state"))
    builder.Connect(ik_system.get_output_port(0), controller.GetInputPort("desired_state"))
    builder.Connect(ik_system.get_output_port(1), controller.GetInputPort("desired_acceleration"))
    builder.Connect(controller.GetOutputPort("generalized_force"), station.GetInputPort("kuka_actuation"))

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
    simulator.AdvanceTo(simulator_context.get_time() + 0.1)
    ctr += 1
    if (ctr == 10):
        ctr = 0
        print(plant.GetPositions(plant_context)[7:13])
        print(plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName("arm_eef")))