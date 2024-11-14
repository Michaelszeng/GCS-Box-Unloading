"""
Template file that allows teleoperation of robots in all 9 scenes.
"""

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
    LoadIrisRegionsYamlFile,
    Rgba,
    RotationMatrix,
    RollPitchYaw,
    InverseKinematics,
    Solve,
    LeafSystem,
    AbstractValue,
    GcsTrajectoryOptimization,
    Point,
    GraphOfConvexSetsOptions
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import iris_yaml
from utils import diagram_visualize_connections

import numpy as np
import time
import importlib
import argparse
import pickle

# TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
# TEST_SCENE = "BOXUNLOADING"

regions_file = f'{TEST_SCENE}_regions.yaml'

regions = LoadIrisRegionsYamlFile(regions_file)

pickle_file = f'testing_positions/{TEST_SCENE}_endpts.pkl'

with open(pickle_file, 'rb') as f:
    endpts = pickle.load(f)
    
all_pts = endpts["start_pts"] + endpts["end_pts"]

num_pts_in_regions = 0
for pt in all_pts:
    in_region = False
    for r in regions.values():
        if r.PointInSet(pt):
            in_region = True
            break
    if in_region:
        num_pts_in_regions += 1

print(f"Num pts in regions: {num_pts_in_regions}")
print(f"Total num pts: {len(all_pts)}")



# src_directory = os.path.dirname(os.path.abspath(__file__))
# parent_directory = os.path.dirname(src_directory)
# data_directory = os.path.join(parent_directory)
# scene_yaml_file = os.path.join(data_directory, "data", "iris_benchmarks_scenes_urdf", "yamls", TEST_SCENE + ".dmd.yaml")



# class TrajectoryPlanner(LeafSystem):
#     def __init__(self, plant, end_pts):
#         super().__init__()
#         self.plant = plant
#         self.end_pts = end_pts
#         self.current_trajectory_idx = 0
#         self.elapsed_time = 0.0
#         self.waiting_period = 2.0  # 2 seconds of waiting before starting next trajectory

#         # Declare output ports for desired state and acceleration
#         self.DeclareVectorOutputPort("desired_state", 2*len(self.plant.GetPositions(self.plant.CreateDefaultContext())), self.CalcDesiredState)
#         self.DeclareVectorOutputPort("desired_acceleration", len(self.plant.GetVelocities(self.plant.CreateDefaultContext())), self.CalcDesiredAcceleration)

#         # Initialize variables for storing trajectory and timestamps
#         self.current_trajectory = None
#         self.start_time = None
#         self.plan_next_trajectory()

#     def CalcDesiredState(self, context, output):
#         if self.current_trajectory:
#             current_time = context.get_time() - self.start_time
#             if current_time > self.current_trajectory.end_time():
#                 # Check if the current trajectory has finished
#                 self.current_trajectory_idx += 1
#                 if self.current_trajectory_idx < len(self.end_pts["start_pts"]):
#                     self.elapsed_time = 0  # Reset elapsed time for waiting period
#                     self.plan_next_trajectory()
#                 else:
#                     self.current_trajectory = None  # No more trajectories
#             else:
#                 # Sample trajectory for desired state at current time
#                 output.SetFromVector(np.append(
#                     self.current_trajectory.value(current_time),
#                     self.current_trajectory.EvalDerivative(current_time)
#                 ))
#         else:
#             self.plan_next_trajectory()
#             output.SetFromVector(np.zeros(len(output.get_value())))

#     def CalcDesiredAcceleration(self, context, output):
#         if self.current_trajectory:
#             current_time = context.get_time() - self.start_time
#             if current_time < self.current_trajectory.end_time():
#                 # Calculate desired acceleration from the trajectory's second derivative
#                 output.SetFromVector(self.current_trajectory.EvalDerivative(current_time, 2))
#             else:
#                 output.SetFromVector(np.zeros(len(output.get_value())))
#         else:
#             output.SetFromVector(np.zeros(len(output.get_value())))

#     def plan_next_trajectory(self):
#         if self.current_trajectory_idx >= len(self.end_pts["start_pts"]):
#             return  # No more trajectories to plan
        
#         # Wait for waiting period before starting next trajectory
#         if self.current_trajectory is None and self.elapsed_time < self.waiting_period:
#             return

#         start_pt = self.end_pts["start_pts"][self.current_trajectory_idx]
#         end_pt = self.end_pts["end_pts"][self.current_trajectory_idx]

#         # Set up GcsTrajectoryOptimization to generate the trajectory
#         gcs = GcsTrajectoryOptimization(len(start_pt))
#         source = gcs.AddRegions([Point(start_pt)], order=0)
#         target = gcs.AddRegions([Point(end_pt)], order=0)
#         gcs.AddEdges(source, target)
#         gcs.AddTimeCost()
#         gcs.AddPathLengthCost()
#         gcs.AddPathContinuityConstraints(2)  # Acceleration continuity

#         options = GraphOfConvexSetsOptions()
#         traj, result = gcs.SolvePath(source, target, options)

#         if result.is_success():
#             self.current_trajectory = traj
#             self.start_time = time.time()  # Reset the start time for the new trajectory
#         else:
#             print(f"Failed to plan trajectory from {start_pt} to {end_pt}.")


# class VectorSplitter(LeafSystem):
#     """
#     Simple LeafSystem that takes a vector input of size out1 + out2 and splits 
#     it into 2 output vectors of size out1 and out2. This is used for multi-robot
#     plants so the output of one controller can control both robots.
#     """
#     def __init__(self, out1, out2=0):
#         super().__init__()
#         self.out1 = out1
#         self.out2 = out2
#         self.DeclareVectorInputPort("input", BasicVector(out1 + out2))
#         self.DeclareVectorOutputPort("output_1", BasicVector(out1), self.Output1)
#         if out2 > 0:
#             self.DeclareVectorOutputPort("output_2", BasicVector(out2), self.Output2)

#     def Output1(self, context, output):
#         input_vector = self.get_input_port(0).Eval(context)
#         output.SetFromVector(input_vector[:self.out1])  # return first `out1` elements

#     def Output2(self, context, output):
#         input_vector = self.get_input_port(0).Eval(context)
#         output.SetFromVector(input_vector[self.out1:])  # return latter `out2` elements


# parser = argparse.ArgumentParser()
# parser.add_argument('--joint_control', default='F', help="T/F; whether to control joint positions (instead of xyz rpy)")
# args = parser.parse_args()
# joint_control = (args.joint_control == 'T')


# meshcat = StartMeshcat()

# builder = DiagramBuilder()

# if TEST_SCENE == "BOXUNLOADING":
#     scenario = load_scenario(data=iris_yaml)
# else:
#     scenario = load_scenario(filename=scene_yaml_file)

# # Hardware station setup
# station = builder.AddSystem(MakeHardwareStation(
#     scenario=scenario,
#     meshcat=meshcat,

#     # This is to be able to load our own models from a local path
#     # we can refer to this using the "package://" URI directive
#     parser_preload_callback=lambda parser: parser.package_map().Add("iris_environments", os.path.join(data_directory, "data", "iris_benchmarks_scenes_urdf", "iris_environments", "assets")),
# ))
# scene_graph = station.GetSubsystemByName("scene_graph")
# plant = station.GetSubsystemByName("plant")

# num_robot_positions = plant.num_positions()
# default_joint_positions = plant.GetPositions(plant.CreateDefaultContext())

# if TEST_SCENE == "3DOFFLIPPER":
#     joint_control = True
# if TEST_SCENE == "5DOFUR3":
#     joint_control = True
#     ee_frame = plant.GetFrameByName("ur_ee_link")
# if TEST_SCENE == "6DOFUR3":
#     ee_frame = plant.GetFrameByName("ur_ee_link")
# if TEST_SCENE == "7DOFIIWA":
#     ee_frame = plant.GetFrameByName("iiwa_link_7")
# if TEST_SCENE == "7DOFBINS":
#     ee_frame = plant.GetFrameByName("iiwa_link_7")
# if TEST_SCENE == "7DOF4SHELVES":
#     ee_frame = plant.GetFrameByName("iiwa_link_7")
# if TEST_SCENE == "14DOFIIWAS":
#     print("Teleop does not yet work for 14DOFIIWAS.")
# if TEST_SCENE == "15DOFALLEGRO":
#     joint_control = True
# if TEST_SCENE == "BOXUNLOADING":
#     ee_frame = plant.GetFrameByName("arm_eef")

# if joint_control:
#     # Add slider for each joint
#     for i in range(num_robot_positions):
#         meshcat.AddSlider(f'q{i}', -np.pi, np.pi, 0.01, default_joint_positions[i])
# else:
#     default_pose = plant.CalcRelativeTransform(plant.CreateDefaultContext(), plant.world_frame(), ee_frame)
#     meshcat.AddSlider('x', -1.0, 2.0, 0.01, default_pose.translation()[0])
#     meshcat.AddSlider('y', -1.0, 1.0, 0.01, default_pose.translation()[1])
#     meshcat.AddSlider('z', 0.0, 3.0, 0.01, default_pose.translation()[2])
#     meshcat.AddSlider('roll', -np.pi, np.pi, 0.01, default_pose.rotation().ToRollPitchYaw().vector()[0])
#     meshcat.AddSlider('pitch', -np.pi, np.pi, 0.01, default_pose.rotation().ToRollPitchYaw().vector()[1])
#     meshcat.AddSlider('yaw', -np.pi, np.pi, 0.01, default_pose.rotation().ToRollPitchYaw().vector()[2])
# meshcat.AddButton("Close")

# # Figure out how many robots there are and how many joints each has
# model_instances_indices_with_actuators = {}
# for actuator_idx in plant.GetJointActuatorIndices():
#     robot_model_instance_idx = plant.get_joint_actuator(actuator_idx).model_instance()
#     if robot_model_instance_idx not in model_instances_indices_with_actuators.keys():
#         model_instances_indices_with_actuators[robot_model_instance_idx] = 1
#     else:
#         model_instances_indices_with_actuators[robot_model_instance_idx] += 1

# # Add controller and splitter (for when there are multiple robots)
# controller = builder.AddSystem(InverseDynamicsController(plant, [100]*num_robot_positions, [0]*num_robot_positions, [50]*num_robot_positions, True))  # True = exposes "desired_acceleration" port
# control_splitter = builder.AddSystem(VectorSplitter(*model_instances_indices_with_actuators.values()))

# # Add Trajetory Planner
# planner = builder.AddSystem(TrajectoryPlanner(plant, endpts))

# # Set controller desired state
# builder.Connect(station.GetOutputPort("state"), controller.GetInputPort("estimated_state"))
# builder.Connect(controller.GetOutputPort("generalized_force"), control_splitter.GetInputPort("input"))
# builder.Connect(planner.GetOutputPort("desired_state"), controller.GetInputPort("desired_state"))

# # Set controller desired accel
# builder.Connect(planner.GetOutputPort("desired_acceleration"), controller.GetInputPort("desired_acceleration"))

# # Connect each output of the splitter to the actuation input for each robot
# for i, (robot_model_instance_idx, num_joints) in enumerate(model_instances_indices_with_actuators.items()):
#     builder.Connect(control_splitter.GetOutputPort(f"output_{i+1}"), station.GetInputPort(f"{plant.GetModelInstanceName(robot_model_instance_idx)}_actuation"))

# diagram = builder.Build()
# diagram_visualize_connections(diagram, "diagram_.svg")
# context = diagram.CreateDefaultContext()

# simulator = Simulator(diagram)
# simulator_context = simulator.get_mutable_context()
# station_context = station.GetMyMutableContextFromRoot(simulator_context)
# plant_context = plant.GetMyMutableContextFromRoot(simulator_context)

# # Main simulation loop
# ctr = 0
# while not meshcat.GetButtonClicks("Close"):
#     simulator.AdvanceTo(simulator_context.get_time() + 0.01)