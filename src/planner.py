from pydrake.all import (
    LeafSystem,
    BasicVector,
    AbstractValue,
    Role,
    JointIndex,
    RigidTransform,
    IrisOptions,
    IrisInConfigurationSpace,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    StartMeshcat,
    Simulator,
    AddDefaultVisualization,
    Trajectory,
    LoadIrisRegionsYamlFile,
    Point
)

from manipulation.meshcat_utils import AddMeshcatTriad

from utils import VisualizePath, ik
from gcs import gcs_traj_opt
from gcs_shortest_walks import *
from poses import grasp_q, deposit_q, get_deposit_poses, get_grasp_poses

import time
import numpy as np
from enum import Enum
from pathlib import Path

class State(Enum):
    RUNNING_TO_GRASP = 0
    PAUSING_TO_GRASP = 1
    RUNNING_TO_DEPOSIT = 3
    PAUSING_TO_DEPOSIT = 4

class MotionPlanner(LeafSystem):
    def __init__(self, meshcat, scene_graph, plant_with_objs, plant, box_randomization_runtime, regions_file, regions_place_file):
        LeafSystem.__init__(self)

        self.robot_state_input_port = self.DeclareVectorInputPort(name="kuka_state", size=12)  # 6 pos, 6 vel
        self.body_poses_input_port = self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))
        self.DeclareVectorOutputPort("kuka_desired_state", 12, self.output_desired_state)  # 6 pos, 6 vel
        self.DeclareVectorOutputPort("kuka_acceleration", 6, self.output_desired_acceleration)

        self.USE_SHORTEST_WALKS = False

        self.plan_stage_ = State.RUNNING_TO_GRASP
        
        self.meshcat = meshcat
        self.scene_graph = scene_graph
        self.plant_with_objs = plant_with_objs
        self.plant = plant
        
        self.plant_with_objs_context = None
        self.plant_context = None
        self.scene_graph_context = None

        self.first_call = True
        self.pause_length = 0.25
        self.start_planning_time = box_randomization_runtime

        self.active_suction_joints = []

        self.obj_num = 0
        self.traj_num = 0
        self.current_traj = None
        self.current_traj_start_time = 0
        
        self.source_regions = LoadIrisRegionsYamlFile(Path(regions_file))
        self.source_regions_place = LoadIrisRegionsYamlFile(Path(regions_place_file))
        
        self.grasp_poses = get_grasp_poses()
        self.deposit_poses = get_deposit_poses()

        self.suctioned_item_name = None
        self.object_in_eef_pose = None


    def set_context(self, scene_graph_context, plant_with_objs_context, controller_plant_context):
        self.scene_graph_context = scene_graph_context
        self.plant_with_objs_context = plant_with_objs_context
        self.plant_context = controller_plant_context
        

    def getNextTrajectory(self, context):
        robot_q = self.robot_state_input_port.Eval(context)[:6]
        
        if self.plan_stage_ == State.RUNNING_TO_GRASP or self.plan_stage_ == State.RUNNING_TO_DEPOSIT:
            deposit_pose = self.deposit_poses[self.obj_num]

            AddMeshcatTriad(self.meshcat, "visuals/deposit_pose", X_PT=deposit_pose)

            if self.USE_SHORTEST_WALKS:
                traj = load_data_for_trajectory(self.traj_num)
            else:
                if self.plan_stage_ == State.RUNNING_TO_GRASP:
                    q_ik = ik(self.plant, self.plant.CreateDefaultContext(), self.grasp_poses[self.obj_num], translation_error=0, rotation_error=0.01, regions=None, pose_as_constraint=True)[0]
                    print(f"q_ik:{q_ik}")
                    q = grasp_q[self.obj_num]
                    print(f"q:{q}")
                    traj = gcs_traj_opt(self.plant, robot_q, [Point(q)], self.source_regions, regions_to_add=None)
                elif self.plan_stage_ == State.RUNNING_TO_DEPOSIT:
                    q_ik = ik(self.plant, self.plant.CreateDefaultContext(), self.deposit_poses[self.obj_num], translation_error=0, rotation_error=0.01, regions=None, pose_as_constraint=True)[0]
                    print(f"q_ik:{q_ik}")
                    q = deposit_q[self.obj_num]
                    print(f"q:{q}")
                    traj = gcs_traj_opt(self.plant, robot_q, [Point(q)], self.source_regions_place, regions_to_add=None)
                
            self.current_traj_start_time = context.get_time()
            self.traj_num += 1
            
            VisualizePath(self.meshcat, self.plant, self.plant.CreateDefaultContext(), traj, f"visuals/traj_{self.traj_num}")

            self.stack_time = context.get_time()
            self.current_traj_duration = traj.end_time()
            self.current_traj_final_pos = traj.value(self.current_traj_duration)

            return traj
        
        elif self.plan_stage_ == State.PAUSING_TO_DEPOSIT or self.plan_stage_ == State.PAUSING_TO_GRASP:
            return [self.pause_length]


    def exit_condition_reached(self, context, exit_threshold_grasp=0.005, exit_threshold_deposit=0.02, time_window=1.0):
        current_time = context.get_time()        
        current_position = self.robot_state_input_port.Eval(context)[:6]

        # Exit condition for grasping/placing is whether we are close enough to the target position
        if self.plan_stage_ == State.RUNNING_TO_GRASP or self.plan_stage_ == State.RUNNING_TO_DEPOSIT:
            plant_context = self.plant.CreateDefaultContext()
            
            # Use forward kinematics to compute current robot pose
            self.plant.SetPositions(plant_context, current_position)
            current_pose_xyz = self.plant.CalcRelativeTransform(plant_context, self.plant.world_frame(), self.plant.GetFrameByName("arm_eef")).translation()
            
            # Use forward kinematics to compute desired robot pose
            if isinstance(self.current_traj, Trajectory):
                self.plant.SetPositions(plant_context, self.current_traj.value(self.current_traj.end_time()))
            else:
                self.plant.SetPositions(plant_context, get_pos_vel_acc_jerk(self.current_traj, get_trajectory_length(self.current_traj))[0])
            target_pose_xyz = self.plant.CalcRelativeTransform(plant_context, self.plant.world_frame(), self.plant.GetFrameByName("arm_eef")).translation()

            current_net_distance_from_target = np.linalg.norm(current_pose_xyz - target_pose_xyz)

            if self.plan_stage_ == State.RUNNING_TO_DEPOSIT:
                if current_net_distance_from_target < exit_threshold_deposit:
                    return True
            else:
                if current_net_distance_from_target < exit_threshold_grasp:
                    return True

        # Exit condition during pauses is simply if pause time has elapsed
        elif self.plan_stage_ == State.PAUSING_TO_GRASP or self.plan_stage_ == State.PAUSING_TO_DEPOSIT:
            if current_time - self.stack_time - self.current_traj_duration - self.current_traj[0] > 0:
                return True
            
        return False
    
    
    def output_desired_state(self, context, output):
        # print(self.plan_stage_)

        if self.first_call:
            print("generating first trajectory.")
            self.stack_time = context.get_time()
            self.current_traj = self.getNextTrajectory(context)
            self.first_call = False

        if self.plan_stage_ == State.RUNNING_TO_GRASP or self.plan_stage_ == State.RUNNING_TO_DEPOSIT:
            if isinstance(self.current_traj, Trajectory):
                new_state = np.concatenate((
                            self.current_traj.value(context.get_time() - self.stack_time),
                            self.current_traj.EvalDerivative(context.get_time() - self.stack_time)
                        ))
                print(f"Desired q: {self.current_traj.value(context.get_time() - self.stack_time).flatten()}")
                print(f"Current q: {self.robot_state_input_port.Eval(context)[:6]}")
            else:
                p,v,a,j = get_pos_vel_acc_jerk(self.current_traj, context.get_time() - self.current_traj_start_time)
                new_state = np.concatenate((p, v))
        else:
            new_state = np.pad(self.current_traj_final_pos.flatten(), (0, 6), 'constant')

        output.SetFromVector(new_state)
        # print(f"outputting desired state: {desired_state}")

        if self.exit_condition_reached(context):
            states = list(State)
            index = states.index(self.plan_stage_)
            next_index = (index + 1) % len(states)
            self.plan_stage_ = states[next_index]

            if self.plan_stage_ == State.RUNNING_TO_GRASP:
                self.obj_num += 1

            # elif self.plan_stage_ == State.PAUSING_TO_GRASP:
            #     body_poses = self.body_poses_input_port.Eval(context)
            #     eef_pose = body_poses[self.plant_with_objs.GetBodyByName("eef").index()]

            #     # Get the geometry ID of the end effector tip
            #     tip_geom_id = self.scene_graph.model_inspector().GetGeometries(
            #         self.plant_with_objs.GetBodyFrameIdOrThrow(self.plant_with_objs.GetBodyByName("eef").index()),
            #         Role.kProximity,
            #     )[0]

            #     query_object = self.scene_graph.get_query_output_port().Eval(
            #         self.scene_graph_context
            #     )
            #     inspector = query_object.inspector()

            #     suction_max_distance = 0.025
            #     closest_pair = None
            #     closest_distance = float('inf')  # Initialize to infinity
            #     closest_ycb_body = None

            #     ycb_link_names = ["base_link_cracker", "base_link_sugar", "base_link_mustard", "base_link_gelatin" , "base_link_meat"]

            #     # Loop to find the closest YCB object
            #     for pair in query_object.ComputeSignedDistancePairwiseClosestPoints(
            #         suction_max_distance
            #     ):
            #         # Identify if either A or B is the end effector
            #         suctioned_geom_id = None
            #         if pair.id_A == tip_geom_id:
            #             suctioned_geom_id = pair.id_B
            #         elif pair.id_B == tip_geom_id:
            #             suctioned_geom_id = pair.id_A
            #         else:
            #             continue

            #         # Get the frame ID and body of the suctioned object
            #         suctioned_frame_id = inspector.GetFrameId(suctioned_geom_id)
            #         suctioned_body = self.plant_with_objs.GetBodyFromFrameId(suctioned_frame_id)
            #         print(f"Suctioned body: {suctioned_body.name()}")

            #         # Check if the suctioned body is in the YCB list
            #         if suctioned_body.name() not in ycb_link_names:
            #             continue  # Ignore objects not in the YCB list

            #         print("suctioned body is in the YCB list")
            #         print("distance: ", pair.distance)

            #         # Check if this is the closest valid YCB object so far
            #         if pair.distance < closest_distance:
            #             closest_distance = pair.distance
            #             closest_pair = pair
            #             closest_ycb_body = suctioned_body

            #     # If a closest valid YCB object is found, proceed to weld it
            #     if closest_ycb_body is not None:
            #         # Lock the joint of the closest YCB object
            #         for i in range(self.plant_with_objs.num_joints()):
            #             joint = self.plant_with_objs.get_joint(JointIndex(i))
            #             if joint.child_body() == closest_ycb_body:
            #                 joint.Lock(self.plant_with_objs_context)
            #                 self.active_suction_joints.append(joint)

            #                 # Save suctioned item name and pose relative to the eef for online iris region generation
            #                 self.suctioned_item_name = closest_ycb_body.name()
            #                 object_pose = body_poses[closest_ycb_body.index()]
            #                 self.object_in_eef_pose = eef_pose.inverse() @ object_pose
            #                 print(f"Suctioned the {closest_ycb_body.name()}")

            # elif self.plan_stage_ == State.PAUSING_TO_DEPOSIT:
            #     print("turning suction off")
            #     for joint in self.active_suction_joints:
            #         joint.Unlock(self.plant_with_objs_context)
            #     self.active_suction_joints = []

            self.current_traj = self.getNextTrajectory(context)


    def output_desired_acceleration(self, context, output):
        if self.plan_stage_ == State.RUNNING_TO_GRASP or self.plan_stage_ == State.RUNNING_TO_DEPOSIT:
            if isinstance(self.current_traj, Trajectory):
                new_accel = self.current_traj.EvalDerivative(context.get_time() - self.start_planning_time, 2)
            else:
                p,v,a,j = get_pos_vel_acc_jerk(self.current_traj, context.get_time() - self.current_traj_start_time)
                new_accel = a
        else:
            new_accel = np.zeros(6)

        output.SetFromVector(new_accel)


    def get_plan_stage(self):
        return self.plan_stage_