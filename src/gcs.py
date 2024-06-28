from pydrake.all import (
    LeafSystem,
    AbstractValue,
    GraphOfConvexSetsOptions,
    GcsTrajectoryOptimization,
    Point,
    LoadIrisRegionsYamlFile,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Solve,
    CompositeTrajectory,
    PiecewisePolynomial,
    PathParameterizedTrajectory,
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

from scenario import scenario_yaml_for_iris, q_nominal
from utils import NUM_BOXES, PREPICK_MARGIN, is_yaml_empty, SuppressOutput
from pick_planner import PickPlanner

import time
import numpy as np
from pathlib import Path


class MotionPlanner(LeafSystem):

    def __init__(self, original_plant, meshcat, robot_pose, box_randomization_runtime):
        LeafSystem.__init__(self)

        kuka_state = self.DeclareVectorInputPort(name="kuka_state", size=12)  # 6 pos, 6 vel

        body_poses = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("kuka_current_pose", body_poses)

        self.traj_idx = self.DeclareAbstractState(
            AbstractValue.Make(CompositeTrajectory([PiecewisePolynomial.FirstOrderHold(
                                                        [0, 1],
                                                        np.array([[0, 0]])
                                                    )]))
        )

        self.DeclareVectorOutputPort(
            "kuka_desired_state", 12, self.output_command  # 6 pos, 6 vel
        )

        self.DeclareVectorOutputPort(
            "kuka_acceleration", 6, self.output_acceleration
        )

        ### Create MBP for IK and Traj. Opt.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        ConfigureParser(parser)
        kuka = parser.AddModelsFromString(scenario_yaml_for_iris, ".dmd.yaml")[1]  # ModelInstance object

        # Weld robot base in place
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", plant.GetModelInstanceByName("robot_base")), robot_pose)
        
        plant.Finalize()
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        # Member variables
        self.plant = plant
        self.plant_context = plant_context
        self.kuka = kuka
        self.robot_pose = robot_pose
        self.original_plant = original_plant
        self.meshcat = meshcat

        self.source_regions = LoadIrisRegionsYamlFile(Path("../data/iris_source_regions.yaml"))
        self.previous_compute_result = None  # BsplineTrajectory object
        self.start_planning_time = box_randomization_runtime
        self.visualize = True
        
        self.box_body_indices = []
        for i in range(NUM_BOXES):
            box_model_idx = original_plant.GetModelInstanceByName(f"Boxes/Box_{i}")  # ModelInstanceIndex
            box_body_idx = original_plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex
            self.box_body_indices.append(box_body_idx)
        self.pick_planner = PickPlanner(self.meshcat, self.robot_pose, self.source_regions, self.box_body_indices, self.plant, self.plant_context)

        self.state = 1  # 1 for picking, 0 for placing
        
        self.q_pick = None
        self.q_place = self.pick_planner.solve_q_place()
        self.target_regions = None
        self.traj = None

        # self.DeclarePeriodicUnrestrictedUpdateEvent(0.025, 0.0, self.compute_command)
        self.DeclarePeriodicUnrestrictedUpdateEvent(1, 0.0, self.compute_command)


    def VisualizePath(self, traj, name):
        """
        Helper function that takes in trajopt basis and control points of Bspline
        and draws spline in meshcat.
        """
        traj_start_time = traj.start_time()
        traj_end_time = traj.end_time()

        # Build matrix of 3d positions by doing forward kinematics at time steps in the bspline
        NUM_STEPS = 80
        pos_3d_matrix = np.zeros((3,NUM_STEPS))
        ctr = 0
        for vis_t in np.linspace(traj_start_time, traj_end_time, NUM_STEPS):
            pos = traj.value(vis_t)
            self.plant.SetPositions(self.plant_context, self.kuka, pos)
            pos_3d = self.plant.CalcRelativeTransform(self.plant_context, self.plant.world_frame(), self.plant.GetFrameByName("arm_eef")).translation()
            pos_3d_matrix[:,ctr] = pos_3d
            ctr += 1

        # Draw line
        if self.visualize:
            self.meshcat.SetLine(name, pos_3d_matrix)


    def perform_gcs_traj_opt(self, q_current, target_regions):
        """
        Define and run a GCS Trajectory Optimization program.

        q-current is a 7D np array containing the robot's current configuration.

        target_regions is a list of ConvexSet objects containing the desired set
        of end configurations for the trajectory optimization.
        """
        # Define GCS Program
        gcs_regions = self.source_regions.copy()
        gcs_regions["start"] = Point(q_current)

        edges = []
        gcs = GcsTrajectoryOptimization(len(q_current))
        gcs_regions = gcs.AddRegions(list(gcs_regions.values()), order=3)
        source = gcs.AddRegions([Point(q_current)], order=0)
        target = gcs.AddRegions(target_regions, order=0)

        edges.append(gcs.AddEdges(source, gcs_regions))
        edges.append(gcs.AddEdges(gcs_regions, target))
        
        gcs.AddTimeCost()
        gcs.AddPathLengthCost()
        gcs.AddPathContinuityConstraints(2)  # Acceleration continuity
        gcs.AddVelocityBounds(
            self.plant.GetVelocityLowerLimits(), self.plant.GetVelocityUpperLimits()
        )
        
        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = 5  # Max number of distinct paths to compare during random rounding; only the lowest cost path is returned.
        start_time = time.time()
        traj, result = gcs.SolvePath(source, target, options)
        print(f"GCS: GCS SolvePath Runtime: {time.time() - start_time}")

        if not result.is_success():
            print("GCS: GCS Fail.")
            return

        # for edge in gcs.graph_of_convex_sets().Edges():
            # print(edge.phi())
            # print(result)
            # print(result.GetSolution(edge.phi()))

        self.VisualizePath(traj, f"GCS Traj")

        return traj
    

    def generate_pick_traj(self, q_current, q_dot_current, q_pick):
        """
        Compute simply trajectory starting from a pre-pick pose (q_current) to
        a pick pose (q_pick).

        q_current is a 6D numpy vector.

        q_dot_current is a 6D numpy vector.

        q_pick is a 6D numpy vector.
        """
        breaks = [0, 1]  # allocate 1 second to go from pre-pick to pick
        samples = np.hstack((q_current.reshape(-1, 1), q_pick.reshape(-1, 1)))
        sample_dot_at_start = q_dot_current
        sample_dot_at_end = np.zeros(6)
        traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(breaks, samples, sample_dot_at_start, sample_dot_at_end)

        self.VisualizePath(traj, f"Pick Traj")

        return traj


    def correct_traj_time(self, traj, context):
        """
        Takes a trajectory and time shifts it to begin at the current time.

        traj is a Trajectory object.

        context is the context passed from compute_command().
        """
        time_shift = context.get_time()  # Time shift value in seconds
        time_scaling_traj = PiecewisePolynomial.FirstOrderHold(
            [time_shift, time_shift + traj.end_time()],  # Assuming two segments: initial and final times
            np.array([[0, traj.end_time() - traj.start_time()]])  # Shifts start and end times by time_shift
        )
        time_shifted_final_traj = PathParameterizedTrajectory(
            traj, time_scaling_traj
        )
        return time_shifted_final_traj


    def compute_command(self, context, state):
        ### Deal with Special Cases
        if context.get_time() < self.start_planning_time:
            print(f"GCS: compute_command returning due to box randomization still occuring.")
            return
        
        ### Read Input Ports
        kuka_state = self.get_input_port(0).Eval(context)
        q_current = kuka_state[:6]
        q_dot_current = kuka_state[6:]

        body_poses = self.get_input_port(1).Eval(context)
        box_poses = {}
        for box_body_idx in self.box_body_indices:
            box_poses[box_body_idx] = body_poses[box_body_idx]
            
        # Plan path either to pick or to placing pose
        if self.state == 1:  # Pre-Pick
            if self.target_regions is None:  # If program has just initialized
                self.target_regions = self.pick_planner.get_viable_pick_poses(box_poses)  # List of Point objects in Configuration Space
                self.traj = self.perform_gcs_traj_opt(q_current, list(self.target_regions.keys()))
            # Check if robot is very close to any of the viable pre-pick positions --> assume it is ready to transition to picking
            for region, pre_pick_pose in self.target_regions.items():
                if np.all(np.isclose(q_current, region.x(), rtol=1e-03, atol=1e-03)):
                    print("GCS: Reached pre-pick pose; switching to picking.")
                    self.state = 2  # Switch to picking
                    self.q_pick = self.pick_planner.solve_q_pick(pre_pick_pose)
                    self.traj = self.correct_traj_time(self.generate_pick_traj(q_current, q_dot_current, self.q_pick), context)  # region.x is the configuration at the pick pose
                    break
        elif self.state == 2:  # Picking
            # Check if we are finished picking up the box --> transition to placing
            if np.all(np.isclose(q_current, self.q_pick, rtol=1e-03, atol=1e-03)):
                print("GCS: Finished picking; switching to placing.")
                self.state = 0  # Switch to placing
                self.traj = self.correct_traj_time(self.perform_gcs_traj_opt(q_current, [Point(self.q_place)]), context)
        else:  # Place
            # Check if we are finished placing --> transition back to pre-pick
            if np.all(np.isclose(q_current, self.q_place, rtol=1e-03, atol=1e-03)):
                print("GCS: Finished placing; switching to picking.")
                self.state = 1  # Switch to picking
                self.target_regions = self.pick_planner.get_viable_pick_poses(box_poses)  # List of Point objects in Configuration Space
                self.traj = self.correct_traj_time(self.perform_gcs_traj_opt(q_current, list(self.target_regions.keys())), context)

        state.get_mutable_abstract_state(int(self.traj_idx)).set_value(self.traj)


    def output_command(self, context, output):
        # Set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self.traj_idx)).get_value()

        if (traj_q.rows() == 1):
            # print("GCS: default traj output.")
            output.SetFromVector(np.append(
                q_nominal,
                np.zeros((6,))
            ))
        else:
            output.SetFromVector(np.append(
                traj_q.value(context.get_time() - self.start_planning_time),
                traj_q.EvalDerivative(context.get_time() - self.start_planning_time)
            ))


    def output_acceleration(self, context, output):
        # Set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self.traj_idx)).get_value()

        if (traj_q.rows() == 1):
            # print("GCS: planner outputting default 0 acceleration")
            output.SetFromVector(np.zeros((6,)))
        else:
            output.SetFromVector(traj_q.EvalDerivative(context.get_time() - self.start_planning_time, 2))