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
    logical_or,
    logical_and
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

from scenario import scenario_yaml_for_iris, q_nominal
from utils import NUM_BOXES, is_yaml_empty, SuppressOutput
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

        self.q_place = self.pick_planner.solve_q_place()
        self.target_regions = None

        self.state = 1  # 1 for picking, 0 for placing

        # self.DeclarePeriodicUnrestrictedUpdateEvent(0.025, 0.0, self.compute_command)
        self.DeclarePeriodicUnrestrictedUpdateEvent(1, 0.0, self.compute_command)


    def VisualizePath(self, traj, name):
        """
        Helper function that takes in trajopt basis and control points of Bspline
        and draws spline in meshcat.
        """
        # Build a new plant to do the forward kinematics to turn this Bspline into 3D coordinates
        # builder = DiagramBuilder()
        # vis_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        # viz_iiwa = Parser(vis_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
        # vis_plant.WeldFrames(vis_plant.world_frame(), vis_plant.GetFrameByName("base"))
        # vis_plant.Finalize()
        # vis_plant_context = vis_plant.CreateDefaultContext()

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


    def compute_command(self, context, state):
        ### Deal with Special Cases
        if context.get_time() < self.start_planning_time:
            print(f"compute_command returning due to box randomization still occuring.")
            return
        
        ### Read Input Ports
        kuka_state = self.get_input_port(0).Eval(context)
        q_current = kuka_state[:6]
        q_dot_current = kuka_state[6:]

        body_poses = self.get_input_port(1).Eval(context)
        box_poses = {}
        for box_body_idx in self.box_body_indices:
            box_poses[box_body_idx] = body_poses[box_body_idx]

        # Define GCS Program
        gcs_regions = self.source_regions.copy()
        gcs_regions["start"] = Point(q_current)

        edges = []
        with SuppressOutput():  # Suppress Gurobi spam
            gcs = GcsTrajectoryOptimization(len(q_current))
            gcs_regions = gcs.AddRegions(list(gcs_regions.values()), order=1)
            source = gcs.AddRegions([Point(q_current)], order=0)
            
        # Plan path either to pick or to placing pose
        if self.state == 1:  # Pick
            if self.target_regions is None:  # If program has just initialized
                self.target_regions = self.pick_planner.get_viable_pick_poses(box_poses)  # List of Point objects in Configuration Space
            # for i in range(len(target_regions)):
                # gcs_regions[f"target_region_{i}"] = target_regions[i]
            with SuppressOutput():  # Suppress Gurobi spam
                target = gcs.AddRegions(self.target_regions, order=0)
            for region in self.target_regions:  # If robot is very close to any of the viable pick positions, assume it has completed a pick.
                if np.all(np.isclose(q_current, region.x(), rtol=1e-05, atol=1e-08)):
                    print("GCS: Finished picking; switching to placing.")
                    self.state = 0  # Switch to placing
                    break
        else:  # Place
            # gcs_regions["goal"] = Point(self.q_place)
            with SuppressOutput():  # Suppress Gurobi spam
                target = gcs.AddRegions([Point(self.q_place)], order=0)
            if np.all(np.isclose(q_current, self.q_place, rtol=1e-05, atol=1e-08)):  # Finished placing
                print("GCS: Finished placing; switching to picking.")
                self.state = 1  # Switch to picking
                self.target_regions = self.pick_planner.get_viable_pick_poses(box_poses)  # List of Point objects in Configuration Space

        with SuppressOutput():  # Suppress Gurobi spam
            edges.append(gcs.AddEdges(source, gcs_regions))
            edges.append(gcs.AddEdges(gcs_regions, target))
        
        gcs.AddTimeCost()
        gcs.AddPathLengthCost()
        gcs.AddVelocityBounds(
            self.plant.GetVelocityLowerLimits(), self.plant.GetVelocityUpperLimits() * 0.25
        )
        
        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = 5  # Max number of distinct paths to compare during random rounding; only the lowest cost path is returned.
        start_time = time.time()
        print("Starting gcs.SolvePath.")
        with SuppressOutput():  # Suppress Gurobi spam
            traj, result = gcs.SolvePath(source, target, options)
        print(f"GCS SolvePath Runtime: {time.time() - start_time}")

        if not result.is_success():
            print("GCS Fail.")
            return

        # for edge in gcs.graph_of_convex_sets().Edges():
            # print(edge.phi())
            # print(result)
            # print(result.GetSolution(edge.phi()))

        self.VisualizePath(traj, f"GCS Traj")

        state.get_mutable_abstract_state(int(self.traj_idx)).set_value(traj)


    def output_command(self, context, output):
        # Set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self.traj_idx)).get_value()

        if (traj_q.rows() == 1):
            # print("default traj output.")
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
            # print("planner outputting default 0 acceleration")
            output.SetFromVector(np.zeros((6,)))
        else:
            output.SetFromVector(traj_q.EvalDerivative(context.get_time() - self.start_planning_time, 2))