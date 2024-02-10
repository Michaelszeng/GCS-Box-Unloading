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

from scenario import scenario_yaml_for_iris
from utils import NUM_BOXES, is_yaml_empty, SuppressOutput

import time
import numpy as np
from pathlib import Path
import pydot
import os


class MotionPlanner(LeafSystem):

    def __init__(self, original_plant, meshcat, robot_pose, box_randomization_runtime):
        LeafSystem.__init__(self)

        kuka_state = self.DeclareVectorInputPort(name="kuka_state", size=12)  # 6 pos, 6 vel

        body_poses = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("kuka_current_pose", body_poses)

        self._traj_index = self.DeclareAbstractState(
            AbstractValue.Make(CompositeTrajectory([PiecewisePolynomial.FirstOrderHold(
                                                        [0, 1],
                                                        np.array([[0, 0]])
                                                    )]))
        )

        self.DeclareVectorOutputPort(
            "kuka_command", 12, self.output_command  # 6 pos, 6 vel
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
        self.q_nominal = np.array([0.0, 0.0, 0.0, 1.5, -1.8, 0.0])  # nominal joint for joint-centering
        self.X_W_Deposit = RigidTransform(RotationMatrix.MakeXRotation(3.14159265), robot_pose.translation() + [0.0, -0.65, 1.0])
        AddMeshcatTriad(meshcat, "X_W_Deposit", X_PT=self.X_W_Deposit, opacity=0.5)
        self.source_regions = LoadIrisRegionsYamlFile(Path("../data/iris_source_regions.yaml"))
        self.previous_compute_result = None  # BpslineTrajectory object
        self.start_planning_time = box_randomization_runtime
        self.visualize = True

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


    def visualize_connectivity(self, iris_regions):
        """
        Create and display SVG graph of IRIS Region connectivity
        """
        numEdges = 0

        graph = pydot.Dot("IRIS region connectivity")
        keys = list(iris_regions.keys())
        for k in keys:
            graph.add_node(pydot.Node(k))
        for i in range(len(keys)):
            v1 = iris_regions[keys[i]]
            for j in range(i + 1, len(keys)):
                v2 = iris_regions[keys[j]]
                if v1.IntersectsWith(v2):
                    numEdges += 1
                    graph.add_edge(pydot.Edge(keys[i], keys[j], dir="both"))

        svg = graph.create_svg()

        with open('../data/iris_connectivity.svg', 'wb') as svg_file:
            svg_file.write(svg)

        return numEdges


    def compute_command(self, context, state):
        ### Deal with Special Cases
        if is_yaml_empty("../data/iris_source_regions.yaml"):
            print(f"compute_command returning due to no IRIS regions ready yet.")
            return
        # if context.get_time() < self.start_planning_time:
        #     print(f"compute_command returning due to box randomization still occuring.")
        #     return

        ### Read Input Ports
        kuka_state = self.get_input_port(0).Eval(context)
        q_current = kuka_state[:6]
        q_dot_current = kuka_state[6:]

        # Solve an IK program for the box deposit position
        # Separate IK program for each region with the constraint that the IK result must be in that region
        ik_start = time.time()
        solve_success = False
        for region in list(self.source_regions.values()):
            ik = InverseKinematics(self.plant, self.plant_context)
            q_variables = ik.q()  # Get variables for MathematicalProgram
            ik_prog = ik.get_mutable_prog()

            # ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_current, q_variables)
            ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), self.q_nominal, q_variables)

            # q_variables must be within half-plane for every half-plane in region
            ik_prog.AddConstraint(logical_and(*[expr <= const for expr, const in zip(region.A() @ q_variables, region.b())]))

            # Pose constraint
            ik.AddPositionConstraint(
                frameA=self.plant.world_frame(),
                frameB=self.plant.GetFrameByName("arm_eef"),
                p_BQ=[0, 0, 0.1],
                p_AQ_lower=self.X_W_Deposit.translation(),
                p_AQ_upper=self.X_W_Deposit.translation(),
            )
            ik.AddOrientationConstraint(
                frameAbar=self.plant.world_frame(),
                R_AbarA=self.X_W_Deposit.rotation(),
                frameBbar=self.plant.GetFrameByName("arm_eef"),
                R_BbarB=RotationMatrix(),
                theta_bound=0.05,
            )

            ik_prog.SetInitialGuess(q_variables, self.q_nominal)
            ik_result = Solve(ik_prog)
            if ik_result.is_success():
                q_goal = ik_result.GetSolution(q_variables)  # (6,) np array
                print(f"IK solve succeeded. q_goal: {q_goal}")
                solve_success = True
            # else:
            #     print(f"ERROR: IK fail: {ik_result.get_solver_id().name()}.")
            #     print(ik_result.GetInfeasibleConstraintNames(ik_prog))

        print(f"IK Runtime: {time.time() - ik_start}")

        if solve_success == False:
            print("IK Solve Failed. GCS unable to work.")
            return        
        
        gcs_regions = self.source_regions.copy()
        gcs_regions["start"] = Point(q_current)
        gcs_regions["goal"] = Point(q_goal)
        if self.visualize:
            with SuppressOutput():  # Suppress Gurobi spam
                self.visualize_connectivity(gcs_regions)
            # print("Connectivity graph saved to ../data/iris_connectivity.svg.")

        edges = []

        with SuppressOutput():  # Suppress Gurobi spam
            gcs = GcsTrajectoryOptimization(len(q_current))
            gcs_regions = gcs.AddRegions(list(gcs_regions.values()), order=1)
            source = gcs.AddRegions([Point(q_current)], order=0)
            target = gcs.AddRegions([Point(q_goal)], order=0)
            edges.append(gcs.AddEdges(source, gcs_regions))
            edges.append(gcs.AddEdges(gcs_regions, target))
        
        gcs.AddTimeCost()
        gcs.AddVelocityBounds(
            self.plant.GetVelocityLowerLimits(), self.plant.GetVelocityUpperLimits() * 0.25
        )
        
        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = 5  # Max number of distinct paths to compare during random rounding; only the lowest cost path is returned.
        start_time = time.time()
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

        state.get_mutable_abstract_state(int(self._traj_index)).set_value(traj)


    def output_command(self, context, output):
        # Set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self._traj_index)).get_value()

        if (traj_q.rows() == 1):
            output.SetFromVector(self.get_input_port(0).Eval(context))
        else:
            output.SetFromVector(np.append(
                traj_q.value(context.get_time()),
                traj_q.EvalDerivative(context.get_time())
            ))


    def output_acceleration(self, context, output):
        # Set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self._traj_index)).get_value()

        if (traj_q.rows() == 1):
            # print("planner outputting default 0 acceleration")
            output.SetFromVector(np.zeros((6,)))
        else:
            output.SetFromVector(traj_q.EvalDerivative(context.get_time(), 2))