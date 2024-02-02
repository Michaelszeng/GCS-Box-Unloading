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
    PiecewisePolynomial
)
from manipulation.meshcat_utils import AddMeshcatTriad

from scenario import robot_yaml
from utils import is_yaml_empty

import os
import time
import numpy as np
from pathlib import Path


class MotionPlanner(LeafSystem):

    def __init__(self, plant, meshcat):
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

        # MBP containing only robot joint for IK and Traj. Opt.
        builder = DiagramBuilder()
        robot_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        kuka = Parser(robot_plant).AddModelsFromString(robot_yaml, ".dmd.yaml")  # ModelInstance object
        robot_plant.Finalize()
        robot_plant_context = robot_plant.CreateDefaultContext()

        # Member variables
        self.plant = plant
        self.robot_plant = robot_plant
        self.robot_plant_context = robot_plant_context
        self.meshcat = meshcat
        self.q_nominal = np.array([0.0, 0.0, 0.0, 1.5, -1.8, 0.0])  # nominal joint for joint-centering
        self.q_end = None
        self.previous_compute_result = None  # BpslineTrajectory object
        self.visualize = True

        # self.DeclarePeriodicUnrestrictedUpdateEvent(0.025, 0.0, self.compute_command)
        self.DeclarePeriodicUnrestrictedUpdateEvent(1, 0.0, self.compute_command)


    def compute_command(self, context, state):
        return
        kuka_state = self.get_input_port(0).Eval(context)
        q_current = kuka_state[:6]
        q_dot_current = kuka_state[6:]

        if is_yaml_empty("../data/iris_source_regions.yaml"):
            print(f"compute_command returning due to no IRIS regions ready yet.")
            return

        regions = LoadIrisRegionsYamlFile(Path("../data/iris_source_regions.yaml"))

        # Use IK to solve for joint positions to drop the box off at
        X_WGoal = RigidTransform([-1.0, 0.07, 1.5])
        ik = InverseKinematics(self.robot_plant, self.robot_plant_context)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        ik_prog = ik.get_mutable_prog()
        ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_current, q_variables)
        ik.AddPositionConstraint(
            frameA=self.robot_plant.world_frame(),
            frameB=self.robot_plant.GetFrameByName("arm_eef"),
            p_BQ=[0, 0, 0.1],
            p_AQ_lower=X_WGoal.translation(),
            p_AQ_upper=X_WGoal.translation(),
        )
        # ik.AddOrientationConstraint(
        #     frameAbar=self.robot_plant.world_frame(),
        #     R_AbarA=X_WGoal.rotation(),
        #     frameBbar=self.robot_plant.GetFrameByName("arm_eef"),
        #     R_BbarB=RotationMatrix(),
        #     theta_bound=0.05,
        # )
        ik_prog.SetInitialGuess(q_variables, self.q_nominal)
        ik_result = Solve(ik_prog)
        if not ik_result.is_success():
            print(f"ERROR: IK fail: {ik_result.get_solver_id().name()}.")
            print(ik_result.GetInfeasibleConstraintNames(ik_prog))

        q_goal = ik_result.GetSolution(q_variables)  # (6,) np array

        edges = []

        gcs = GcsTrajectoryOptimization(len(q_current))
        regions = gcs.AddRegions(list(regions.values()), order=1)
        source = gcs.AddRegions([Point(q_current)], order=0)
        target = gcs.AddRegions([Point(q_goal)], order=0)
        edges.append(gcs.AddEdges(source, regions))
        edges.append(gcs.AddEdges(regions, target))
        
        gcs.AddTimeCost()
        print(f"self.robot_plant.GetVelocityLowerLimits(): {self.robot_plant.GetVelocityLowerLimits()}")
        gcs.AddVelocityBounds(
            self.robot_plant.GetVelocityLowerLimits(), self.robot_plant.GetVelocityUpperLimits() * 0.25
        )
        
        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = 5  # Max number of distinct paths to compare during random rounding; only the lowest cost path is returned.
        start_time = time.time()
        print("Running GCS")
        traj, result = gcs.SolvePath(source, target, options)
        print(f"GCS SolvePath Runtime: {time.time() - start_time}")

        for edge in gcs.graph_of_convex_sets().Edges():
            print(result.GetSolution(edge.phi()))

        state.get_mutable_abstract_state(int(self._traj_index)).set_value(traj)


    def output_command(self, context, output):
        # Set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self._traj_index)).get_value()

        if (traj_q.rows() == 1):
            output.SetFromVector(np.append(
                self.plant.GetPositions(self.plant.CreateDefaultContext(), self.plant.GetModelInstanceByName("kuka")),
                np.zeros((6,))
            ))
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
            output.SetFromVector(np.zeros((7,)))
        else:
            output.SetFromVector(traj_q.EvalDerivative(context.get_time(), 2))