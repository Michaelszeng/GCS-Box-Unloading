from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BsplineTrajectory,
    DiagramBuilder,
    IrisInConfigurationSpace,
    IrisOptions,
    MathematicalProgram,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PositionConstraint,
    Rgba,
    RigidTransform,
    Role,
    Solve,
    StartMeshcat,
)
from pydrake.multibody import inverse_kinematics
from manipulation.meshcat_utils import AddMeshcatTriad

from scenario import scenario_yaml_for_source_regions

import time
import numpy as np

def test_iris_region(plant, meshcat, regions):
    """
    Test some key points and see if these are contained in the IRIS region.
    """
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("arm_eef")
    q_nominal = np.array([0.0, 0.0, 0.0, 1.5, -1.8, 0.0])  # nominal joint for joint-centering

    keypoints = [[-1,-1,0.25],
                 [-1,1,0.25],
                 [0,-1,0.25],
                 [0,1,0.25],
                 [-1,-1,2.5],
                 [-1,1,2.5],
                 [0,-1,2.5],
                 [0,1,2.5]]

    qs = {}
    for kp in keypoints:
        AddMeshcatTriad(meshcat, str(kp), X_PT=RigidTransform(kp), opacity=0.5)

        # Use IK to turn this into joint coords
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        ik_prog = ik.prog()
        ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_nominal, q_variables)
        ik.AddPositionConstraint(
            frameA=world_frame,
            frameB=ee_frame,
            p_BQ=[0, 0, 0.1],
            p_AQ_lower=kp,
            p_AQ_upper=kp,
        )
        ik_prog.SetInitialGuess(q_variables, q_nominal)
        start = time.time()
        ik_result = Solve(ik_prog)
        print(f"ik time: {time.time()-start}")
        if not ik_result.is_success():
            print(f"ERROR: IK fail on kp {kp}: {ik_result.get_solver_id().name()}.")
            print(ik_result.GetInfeasibleConstraintNames(ik_prog))
        else:
            print(f"IK success on kp {kp}.")

        q = ik_result.GetSolution(q_variables)  # (6,) np array
        qs[tuple(kp)] = q

    for region in regions:
        for p, q in qs.items():
            if region.PointInSet(q):
                print(f"region {region} contains point {p}.")
            else:
                print(f"region {region} does NOT contains point {p}.")

    

def generate_source_iris_regions(meshcat):
    """
    Source IRIS regions are defined as the regions considering only self-
    collision with the robot, and collision with the walls of the empty truck
    trailed (excluding the back wall).
    """

    # Create new MBP containing just robot and truck trailer walls
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    print(scenario_yaml_for_source_regions)
    parser.AddModelsFromString(scenario_yaml_for_source_regions, ".dmd.yaml")
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # Run IRIS
    regions = []

    options = IrisOptions()
    options.num_collision_infeasible_samples = 1
    options.require_sample_point_is_contained = True
    region = IrisInConfigurationSpace(plant, plant_context, options)
    regions.append(region)

    test_iris_region(plant, meshcat, regions)
    
