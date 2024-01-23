from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BsplineTrajectory,
    DiagramBuilder,
    IrisInConfigurationSpace,
    IrisOptions,
    Hyperellipsoid,
    MathematicalProgram,
    Parser,
    PositionConstraint,
    Rgba,
    RigidTransform,
    Solve,
    Sphere,
    InverseKinematics,
    MeshcatVisualizer,
    StartMeshcat,
    Simulator
)
from manipulation.meshcat_utils import AddMeshcatTriad

from scenario import scenario_yaml_for_source_regions

import time
import numpy as np

# Corners of cuboid
keypoints = [[0.5,-1,0.25],
                [0.5,1,0.25],
                [-0.5,-1,0.25],
                [-0.5,1,0.25],
                [0.5,-1,2.5],
                [0.5,1,2.5],
                [-0.5,-1,2.5],
                [-0.5,1,2.5]]

def test_iris_region(plant, plant_context, meshcat, regions):
    """
    Plot small spheres in the volume of each region. (we are using forward
    kinematics to return from configuration space to task space.)
    """
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("arm_eef")
    q_nominal = np.array([0.0, 0.0, 0.0, 1.5, -1.8, 0.0])  # nominal joint for joint-centering
    
    for kp in keypoints:
        AddMeshcatTriad(meshcat, str(kp), X_PT=RigidTransform(kp), opacity=0.5)
    
    # Finding the minimum and maximum for each dimension
    x_min = min(keypoint[0] for keypoint in keypoints)
    x_max = max(keypoint[0] for keypoint in keypoints)
    y_min = min(keypoint[1] for keypoint in keypoints)
    y_max = max(keypoint[1] for keypoint in keypoints)
    z_min = min(keypoint[2] for keypoint in keypoints)
    z_max = max(keypoint[2] for keypoint in keypoints)

    # Step size
    step = 0.1

    # Iterating through the volume
    for x in np.arange(x_min, x_max, step):
        for y in np.arange(y_min, y_max, step):
            for z in np.arange(z_min, z_max, step):
                kp = [x, y, z]
                
                # Use IK to turn this into joint coords
                ik = InverseKinematics(plant, plant_context)
                q_variables = ik.q()  # Get variables for MathematicalProgram
                ik_prog = ik.get_mutable_prog()
                ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_nominal, q_variables)
                ik.AddPositionConstraint(
                    frameA=world_frame,
                    frameB=ee_frame,
                    p_BQ=[0, 0, 0.1],
                    p_AQ_lower=kp,
                    p_AQ_upper=kp,
                )
                ik_prog.SetInitialGuess(q_variables, q_nominal)
                ik_result = Solve(ik_prog)
                if not ik_result.is_success():
                    print(f"ERROR: IK fail on kp {kp}: {ik_result.get_solver_id().name()}.")
                    print(ik_result.GetInfeasibleConstraintNames(ik_prog))

                q = ik_result.GetSolution(q_variables)  # (6,) np array

                for i in range(len(regions)):
                    region = regions[i]
                    if region.PointInSet(q):
                        print(f"region {i} contains point {kp}.")
                        meshcat.SetObject(f"IRISRegionSpheres/region {i}, {kp}", Sphere(0.05), Rgba(0.5, 0, 0, 0.5))
                        meshcat.SetTransform(f"IRISRegionSpheres/region {i}, {kp}", RigidTransform(kp))
                    # else:
                    #     print(f"region {i} does NOT contain point {kp}.")


def generate_source_iris_regions(meshcat, robot_pose, visualize_iris_scene=True):
    """
    Source IRIS regions are defined as the regions considering only self-
    collision with the robot, and collision with the walls of the empty truck
    trailer (excluding the back wall).
    """

    # Create new MBP containing just robot and truck trailer walls
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    print(scenario_yaml_for_source_regions)
    parser.AddModelsFromString(scenario_yaml_for_source_regions, ".dmd.yaml")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", plant.GetModelInstanceByName("robot_base")), robot_pose)
    plant.Finalize()
    if visualize_iris_scene:
        meshcat2 = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat2)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    if visualize_iris_scene:
        simulator = Simulator(diagram)
        simulator.AdvanceTo(0.1)

    # Run IRIS
    regions = []
    options = IrisOptions()
    options.num_collision_infeasible_samples = 1
    options.require_sample_point_is_contained = True
    seed_center = [sum(kp[0] for kp in keypoints)/8, sum(kp[1] for kp in keypoints)/8, sum(kp[2] for kp in keypoints)/8]
    options.starting_ellipse = Hyperellipsoid(np.identity(3), seed_center)
    region = IrisInConfigurationSpace(plant, plant_context, options)
    regions.append(region)

    test_iris_region(plant, plant_context, meshcat, regions)
    
