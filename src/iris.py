from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BsplineTrajectory,
    DiagramBuilder,
    IrisInConfigurationSpace,
    IrisOptions,
    Hyperellipsoid,
    HPolyhedron,
    RandomGenerator,
    PointCloud,
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

def test_iris_region(plant, plant_context, meshcat, regions, seed=42, num_sample=1000):
    """
    Plot small spheres in the volume of each region. (we are using forward
    kinematics to return from configuration space to task space.)
    """
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("arm_eef")

    rng = RandomGenerator(seed)

    for i in range(len(regions)):
        region = regions[i]

        xyzs = []  # List to hold XYZ positions of configurations in the IRIS region

        q_sample = region.UniformSample(rng)
        prev_sample = q_sample

        plant.SetPositions(plant_context, q_sample)
        xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame).translation())

        for _ in range(num_sample-1):
            q_sample = region.UniformSample(rng, prev_sample)
            prev_sample = q_sample

            plant.SetPositions(plant_context, q_sample)
            xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame).translation())
        
        # Create pointcloud from sampled point in IRIS region in order to plot in Meshcat
        xyzs = np.array(xyzs)
        pc = PointCloud(len(xyzs))
        pc.mutable_xyzs()[:] = xyzs.T
        meshcat.SetObject(f"region {i}", pc, point_size=0.01, rgba=Rgba(0.5,0.0,0.0,0.5))



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
    model_instances = parser.AddModelsFromString(scenario_yaml_for_source_regions, ".dmd.yaml")
    print(f"model_instances: {model_instances}")
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
    # region = IrisInConfigurationSpace(plant, plant_context, options)

    region = HPolyhedron.MakeBox([-0.1,-0.1,-0.1,1.2,-2.1,-0.1], [0.1,0.1,0.1,1.8,-1.5,0.1])

    regions.append(region)

    test_iris_region(plant, plant_context, meshcat, regions)



    # params = dict(edge_step_size=0.125)
    # builder = RobotDiagramBuilder()
    # params["robot_model_instances"] = builder.parser().AddModelsFromString(
    #     limits_urdf, "urdf"
    # )
    # params["model"] = builder.Build()
    # checker = SceneGraphCollisionChecker(**params)

    # options = mut.IrisFromCliqueCoverOptions()
    # options.num_builders = 3  # set to 1 fewer than number of cores on computer
    # options.num_points_per_coverage_check = 1000
    # options.num_points_per_visibility_round = 500  # 1000
    # options.coverage_termination_threshold = 0.2  # set low threshold at first for faster debugging

    # # import logging
    # # logger = logging.getLogger("drake")
    # # logger.setLevel(logging.DEBUG)
    # # from pydrake.common import configure_logging
    # # configure_logging()

    # sets = mut.IrisInConfigurationSpaceFromCliqueCover(
    #     checker=checker, options=options, sets=[]
    # )
    
