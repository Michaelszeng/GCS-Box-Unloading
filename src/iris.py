from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BsplineTrajectory,
    DiagramBuilder,
    IrisInConfigurationSpace,
    IrisOptions,
    Hyperellipsoid,
    HPolyhedron,
    IrisFromCliqueCoverOptions,
    IrisInConfigurationSpaceFromCliqueCover,
    SceneGraphCollisionChecker,
    SaveIrisRegionsYamlFile,
    LoadIrisRegionsYamlFile,
    RandomGenerator,
    PointCloud,
    RobotDiagramBuilder,
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
    Simulator,
)
from manipulation.meshcat_utils import AddMeshcatTriad

from scenario import scenario_yaml_for_iris

import os
import time
import numpy as np
from pathlib import Path
import pydot
from IPython.display import SVG, display_jpeg, display_pdf, display_png, Image


def test_iris_region(plant, plant_context, meshcat, regions, seed=42, num_sample=50000):
    """
    Plot small spheres in the volume of each region. (we are using forward
    kinematics to return from configuration space to task space.)
    """
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("arm_eef")

    rng = RandomGenerator(seed)

    colors = [Rgba(0.5,0.0,0.0,0.5),
              Rgba(0.0,0.5,0.0,0.5),
              Rgba(0.0,0.0,0.5,0.5),
              Rgba(0.5,0.5,0.0,0.5),
              Rgba(0.5,0.0,0.5,0.5),
              Rgba(0.0,0.5,0.5,0.5),
              Rgba(0.2,0.2,0.2,0.5),
              Rgba(0.5,0.2,0.0,0.5),
              Rgba(0.2,0.5,0.0,0.5),
              Rgba(0.5,0.0,0.2,0.5),
              Rgba(0.2,0.0,0.5,0.5),
              Rgba(0.0,0.5,0.2,0.5),
              Rgba(0.0,0.2,0.5,0.5),
              ]

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
        meshcat.SetObject(f"region {i}", pc, point_size=0.025, rgba=colors[i % len(colors)])


def VisualizeConnectivity(iris_regions):
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


def generate_source_iris_regions(meshcat, robot_pose, box_poses, minimum_clique_size=7, use_previous_saved_sets=True, visualize_connectivity=True, visualize_iris_scene=False):
    """
    Source IRIS regions are defined as the regions considering only self-
    collision with the robot, and collision with the walls of the empty truck
    trailer (excluding the back wall).

    Note: visualize_iris_scene is not working right now.
    """
    params = dict(edge_step_size=0.125)
    robot_diagram_builder = RobotDiagramBuilder()
    diagram_builder = robot_diagram_builder.builder()

    # Add boxes to scenario yaml
    yaml = scenario_yaml_for_iris
    for i in range(len(box_poses)):
        relative_path_to_box = '../data/Box_0_5_0_5_0_5.sdf'
        absolute_path_to_box = os.path.abspath(relative_path_to_box)

        yaml += f"""
- add_model: 
    name: Boxes/Box_{i}
    file: file://{absolute_path_to_box}
"""

    params["robot_model_instances"] = robot_diagram_builder.parser().AddModelsFromString(yaml, ".dmd.yaml")
    plant = robot_diagram_builder.plant()

    # Set Pose of each box (from simulating the box randomization) and weld
    for i in range(len(box_poses)):
        box_model_idx = plant.GetModelInstanceByName(f"Boxes/Box_{i}")  # ModelInstanceIndex
        box_frame = plant.GetFrameByName("Box_0_5_0_5_0_5", box_model_idx)
        plant.WeldFrames(plant.world_frame(), box_frame, box_poses[i])

    scene_graph = robot_diagram_builder.scene_graph()
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", plant.GetModelInstanceByName("robot_base")), robot_pose)
    if visualize_iris_scene:
        meshcat2 = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(diagram_builder, scene_graph, meshcat2)
    diagram = robot_diagram_builder.Build()
    params["model"] = diagram
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    checker = SceneGraphCollisionChecker(**params)
    options = IrisFromCliqueCoverOptions()
    options.num_builders = 7  # set to 1 fewer than number of cores on computer
    options.num_points_per_coverage_check = 1000
    options.num_points_per_visibility_round = 250  # 1000
    options.coverage_termination_threshold = 0.35
    options.minimum_clique_size = minimum_clique_size  # minimum of 7 points needed to create a shape with volume in 6D

    options.iris_options.random_seed = 0

    if use_previous_saved_sets:
        print("Using saved iris regions.")
        sets = LoadIrisRegionsYamlFile(Path("../data/iris_source_regions.yaml"))
        sets = [hpolytope for hpolytope in sets.values()]
    else:
        sets = []

    sets = IrisInConfigurationSpaceFromCliqueCover(
        checker=checker, options=options, sets=sets
    )

    sets_dict ={f"set{i}" : sets[i] for i in range(len(sets))}
    SaveIrisRegionsYamlFile(Path("../data/iris_source_regions.yaml"), sets_dict)

    if visualize_connectivity:
        VisualizeConnectivity(sets_dict)

    test_iris_region(plant, plant_context, meshcat, sets)
    
