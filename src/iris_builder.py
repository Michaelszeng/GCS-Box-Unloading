from pydrake.all import (
    StartMeshcat,
    AddDefaultVisualization,
    Simulator,
    VisibilityGraph,
    RobotDiagramBuilder,
    VPolytope,
    HPolyhedron,
    SceneGraphCollisionChecker,
    RandomGenerator,
    PointCloud,
    Rgba,
    Quaternion,
    RigidTransform,
    FastIris,
    FastIrisOptions,
    SaveIrisRegionsYamlFile,
    LoadIrisRegionsYamlFile,
    Hyperellipsoid,
    RotationMatrix,
    QuaternionFloatingJoint,
    ApplyMultibodyPlantConfig,
    MultibodyPlantConfig,
    MultibodyPlant,
    Parser,
    GeometrySet,
    Role,
    CollisionFilterDeclaration,
)

from manipulation.meshcat_utils import AddMeshcatTriad

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import NUM_BOXES, get_fast_box_poses, scenario_yaml_with_boxes, BOX_DIM, q_nominal, q_place_nominal, scenario_yaml, robot_yaml, scenario_yaml_for_iris, robot_pose, set_hydroelastic, set_up_scene, get_W_X_eef
from utils import ik
from iris import IrisRegionGenerator
from poses import get_grasp_poses, get_deposit_poses, grasp_q, deposit_q

import numpy as np
import importlib
from scipy.spatial.transform import Rotation
from scipy.sparse import find
import time

JUST_VISUALIZE_EXISTING_REGIONS = True
# JUST_VISUALIZE_EXISTING_REGIONS = False

box_names = ["Boxes/Box_4", "Boxes/Box_16", "Boxes/Box_17", "Boxes/Box_12"]

meshcat = StartMeshcat()

if not JUST_VISUALIZE_EXISTING_REGIONS:
    rng = RandomGenerator(1234)
    np.random.seed(1234)

    seeds = get_grasp_poses() + [get_deposit_poses()[0]] + [RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.84, 0, 1.74]),
                                                            RigidTransform(RotationMatrix.MakeXRotation(np.pi), [1.5, 0, 1.0])]

    regions_dict = {}
    for i, seed in enumerate(seeds):

        robot_diagram_builder = RobotDiagramBuilder()
        parser = robot_diagram_builder.parser()
        scene_graph = robot_diagram_builder.scene_graph()
        robot_model_instances = parser.AddModelsFromString(scenario_yaml_with_boxes, ".dmd.yaml")

        plant = robot_diagram_builder.plant()

        plant_config = MultibodyPlantConfig(
                discrete_contact_solver="sap",
                time_step=0.001
            )
        ApplyMultibodyPlantConfig(plant_config, plant)


        fast_box_poses = get_fast_box_poses()  # Get pre-computed box poses

        # Set poses for all boxes
        for j in range(NUM_BOXES):
            box_name = f"Boxes/Box_{j}"
            box_model_idx = plant.GetModelInstanceByName(box_name)  # ModelInstanceIndex
            box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex
            
            box_frame = plant.GetFrameByName("Box_0_5_0_5_0_5", box_model_idx)
            
            if i < 4 and box_name in box_names[:i]:
                plant.WeldFrames(plant.world_frame(), box_frame, RigidTransform([100,0,0]))
            else:
                plant.WeldFrames(plant.world_frame(), box_frame, fast_box_poses[j])


        plant.Finalize()
        AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
        diagram = robot_diagram_builder.Build()



        simulator = Simulator(diagram)

        simulator_context = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(simulator_context)


        num_robot_positions = plant.num_positions()

        robot_model_instances = robot_model_instances[:2]
        for x in robot_model_instances:
            print(plant.GetModelInstanceName(x))
        print(robot_model_instances)
        collision_checker_params = {}
        collision_checker_params["robot_model_instances"] = robot_model_instances
        collision_checker_params["model"] = diagram
        collision_checker_params["edge_step_size"] = 0.125
        collision_checker = SceneGraphCollisionChecker(**collision_checker_params)

        # Roll forward sim a bit to show the visualization
        simulator.AdvanceTo(0.001)

        # RUN IRIS
        options = FastIrisOptions()
        options.random_seed = 0
        options.verbose = False
        options.require_sample_point_is_contained = True
        domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(),
                                    plant.GetPositionUpperLimits())
        kEpsilonEllipsoid = 1e-5
        AddMeshcatTriad(meshcat, f"{i}", X_PT=seed)
        q = ik(plant, plant_context, seed, translation_error=0, rotation_error=0.05, regions=None, pose_as_constraint=True)[0]
        print(f"seed: {q.flatten()}")
        clique_ellipse = Hyperellipsoid.MakeHypersphere(kEpsilonEllipsoid, q)
        region = FastIris(collision_checker, clique_ellipse, domain, options)

        regions_dict[f"set{i}"] = region
        
        
    SaveIrisRegionsYamlFile("IRIS_REGIONS.yaml", regions_dict)

else:
    robot_diagram_builder = RobotDiagramBuilder()
    parser = robot_diagram_builder.parser()
    scene_graph = robot_diagram_builder.scene_graph()
    robot_model_instances = parser.AddModelsFromString(scenario_yaml_with_boxes, ".dmd.yaml")

    plant = robot_diagram_builder.plant()

    plant_config = MultibodyPlantConfig(
            discrete_contact_solver="sap",
            time_step=0.001
        )
    ApplyMultibodyPlantConfig(plant_config, plant)


    fast_box_poses = get_fast_box_poses()  # Get pre-computed box poses

    # Set poses for all boxes
    for j in range(NUM_BOXES):
        box_name = f"Boxes/Box_{j}"
        box_model_idx = plant.GetModelInstanceByName(box_name)  # ModelInstanceIndex
        box_body_idx = plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex
        
        box_frame = plant.GetFrameByName("Box_0_5_0_5_0_5", box_model_idx)
        
        plant.WeldFrames(plant.world_frame(), box_frame, fast_box_poses[j])

    plant.Finalize()
    AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
    diagram = robot_diagram_builder.Build()
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(simulator_context)

    # Roll forward sim a bit to show the visualization
    simulator.AdvanceTo(0.001)

regions_dict = LoadIrisRegionsYamlFile("IRIS_REGIONS.yaml")

qs = grasp_q + deposit_q
for i in range(5):
    in_set = False
    for s in regions_dict.values():
        if s.PointInSet(qs[i]):
            in_set = True
    print(in_set)

# for i in range(4):
#     print(regions_dict[f"set{i}"].PointInSet(grasp_q[i]))
# print(regions_dict[f"set{4}"].PointInSet(deposit_q[0]))

# Visualize IRIS regions
IrisRegionGenerator.visualize_connectivity(regions_dict, 0.0, output_file='IRIS_REGIONS.svg', skip_svg=False)
IrisRegionGenerator.visualize_iris_region(plant, plant_context, meshcat, regions_dict)

time.sleep(100)