"""
Template file that shows how to build a generic MultibodyPlant containing one of
the 9 test scenes.
"""

from pydrake.all import (
    StartMeshcat,
    AddDefaultVisualization,
    Simulator,
    VisibilityGraph,
    RobotDiagramBuilder,
    VPolytope,
    HPolyhedron,
    Hyperrectangle,
    SceneGraphCollisionChecker,
    RandomGenerator,
    PointCloud,
    Rgba,
    Quaternion,
    RigidTransform,
    RationalForwardKinematics,
    IrisInConfigurationSpaceFromCliqueCover,
    IrisFromCliqueCoverOptions,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import scenario_yaml_for_iris
from iris import IrisRegionGenerator
from utils import ik
from rrt import *
from rrt_star import *

import numpy as np
import importlib
from scipy.spatial.transform import Rotation
from scipy.sparse import find
import pickle
import time

TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
# TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
# TEST_SCENE = "BOXUNLOADING"

rng = RandomGenerator(1234)

src_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_directory = os.path.dirname(src_directory)
data_directory = os.path.join(parent_directory)
scene_yaml_file = os.path.join(data_directory, "data", "iris_benchmarks_scenes_urdf", "yamls", TEST_SCENE + ".dmd.yaml")

meshcat = StartMeshcat()


robot_diagram_builder = RobotDiagramBuilder()
parser = robot_diagram_builder.parser()
iris_environement_assets = os.path.join(data_directory, "data", "iris_benchmarks_scenes_urdf", "iris_environments", "assets")
parser.package_map().Add("iris_environments",iris_environement_assets)
if TEST_SCENE == "BOXUNLOADING":
    robot_model_instances = parser.AddModelsFromString(scenario_yaml_for_iris, ".dmd.yaml")
else:
    robot_model_instances = parser.AddModels(scene_yaml_file)
plant = robot_diagram_builder.plant()
plant.Finalize()
num_robot_positions = plant.num_positions()
AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
diagram = robot_diagram_builder.Build()
context = diagram.CreateDefaultContext()

# Roll forward sim a bit to show the visualization
simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
plant_context = diagram.GetMutableSubsystemContext(plant, context)

ambient_dim = plant.num_positions()

if ambient_dim == 3:
    cspace_meshcat = StartMeshcat()

    # Plot bounding box around C-space
    lower_corner = plant.GetPositionLowerLimits()
    upper_corner = plant.GetPositionUpperLimits()

    # Define the 8 corners of the box
    box_corners = np.array([
        [[lower_corner[0]], [lower_corner[1]], [lower_corner[2]]],  # (xmin, ymin, zmin)
        [[upper_corner[0]], [lower_corner[1]], [lower_corner[2]]],  # (xmax, ymin, zmin)
        [[lower_corner[0]], [upper_corner[1]], [lower_corner[2]]],  # (xmin, ymax, zmin)
        [[upper_corner[0]], [upper_corner[1]], [lower_corner[2]]],  # (xmax, ymax, zmin)
        [[lower_corner[0]], [lower_corner[1]], [upper_corner[2]]],  # (xmin, ymin, zmax)
        [[upper_corner[0]], [lower_corner[1]], [upper_corner[2]]],  # (xmax, ymin, zmax)
        [[lower_corner[0]], [upper_corner[1]], [upper_corner[2]]],  # (xmin, ymax, zmax)
        [[upper_corner[0]], [upper_corner[1]], [upper_corner[2]]],  # (xmax, ymax, zmax)
    ])

    # Draw lines between the corners to form the edges of the bounding box
    box_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    for edge in box_edges:
        cspace_meshcat.SetLine(f"bounding_box/box_edge_{edge[0]}_{edge[1]}", np.hstack((box_corners[edge[0]], box_corners[edge[1]])), rgba=Rgba(0, 0, 0, 1))

simulator.AdvanceTo(0.001)

collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)

if TEST_SCENE == "3DOFFLIPPER":
    joint_control = True
if TEST_SCENE == "5DOFUR3":
    joint_control = True
    ee_frame = plant.GetFrameByName("ur_ee_link")
    ee_body = plant.GetBodyByName("ur_ee_link")
if TEST_SCENE == "6DOFUR3":
    ee_frame = plant.GetFrameByName("ur_ee_link")
    ee_body = plant.GetBodyByName("ur_ee_link")
if TEST_SCENE == "7DOFIIWA":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
    ee_body = plant.GetBodyByName("iiwa_link_7")
if TEST_SCENE == "7DOFBINS":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
    ee_body = plant.GetBodyByName("iiwa_link_7")
if TEST_SCENE == "7DOF4SHELVES":
    ee_frame = plant.GetFrameByName("iiwa_link_7")
    ee_body = plant.GetBodyByName("iiwa_link_7")
if TEST_SCENE == "14DOFIIWAS":
    print("Teleop does not yet work for 14DOFIIWAS.")
if TEST_SCENE == "15DOFALLEGRO":
    joint_control = True
if TEST_SCENE == "BOXUNLOADING":
    ee_frame = plant.GetFrameByName("arm_eef")
    ee_body = plant.GetBodyByName("arm_eef")

pickle_file = f'{TEST_SCENE}_endpts.pkl'
with open(pickle_file, 'rb') as f:
    endpts = pickle.load(f)

def check_collision_free(q):
    return collision_checker.CheckConfigCollisionFree(q)
    
def make_sample_q():
    domain = Hyperrectangle(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    last_polytope_sample = domain.UniformSample(rng)

    def sample_q():
        return domain.UniformSample(rng)
    
    return sample_q

assert len(endpts['start_pts']) == len(endpts['end_pts'])

all_path_pts = np.array(endpts['start_pts'][0]).reshape(3,1)  # Initialize ambient_dim x N matrix to hold all points in RRTs
for i in range(len(endpts['start_pts'])):
    start_q = endpts['start_pts'][i]
    end_q = endpts['end_pts'][i]

    print(f"Start: {start_q}")
    print(f"End: {end_q}")

    # rrt_options = RRTOptions(step_size=1e-1, 
    #                          check_size=1e-2, 
    #                         #  max_vertices=1e3,
    #                          max_vertices=1
    #                          max_iters=1e4, 
    #                          goal_sample_frequency=0.05, 
    #                          always_swap=False,
    #                          timeout=np.inf)

    # rrt = RRT(make_sample_q(), check_collision_free, meshcat=cspace_meshcat)

    rrt_options = RRTOptions(step_size=1e-1, 
                             check_size=1e-2, 
                             max_vertices=500,
                             max_iters=1e4, 
                             goal_sample_frequency=0.05, 
                             timeout=np.inf,
                             index=i)

    rrt = RRTStar(make_sample_q(), check_collision_free, meshcat=cspace_meshcat)

    path = rrt.plan(start_q, end_q, rrt_options)

    print(f"Found path: {path != []}")
    
    print(np.shape(np.array(path)))
    
    all_path_pts = np.hstack((all_path_pts, np.array(path).T))
    
vpoly = VPolytope(all_path_pts)
hpoly = HPolyhedron(vpoly)
IrisRegionGenerator.visualize_iris_region(plant, plant_context, cspace_meshcat if ambient_dim == 3 else meshcat, [hpoly], name="rrt_convex_hull", task_space=(ambient_dim!=3), scene=TEST_SCENE)

options = IrisFromCliqueCoverOptions()
options.num_points_per_coverage_check = 1000
options.num_points_per_visibility_round = 1000
options.coverage_termination_threshold = 0.7
options.minimum_clique_size = 8
options.iteration_limit = 1
options.fast_iris_options.max_iterations = 1
# options.fast_iris_options.require_sample_point_is_contained = True
options.fast_iris_options.mixing_steps = 50
options.fast_iris_options.random_seed = 0
options.fast_iris_options.verbose = True
options.use_fast_iris = True

# Very scuffed way of setting the domain for clique covers
options.iris_options.bounding_region = hpoly

regions = IrisInConfigurationSpaceFromCliqueCover(
    checker=collision_checker, options=options, generator=RandomGenerator(0), sets=[]
)  # List of HPolyhedrons

IrisRegionGenerator.visualize_iris_region(plant, plant_context, cspace_meshcat if ambient_dim == 3 else meshcat, regions, task_space=(ambient_dim!=3), scene=TEST_SCENE)

time.sleep(10)