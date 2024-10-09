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
    Hyperellipsoid,
    SceneGraphCollisionChecker,
    ConfigurationSpaceObstacleCollisionChecker,
    RandomGenerator,
    PointCloud,
    Sphere,
    Rgba,
    Quaternion,
    RigidTransform,
    PRM,
    FastCliqueInflation,
    FastCliqueInflationOptions,
    FastIris,
    FastIrisOptions,
    IrisInConfigurationSpace,
    IrisOptions,
    LoadIrisRegionsYamlFile,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import scenario_yaml_for_iris
from iris import IrisRegionGenerator
from utils import ik

import numpy as np
import time
from scipy.spatial.transform import Rotation
from scipy.sparse import csc_matrix, lil_matrix

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

scene_yaml_file = os.path.dirname(os.path.abspath(__file__)) + "/../../data/iris_benchmarks_scenes_urdf/yamls/" + TEST_SCENE + ".dmd.yaml"

# Load restricted domain if it exists
if os.path.exists(f"{TEST_SCENE}.yaml"):
    domain = list(LoadIrisRegionsYamlFile(f"{TEST_SCENE}.yaml").values())[0]

meshcat = StartMeshcat()


robot_diagram_builder = RobotDiagramBuilder()
parser = robot_diagram_builder.parser()
parser.package_map().Add("iris_environments", os.path.dirname(os.path.abspath(__file__)) + "/../../data/iris_benchmarks_scenes_urdf/iris_environments/assets")
if TEST_SCENE == "BOXUNLOADING":
    robot_model_instances = parser.AddModelsFromString(scenario_yaml_for_iris, ".dmd.yaml")
else:
    robot_model_instances = parser.AddModels(scene_yaml_file)
plant = robot_diagram_builder.plant()
scene_graph = robot_diagram_builder.scene_graph()
plant.Finalize()
AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
diagram = robot_diagram_builder.Build()

# Roll forward sim a bit to show the visualization
simulator = Simulator(diagram)
simulator.AdvanceTo(0.001)

plant_context = plant.GetMyMutableContextFromRoot(diagram.CreateDefaultContext())

ambient_dim = plant.num_positions()

if ambient_dim == 3:
    cspace_meshcat = StartMeshcat()

collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)
cspace_obstacle_collision_checker = ConfigurationSpaceObstacleCollisionChecker(collision_checker, [])

if not os.path.exists(f"{TEST_SCENE}.yaml"):
    domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    # domain = HPolyhedron.MakeBox([plant.GetPositionLowerLimits()[0], plant.GetPositionLowerLimits()[1], 1.5], plant.GetPositionUpperLimits())  # reduce size of domain to make it easer to tell what's going on

# Sample to build PRM
N = 60
points = np.zeros((ambient_dim, N))  # ambient_dim x N
last_polytope_sample = domain.UniformSample(rng, domain.ChebyshevCenter())
for i in range(np.shape(points)[1]):
    last_polytope_sample = domain.UniformSample(rng, last_polytope_sample)

    while not cspace_obstacle_collision_checker.CheckConfigCollisionFree(last_polytope_sample):
        last_polytope_sample = domain.UniformSample(rng, last_polytope_sample)

    points[:, i] = last_polytope_sample

# Build PRM
RADIUS = np.pi/2  # Radians
prm = PRM(cspace_obstacle_collision_checker, points, RADIUS)  # Adjacency Mat. Scipy csc_matrix

# Remove self edges from PRM
non_diag_mask = prm.nonzero()[0] != prm.nonzero()[1]
prm = csc_matrix((prm.data[non_diag_mask], (prm.nonzero()[0][non_diag_mask], prm.nonzero()[1][non_diag_mask])), shape=prm.shape)

# Prune edges between two vertices that both have more than MAX_NEIGHBORS neighbors
MAX_NEIGHBORS = 3
for i in range(N):
    num_neighbors = np.count_nonzero(prm.getrow(i).toarray().flatten())

    if num_neighbors <= MAX_NEIGHBORS:
        continue
    
    neighbors = prm.getrow(i)
    for neighbor_idx, val in zip(neighbors.indices, neighbors.data):
        if val == 1 and np.count_nonzero(prm.getrow(neighbor_idx).toarray().flatten()) > MAX_NEIGHBORS:  # both current vtx and neighbor have degree more than 3
            prm[i,neighbor_idx] = 0
            prm[neighbor_idx,i] = 0
        
        # Check if current vtx now has MAX_NEIGHBORS or less neighbors
        if np.count_nonzero(prm.getrow(i).toarray().flatten()) <= MAX_NEIGHBORS:
            break

print(f"\n{prm.toarray().astype(int)}\n")

if ambient_dim == 3:  # Visualize PRM in 3D cspace
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

    # Draw edges of PRM
    for i in range(N):
        for j in range(i + 1, N):
            if prm[i, j] == 1:
                cspace_meshcat.SetLine(f"prm/edges/({i},{j})", np.hstack((points[:,i:i+1], points[:,j:j+1])), rgba=Rgba(0, 0, 1, 1))

    # Draw nodes of PRM
    for i in range(N):
        cspace_meshcat.SetObject(f"prm/points/{i}", Sphere(radius=0.02), rgba=Rgba(0, 0, 1, 1))
        cspace_meshcat.SetTransform(f"prm/points/{i}", RigidTransform(points[:,i]))
else:  # Visualize PRM in task space
    if TEST_SCENE == "5DOFUR3":
        ee_frame = plant.GetFrameByName("ur_ee_link")
    if TEST_SCENE == "6DOFUR3":
        ee_frame = plant.GetFrameByName("ur_ee_link")
    if TEST_SCENE == "7DOFIIWA":
        ee_frame = plant.GetFrameByName("iiwa_link_7")
    if TEST_SCENE == "7DOFBINS":
        ee_frame = plant.GetFrameByName("iiwa_link_7")
    if TEST_SCENE == "7DOF4SHELVES":
        ee_frame = plant.GetFrameByName("iiwa_link_7")
    if TEST_SCENE == "14DOFIIWAS":
        print("visualize_iris_region for 14DOFIIWAS in task space to be implemented.")
    if TEST_SCENE == "15DOFALLEGRO":
        print("visualize_iris_region for 15DOFALLEGRO in task space to be implemented.")
    if TEST_SCENE == "BOXUNLOADING":
        ee_frame = plant.GetFrameByName("arm_eef")

    # Draw edges of PRM
    for i in range(N):
        for j in range(i + 1, N):
            if prm[i, j] == 1:
                plant.SetPositions(plant_context, points[:,i])
                p1 = plant.CalcRelativeTransform(plant_context, frame_A=plant.world_frame(), frame_B=ee_frame).translation()

                plant.SetPositions(plant_context, points[:,j])
                p2 = plant.CalcRelativeTransform(plant_context, frame_A=plant.world_frame(), frame_B=ee_frame).translation()

                meshcat.SetLine(f"prm/edges/({i},{j})", np.hstack((p1.reshape(3,1), p2.reshape(3,1))), rgba=Rgba(0, 0, 1, 1))

    # Draw nodes of PRM
    for i in range(N):
        meshcat.SetObject(f"prm/points/{i}", Sphere(radius=0.02), rgba=Rgba(0, 0, 1, 1))

        plant.SetPositions(plant_context, points[:,i])
        p = plant.CalcRelativeTransform(plant_context, frame_A=plant.world_frame(), frame_B=ee_frame)
        
        meshcat.SetTransform(f"prm/points/{i}", p)

# Jump to a new "path" if the current path ends
options = FastCliqueInflationOptions()
# options = IrisOptions()
# options.require_sample_point_is_contained = True

fast_iris_options = FastIrisOptions()

options.configuration_space_margin = 1e-3
last_point_idx = 0  # Start with the first point
cspace_coverage = 0
COVERAGE_THRESH = 0.35
# Used for IRIS utility functions
iris_gen = IrisRegionGenerator(meshcat, cspace_obstacle_collision_checker, f"data/iris_regions_prm_{TEST_SCENE}.yaml", DEBUG=True)

print(f"Number of edges in PRM: {np.sum(prm)/2}")

regions = {}
region_point_containment = {}
for i in range(N):
    for j in range(i+1, N):
        if prm[i, j]:  # Find neighbors of point i

            # Build region around (i,j)
            print(f"Building region around ({i}, {j})")
            line_clique = np.hstack((points[:,i:i+1], points[:,j:j+1]))  # ambient_dim x 2

            initial_hpoly = FastCliqueInflation(cspace_obstacle_collision_checker, line_clique, domain, options)
            hpoly_inscribed_ellipsoid = initial_hpoly.MaximumVolumeInscribedEllipsoid()

            # plant.SetPositions(plant_context, (points[:,i] +  points[:,j]) / 2)
            # fastiris_ellipse = Hyperellipsoid.MakeHypersphere(1e-5, plant.GetPositions(plant_context))
            hpoly = FastIris(cspace_obstacle_collision_checker, hpoly_inscribed_ellipsoid, domain, fast_iris_options)

            # plant.SetPositions(plant_context, (points[:,i] +  points[:,j]) / 2)
            # start_time = time.time()
            # hpoly = IrisInConfigurationSpace(plant, plant_context, options)

            regions[f"{i},{j}"] = hpoly

            # Check whether any other points are now covered by this region
            # hpoly_scaled = hpoly.Scale(3.0)
            # M = hpoly_scaled.A() @ points <= hpoly_scaled.b()[:, None]
            M = hpoly.A() @ points <= hpoly.b()[:, None]

            C = np.all(M, axis=0)  # (N,) array of truths

            # Remove edges between every point pair in C
            prm_lil = prm.tolil()
            prm_lil[np.ix_(C, C)] = 0  # Remove all edges covered by region
            # prm_lil[C,:] = 0  # Remove all points covered by region
            prm = prm_lil.tocsc()

print(f"Number of regions generated: {len(regions)}")

IrisRegionGenerator.visualize_iris_region(plant, plant_context, cspace_meshcat if ambient_dim == 3 else meshcat, regions, task_space=(ambient_dim!=3), scene=TEST_SCENE)
coverage = IrisRegionGenerator.estimate_coverage(plant, cspace_obstacle_collision_checker, regions)
IrisRegionGenerator.visualize_connectivity(regions, coverage, output_file='prm_regions_connectivity.svg', skip_svg=False)

print(f"c-space Coverage: {coverage}")


print(f"{cspace_meshcat.web_url() if ambient_dim == 3 else meshcat.web_url()}/download")
time.sleep(10)


# while cspace_coverage < COVERAGE_THRESH:
#     # Find the first point that connects to last_point_idx
#     found_edge = False
#     for i in range(N):
#         if i == last_point_idx:
#             continue

#         if prm[last_point_idx, i] != 0:
#             # Remove that edge from prm to ensure we don't find it again
#             prm[last_point_idx, i] = 0
#             prm[i, last_point_idx] = 0

#             # Only inflate a region here if both points forming the clique are not in other regions
#             print(regions.values())
#             if all(not (r.PointInSet(points[:, last_point_idx]) or r.PointInSet(points[:, i])) for r in regions.values()):            
#                 line_clique = np.hstack((points[:,last_point_idx:last_point_idx+1], points[:,i:i+1]))  # ambient_dim x 2
#                 region = FastCliqueInflation(cspace_obstacle_collision_checker, line_clique, domain, options)
#                 regions[f"{last_point_idx},{i}"] = region

#             # Update last_point_idx
#             last_point_idx = i
#             found_edge = True
#             break
    
#     # If this point had no edges, pick another next
#     if not found_edge:
#         last_point_idx += 1

#     cspace_coverage = iris_gen.estimate_coverage(regions, num_samples=500)