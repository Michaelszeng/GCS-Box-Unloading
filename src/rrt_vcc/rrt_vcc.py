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
    ConfigurationSpaceObstacleCollisionChecker,
    RandomGenerator,
    PointCloud,
    Rgba,
    Quaternion,
    RigidTransform,
    RationalForwardKinematics,
    IrisInConfigurationSpaceFromCliqueCover,
    IrisFromCliqueCoverOptions,
    FastCliqueInflation,
    FastCliqueInflationOptions,
    FastIris,
    FastIrisOptions,
    IrisInConfigurationSpace,
    IrisOptions,
    SaveIrisRegionsYamlFile,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import scenario_yaml_for_iris
from iris import IrisRegionGenerator
from utils import ik
from rrt_master import *

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
import pickle
import time

# TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
# TEST_SCENE = "BOXUNLOADING"

rng = RandomGenerator(0)
np.random.seed(0)

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
cspace_obstacle_collision_checker = ConfigurationSpaceObstacleCollisionChecker(collision_checker, [])

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

def forward_kinematics(q):
    plant.SetPositions(plant_context, q)
    p = plant.CalcRelativeTransform(plant_context, frame_A=plant.world_frame(), frame_B=ee_frame).translation()
    return p

print(endpts['end_pts'])
assert len(endpts['start_pts']) == len(endpts['end_pts'])
print(f"Number of user-seeded start and end points: {len(endpts['start_pts'])}")


def mark_corners(points, tolerance=1e-5):
    """
    Return the path with only "corners", ignoring intermediate nodes along 
    straight line edges.
    """
    def is_corner(p1, p2, p3, tolerance=1e-5):
        """
        Check if the point p2 is a corner, i.e., if the direction changes between p1->p2 and p2->p3.
        """
        vec1 = p2 - p1
        vec2 = p3 - p2
        
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Check if the two vectors are collinear
        dot_product = np.dot(vec1_norm, vec2_norm)
        return not (np.abs(dot_product - 1) < tolerance or np.abs(dot_product + 1) < tolerance)

    if len(points) < 3:
        # If there are fewer than 3 points, consider all of them corners
        return [(point, True) for point in points]
    
    result = []
    
    # The start point is always a corner
    result.append(points[0])
    
    for i in range(1, len(points) - 1):
        if is_corner(points[i-1], points[i], points[i+1], tolerance):
            result.append(points[i])
    
    result.append(points[-1])
    
    return result

all_path_pts = np.empty((ambient_dim, 0))  # ambient_dim x N
all_paths = []  # list of ambient_dim x n numpy arrays
adj_mat = None
for i in range(len(endpts['start_pts'])):
    start_q = endpts['start_pts'][i]
    end_q = endpts['end_pts'][i]

    print(f"Start: {start_q}")
    print(f"End: {end_q}")

    rrt_options = RRTMasterOptions(step_size=1e-1, 
                                   check_size=5e-2, 
                                   neighbor_radius=0.2,
                                   min_vertices=1000,
                                   max_vertices=5000,
                                   goal_sample_frequency=0.1, 
                                   timeout=np.inf,
                                   index=i,
                                   draw_rrt=False,
                                   use_rrt_star=False,
                                   use_bi_rrt=True
                                  )

    path = RRTMaster(rrt_options, start_q, end_q, make_sample_q(), check_collision_free, ForwardKinematics=forward_kinematics, meshcat=cspace_meshcat if ambient_dim==3 else meshcat)
    path = mark_corners(path)

    print(f"Found path: {path != []}")
    # print(path)
    
    # Compile all path points and paths
    all_path_pts = np.hstack((all_path_pts, np.array(path).T))
    all_paths.append(np.array(path).T)
    
    # Compile all paths into single graph adjacency matrix
    temp_n = len(path)
    temp = scipy.sparse.diags(
        [[True]*temp_n, [True]*(temp_n-1), [True]*(temp_n-1)],
        [0, 1, -1],
        shape=(temp_n, temp_n),
        dtype=bool
    )
    if adj_mat is None:
        adj_mat = temp
    else:
        adj_mat = scipy.sparse.block_diag([adj_mat, temp])
    
# Remove self edges from adj_mat
non_diag_mask = adj_mat.nonzero()[0] != adj_mat.nonzero()[1]
adj_mat = csc_matrix((adj_mat.data[non_diag_mask], (adj_mat.nonzero()[0][non_diag_mask], adj_mat.nonzero()[1][non_diag_mask])), shape=adj_mat.shape)

np.set_printoptions(threshold=10000, linewidth=200)
print(f"\n{adj_mat.toarray().astype(int)}\n")
N = adj_mat.shape[0]
print(f"Graph vertex set size: {N}")

# Generate convex hull of RRT paths --> sampling domain for Clique Covers
vpoly = VPolytope(all_path_pts)
hpoly = HPolyhedron(vpoly)
IrisRegionGenerator.visualize_iris_region(plant, plant_context, cspace_meshcat if ambient_dim==3 else meshcat, [hpoly], colors = [Rgba(0.0,0.0,0.0,0.5)], name="rrt_convex_hull", task_space=(ambient_dim!=3), scene=TEST_SCENE)

options = IrisFromCliqueCoverOptions()
options.sampling_domain = hpoly  # Set domain for clique covers point samples, but not for iris regions
options.num_points_per_coverage_check = 1000
options.num_points_per_visibility_round = 1000
options.coverage_termination_threshold = 0.8
options.minimum_clique_size = ambient_dim + 1
options.iteration_limit = 5
options.fast_iris_options.max_iterations = 1
options.fast_iris_options.require_sample_point_is_contained = True
options.fast_iris_options.mixing_steps = 50
options.fast_iris_options.random_seed = 0
options.fast_iris_options.verbose = True
options.use_fast_iris = True

# Run 1 iteration of Clique Covers to cover large areas
clique_covers_regions = IrisInConfigurationSpaceFromCliqueCover(
    checker=collision_checker, options=options, generator=RandomGenerator(0), sets=[]
)  # List of HPolyhedrons
clique_covers_regions =  {str(i): clique_covers_regions[i] for i in range(len(clique_covers_regions))}  # Convert to dict for uniformity

IrisRegionGenerator.visualize_iris_region(plant, plant_context, cspace_meshcat if ambient_dim == 3 else meshcat, clique_covers_regions, name="clique_covers_regions", task_space=(ambient_dim!=3), scene=TEST_SCENE)

# Run clique inflation on remaining uncovered RRT path edges
fast_clique_inflation_options = FastCliqueInflationOptions()

fast_iris_options = FastIrisOptions()
fast_iris_options.require_sample_point_is_contained = True
fast_iris_options.configuration_space_margin = 1e-3

rrt_inflation_regions = {}
region_point_containment = {}
iris_domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())  # IRIS allowed to inflate wherever it wants

# edges_to_cover = []  # List of ambient_dim x 2 edges
# for path in all_paths:
#     n = np.shape(path)[1]  # Number of points in this path
#     for i in range(n-1):
#         # Skip/remove edge if already covered by the clique covers regions
#         step = 2e-2
#         prev_pt = None  # Keep track of previous pt that we can cut the edge at if we find it is partially covered by the clique covers regions
#         for pt in np.linspace(path[:,i], path[:,i+1], int(np.linalg.norm(path[:,i] - path[:,i+1]) / step)):
#             pt_in_sets = False
#             for r in clique_covers_regions.values():
#                 if r.PointInSet(pt):
#                     pt_in_sets = True
#             if not pt_in_sets:
#                 adj_mat = adj_mat.tolil()
#                 adj_mat[i, j] = 0
#                 adj_mat[j, i] = 0
#                 adj_mat = adj_mat.tocsc()
#                 print(f"skipping inflation for edge ({i}, {j}) due to being covered by clique covers regions.")
                    
                    

for i in range(N):
    for j in range(i+1, N):
        if adj_mat[i, j]:  # Find neighbors of point i
            # Skip/remove edge if already covered by the clique covers regions
            step = 2e-2
            for pt in np.linspace(all_path_pts[:,i:i+1], all_path_pts[:,j:j+1], int(np.linalg.norm(all_path_pts[:,i:i+1] - all_path_pts[:,i:i+1]) / step)):
                pt_in_sets = False
                for r in clique_covers_regions.values():
                    if r.PointInSet(pt):
                        pt_in_sets = True
                if not pt_in_sets:
                    adj_mat = adj_mat.tolil()
                    adj_mat[i, j] = 0
                    adj_mat[j, i] = 0
                    adj_mat = adj_mat.tocsc()
                    print(f"skipping inflation for edge ({i}, {j}) due to being covered by clique covers regions.")
                    continue
                    

            # Build region around (i,j)
            print(f"Building region around ({i}, {j})")
            if np.linalg.norm(all_path_pts[:,i:i+1] - all_path_pts[:,j:j+1]) < 1e-6:
                print("Skipping edge inflation due to zero length.")
                continue                
            line_clique = np.hstack((all_path_pts[:,i:i+1], all_path_pts[:,j:j+1]))  # ambient_dim x 2

            try:
                initial_hpoly = FastCliqueInflation(cspace_obstacle_collision_checker, line_clique, iris_domain, fast_clique_inflation_options)
                hpoly_inscribed_ellipsoid = initial_hpoly.MaximumVolumeInscribedEllipsoid()
                
                if not initial_hpoly.PointInSet(all_path_pts[:,i]) or not initial_hpoly.PointInSet(all_path_pts[:,j]):
                    print("INFLATED EDGE DOES NOT CONTAIN CLIQUE POINTS")
            except Exception as e: 
                print(f"FastCliqueInflation fail; skipping inflation of edge ({all_path_pts[:,i], all_path_pts[:,j]}). Error Message: {e}")
                continue

            try:
                hpoly = FastIris(cspace_obstacle_collision_checker, hpoly_inscribed_ellipsoid, iris_domain, fast_iris_options)
                
                if not hpoly.PointInSet(all_path_pts[:,i]) or not hpoly.PointInSet(all_path_pts[:,j]):
                    print("INFLATED REGION DOES NOT CONTAIN CLIQUE INFLATION CLIQUE POINTS")
            except Exception as e: 
                hpoly = initial_hpoly
                print(f"FastIRIS fail; defaulting to clique inflation region. Error Message: {e}")
                
            rrt_inflation_regions[f"{i},{j}"] = hpoly

            # Check whether any other points are now covered by this region
            # hpoly_scaled = hpoly.Scale(3.0)
            # M = hpoly_scaled.A() @ points <= hpoly_scaled.b()[:, None]
            M = hpoly.A() @ all_path_pts <= hpoly.b()[:, None]

            C = np.all(M, axis=0)  # (N,) array of truths

            # Remove edges between every point pair in C
            adj_mat_lil = adj_mat.tolil()
            adj_mat_lil[np.ix_(C, C)] = 0  # Remove all edges covered by region
            # adj_mat_lil[C,:] = 0  # Remove all points covered by region
            adj_mat = adj_mat_lil.tocsc()

print(f"Number of regions generated by clique covers: {len(clique_covers_regions.values())}")
print(f"Number of regions generated by RRT inflation: {len(rrt_inflation_regions.values())}")

combined_regions = {**clique_covers_regions, **rrt_inflation_regions}

SaveIrisRegionsYamlFile(f'{TEST_SCENE}_regions.yaml', combined_regions)

IrisRegionGenerator.visualize_iris_region(plant, plant_context, cspace_meshcat if ambient_dim == 3 else meshcat, rrt_inflation_regions, density=0, name="rrt_inflation_regions", task_space=(ambient_dim!=3), scene=TEST_SCENE)

coverage = IrisRegionGenerator.estimate_coverage(plant, cspace_obstacle_collision_checker, combined_regions)
print(f"c-space Coverage: {coverage}")

IrisRegionGenerator.visualize_connectivity(combined_regions, coverage, output_file=f'{TEST_SCENE}rrt_vcc_inflation_connectivity.svg', skip_svg=False)
print(f"Saved connectivity graph svg.")

print(f"{cspace_meshcat.web_url() if ambient_dim == 3 else meshcat.web_url()}/download")
time.sleep(100000)