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
    Rgba,
    Quaternion,
    RigidTransform,
    LoadIrisRegionsYamlFile,
    FastIris,
    FastIrisOptions,
    MathematicalProgram,
    Solve,
    Sphere,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import scenario_yaml_for_iris
from utils import ik

import numpy as np
import importlib
from scipy.spatial import ConvexHull

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
np.random.seed(1234)

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
AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
diagram = robot_diagram_builder.Build()

# Roll forward sim a bit to show the visualization
simulator = Simulator(diagram)
simulator.AdvanceTo(0.001)

plant_context = plant.CreateDefaultContext()

ambient_dim = plant.num_positions()
default_joint_positions = plant.GetPositions(plant.CreateDefaultContext())

collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)
cspace_obstacle_collision_checker = ConfigurationSpaceObstacleCollisionChecker(collision_checker, [])


################################################################################
### Begin Sequential Region Generation
################################################################################
# Load restricted domain if it exists
if os.path.exists(f"{TEST_SCENE}.yaml"):
    domain = list(LoadIrisRegionsYamlFile(f"{TEST_SCENE}.yaml").values())[0]
else:
    domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    
manual_seed = default_joint_positions  # The location at which to build the first region

fast_iris_options = FastIrisOptions()
manual_seed_ellipsoid = Hyperellipsoid.MakeUnitBall(ambient_dim)
hpoly = FastIris(cspace_obstacle_collision_checker, manual_seed_ellipsoid, domain, fast_iris_options)

MAX_NUM_SAMPLES_PER_FACE = 500
MIXING_STEPS = 50

face_areas = []
max_area_idx = 0
face_samples = []

# Sample uniformly on surface of polytope
# To do so, we sample uniformly on each face of the polytope.
A = hpoly.A()  # N x ambient_dim
b = hpoly.b()  # N
N = np.shape(b)[0]
# Iterate through each hyperplane in polytope
for i in range(N):
    print(f"{i=}")
    # First, we find any point on surface i of the polytope
    A_slash_i = np.delete(A, i, axis=0)  # N x ambient_dim
    b_slash_i = np.delete(b, i, axis=0)  # N
    
    # The program is as follows:
    # Find x
    # s.t. A_{\i} x ≤ b_{\i}
    #        A_{i}x = b_{i}
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(ambient_dim)
    for Ai, bi in zip(A_slash_i, b_slash_i):
        prog.AddLinearConstraint(Ai @ x <= bi)
    prog.AddLinearEqualityConstraint(A[i] @ x == b[i])
    
    result = Solve(prog)
    # print(f"Solver {result.get_solver_id().name()} Success: {result.is_success()}")
    # print('x* = ', result.GetSolution(x))
    
    # If the solver fails, assume that face of the polytope is just redundant
    if not result.is_success():
        continue
        
    # Seed point on face of the polytope for hit and run sampling
    current_sample = result.GetSolution(x)
    
    # Next, using this initial point on the face, use hit-and-run sampling
    # to generate more points on the face.
    samples = np.empty((ambient_dim, MAX_NUM_SAMPLES_PER_FACE))
    for j in range(MAX_NUM_SAMPLES_PER_FACE):
        for _ in range(MIXING_STEPS):
            direction = np.random.normal(0, 1, ambient_dim)
            direction = direction - ((A[i] @ direction - b[i]) / (A[i] @ A[i])) * A[i]  # Project onto face of polytope
            direction = direction / np.linalg.norm(direction)
            
            line_b = b_slash_i - A_slash_i @ current_sample
            line_a = A_slash_i @ direction
            theta_max = float('inf')
            theta_min = float('-inf')
            for k in range(np.shape(line_a)[0]):
                if line_a[k] < 0.0:
                    theta_min = max(theta_min, line_b[k] / line_a[k])
                elif line_a[k] >= 0.0:
                    theta_max = min(theta_max, line_b[k] / line_a[k])
            if theta_max == float('inf') or theta_min == float('-inf') or theta_max < theta_min:
                if theta_max < theta_min:
                    raise Exception("Hit and run fail; theta_max < theta_min.")
                raise Exception(f"Hit and run fail. theta_max: {theta_max}. theta_min: {theta_min}")
        
            theta = np.random.uniform(theta_min, theta_max)
            current_sample = current_sample + theta * direction
            
        samples[:, j] = current_sample
        
    # Now, estimate the surface area of the polytope and store it. We will
    # need it later to rescale the number of samples we keep for each face
    # of the polytope to ensure uniform distribution.
    print(f"np.shape(samples): {np.shape(samples)}")
    face_area = ConvexHull(samples.T).area
    face_areas.append(face_area)
    if face_area > face_areas[max_area_idx]:  # Keep track of largest face
        max_area_idx = len(face_areas)-1  # Current index in face_areas
        
print(f"Number of non-redundant faces on polytope: {len(face_areas)}")
print(f"{face_areas=}")
    
# Now, with samples and area for each face, we reduce the number of samples
# on each face proportional to area of the face.
# We also aggregate all samples on surface of polytope into a single np array
surface_samples = np.empty(shape=(ambient_dim,0))
for i in range(len(face_areas)):
    area_ratio = face_areas[i] / face_areas[max_area_idx]
    num_face_samples_to_keep = int(area_ratio * MAX_NUM_SAMPLES_PER_FACE)
    # Keep lower ratio of samples (generated after more mixing steps)
    surface_samples.hstack((surface_samples, face_samples[:, -num_face_samples_to_keep:]))
    
print(np.shape(surface_samples))

for i in range(np.shape(surface_samples)[1]):
    meshcat.SetObject(
        f"sample_{i}",
        Sphere(radius=0.01 if ambient_dim==3 else 0.005),
        rgba=Rgba(1, 0, 0, 1),
    )
    meshcat.SetTransform(
        f"sample_{i}", RigidTransform(surface_samples[:,i])
    )