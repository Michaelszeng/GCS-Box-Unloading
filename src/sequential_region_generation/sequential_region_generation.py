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
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from station import MakeHardwareStation, load_scenario
from scenario import scenario_yaml_for_iris
from utils import ik

import numpy as np
import importlib
from scipy.spatial.transform import Rotation
from scipy.sparse import find

# TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
TEST_SCENE = "7DOFIIWA"
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

NUM_ANGLE_INCREMENTS = 100

while True:
    # Sample uniformly on surface of polytope
    # To do so, we sample uniformly on each face of the polytope.
    A = hpoly.A()  # N x ambient_dim
    b = hpoly.b()  # N
    N = np.shape(b)[0]
    # Iterate through each hyperplane in polytope
    for i in range(N):
        # First, we find any point on surface i of the polytope
        A_slash_i = np.delete(A, i, axis=0)
        b_slash_i = np.delete(b, i, axis=0)
        
        # The program is as follows:
        # Find x
        # s.t. A_{\i} x ≤ b_{\i}
        #        A_{i}x = b_{i}
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(ambient_dim)
        for row, bi in zip(A_slash_i, b_slash_i):
            prog.AddLinearConstraint(row @ x <= bi)
        prog.AddLinearEqualityConstraint(A[i] @ x == b[i])
        
        result = Solve(prog)
        # print(f"Solver {result.get_solver_id().name()} Success: {result.is_success()}")
        # print('x* = ', result.GetSolution(x))
        
        # If the solver fails, assume that face of the polytope is just redundant
        if not result.is_success():
            continue
        
        