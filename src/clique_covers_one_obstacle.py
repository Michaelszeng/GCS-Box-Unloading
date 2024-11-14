"""
This test takes an IRIS region (specifically, the one generated around q_nominal),
places a box that intersect with the region, and then runs clique covers within
the region to divide that region into sub-regions avoiding that obstacle.

We are measuring the runtime of this operation (it will be critical to dynamic
IRIS).
"""

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
    MaxCliqueSolverViaMip,
    SolverOptions,
    CommonSolverOption,
    GurobiSolver,
    SaveIrisRegionsYamlFile,
    RandomGenerator,
    PointCloud,
    RobotDiagramBuilder,
    MathematicalProgram,
    Parser,
    PositionConstraint,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Solve,
    Sphere,
    InverseKinematics,
    MeshcatVisualizer,
    StartMeshcat,
    Simulator,
    configure_logging,
)
from manipulation.meshcat_utils import AddMeshcatTriad

from scenario import iris_yaml, q_nominal

import logging
import os
import time
import numpy as np
from pathlib import Path


def test_iris_region(plant, plant_context, meshcat, regions, seed=42, num_sample=50000, colors=None):
    """
    Plot small spheres in the volume of each region. (we are using forward
    kinematics to return from configuration space to task space.)
    """
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("arm_eef")

    rng = RandomGenerator(seed)

    # Allow caller to input custom colors
    if colors is None:
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


configure_logging()
log = logging.getLogger("drake")
log.setLevel("DEBUG")

########################
###    Scene Setup   ###
########################
robot_pose = RigidTransform([0.0,0.0,0.58])

# Compute box transform (box is hand-picked to intersect with IRIS region)
box_rot = np.array([
    [0.00017050116307726348, -0.7764898345413309, 0.630129754719875],
    [-0.0001972651018409266, 0.6301297255023258, 0.7764898519136247],
    [-0.9999999660079157, -0.00025669503310665753, -4.573649520145384e-05],
])
box_pos = [0.9435565551763379, -1.1228571445027955, 2.525706407385175]
box_pose = RigidTransform(RotationMatrix(box_rot), box_pos)

meshcat = StartMeshcat()

collision_checker_params = dict(edge_step_size=0.125)
robot_diagram_builder = RobotDiagramBuilder()
diagram_builder = robot_diagram_builder.builder()

# Add oen box to scenario yaml
relative_path_to_box = '../data/Box_0_5_0_5_0_5.sdf'
absolute_path_to_box = os.path.abspath(relative_path_to_box)

iris_yaml += f"""
- add_model: 
    name: Box
    file: file://{absolute_path_to_box}
"""

collision_checker_params["robot_model_instances"] = robot_diagram_builder.parser().AddModelsFromString(iris_yaml, ".dmd.yaml")
plant = robot_diagram_builder.plant()

# Set Pose of box and weld
box_model_idx = plant.GetModelInstanceByName(f"Box")  # ModelInstanceIndex
box_frame = plant.GetFrameByName("Box_0_5_0_5_0_5", box_model_idx)
plant.WeldFrames(plant.world_frame(), box_frame, box_pose)

scene_graph = robot_diagram_builder.scene_graph()

visualizer = MeshcatVisualizer.AddToBuilder(diagram_builder, scene_graph, meshcat)

# Weld robot base in place
diagram = robot_diagram_builder.Build()
collision_checker_params["model"] = diagram
context = diagram.CreateDefaultContext()
# plant_context = plant.GetMyContextFromRoot(context)

########################
### Simulation Setup ###
########################
simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)
simulator.set_target_realtime_rate(1)
simulator.set_publish_every_time_step(True)
meshcat.StartRecording()
simulator.AdvanceTo(0.1)


########################
###   Clique Cover   ###
########################
# Load iris test region
regions_file = Path("../data/iris_test_region.yaml")
regions = LoadIrisRegionsYamlFile(regions_file)
regions = [hpolytope for hpolytope in regions.values()]

checker = SceneGraphCollisionChecker(**collision_checker_params)

options = IrisFromCliqueCoverOptions()
options.num_points_per_coverage_check = 500
options.num_points_per_visibility_round = 10  # 1000

options.coverage_termination_threshold = 0.75  # should do fairly well with coverage since the source region itself is easy to cover
options.minimum_clique_size = 10  # minimum of 7 points needed to create a shape with volume in 6D
options.iteration_limit = 25

# Set work and time limits on clique solver
clique_solver_options = SolverOptions()
clique_solver_options.SetOption(GurobiSolver().solver_id(), "WorkLimit", 5.5)  # Complete guess
clique_solver_options.SetOption(GurobiSolver().solver_id(), "MIPGap", 0.1)
clique_solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
clique_solver = MaxCliqueSolverViaMip()
clique_solver.SetSolverOptions(clique_solver_options)
options.max_clique_solver.SetSolverOptions(clique_solver_options)
print(options.max_clique_solver.GetSolverOptions().get_print_to_console())


options_internal = IrisOptions()
options_internal.random_seed = 0
options_internal.bounding_region = regions[0]  # regions should be length 1

options.iris_options = options_internal

print("starting IrisInConfigurationSpaceFromCliqueCover().")
start = time.time()
regions = IrisInConfigurationSpaceFromCliqueCover(
    checker=checker, options=options, generator=RandomGenerator(42), sets=[]  # start without any regions since we're regenerating regions within the source region
)  # List of HPolyhedrons
print(f"IrisInConfigurationSpaceFromCliqueCover() runtime: {time.time() - start}")

regions_dict = {f"set{i}" : regions[i] for i in range(len(regions))}
SaveIrisRegionsYamlFile(regions_file, regions_dict)

test_iris_region(plant, plant_context, meshcat, regions)

simulator.AdvanceTo(0.2)
meshcat.PublishRecording()