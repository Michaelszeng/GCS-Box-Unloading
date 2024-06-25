from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    IrisInConfigurationSpace,
    IrisOptions,
    Hyperellipsoid,
    HPolyhedron,
    VPolytope,
    Intersection,
    IrisFromCliqueCoverOptions,
    IrisInConfigurationSpaceFromCliqueCover,
    SceneGraphCollisionChecker,
    SaveIrisRegionsYamlFile,
    LoadIrisRegionsYamlFile,
    RandomGenerator,
    PointCloud,
    RobotDiagramBuilder,
    MathematicalProgram,
    Rgba,
    RigidTransform,
)
from manipulation.meshcat_utils import AddMeshcatTriad

from scenario import scenario_yaml_for_iris, q_nominal
from utils import NUM_BOXES

import numpy as np
from pathlib import Path
import pydot
import matplotlib.pyplot as plt


class IrisRegionGenerator():

    def __init__(self, meshcat, robot_pose, regions_file="../data/iris_source_regions.yaml", DEBUG=True):
        self.meshcat = meshcat
        robot_diagram_builder = RobotDiagramBuilder()
        diagram_builder = robot_diagram_builder.builder()

        self.robot = robot_diagram_builder.parser().AddModelsFromString(scenario_yaml_for_iris, ".dmd.yaml")
        self.plant = robot_diagram_builder.plant()

        # scene_graph = robot_diagram_builder.scene_graph()
        # Weld robot base in place
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("base_link", self.plant.GetModelInstanceByName("robot_base")), robot_pose)
        self.diagram = robot_diagram_builder.Build()
        context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(context)

        self.regions_file = Path(regions_file)

        self.DEBUG = DEBUG


    def generate_overlap_histogram(self, plant, regions, num_samples=10000, seed=42):
        """
        Measure region overlap by randomly sampling over union of all sets, and
        creating a histogram of how many sets of sample falls in. The fewer, the
        better.
        """
        rng = RandomGenerator(seed)

        sampling_domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())

        data = {}

        last_sample = sampling_domain.UniformSample(rng)
        for i in range(num_samples):
            last_sample = sampling_domain.UniformSample(rng, last_sample)
            last_sample_num_regions = 0
            # Count the number of sets the sample appears in
            for r in regions:
                if r.PointInSet(last_sample):
                    last_sample_num_regions += 1
            
            if last_sample_num_regions == 0:
                continue

            if last_sample_num_regions in data.keys():
                data[last_sample_num_regions] += 1
            else:
                data[last_sample_num_regions] = 0

        # Plot data
        num_regions = list(data.keys())
        samples = list(data.values())

        # Plotting the histogram
        if self.DEBUG:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(num_regions, samples, color='blue', edgecolor='black')
            plt.xlabel('Number of Regions Sample Appears In')
            plt.ylabel('Number of Samples')
            plt.title('Histogram of Sample Distribution Across Regions')
            plt.xticks(num_regions)  # Ensure all x-axis labels are shown
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            # Add text annotations on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')
            plt.show()



    def visualize_connectivity(self, iris_regions):
        """
        Create and save SVG graph of IRIS Region connectivity.
        """
        numEdges = 0
        numNodes = 0

        graph = pydot.Dot("IRIS region connectivity")
        for i in range(len(iris_regions)):
            numNodes += 1
            graph.add_node(pydot.Node(i))
            v1 = iris_regions[i]
            for j in range(i + 1, len(iris_regions)):
                v2 = iris_regions[j]
                if v1.IntersectsWith(v2):
                    numEdges += 1
                    graph.add_edge(pydot.Edge(i, j, dir="both"))

        svg = graph.create_svg()

        with open('../iris_connectivity.svg', 'wb') as svg_file:
            svg_file.write(svg)

        return numNodes, numEdges


    def test_iris_region(self, plant, plant_context, meshcat, regions, seed=42, num_sample=50000, colors=None):
        """
        Plot small spheres in the volume of each region. (we are using forward
        kinematics to return from configuration space to task space.)
        """
        if not self.DEBUG:
            return

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
            meshcat.SetObject(f"regions/region {i}", pc, point_size=0.025, rgba=colors[i % len(colors)])

        num_nodes, num_edges = self.visualize_connectivity(regions)
        print("Connectivity graph saved to ../iris_connectivity.svg.")
        print(f"Number of nodes and edges: {num_nodes}, {num_edges}")

        self.generate_overlap_histogram(plant, regions)


    def load_and_test_regions(self):
        regions = LoadIrisRegionsYamlFile(self.regions_file)
        regions = [hpolyhedron for hpolyhedron in regions.values()]
        self.test_iris_region(self.plant, self.plant_context, self.meshcat, regions)


    def generate_source_region_at_q_nominal(self):
        """
        Generate a region around q_nominal so we guarantee good coverage around 
        nominal pose.
        """
        # Explicitely set plant positions at q_nominal as as seed for IRIS
        self.plant.SetPositions(self.plant_context, q_nominal)

        iris_options = IrisOptions()
        iris_options.require_sample_point_is_contained = True
        iris_options.random_seed = 0

        region = IrisInConfigurationSpace(self.plant, self.plant_context, iris_options)
        regions_dict = {f"set0" : region}
        SaveIrisRegionsYamlFile(self.regions_file, regions_dict)
        
        # This source region will be drawn in black
        self.test_iris_region(self.plant, self.plant_context, self.meshcat, [region], colors=[Rgba(0.0,0.0,0.0,0.5)])


    def generate_source_iris_regions(self, minimum_clique_size=12, coverage_threshold=0.35, use_previous_saved_regions=True):
        """
        Source IRIS regions are defined as the regions considering only self-
        collision with the robot, and collision with the walls of the empty truck
        trailer (excluding the back wall).

        This function automatically searches the regions_file for existing
        regions, and 
        """
        collision_checker_params = dict(edge_step_size=0.125)
        collision_checker_params["robot_model_instances"] = self.robot
        # Must clone diagram so we don't pass ownership of the original digram to SceneGraphCollisionChecker (preventing us from ever using the digram again)
        collision_checker_params["model"] = self.diagram.Clone()
        
        checker = SceneGraphCollisionChecker(**collision_checker_params)
        options = IrisFromCliqueCoverOptions()
        options.num_points_per_coverage_check = 500
        options.num_points_per_visibility_round = 250  # 1000
        options.coverage_termination_threshold = coverage_threshold
        options.minimum_clique_size = minimum_clique_size  # minimum of 7 points needed to create a shape with volume in 6D

        options.iris_options.random_seed = 0

        if use_previous_saved_regions:
            print("Using saved iris regions.")
            regions = LoadIrisRegionsYamlFile(self.regions_file)
            regions = [hpolyhedron for hpolyhedron in regions.values()]

            # Scale down previous regions and use as obstacles in new round of Clique Covers
            # Encourages exploration while still allowing small degree of region overlap
            region_obstacles = [hpolyhedron.Scale(0.975) for hpolyhedron in regions]

            # Set previous regions as obstacles to encourage exploration
            options.iris_options.configuration_obstacles = region_obstacles
        else:
            regions = []

        regions = IrisInConfigurationSpaceFromCliqueCover(
            checker=checker, options=options, generator=RandomGenerator(42), sets=regions
        )  # List of HPolyhedrons

        regions_dict = {f"set{i}" : regions[i] for i in range(len(regions))}
        SaveIrisRegionsYamlFile(self.regions_file, regions_dict)

        self.test_iris_region(self.plant, self.plant_context, self.meshcat, regions)
        
