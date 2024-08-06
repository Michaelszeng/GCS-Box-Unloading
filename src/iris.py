from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Hyperellipsoid,
    HPolyhedron,
    VPolytope,
    Intersection,
    IrisFromCliqueCoverOptions,
    IrisInConfigurationSpaceFromCliqueCover,
    FastIris,
    FastIrisOptions,
    IrisInConfigurationSpace,
    IrisOptions,
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

import numpy as np
from pathlib import Path
import pydot
import matplotlib.pyplot as plt


class IrisRegionGenerator():
    def __init__(self, meshcat, collision_checker, regions_file="../data/iris_source_regions.yaml", DEBUG=False):
        self.meshcat = meshcat
        self.collision_checker = collision_checker
        self.plant = collision_checker.plant()
        self.plant_context = collision_checker.plant_context()

        self.regions_file = Path(regions_file)

        self.DEBUG = DEBUG


    @staticmethod
    def visualize_connectivity(iris_regions):
        """
        Create and save SVG graph of IRIS Region connectivity.

        iris_regions is a list of ConvexSets.
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
            plt.show(block=False)


    def test_iris_region(self, plant, plant_context, meshcat, regions, seed=42, num_sample=50000, colors=None):
        """
        Plot small spheres in the volume of each region. (we are using forward
        kinematics to return from configuration space to task space.)
        """
        if not self.DEBUG:
            print("IrisRegionGenerator: DEBUG set to False; skipping region visualization.")
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

        num_nodes, num_edges = IrisRegionGenerator.visualize_connectivity(regions)
        print("Connectivity graph saved to ../iris_connectivity.svg.")
        print(f"Number of nodes and edges: {num_nodes}, {num_edges}")
        print("\n\n")

        self.generate_overlap_histogram(plant, regions)


    def load_and_test_regions(self):
        regions = LoadIrisRegionsYamlFile(self.regions_file)
        regions = [hpolyhedron for hpolyhedron in regions.values()]
        self.test_iris_region(self.plant, self.plant_context, self.meshcat, regions)


    def generate_source_region_at_q_nominal(self, q):
        """
        Generate a region around a nominal position so we guarantee good
        coverage around that position.
        """
        # Explicitely set plant positions at q as as seed for IRIS
        self.plant.SetPositions(self.plant_context, q)

        options = FastIrisOptions()
        options.random_seed = 0
        options.verbose = True
        domain = HPolyhedron.MakeBox(self.plant.GetPositionLowerLimits(),
                                     self.plant.GetPositionUpperLimits())
        kEpsilonEllipsoid = 1e-5
        clique_ellipse = Hyperellipsoid.MakeHypersphere(kEpsilonEllipsoid, self.plant.GetPositions(self.plant_context))
        region = FastIris(self.collision_checker, clique_ellipse, domain, options)

        regions_dict = {f"set0" : region}
        SaveIrisRegionsYamlFile(self.regions_file, regions_dict)
        
        # This source region will be drawn in black
        self.test_iris_region(self.plant, self.plant_context, self.meshcat, [region], colors=[Rgba(0.0,0.0,0.0,0.5)])


    def generate_source_iris_regions(self, minimum_clique_size=12, coverage_threshold=0.35, num_points_per_visibility_round=500, use_previous_saved_regions=True):
        """
        Source IRIS regions are defined as the regions considering only self-
        collision with the robot, and collision with the walls of the empty truck
        trailer (excluding the back wall).

        This function automatically searches the regions_file for existing
        regions, and 
        """
        options = IrisFromCliqueCoverOptions()
        options.num_points_per_coverage_check = 500
        options.num_points_per_visibility_round = num_points_per_visibility_round
        options.coverage_termination_threshold = coverage_threshold
        options.minimum_clique_size = minimum_clique_size  # minimum of 7 points needed to create a shape with volume in 6D
        options.iteration_limit = 1  # Only build 1 visibility graph --> cliques --> region in order not to have too much region overlap
        options.fast_iris_options.max_iterations = 1
        options.fast_iris_options.mixing_steps = 25  # default 50
        options.fast_iris_options.random_seed = 0
        options.fast_iris_options.verbose = True
        options.use_fast_iris = True

        if use_previous_saved_regions:
            print("Using saved iris regions.")
            regions = LoadIrisRegionsYamlFile(self.regions_file)
            regions = [hpolyhedron for hpolyhedron in regions.values()]

            # Scale down previous regions and use as obstacles in new round of Clique Covers
            # Encourages exploration while still allowing small degree of region overlap
            region_obstacles = [hpolyhedron.Scale(0.99) for hpolyhedron in regions]

            # Set previous regions as obstacles to encourage exploration
            options.iris_options.configuration_obstacles = region_obstacles
        else:
            regions = []

        regions = IrisInConfigurationSpaceFromCliqueCover(
            checker=self.collision_checker, options=options, generator=RandomGenerator(42), sets=regions
        )  # List of HPolyhedrons

        regions_dict = {f"set{i}" : regions[i] for i in range(len(regions))}
        SaveIrisRegionsYamlFile(self.regions_file, regions_dict)

        self.test_iris_region(self.plant, self.plant_context, self.meshcat, regions)
        
