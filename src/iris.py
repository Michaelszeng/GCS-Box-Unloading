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
import matplotlib
import pyvista as pv
import time
matplotlib.use("tkagg")


class IrisRegionGenerator():
    def __init__(self, meshcat, collision_checker, regions_file, DEBUG=False):
        self.meshcat = meshcat
        self.collision_checker = collision_checker  # ConfigurationObstacleCollisionChecker
        self.plant = collision_checker.plant()
        self.plant_context = collision_checker.plant_context()

        self.regions_file = Path(regions_file)

        self.DEBUG = DEBUG


    @staticmethod
    def visualize_connectivity(regions, coverage, output_file='../iris_connectivity.svg', skip_svg=False):
        """
        Create and save SVG graph of IRIS Region connectivity.

        regions can be a list of ConvexSets or a dictionary with keys as
        labels and values as ConvexSets.
        """
        numEdges = 0
        numNodes = 0

        graph = pydot.Dot("IRIS region connectivity")

        if isinstance(regions, dict):
            items = list(regions.items())
        else:
            items = list(enumerate(regions))

        for i, (label1, v1) in enumerate(items):
            numNodes += 1
            graph.add_node(pydot.Node(label1))
            for j in range(i + 1, len(items)):
                label2, v2 = items[j]
                if v1.IntersectsWith(v2):
                    numEdges += 1
                    graph.add_edge(pydot.Edge(label1, label2, dir="both"))

        # Add text annotations for numNodes and numEdges
        annotation = f"Nodes: {numNodes}, Edges: {numEdges}, Coverage: {coverage}"
        graph.add_node(pydot.Node("annotation", label=annotation, shape="none", fontsize="12", pos="0,-1!", margin="0"))

        if not skip_svg:
            svg = graph.create_svg()

            with open(output_file, 'wb') as svg_file:
                svg_file.write(svg)

        return numNodes, numEdges
    

    @staticmethod
    def estimate_coverage(plant, collision_checker, regions, num_samples=10000, seed=42):
        """
        Estimate coverage fraction of the regions in the entire cspace via
        sampling.

        regions can be a list of ConvexSets or a dictionary with keys as
        labels and values as ConvexSets.
        """
        if isinstance(regions, dict):
            regions = list(regions.values())

        rng = RandomGenerator(seed)
        sampling_domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
        last_sample = sampling_domain.UniformSample(rng)

        collision_checker.SetConfigurationSpaceObstacles([])  # We don't want to account for any c-space obstacles during the coverge estimate

        num_samples_in_regions = 0
        num_samples_collision_free = 0
        for _ in range(num_samples):
            last_sample = sampling_domain.UniformSample(rng, last_sample)

            # Check if sample is in collision
            if collision_checker.CheckConfigCollisionFree(last_sample):
                num_samples_collision_free += 1

                # If sample is collsion-free, check if sample falls in regions
                for r in regions:
                    if r.PointInSet(last_sample):
                        num_samples_in_regions += 1
                        break

        return num_samples_in_regions / num_samples_collision_free


    def visualize_cspace(self, num_samples=100000, seed=42):
        rng = RandomGenerator(seed)
        cspace_dim = self.plant.num_positions()
        sampling_domain = HPolyhedron.MakeBox(self.plant.GetPositionLowerLimits(), self.plant.GetPositionUpperLimits())
        last_sample = sampling_domain.UniformSample(rng)

        self.collision_checker.SetConfigurationSpaceObstacles([])  # We don't want to account for any c-space obstacles during this visualization

        collision_free_samples = None  # N x cspace_dim array
        for _ in range(num_samples):
            last_sample = sampling_domain.UniformSample(rng, last_sample)

            # Check if the sample is in collision
            if self.collision_checker.CheckConfigCollisionFree(last_sample):
                if collision_free_samples is None:
                    collision_free_samples = last_sample[np.newaxis, :]
                else:
                    collision_free_samples = np.vstack((collision_free_samples, last_sample))

        # print(f"Collision-free fraction: {np.shape(collision_free_samples)[0] / num_samples}")  # ~11%

        if (cspace_dim == 6):  # 6 choose 3 = 20; make a 4x5 grid of plots
            plotter = pv.Plotter(shape=(4, 5), notebook=False)

            for i in range(20):
                # Update the index each axis of the current plot represents so each of the (cspace_dim choose 3) plots is unique
                if i == 0:
                    x_idx = 0
                    y_idx = 1
                    z_idx = 2
                else:
                    if z_idx != cspace_dim-1:
                        z_idx += 1
                    elif y_idx != cspace_dim-2:  # and z_idx == cspace_dim-1
                        y_idx += 1
                        z_idx = y_idx + 1
                    else:  # and z_idx == cspace_dim-1 and y_idx == cspace_dim-2
                        x_idx += 1
                        y_idx = x_idx + 1
                        z_idx = y_idx + 1

                plotter.subplot(i // 5, i % 5)

                plotter.add_mesh(pv.PolyData(collision_free_samples[:, [x_idx, y_idx, z_idx]]), 
                                render_points_as_spheres=True, point_size=2.5)
                
                plotter.camera_position = 'xy'
                plotter.camera.azimuth = 45
                plotter.camera.elevation = 45
                plotter.show_grid()
                plotter.show_bounds(
                    grid='back',
                    axes_ranges=[-3.15, 3.15, -3.15, 3.15, -3.15, 3.15],
                    location='outer',
                    ticks='both',
                    show_xlabels=False,
                    show_ylabels=False,
                    show_zlabels=False,
                    xtitle=f"idx={x_idx}",
                    ytitle=f"idx={y_idx}",
                    ztitle=f"idx={z_idx}",
                )

            plotter.show()


    def generate_overlap_histogram(self, regions, seed=42):
        """
        Measure region overlap by randomly sampling 100 points in each region
        and checking how many other regions that sample also falls in.
        Generally, the less overlap the better.
        """
        rng = RandomGenerator(seed)

        data = {}

        for r in regions:
            last_sample = r.UniformSample(rng, mixing_steps=5)
            for _ in range(100):
                last_sample = r.UniformSample(rng, last_sample, mixing_steps=5)
                last_sample_num_regions = 0

                # Count the number of sets the sample appears in
                for r_ in regions:
                    if r_.PointInSet(last_sample):
                        last_sample_num_regions += 1

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
            plt.pause(1)  # Allow the plot to be displayed


    @staticmethod
    def visualize_iris_region(plant, plant_context, meshcat, regions, seed=42, density=1, colors=None, name="regions", task_space=True, scene="BOXUNLOADING"):
        """
        Plot dense point clouds to visualize the volume of each region. If 
        `task_space` is set to True, we use forward kinematics to return from
        configuration space to task space.

        regions can either be a list of HPolyhedrons or a dictionary mapping
        names to HPolyhedrons.
        """
        if task_space:            
            if scene == "3DOFFLIPPER":
                print("visualize_iris_region for 3DOFFLIPPER in task space to be implemented.")
                return
            if scene == "5DOFUR3":
                ee_frame = plant.GetFrameByName("ur_ee_link")
            if scene == "6DOFUR3":
                ee_frame = plant.GetFrameByName("ur_ee_link")
            if scene == "7DOFIIWA":
                ee_frame = plant.GetFrameByName("iiwa_link_7")
            if scene == "7DOFBINS":
                ee_frame = plant.GetFrameByName("iiwa_link_7")
            if scene == "7DOF4SHELVES":
                ee_frame = plant.GetFrameByName("iiwa_link_7")
            if scene == "14DOFIIWAS":
                print("visualize_iris_region for 14DOFIIWAS in task space to be implemented.")
                return
            if scene == "15DOFALLEGRO":
                print("visualize_iris_region for 15DOFALLEGRO in task space to be implemented.")
                return
            if scene == "BOXUNLOADING":
                ee_frame = plant.GetFrameByName("arm_eef")

        rng = RandomGenerator(seed)

        # Allow caller to input custom colors
        if colors is None:
            colors = [
                Rgba(0.5,0.0,0.0,0.5),
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

        if isinstance(regions, list):
            # Make regions a dictionry with keys as a counter
            regions = {i: regions[i] for i in range(len(regions))}

        for i, (region_name, region) in enumerate(regions.items()):

            xyzs = []  # List to hold XYZ positions of configurations in the IRIS region

            q_sample = region.UniformSample(rng)
            prev_sample = q_sample

            if task_space:
                plant.SetPositions(plant_context, q_sample)
                xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=plant.world_frame(), frame_B=ee_frame).translation())
            else:
                xyzs.append(prev_sample)

            num_samples = int(2e4 * density * region.CalcVolumeViaSampling(RandomGenerator(0), desired_rel_accuracy=0.01, max_num_samples=1000).volume)

            for _ in range(num_samples-1):
                q_sample = region.UniformSample(rng, prev_sample)
                prev_sample = q_sample

                if task_space:
                    plant.SetPositions(plant_context, q_sample)
                    xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=plant.world_frame(), frame_B=ee_frame).translation())
                else:
                    xyzs.append(prev_sample)
                    
            # Create pointcloud from sampled point in IRIS region in order to plot in Meshcat
            xyzs = np.array(xyzs)
            pc = PointCloud(len(xyzs))
            pc.mutable_xyzs()[:] = xyzs.T
            meshcat.SetObject(f"{name}/{region_name}", pc, point_size=0.025, rgba=colors[i % len(colors)])


    def test_iris_region(self, plant, plant_context, meshcat, collision_checker, regions, seed=42, colors=None, name="regions", coverage=True, histogram=True, connectivity=True, svg=True, task_space_render=True):
        """
        Run a series of tests on a given list of IRIS regions.
        """
        if not self.DEBUG:
            print("IrisRegionGenerator: DEBUG set to False; skipping region visualization.")
            return
        
        if coverage:
            coverage = self.estimate_coverage(plant, collision_checker, regions)
            print(f"Estimated region coverage fraction: {coverage}")

        if histogram:
            self.generate_overlap_histogram(regions)
        
        if connectivity:
            num_nodes, num_edges = IrisRegionGenerator.visualize_connectivity(regions, coverage, skip_svg=(not svg))
            if svg:
                print("Connectivity graph saved to ../iris_connectivity.svg.")
            print(f"Number of nodes and edges: {num_nodes}, {num_edges}")
            print("\n\n")

        if task_space_render:
            IrisRegionGenerator.visualize_iris_region(plant, plant_context, meshcat, regions, seed, colors, name)


    def load_and_test_regions(self, name="regions"):
        regions = LoadIrisRegionsYamlFile(self.regions_file)

        # To control how many sets to evaluate
        num_sets = 9999
        regions = {k: v for k, v in regions.items() if k.startswith("set") and k[3:].isdigit() and 0 <= int(k[3:]) <= num_sets}

        regions = [hpolyhedron for hpolyhedron in regions.values()]

        volumes = []
        for r in regions:
            volumes.append(r.CalcVolumeViaSampling(RandomGenerator(0), desired_rel_accuracy=0.01, max_num_samples=1000000).volume)
        print("volumes:", volumes)

        self.test_iris_region(self.plant, self.plant_context, self.collision_checker, self.meshcat, regions, name=name)


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

        regions_dict = {"set0" : region}
        SaveIrisRegionsYamlFile(self.regions_file, regions_dict)
        
        # This source region will be drawn in black
        self.test_iris_region(self.plant, self.plant_context, self.meshcat, self.collision_checker, [region], colors=[Rgba(0.0,0.0,0.0,0.5)], coverage=True, histogram=False, connectivity=False, svg=False, task_space_render=False)


    def generate_source_iris_regions(self, 
                                     minimum_clique_size=12, 
                                     coverage_threshold=0.35, 
                                     num_points_per_visibility_round=500, 
                                     clique_covers_seed=0, 
                                     use_previous_saved_regions=True, 
                                     coverage_check_only=False):
        """
        Source IRIS regions are defined as the regions considering only self-
        collision with the robot, and collision with the walls of the empty truck
        trailer (excluding the back wall).

        This function automatically searches the regions_file for existing
        regions, and begins with those.
        """
        options = IrisFromCliqueCoverOptions()
        options.num_points_per_coverage_check = 1000
        options.num_points_per_visibility_round = num_points_per_visibility_round
        options.coverage_termination_threshold = coverage_threshold
        options.minimum_clique_size = minimum_clique_size  # minimum of 7 points needed to create a shape with volume in 6D
        options.iteration_limit = 1  # Only build 1 visibility graph --> cliques --> region in order not to have too much region overlap
        options.fast_iris_options.max_iterations = 1
        options.fast_iris_options.require_sample_point_is_contained = True
        options.fast_iris_options.mixing_steps = 10  # default 50
        options.fast_iris_options.random_seed = 0
        options.fast_iris_options.verbose = True
        options.use_fast_iris = True

        if coverage_check_only:
            options.iteration_limit = 0
            regions = LoadIrisRegionsYamlFile(self.regions_file)
            regions = [hpolyhedron for hpolyhedron in regions.values()]
            self.collision_checker.SetConfigurationSpaceObstacles([])  # We don't want to account for any c-space obstacles during the coverge estimate
        elif use_previous_saved_regions:
            regions = LoadIrisRegionsYamlFile(self.regions_file)
            regions = [hpolyhedron for hpolyhedron in regions.values()]

            # Scale down previous regions and use as obstacles in new round of Clique Covers
            # Encourages exploration while still allowing small degree of region overlap
            region_obstacles = [hpolyhedron.Scale(0.9) for hpolyhedron in regions]
            # region_obstacles = [hpolyhedron.MaximumVolumeInscribedEllipsoid() for hpolyhedron in regions]

            # Set previous regions as obstacles to encourage exploration
            # options.iris_options.configuration_obstacles = region_obstacles  # No longer needed bc of the line below
            self.collision_checker.SetConfigurationSpaceObstacles(region_obstacles)  # Set config. space obstacles in collision checker so FastIRIS will also respect them
        else:
            regions = []

        regions = IrisInConfigurationSpaceFromCliqueCover(
            checker=self.collision_checker, options=options, generator=RandomGenerator(clique_covers_seed), sets=regions
        )  # List of HPolyhedrons

        # Remove redundant hyperplanes
        regions = [r.ReduceInequalities() for r in regions]

        if not coverage_check_only:
            regions_dict = {f"set{i}" : regions[i] for i in range(len(regions))}
            SaveIrisRegionsYamlFile(self.regions_file, regions_dict)

            self.test_iris_region(self.plant, self.plant_context, self.meshcat, self.collision_checker, regions, coverage=True, histogram=False, connectivity=True, svg=False, task_space_render=False)
        
    
    @staticmethod
    def post_process_iris_regions(regions_dict, edge_count_threshold=0.75):
        """
        Simplify IRIS regions using SimplifyByIncrementalFaceTranslation()
        procedure. This reduces the number of faces on the HPolyhedron which
        potentially removes region intersections. This function is meant to be
        used in line with any calls to `LoadIrisRegionsYamlFile()`.

        Note: we intentionally do not saved the simplified HPolytopes to file
        since we want to keep the detailed original polyhedrons.

        regions_dict is a dictionary that maps region names to regions.

        edge_count_threshold is a tunable value that controls what fraction of
        edges relative to the average warrants allowing that vertex's edges
        to be removed. Lower --> more edges are removed.
        """
        # First find number of edges on each region
        edge_counts = {}
        for s, r in regions_dict.items():
            for s_, r_ in regions_dict.items():
                if r.IntersectsWith(r_):
                    if s not in edge_counts.keys():
                        edge_counts[s] = 0.5  # Add 0.5 instead of 1 since we're going to double count every edge
                    else:
                        edge_counts[s] += 0.5
                    
                    if s_ not in edge_counts.keys():
                        edge_counts[s_] = 0.5
                    else:
                        edge_counts[s_] += 0.5

        avg_edge_count = sum(ct for ct in edge_counts.values()) / len(edge_counts)
        print(f"IRIS region avg_edge_count: {avg_edge_count}")
                    
        # Then perform simplifications on each HPolyhedron
        output_regions = {}
        for s, r in regions_dict.items():
            intersecting_polytopes = []
            for s_, r_ in regions_dict.items():
                if r.IntersectsWith(r_) and edge_counts[s_] < avg_edge_count * edge_count_threshold:
                    intersecting_polytopes.append(r_)

            r_simplified = r.SimplifyByIncrementalFaceTranslation(min_volume_ratio=0.1,
                                                                  max_iterations=1,
                                                                  intersecting_polytopes=intersecting_polytopes,
                                                                  random_seed=42)
            print("finished call to SimplifyByIncrementalFaceTranslation.")
            
            output_regions[s] = r_simplified
        
        # FOR TESTING ONLY
        SaveIrisRegionsYamlFile("../data/TEMPORARY.yaml", output_regions)
        IrisRegionGenerator.visualize_connectivity(output_regions, "n/a", output_file='../TEMPORARY.svg')

        return output_regions

