from pydrake.all import (
    GcsTrajectoryOptimization,
    GraphOfConvexSetsOptions,
    Point,
    CompositeTrajectory,
    Trajectory,
    CommonSolverOption,
)

from iris import IrisRegionGenerator

import time
import numpy as np



def gcs_traj_opt(self, q_start, target_regions, gcs_regions, regions_to_add=None, vel_lim=1.0, DETAILED_LOGS=False):
    """
    Define and run a GCS Trajectory Optimization program.

    q_current is a 7D np array containing the robot's current configuration.

    target_regions is a list of ConvexSet objects containing the desired set
    of end configurations for the trajectory optimization.

    gcs_regions is a dictionary mapping convex set names to convex sets.
    """
    gcs = GcsTrajectoryOptimization(6)
    gcs.AddTimeCost()
    gcs.AddPathLengthCost()
    gcs.AddPathContinuityConstraints(2)  # Acceleration continuity
    gcs.AddVelocityBounds(
        self.plant.GetVelocityLowerLimits() * vel_lim, 
        self.plant.GetVelocityUpperLimits() * vel_lim
    )
    
    gcs_regions = gcs.AddRegions(list(gcs_regions.values()), order=3)
    
    if regions_to_add:
        added_regions = gcs.AddRegions(regions_to_add, order=3, name="added-regions")

    source = gcs.AddRegions([Point(q) for q in q_start], order=0, name="source")
    target = gcs.AddRegions(target_regions, order=0)

    if regions_to_add:
        all_regions = {**{f"{i}": obj for i, obj in enumerate(gcs_regions, 1)},
                    **{f"added-region{i}": obj for i, obj in enumerate(added_regions.regions(), 1)},
                    **{f"source{i}": obj for i, obj in enumerate(source.regions(), 1)},
                    **{f"target{i}": obj for i, obj in enumerate(target.regions(), 1)}}
    else:
        all_regions = {**{f"{i}": obj for i, obj in enumerate(gcs_regions, 1)},
                    **{f"source{i}": obj for i, obj in enumerate(source.regions(), 1)},
                    **{f"target{i}": obj for i, obj in enumerate(target.regions(), 1)}}
    IrisRegionGenerator.visualize_connectivity(all_regions)
    print("Connectivity Graph for GCS fail saved to '../iris_connectivity.svg'.")

    gcs.AddEdges(source, self.regions_).Edges()
    gcs.AddEdges(self.regions_, target).Edges()

    if regions_to_add:
        gcs.AddEdges(source, added_regions).Edges()
        gcs.AddEdges(self.regions_, added_regions).Edges()
        gcs.AddEdges(added_regions, self.regions_).Edges()
        gcs.AddEdges(added_regions, target).Edges()
    
    options = GraphOfConvexSetsOptions()
    if (DETAILED_LOGS):
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    options.preprocessing = True
    options.max_rounded_paths = 5  # Max number of distinct paths to compare during random rounding; only the lowest cost path is returned.
    
    start_time = time.time()
    traj, result = gcs.SolvePath(source, target, options)
    
    if not result.is_success():
        print("Could not find a feasible path from q_start to q_goal")
    else:
        print(f"Solved GCS in {time.time() - start_time}.")
        gcs.RemoveSubgraph(source)
        gcs.RemoveSubgraph(target)
        if regions_to_add:
            gcs.RemoveSubgraph(added_regions)
        return traj