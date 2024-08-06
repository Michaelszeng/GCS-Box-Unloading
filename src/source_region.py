class SourceRegion:
    """
    Object representing a "source region"--an HPolytope in collision-free
    configuration space when ignoring all obstacles (boxes). The source region 
    consists of sub-regions--HPolytope subsets of the source region--that result
    from adding hyperplanes due to the presence of obstacles. 
    """
    def __init__(self, source_polytope):
        """
        polytope is an HPolytope representing the source region itself.
        """
        self.source_polytope = source_polytope
        self.sub_regions = []  # List of tuples with (obstacle name, list of HPolytopes representing subdivisions of region resulting from obstacle)
        self.sub_regions_during_grasp = []  # List of tuples with (obstacle name, list of HPolytopes representing subdivisions of region resulting from obstacle)

    
    def remove_obstacle(self, removed_obs_name):
        """
        Remove all sub-regions from this source region relating to
        removed_obs_name.

        removed_obs_name is the name of the obstacle to be removed. 
        """

    
    def return_final_regions(self, during_grasp=False):
        """
        Takes intersection of all sub-regions and performs reductions to remove
        fully enclosed polytopes. Returns a list of HPolytopes.
        """
        pass