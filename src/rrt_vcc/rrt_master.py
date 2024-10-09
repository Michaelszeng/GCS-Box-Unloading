"""
Helper Class to aggregate all the versions of RRT.

This is the actual user-facing API. The other RRT python files should not be
interfaced directly.
"""

import numpy as np

from rrt import *
from rrt_star import *

class RRTMasterOptions:
    def __init__(
        self,
        step_size=1e-1,
        check_size=1e-2,
        min_vertices=1e3,  # For RRT* only
        max_vertices=1e3,
        goal_sample_frequency=0.05,
        neighbor_radius=0.2,  # For RRT* only
        timeout=np.inf,
        index=0,
        draw_rrt=True,
        use_rrt_star=False,
        use_bi_rrt=True,  # This only matters if use_rrt_star=False
    ):
        self.step_size = step_size
        self.check_size = check_size
        self.min_vertices = int(min_vertices)
        self.max_vertices = int(max_vertices)
        self.goal_sample_frequency = goal_sample_frequency
        self.neighbor_radius = neighbor_radius  # Added for RRT*
        self.timeout = timeout
        self.index = index
        self.draw_rrt = draw_rrt
        self.use_rrt_star = use_rrt_star
        self.use_bi_rrt = use_bi_rrt
        assert self.goal_sample_frequency >= 0
        assert self.goal_sample_frequency <= 1
        

def RRTMaster(master_options, start, goal, RandomConfig, ValidityChecker, Distance=None, ForwardKinematics=None, meshcat=None):
    if master_options.use_rrt_star:
        rrt = RRTStar(RandomConfig, ValidityChecker, Distance=Distance, ForwardKinematics=ForwardKinematics, meshcat=meshcat)
        
        options = RRTStarOptions(
            step_size=master_options.step_size,
            check_size=master_options.check_size,
            min_vertices=master_options.min_vertices,
            max_vertices=master_options.max_vertices,
            goal_sample_frequency=master_options.goal_sample_frequency,
            neighbor_radius=master_options.neighbor_radius,
            timeout=master_options.timeout,
            index=master_options.index,
            draw_rrt=master_options.draw_rrt
        )
    else:
        options = RRTOptions(
            step_size=master_options.step_size,
            check_size=master_options.check_size,
            max_vertices=master_options.max_vertices,
            goal_sample_frequency=master_options.goal_sample_frequency,
            timeout=master_options.timeout,
            index=master_options.index,
            draw_rrt=master_options.draw_rrt
        )
        
        if master_options.use_bi_rrt:
            rrt = BiRRT(RandomConfig, ValidityChecker, Distance=Distance, meshcat=meshcat)
        else:
            rrt = RRT(RandomConfig, ValidityChecker, Distance=Distance, meshcat=meshcat)
            
    return rrt.plan(start, goal, options)