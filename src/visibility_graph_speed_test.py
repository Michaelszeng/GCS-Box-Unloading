from pydrake.all import (
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
)

import numpy as np
import time

from scenario import scenario_yaml_for_iris, robot_pose


# Generate regions with no obstacles at all
robot_diagram_builder = RobotDiagramBuilder()
robot_model_instances = robot_diagram_builder.parser().AddModelsFromString(scenario_yaml_for_iris, ".dmd.yaml")
robot_diagram_builder_plant = robot_diagram_builder.plant()
robot_diagram_builder_plant.WeldFrames(robot_diagram_builder_plant.world_frame(), robot_diagram_builder_plant.GetFrameByName("base_link", robot_diagram_builder_plant.GetModelInstanceByName("robot_base")), robot_pose)
robot_diagram_builder_diagram = robot_diagram_builder.Build()

collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = robot_diagram_builder_diagram
collision_checker_params["edge_step_size"] = 0.01
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)


from pydrake.all import HPolyhedron, RandomGenerator, VisibilityGraph, IrisFromCliqueCoverOptions, IrisInConfigurationSpaceFromCliqueCover
from time import time
generator = RandomGenerator(0)
domain = HPolyhedron.MakeBox(collision_checker.plant().GetPositionLowerLimits(),
                           collision_checker.plant().GetPositionUpperLimits())

for n in [int(1e1), int(1e2), int(1e3), int(1e4)]:
    def get_points():
        points = []
        last_point = domain.ChebyshevCenter()
        while len(points) < n:
            point = domain.UniformSample(generator, last_point)
            if collision_checker.CheckConfigCollisionFree(point):
                points.append(point)
            last_point = point
        return np.array(points).T
    t0 = time()
    points = get_points()
    t1 = time()
    print(f"{n = }")
    print(f"time to get points = {t1 - t0}")
    G = VisibilityGraph(collision_checker, points)
    t2 = time()

    print(f"time for visibility_graph = {t2-t1}")
    print()











