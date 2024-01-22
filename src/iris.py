from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BsplineTrajectory,
    DiagramBuilder,
    InverseKinematics,
    IrisInConfigurationSpace,
    IrisOptions,
    MathematicalProgram,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PositionConstraint,
    Rgba,
    RigidTransform,
    Role,
    Solve,
    StartMeshcat,
)
from manipulation.utils import ConfigureParser

from scenario import scenario_yaml_for_source_regions

import time
import numpy as np

def generate_source_iris_regions():
    """
    Source IRIS regions are defined as the regions considering only self-
    collision with the robot, and collision with the walls of the empty truck
    trailed (excluding the back wall).
    """

    # Create new MBP containing just robot and truck trailer walls
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    print(scenario_yaml_for_source_regions)
    parser.AddModelsFromString(scenario_yaml_for_source_regions, ".dmd.yaml")
    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    plant_context = plant.GetMyContextFromRoot(context)

    options = IrisOptions()
    options.num_collision_infeasible_samples = 1
    options.require_sample_point_is_contained = True
    region = IrisInConfigurationSpace(plant, plant_context, options)
    print(f"region volume: {region.CalcVolumeViaSampling()}")
    
