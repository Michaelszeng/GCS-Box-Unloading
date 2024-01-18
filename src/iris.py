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

import time
import numpy as np