from pydrake.all import (
    LeafSystem,
    AbstractValue,
    GraphOfConvexSetsOptions,
    GcsTrajectoryOptimization,
    Point,
    LoadIrisRegionsYamlFile,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Solve,
    CompositeTrajectory,
    PiecewisePolynomial,
    logical_or,
    logical_and
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

from scenario import scenario_yaml_for_iris
from utils import NUM_BOXES, is_yaml_empty, SuppressOutput

import time
import numpy as np
from pathlib import Path
import pydot
import os

class Debugger(LeafSystem):
    """
    Simple Leafsystem that prints information from input port for debugging
    purposes.
    """

    def __init__(self, print_frequency=0.025):
        LeafSystem.__init__(self)

        kuka_state = self.DeclareVectorInputPort(name="kuka_state", size=12)  # 6 pos, 6 vel

        body_poses = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("kuka_current_pose", body_poses)

        kuka_actuation = self.DeclareVectorInputPort(name="kuka_actuation", size=6)

        self.DeclarePeriodicUnrestrictedUpdateEvent(print_frequency, 0.0, self.debug_print)


    def debug_print(self, context, state):
        kuka_state = self.get_input_port(0).Eval(context)
        q_current = kuka_state[:6]
        q_dot_current = kuka_state[6:]
        print(f"q_current: {q_current}")
        print(f"q_dot_current: {q_dot_current}")

        kuka_actuation = self.get_input_port(2).Eval(context)
        print(f"kuka_actuation: {kuka_actuation}")