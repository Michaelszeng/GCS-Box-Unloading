""" Miscellaneous Utility functions """

from enum import Enum
from typing import BinaryIO, Optional, Union, Tuple
from pydrake.all import (
    Diagram,
    RigidTransform,
    RotationMatrix,
    SpatialVelocity,
    MultibodyPlant,
    Context,
    AngleAxis,
)
from dataclasses import dataclass, field
from pydrake.common.yaml import yaml_load_typed
import numpy as np
import numpy.typing as npt
import pydot
import matplotlib.pyplot as plt


def diagram_visualize_connections(diagram: Diagram, file: Union[BinaryIO, str]) -> None:
    """
    Create SVG file of system diagram.
    """
    if type(file) is str:
        file = open(file, "bw")
    graphviz_str = diagram.GetGraphvizString()
    svg_data = pydot.graph_from_dot_data(
        diagram.GetGraphvizString())[0].create_svg()
    file.write(svg_data)