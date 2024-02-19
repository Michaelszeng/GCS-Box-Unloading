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
import yaml
import os
import sys

NUM_BOXES = 40
BOX_DIM = 0.5


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


def is_yaml_empty(file_path):
    with open(file_path, 'r') as file:
        try:
            content = yaml.safe_load(file)
            # Check if the content is None or an empty collection
            return content is None or not content
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return False  # Assuming the file is not empty if there's a parsing error
        

class SuppressOutput:
    def __enter__(self):
        self.original_stdout_fd = sys.stdout.fileno()
        self.original_stderr_fd = sys.stderr.fileno()

        # Flush the Python stdout and stderr buffers
        sys.stdout.flush()
        sys.stderr.flush()

        self.devnull = os.open(os.devnull, os.O_WRONLY)

        # Duplicate the file descriptors to restore later
        self.saved_stdout_fd = os.dup(self.original_stdout_fd)
        self.saved_stderr_fd = os.dup(self.original_stderr_fd)

        # Replace stdout and stderr with /dev/null
        os.dup2(self.devnull, self.original_stdout_fd)
        os.dup2(self.devnull, self.original_stderr_fd)

    def __exit__(self, exc_type, exc_value, traceback):
        # Flush any C++ buffered outputs
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore the original stdout and stderr file descriptors
        os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
        os.dup2(self.saved_stderr_fd, self.original_stderr_fd)

        # Close the temporary file descriptors and /dev/null
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        os.close(self.devnull)
