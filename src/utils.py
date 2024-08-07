""" Miscellaneous Utility functions """
from pydrake.all import (
    Diagram,
    RigidTransform,
    RotationMatrix,
    SpatialVelocity,
    MultibodyPlant,
    Context,
    AngleAxis,
    VPolytope,
    Point,
    InverseKinematics,
    Solve,
    logical_or,
    logical_and
)
from pydrake.common.yaml import yaml_load_typed

from enum import Enum
from typing import BinaryIO, Optional, Union, Tuple
import numpy as np
import numpy.typing as npt
import pydot
import matplotlib.pyplot as plt
import yaml
import os
import sys
import time

from scenario import q_nominal


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


def ik(plant, plant_context, pose, translation_error=0, rotation_error=0.05, regions=None):
    """
    Use Inverse Kinematics to solve for a configuration that satisfies a
    task-space pose. 
    
    If regions is not None, this function also ensures the configuration is
    reachable within one of the regions (or return None if this isn't possible).
    """
    satisfy_regions_constraint = regions is not None
    if regions is None:  # Make regions not None so that the for loop below runs at least once
        regions = {"_": Point(np.zeros(6))}

    # Separate IK program for each region with the constraint that the IK result must be in that region
    ik_start = time.time()
    solve_success = False
    for region in list(regions.values()):
        ik = InverseKinematics(plant, plant_context)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        ik_prog = ik.get_mutable_prog()

        ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_nominal, q_variables)

        # q_variables must be within half-plane for every half-plane in region
        if satisfy_regions_constraint:
            ik_prog.AddConstraint(logical_and(*[expr <= const for expr, const in zip(region.A() @ q_variables, region.b())]))

        # Pose constraint
        ik.AddPositionConstraint(
            frameA=plant.world_frame(),
            frameB=plant.GetFrameByName("arm_eef"),
            p_BQ=[0, 0, 0.1],
            p_AQ_lower=pose.translation() - translation_error,
            p_AQ_upper=pose.translation() + translation_error,
        )
        ik.AddOrientationConstraint(
            frameAbar=plant.world_frame(),
            R_AbarA=pose.rotation(),
            frameBbar=plant.GetFrameByName("arm_eef"),
            R_BbarB=RotationMatrix(),
            theta_bound=rotation_error,
        )

        ik_prog.SetInitialGuess(q_variables, q_nominal)
        ik_result = Solve(ik_prog)
        if ik_result.is_success():
            q = ik_result.GetSolution(q_variables)  # (6,) np array
            print(f"IK solve succeeded. q: {q}")
            print(pose.translation())
            print(pose.rotation())
            solve_success = True
            break
        # else:
            # print(f"ERROR: IK fail: {ik_result.get_solver_id().name()}: {ik_result.GetInfeasibleConstraintNames(ik_prog)}")

    # print(f"IK Runtime: {time.time() - ik_start}")

    if solve_success == False:
        # print(f"ERROR: IK fail: {ik_result.get_solver_id().name()}.")
        return None
    
    return q