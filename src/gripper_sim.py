from pydrake.all import (
    LeafSystem,
    AbstractValue,
    RigidTransform,
    RotationMatrix,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    ExternallyAppliedSpatialForce,
    SpatialForce,
    BodyIndex,
    Rgba,
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

import numpy as np

from scenario import BOX_DIM, NUM_BOXES, PREPICK_MARGIN


class GripperSimulator(LeafSystem):
    """
    Simulates the suction gripper by applying external forces on the box being
    grabbed.

    Also serves the role of applying forces on boxes during box randomization
    procedure.
    """

    def __init__(self, plant, meshcat, randomize_boxes, box_fall_runtime, box_randomization_runtime):
        LeafSystem.__init__(self)

        motion_planner_state = self.DeclareVectorInputPort(name="motion_planner_state", size=1)  # 1 for pre-picking, 2 for picking, 0 for placing

        target_box_body_idx = AbstractValue.Make(BodyIndex())
        self.DeclareAbstractInputPort("target_box_body_idx", target_box_body_idx)

        target_box_X_pick = AbstractValue.Make(RigidTransform())
        self.DeclareAbstractInputPort("target_box_X_pick", target_box_X_pick)

        body_poses = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("body_poses", body_poses)

        self.DeclareAbstractOutputPort(
            "applied_spatial_force",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self.output_forces,
        )

        self.plant = plant
        self.meshcat = meshcat
        self.randomize_boxes = randomize_boxes
        self.box_fall_runtime = box_fall_runtime
        self.box_randomization_runtime = box_randomization_runtime


    def compute_box_randomization_forces(self):
        box_forces = []
        for i in range(NUM_BOXES):
            force = ExternallyAppliedSpatialForce()
            zero_force = ExternallyAppliedSpatialForce()

            box_model_idx = self.plant.GetModelInstanceByName(f"Boxes/Box_{i}")  # ModelInstanceIndex
            box_body_idx = self.plant.GetBodyIndices(box_model_idx)[0]  # BodyIndex

            force.body_index = box_body_idx
            force.p_BoBq_B = [0,0,0]
            force.F_Bq_W = SpatialForce(tau=[0,0,0], f=[1000,0,0])
            box_forces.append(force)

        return box_forces

    
    def output_forces(self, context, output):
        # Apply backward pushing force during box randomization
        if self.randomize_boxes and context.get_time() > self.box_fall_runtime+1.0 and context.get_time() < self.box_fall_runtime:
            output.set_value(self.compute_box_randomization_forces())
            return

        # Otherwise, apply forces as needed simulate gripper suction on target box
        motion_planner_state = self.get_input_port(0).Eval(context)
        target_box_body_idx = self.get_input_port(1).Eval(context)

        force = ExternallyAppliedSpatialForce()

        if motion_planner_state == 0:  # Placing --> apply gripper force
            body_poses = self.get_input_port(3).Eval(context)
            target_box_pose = body_poses[target_box_body_idx]
            target_box_center = RigidTransform(target_box_pose.rotation(), target_box_pose.translation() + target_box_pose.rotation() @ np.array([BOX_DIM/2, BOX_DIM/2, -BOX_DIM/2]))  # Because box_pose is at the corner of the box

            eef_model_idx = self.plant.GetModelInstanceByName("kuka")  # ModelInstanceIndex
            eef_body_idx = self.plant.GetBodyIndices(eef_model_idx)[-1]  # BodyIndex
            
            # We find the direction to apply to apply the force on the box by rotating the vector from eef to box center at pick
            # by the amount the box has rotated since being picked up
            force_direction = (body_poses[eef_body_idx].translation() - target_box_center.translation()) / np.linalg.norm(body_poses[eef_body_idx].translation() - target_box_center.translation())

            self.meshcat.SetLine("gripper force", np.vstack((target_box_center.translation(), target_box_center.translation()+force_direction)).T, line_width=50.0, rgba=Rgba(1.0, 0.75, 0.0, 0.5))

            force.body_index = target_box_body_idx
            force.p_BoBq_B = [BOX_DIM/2, BOX_DIM/2, -BOX_DIM/2]  # apply force at box center
            print(f"Applying force: {400*force_direction}")
            force.F_Bq_W = SpatialForce(tau=[0,0,0], f=400*force_direction)

        else:  # Remove gripper force on target box
            force.body_index = target_box_body_idx
            force.p_BoBq_B = [BOX_DIM/2, BOX_DIM/2, -BOX_DIM/2]  # apply force at box center
            force.F_Bq_W = SpatialForce(tau=[0,0,0], f=[0,0,0])

        output.set_value([force])
