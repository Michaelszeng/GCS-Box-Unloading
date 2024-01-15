###############
# Robot Setup #
###############

from pydrake.all import (
    GeometrySet,
    CollisionFilterDeclaration,
    Role
)

import os

relative_path_to_robot = '../data/unload-gen0/robot.urdf'

""" Note: It is necessary to split the truck trailer into individual parts since
    Drake automatically takes the convex hull of the collision geometry, which 
    would make the hollow shipping container no longer hollow."""
relative_path_to_truck_trailer_floor = '../data/Truck_Trailer_Floor.sdf'
relative_path_to_truck_trailer_back = '../data/Truck_Trailer_Back.sdf'
relative_path_to_truck_trailer_right_side = '../data/Truck_Trailer_Right_Side.sdf'
relative_path_to_truck_trailer_left_side = '../data/Truck_Trailer_Left_Side.sdf'
relative_path_to_truck_trailer_roof = '../data/Truck_Trailer_Roof.sdf'

absolute_path_to_robot = os.path.abspath(relative_path_to_robot)
absolute_path_to_truck_trailer_floor = os.path.abspath(relative_path_to_truck_trailer_floor)
absolute_path_to_truck_trailer_back = os.path.abspath(relative_path_to_truck_trailer_back)
absolute_path_to_truck_trailer_right_side = os.path.abspath(relative_path_to_truck_trailer_right_side)
absolute_path_to_truck_trailer_left_side = os.path.abspath(relative_path_to_truck_trailer_left_side)
absolute_path_to_truck_trailer_roof = os.path.abspath(relative_path_to_truck_trailer_roof)

scenario_yaml = f"""
model_drivers:
    kuka: !ForceDriver {{}}  # ForceDriver allows access to desired_state and desired_acceleration input ports for station (results in better traj following)

directives:
- add_model:
    name: kuka
    file: file://{absolute_path_to_robot}
    default_joint_positions:
        left_wheel: [0.0]
        right_wheel: [0.0]
        arm_a6: [0.0]
        arm_a5: [0.0]
        arm_a4: [0.0]
        arm_a3: [1.0]
        arm_a2: [-1.0]
        arm_a1: [0.0]

- add_weld:
    parent: world
    child: kuka::base_link

    

- add_model: 
    name: Truck_Trailer_Floor
    file: file://{absolute_path_to_truck_trailer_floor}

- add_weld:
    parent: world
    child: Truck_Trailer_Floor::Truck_Trailer_Floor

    

- add_model: 
    name: Truck_Trailer_Back
    file: file://{absolute_path_to_truck_trailer_back}

- add_weld:
    parent: world
    child: Truck_Trailer_Back::Truck_Trailer_Back



- add_model: 
    name: Truck_Trailer_Right_Side
    file: file://{absolute_path_to_truck_trailer_right_side}

- add_weld:
    parent: world
    child: Truck_Trailer_Right_Side::Truck_Trailer_Right_Side



- add_model: 
    name: Truck_Trailer_Left_Side
    file: file://{absolute_path_to_truck_trailer_left_side}

- add_weld:
    parent: world
    child: Truck_Trailer_Left_Side::Truck_Trailer_Left_Side
    
    

- add_model: 
    name: Truck_Trailer_Roof
    file: file://{absolute_path_to_truck_trailer_roof}
"""



def filterCollisionGeometry(scene_graph, context=None):
    """Some robot models may appear to have self collisions due to overlapping 
    collision geometries.
    This function filters out such problems for our PR2 model."""
    if context is None:
        filter_manager = scene_graph.collision_filter_manager()
    else:
        filter_manager = scene_graph.collision_filter_manager(context)
    inspector = scene_graph.model_inspector()

    robot = {}

    for gid in inspector.GetGeometryIds(
        GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
    ):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))
        print(f"gid_name: {gid_name}")
        if "arm" in gid_name or "base" in gid_name or "motor" in gid_name:
            link_name = gid_name.split("::")[1]
            robot[link_name] = [gid]

    print(f"robot: {robot}")

    def add_exclusion(set1, set2=None):
        if set2 is None:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
            )
        else:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    GeometrySet(set1), GeometrySet(set2)
                )
            )

    # Robot-to-self collisions
    # add_exclusion(
    #     arm["base_link"],
    #     arm["l_shoulder_pan_link"]
    #     + arm["r_shoulder_pan_link"]
    #     + arm["l_upper_arm_link"]
    #     + arm["r_upper_arm_link"]
    #     + arm["head_pan_link"]
    #     + arm["head_tilt_link"],
    # )
    # add_exclusion(
    #     arm["torso_lift_link"], arm["head_pan_link"] + arm["head_tilt_link"]
    # )
    # add_exclusion(
    #     arm["l_shoulder_pan_link"] + arm["torso_lift_link"],
    #     arm["l_upper_arm_link"],
    # )
    # add_exclusion(
    #     arm["r_shoulder_pan_link"] + arm["torso_lift_link"],
    #     arm["r_upper_arm_link"],
    # )
    add_exclusion(robot["base"], robot["arm_a2"])
    add_exclusion(robot["arm_a4"], robot["arm_a6"])