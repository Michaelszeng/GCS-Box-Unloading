"""Simulation scenario YAML definitions."""

import os

relative_path_to_robot_base = '../data/unload-gen0/robot_base.urdf'
relative_path_to_robot_arm = '../data/unload-gen0/robot_arm.urdf'

""" Note: It is necessary to split the truck trailer into individual parts since
    Drake automatically takes the convex hull of the collision geometry, which 
    would make the hollow shipping container no longer hollow."""
relative_path_to_truck_trailer_floor = '../data/Truck_Trailer_Floor.sdf'
relative_path_to_truck_trailer_back = '../data/Truck_Trailer_Back.sdf'
relative_path_to_truck_trailer_right_side = '../data/Truck_Trailer_Right_Side.sdf'
relative_path_to_truck_trailer_left_side = '../data/Truck_Trailer_Left_Side.sdf'
relative_path_to_truck_trailer_roof = '../data/Truck_Trailer_Roof.sdf'

absolute_path_to_robot_base = os.path.abspath(relative_path_to_robot_base)
absolute_path_to_robot_arm = os.path.abspath(relative_path_to_robot_arm)
absolute_path_to_truck_trailer_floor = os.path.abspath(relative_path_to_truck_trailer_floor)
absolute_path_to_truck_trailer_back = os.path.abspath(relative_path_to_truck_trailer_back)
absolute_path_to_truck_trailer_right_side = os.path.abspath(relative_path_to_truck_trailer_right_side)
absolute_path_to_truck_trailer_left_side = os.path.abspath(relative_path_to_truck_trailer_left_side)
absolute_path_to_truck_trailer_roof = os.path.abspath(relative_path_to_truck_trailer_roof)

scenario_yaml = f"""
directives:
- add_model:
    name: robot_base
    file: file://{absolute_path_to_robot_base}
- add_model:
    name: kuka
    file: file://{absolute_path_to_robot_arm}
    default_joint_positions:
        arm_a1: [0.0]
        arm_a2: [-2.5]
        arm_a3: [2.8]
        arm_a4: [0.0]
        arm_a5: [1.2]
        arm_a6: [0.0]
- add_weld:
    parent: robot_base::base
    child: kuka::base_link        


- add_model: 
    name: Truck_Trailer_Floor
    file: file://{absolute_path_to_truck_trailer_floor}
- add_weld:
    parent: world
    child: Truck_Trailer_Floor::Truck_Trailer_Floor


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


- add_model: 
    name: Truck_Trailer_Back
    file: file://{absolute_path_to_truck_trailer_back}
- add_weld:
    parent: world
    child: Truck_Trailer_Back::Truck_Trailer_Back
"""


scenario_yaml_for_iris = scenario_yaml.replace(
f"""
model_drivers:
    kuka: !ForceDriver {{}}  # ForceDriver allows access to desired_state and desired_acceleration input ports for station (results in better traj following)
""",
""
)

scenario_yaml_for_iris = scenario_yaml_for_iris.replace(
f"""
- add_model: 
    name: Truck_Trailer_Roof
    file: file://{absolute_path_to_truck_trailer_roof}
""",
f"""
- add_model: 
    name: Truck_Trailer_Roof
    file: file://{absolute_path_to_truck_trailer_roof}
- add_weld:
    parent: world
    child: Truck_Trailer_Roof::Truck_Trailer_Roof
"""
)


robot_yaml = f"""
directives:
- add_model:
    name: kuka
    file: file://{absolute_path_to_robot_arm}
    default_joint_positions:
        arm_a1: [0.0]
        arm_a2: [-2.5]
        arm_a3: [2.8]
        arm_a4: [0.0]
        arm_a5: [1.2]
        arm_a6: [0.0]

- add_weld:
    parent: world
    child: kuka::base_link
"""