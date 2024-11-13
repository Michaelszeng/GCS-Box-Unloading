from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    ExternallyAppliedSpatialForce,
    SpatialForce,
)


import re

# Grasp poses for 4 hand-picked boxes
grasp_poses = """
RigidTransform(
  R=RotationMatrix([
    [0.8213397604015076, 0.38786737397304466, -0.4182820796912635],
    [0.10213533050510548, -0.8214054877967768, -0.5611251187390847],
    [-0.5612213219810941, 0.4181529921267972, -0.7142679489723726],
  ]),
  p=[1.3515518327828233, -0.6830396934642241, 0.7732747306660809],
)
RigidTransform(
  R=RotationMatrix([
    [0.5051888443618475, -0.8630077441535514, -0.0013656732182894973],
    [-0.8630088185192302, -0.5051883977863635, -0.0006796323193542573],
    [-0.00010339431026764047, 0.0015219306966048611, -0.9999988365176088],
  ]),
  p=[0.9408868488583103, -0.4055331601085078, 0.6621035288738676],
)
RigidTransform(
  R=RotationMatrix([
    [0.5349355255721618, -0.8448912661889731, -0.0016528153039417847],
    [-0.8448928343751652, -0.53493513394237, -0.0007077397533158268],
    [-0.00028618583968483024, 0.0017750469437543357, -0.9999983836517],
  ]),
  p=[1.2558133611139781, -0.10121101664304441, 0.6752136442434727],
)
RigidTransform(
  R=RotationMatrix([
    [0.7525779124183528, -0.27458072926953603, 0.5985247771428047],
    [-0.39115836673059756, -0.9175897007434801, 0.07088210794087654],
    [0.5297373102554565, -0.2874622830969315, -0.7979622910388773],
  ]),
  p=[1.9525309976587801, -0.9099933848932932, 1.0850946343557428],
)
"""

# Deposit pose on conveyor
deposit_poses="""
RigidTransform(
  R=RotationMatrix([
    [0.9999391951169945, -0.009430914693448027, 0.005715235500137951],
    [-0.00913119517750239, -0.9986886114567536, -0.0503753771322042],
    [0.0061828264901652, 0.05032012713245413, -0.9987140018353432],
  ]),
  p=[0.05000000090861889, -0.6900000045101764, 1.6000000571600599],
)
RigidTransform(
  R=RotationMatrix([
    [0.9999391951169945, -0.009430914693448027, 0.005715235500137951],
    [-0.00913119517750239, -0.9986886114567536, -0.0503753771322042],
    [0.0061828264901652, 0.05032012713245413, -0.9987140018353432],
  ]),
  p=[0.05000000090861889, -0.6900000045101764, 1.6000000571600599],
)
RigidTransform(
  R=RotationMatrix([
    [0.9999391951169945, -0.009430914693448027, 0.005715235500137951],
    [-0.00913119517750239, -0.9986886114567536, -0.0503753771322042],
    [0.0061828264901652, 0.05032012713245413, -0.9987140018353432],
  ]),
  p=[0.05000000090861889, -0.6900000045101764, 1.6000000571600599],
)
RigidTransform(
  R=RotationMatrix([
    [0.9999391951169945, -0.009430914693448027, 0.005715235500137951],
    [-0.00913119517750239, -0.9986886114567536, -0.0503753771322042],
    [0.0061828264901652, 0.05032012713245413, -0.9987140018353432],
  ]),
  p=[0.05000000090861889, -0.6900000045101764, 1.6000000571600599],
)
"""

def get_grasp_poses(offsets = None):
    return get_poses_from_string(grasp_poses, offsets)

def get_deposit_poses(offsets = None):
    return get_poses_from_string(deposit_poses, offsets)

def get_poses_from_string(string, offsets = None):
    # Regex pattern to capture the body index, rotation matrix, and translation vector
    # pattern = re.compile(
    #     r'RigidTransform\(\s*R=RotationMatrix\(\[\s*\[\s*([^]]+)\s*\],\s*\[\s*([^]]+)\s*\],\s*\[\s*([^]]+)\s*\]\s*\]\s*\),\s*p=\[\s*([^\]]+)\s*\]\s*\)'
    # )
    pattern = re.compile(
        r'RigidTransform\(\s*R=RotationMatrix\(\[\s*\[\s*([^]]+)\s*\],\s*\[\s*([^]]+)\s*\],\s*\[\s*([^]]+)\s*\],\s*\]\s*\),\s*p=\[\s*([^\]]+)'
    )

    string = '\n'.join(line for line in string.splitlines() if not line.strip().startswith("#"))

    # Find all matches in the string
    matches = pattern.findall(string)
    
    poses = []
    if offsets is None:
        offsets = [0,0,0]
    
    for match in matches:
        # Extract the rotation matrix rows and translation vector
        r1 = list(map(float, match[0].split(',')))
        r2 = list(map(float, match[1].split(',')))
        r3 = list(map(float, match[2].split(',')))
        p = list(map(float, match[3].split(',')))
        p[0] += offsets[0]
        p[1] += offsets[1]
        p[2] += offsets[2]
        
        # Create the RotationMatrix and RigidTransform objects
        R = RotationMatrix([r1, r2, r3])
        T = RigidTransform(R=R, p=p)
        
        # Append the RigidTransform to the list
        poses.append(T)
    
    return poses