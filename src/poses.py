from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    ExternallyAppliedSpatialForce,
    SpatialForce,
)


import re

"""
Box order:
4
16
17
12
"""

# Grasp poses for 4 hand-picked boxes
grasp_poses = """
RigidTransform(
  R=RotationMatrix([
    [0.7655006608345186, 0.6033286552162619, -0.22361411417183036],
    [0.1955034325446328, -0.5491954750098648, -0.8125039926621623],
    [-0.6130148009022148, 0.5782550164275498, -0.5383623220946954],
  ]),
  p=[1.3899999954008606, -0.5599213410433532, 0.9600000269205806],
)
RigidTransform(
  R=RotationMatrix([
    [0.6963508701755703, 0.7177011471328886, 0.0007273306414716029],
    [0.7177014882620324, -0.6963507661048509, -0.0004292920346258935],
    [0.00019837386369428523, 0.0008209441657139438, -0.9999996433491799],
  ]),
  p=[1.0483833539799166, -0.5797910216507431, 0.7199896525717142],
)
RigidTransform(
  R=RotationMatrix([
    [0.8679219304781242, 0.4967002991599934, 0.0005791454905856858],
    [0.4967005893072739, -0.8679207476413076, -0.0014492741495745302],
    [-0.00021720251647642678, 0.0015455187241592103, -0.9999987820967284],
  ]),
  p=[1.3399998775330113, -0.15999977751546973, 0.7200221170225312],
)
RigidTransform(
  R=RotationMatrix([
    [0.8309219926570796, -0.10998780375267629, 0.5454093189013653],
    [-0.13426736076954657, -0.9909338952861583, 0.00472133508466025],
    [0.539945291627558, -0.07715373094455004, -0.8381565389893268],
  ]),
  p=[1.9499685550255965, -0.9500328401899641, 1.0804003073177475],
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