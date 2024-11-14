import typing as T
import pickle
import numpy as np


def load_data_for_trajectory(traj_num: int):
    """
    pass the trajectry index. 
    0th trajectory moves arm with object from grasp of object 0 to deposit of 0
    1st trajectory moves arm without object from deposit of object 0 to grasp of 1
    2nd trajectory moves arm with object from grasp of object 1 to deposit of 1
    3rd trajectory moves arm without object from deposit of object 1 to grasp of 2
    etc
    """
    file_name = "shortest_walks_trajectories/traj_" + str(traj_num) + ".pkl"
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_trajectory_length(data):
    """
    tells you how long trajectory takes
    """
    return data["time_trajectory"][-1]

def get_pos_vel_acc_jerk(data, t:float):
    """
    pass the data (output of load_data_for_trajectory function)
    and the time since the beginning of the current trajectory.

    NOTE: not the time since start of execution, but time since beginning of this specific trajectory!
    
    returns pos, vel, acc, jerk
    """
    if t < 0:
        p = data["position_trajectory"][0]
        v = data["velocity_trajectory"][0]
        a = data["acceleration_trajectory"][0]
        j = np.zeros(6)
    elif t >= data["time_trajectory"][-1]:
        p = data["position_trajectory"][-1]
        v = data["velocity_trajectory"][-1]
        a = data["acceleration_trajectory"][-1]
        j = np.zeros(6)
    else:
        index = None
        for i, tstep in enumerate(data["time_trajectory"]):
            if tstep > t:
                index = i-1
                break
        dt = t - data["time_trajectory"][index]
        j = data["jerk_trajectory"][index]
        p = data["position_trajectory"][index] + data["velocity_trajectory"][index] * dt +  data["acceleration_trajectory"][index] * dt ** 2 / 2 + j*dt**3/6
        v = data["velocity_trajectory"][index] + data["acceleration_trajectory"][index] * dt +  j * dt ** 2 / 2
        a = data["acceleration_trajectory"][index] + j * dt
    return p,v,a,j