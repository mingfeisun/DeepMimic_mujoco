import math
import numpy as np
from pyquaternion import Quaternion

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

BODY_JOINTS_IN_DP_ORDER = ["chest", "neck", "right_hip", "right_knee",
                        "right_ankle", "right_shoulder", "right_elbow", "left_hip", 
                        "left_knee", "left_ankle", "left_shoulder", "left_elbow"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

BODY_DEFS = ["root", "chest", "neck", "right_hip", "right_knee", 
             "right_ankle", "right_shoulder", "right_elbow", "right_wrist", "left_hip", 
             "left_knee", "left_ankle", "left_shoulder", "left_elbow", "left_wrist"]

PARAMS_KP_KD = {"chest": [1000, 100], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30], 
        "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50], "right_knee": [500, 50], 
        "right_ankle": [400, 40], "left_hip": [500, 50], "left_knee": [500, 50], "left_ankle": [400, 40]}

JOINT_WEIGHT = {"root": 1, "chest": 0.5, "neck": 0.3, "right_hip": 0.5, 
                "right_knee": 0.3, "right_ankle": 0.2, "right_shoulder": 0.3, "right_elbow": 0.2, 
                "right_wrist": 0.0, "left_hip": 0.5, "left_knee": 0.3, "left_ankle": 0.2, 
                "left_shoulder": 0.3, "left_elbow": 0.2, "left_wrist": 0.0}

def align_rotation(rot):
    q_input = Quaternion(rot[0], rot[1], rot[2], rot[3])
    q_align_right = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                                [0.0, 0.0, 1.0], 
                                                [0.0, -1.0, 0.0]]))
    q_align_left = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                               [0.0, 0.0, -1.0], 
                                               [0.0, 1.0, 0.0]]))
    q_output = q_align_left * q_input * q_align_right
    return q_output.elements

def align_position(pos):
    assert len(pos) == 3
    left_matrix = np.array([[1.0, 0.0, 0.0], 
                            [0.0, 0.0, -1.0], 
                            [0.0, 1.0, 0.0]])
    pos_output = np.matmul(left_matrix, pos)
    return pos_output

def calc_angular_vel_from_quaternion(orien_0, orien_1, dt):
    seg0 = align_rotation(orien_0)
    seg1 = align_rotation(orien_1)

    q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
    q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])

    q_diff =  q_0.conjugate * q_1
    # q_diff =  q_1 * q_0.conjugate
    axis = q_diff.axis
    angle = q_diff.angle
    
    tmp_vel = (angle * 1.0)/dt * axis
    vel_angular = [tmp_vel[0], tmp_vel[1], tmp_vel[2]]

    return vel_angular

def calc_diff_from_quaternion(orien_0, orien_1):
    seg0 = align_rotation(orien_0)
    seg1 = align_rotation(orien_1)

    q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
    q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])

    q_diff =  q_0.conjugate * q_1
    # q_diff =  q_1 * q_0.conjugate
    angle = q_diff.angle
    return angle
# 
# def xyzrot2quat(xyzrot):
#     sx, cx = math.sin(xyzrot[0]), math.cos(xyzrot[0])
#     sy, cy = math.sin(xyzrot[1]), math.cos(xyzrot[1])
#     sz, cz = math.sin(xyzrot[2]), math.cos(xyzrot[2])
# 
#     R_x = np.array([[1.0, 0.0, 0.0],
#                     [0.0, cx, -sx],
#                     [0.0, sx, cx]])
# 
#     R_y = np.array([[cy, 0.0, sy],
#                     [0.0, 1, 0.0],
#                     [-sy, 0.0, cy]])
# 
#     R_z = np.array([[cz, -sz, 0.0],
#                     [sz, cz, 0.0],
#                     [0.0, 0.0, 1.0]])
#     
#     rot = np.matmul( np.matmul(R_x, R_y), R_z)
# 
#     quat = Quaternion(matrix=rot)
# 
#     return  quat.elements