#!/usr/bin/env python3

import os
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import json
import copy
import numpy as np
from pyquaternion import Quaternion

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
               "left_shoulder", "left_elbow", "right_hip", "right_knee", 
               "right_ankle", "left_hip", "left_knee", "left_ankle"]

BODY_JOINTS_IN_DP_ORDER = ["chest", "neck", "right_hip", "right_knee",
                           "right_ankle", "right_shoulder", "right_elbow", "left_hip", 
                           "left_knee", "left_ankle", "left_shoulder", "left_elbow"]

DOF_DEF = {"chest": 3, "neck": 3, "right_shoulder": 3, "right_elbow": 1, 
           "left_shoulder": 3, "left_elbow": 1, "right_hip": 3, "right_knee": 1, 
           "right_ankle": 3, "left_hip": 3, "left_knee": 1, "left_ankle": 3}

PARAMS_KP_KD = {"chest": [1000, 100], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30], 
           "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50], "right_knee": [500, 50], 
           "right_ankle": [400, 40], "left_hip": [500, 50], "left_knee": [500, 50], "left_ankle": [400, 40]}

'''
Some reference for KD controller:
1. 
mjtNum*   qfrc_bias;            // C(qpos,qvel)                             (nv x 1)

2.
mj_rne
void mj_rne(const mjModel* m, mjData* d, int flg_acc, mjtNum* result);
RNE: compute M(qpos)*qacc + C(qpos,qvel); flg_acc=0 removes inertial term.

3. 
mj_solveM
void mj_solveM(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* y, int n);
Solve linear system M * x = y using factorization: x = inv(L'*D*L)*y
'''

file_path = 'humanoid_deepmimic/envs/asset/humanoid_deepmimic.xml'
with open(file_path) as fin:
    MODEL_XML = fin.read()

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)


def calc_linear_vel_from_frames(frame_0, frame_1, dt):
    curr_idx = 0
    offset_idx = 0 # root joint offset: 3 (position) + 4 (orientation)
    vel_linear = []

    curr_idx = offset_idx
    offset_idx += 3 # position is 3D
    vel_linear = (frame_1[curr_idx:offset_idx] - frame_0[curr_idx:offset_idx])*1.0/dt
    vel_linear = align_position(vel_linear)

    return vel_linear

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

def read_positions():
    motions = None
    all_states = []

    durations = []

    with open('./motions/humanoid3d_dance_b.txt') as fin:
        data = json.load(fin)
        motions = np.array(data["Frames"])
        total_time = 0.0
        for each_frame in motions:
            duration = each_frame[0]
            each_frame[0] = total_time
            total_time += duration
            durations.append(duration)

        for each_frame in motions:
            curr_idx = 1
            offset_idx = 8
            state = {}
            state['root_pos'] = align_position(each_frame[curr_idx:curr_idx+3])
            state['root_rot'] = align_rotation(each_frame[curr_idx+3:offset_idx])
            for each_joint in BODY_JOINTS_IN_DP_ORDER:
                curr_idx = offset_idx
                dof = DOF_DEF[each_joint]
                if dof == 1:
                    offset_idx += 1
                    state[each_joint] = each_frame[curr_idx:offset_idx]
                elif dof == 3:
                    offset_idx += 4
                    state[each_joint] = align_rotation(each_frame[curr_idx:offset_idx])
            all_states.append(state)

    return all_states, durations

def read_velocities(dt=None):
    states, durations = read_positions()

    offset_root_joint_vel = 6 # 6 = 3 (linear velocity) + 3 (angular velocity)
    n_states = len(states)
    n_vel = sum(DOF_DEF.values()) + offset_root_joint_vel 
    velocities = np.zeros((n_states, n_vel))

    for idx in range(n_states - 1):
        state_0 = states[idx]
        state_1 = states[idx+1]

        if not dt:
            dt = durations[idx]

        curr_idx = 0
        offset_idx = 6
        
        velocities[idx, 0:3] = calc_linear_vel_from_frames(state_0["root_pos"], state_1["root_pos"], dt)
        velocities[idx, 3:6] = calc_angular_vel_from_frames(state_0["root_rot"], state_1["root_rot"], dt)

        for each_joint in BODY_JOINTS:
            curr_idx = offset_idx
            pos_0 = state_0[each_joint]
            pos_1 = state_1[each_joint]
            dof = DOF_DEF[each_joint]

            # velocities[idx, 26] = (np.random.rand() - 0.5) * 10

            if dof == 1:
                offset_idx += 1
                velocities[idx, curr_idx:offset_idx] = (pos_1 - pos_0) * 1.0 / dt
            elif dof == 3:
                offset_idx += 3
                velocities[idx, curr_idx:offset_idx] = calc_angular_vel_from_frames(pos_0, pos_1, dt)

    return velocities

def render_from_pos():
    states, durations = read_positions()

    from time import sleep

    phase_offset = np.array([0.0, 0.0, 0.0])

    while True:
        for k in range(len(states)):
            state = states[k]
            dura = durations[k]
            sim_state = sim.get_state()

            sim_state.qpos[:3] = np.array(state['root_pos']) + phase_offset
            sim_state.qpos[3:7] = state['root_rot']

            for each_joint in BODY_JOINTS:
                idx = sim.model.get_joint_qpos_addr(each_joint)
                tmp_val = state[each_joint]
                if isinstance(idx, np.int32):
                    assert 1 == len(tmp_val)
                    sim_state.qpos[idx] = state[each_joint]
                elif isinstance(idx, tuple):
                    assert idx[1] - idx[0] == len(tmp_val)
                    sim_state.qpos[idx[0]:idx[1]] = state[each_joint]

            # print(sim_state.qpos)
            sim.set_state(sim_state)
            sim.forward()
            viewer.render()

        sim_state = sim.get_state()
        phase_offset = sim_state.qpos[:3]
        phase_offset[2] = 0

        if os.getenv('TESTING') is not None:
            break

def render_from_vel():
    velocities = read_velocities()

    while True:
        for each_vel in velocities:
            sim_state = sim.get_state()
            sim_state.qvel[:] = each_vel[:]
            sim.set_state(sim_state)
            sim.forward()
            viewer.render()

        if os.getenv('TESTING') is not None:
            break

def render_from_torques():
    kp, kd = [], []
    for each_joint in BODY_JOINTS:
        kp += [PARAMS_KP_KD[each_joint][0] for _ in range(DOF_DEF[each_joint])]
        kd += [PARAMS_KP_KD[each_joint][1] for _ in range(DOF_DEF[each_joint])]

    kp = np.array(kp)
    kd = np.array(kd)

    pos_err = calc_pos_err_test()
    vel_err = calc_vel_err_test()
    while True:
        for (p_err, v_err) in zip(pos_err, vel_err):
            torque = kp * p_err[6:] + kd * v_err[6:]
            # torque = kp * p_err[6:]
            print(torque)
            sim.data.ctrl[:] = torque[:]
            # sim.forward()
            sim.step()
            viewer.render()

def calc_pos_err_test():
    # pos_states, _ = read_positions()
    # curr_velocities = read_velocities()
    pos_err = read_velocities(dt=1.0)
    return pos_err[:-1, :]

def calc_vel_err_test():
    velocities = read_velocities()
    vel_err = velocities[1:] - velocities[:-1]
    return vel_err

def action2torques(action):
    pass


def calc_angular_vel_from_frames(orien_0, orien_1, dt):
    seg0 = align_rotation(orien_0)
    seg1 = align_rotation(orien_1)

    q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
    q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])

    q_diff =  q_0.conjugate * q_1
    # q_diff =  q_1 * q_0.conjugate
    axis = q_diff.axis
    angle = q_diff.angle
    
    tmp_vel = (angle * 1.0)/dt * axis
    vel_angular = np.array([tmp_vel[0], tmp_vel[1], tmp_vel[2]])

    return vel_angular


def calc_pos_err(now_pos, next_pos):
    curr_idx = 0
    offset_idx = 6
    assert len(now_pos) == len(next_pos)
    err = np.zeros_like(now_pos)

    for each_joint in BODY_JOINTS:
        curr_idx = offset_idx
        dof = DOF_DEF[each_joint]
        if dof == 1:
            offset_idx += 1
            seg_0 = now_pos[curr_idx:offset_idx]
            seg_1 = next_pos[curr_idx:offset_idx]
            err[curr_idx:offset_idx] = (seg_1 - seg_0) * 1.0
        elif dof == 3:
            offset_idx += 4
            seg_0 = now_pos[curr_idx:offset_idx]
            seg_1 = next_pos[curr_idx:offset_idx]
            err[curr_idx:offset_idx] = calc_angular_vel_from_frames(seg_0, seg_1, 1.0)
    return err


def align_pos(pos_input):
    offset_map = {}
    offset_idx = 0
    for each_joint in BODY_JOINTS_IN_DP_ORDER:
        offset_map[each_joint] = offset_idx
        if DOF_DEF[each_joint] == 1:
            offset_idx += 1
        elif DOF_DEF[each_joint] == 3:
            offset_idx += 4
        else:
            raise NotImplementedError
    pos_output = []
    for each_joint in BODY_JOINTS:
        offset_idx = offset_map[each_joint]
        dof = DOF_DEF[each_joint]
        tmp_seg = []
        if dof == 1:
            tmp_seg = [pos_input[offset_idx]]
        elif dof == 3:
            tmp_seg = pos_input[offset_idx:offset_idx+dof+1]
        else:
            raise NotImplementedError
        pos_output += tmp_seg

    return pos_output

def calc_vel_err(now_vel, next_vel):
    return next_vel - now_vel

if __name__ == "__main__":
    # render_from_torques()
    render_from_pos()
    # render_from_vel()