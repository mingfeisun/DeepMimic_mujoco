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

file_path = '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/humanoid_deepmimic.xml'
with open(file_path) as fin:
    MODEL_XML = fin.read()

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)

offset_root_joint = 7

def calc_vel_from_frames(frame_0, frame_1, dt):

    curr_idx = 0
    offset_idx = 7 # root joint offset: 3 (position) + 4 (orientation)
    vel = []
    for each_joint in BODY_JOINTS:
        curr_idx = offset_idx
        dof = DOF_DEF[each_joint]
        if dof == 1:
            offset_idx += dof
            tmp_vel = (frame_1[curr_idx:offset_idx] - frame_0[curr_idx:offset_idx])*1.0/dt
            vel += [tmp_vel]
        elif dof == 3:
            offset_idx = offset_idx + dof + 1
            seg0 = frame_0[curr_idx:offset_idx]
            seg1 = frame_1[curr_idx:offset_idx]
            q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
            q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])
            q_diff = q_0.conjugate * q_1
            axis = q_diff.axis
            angle = q_diff.angle
            
            tmp_vel = (angle * 1.0)/dt * axis
            vel += [tmp_vel[0], tmp_vel[1], tmp_vel[2], 0]

    return np.array(vel)

def read_velocities():
    motions = None
    velocities = None

    with open('humanoid3d_backflip.txt') as fin:
        data = json.load(fin)
        motions = np.array(data["Frames"])
        velocities = np.zeros_like(motions)
        total_time = 0.0
        for each_frame in motions:
            duration = each_frame[0]
            each_frame[0] = total_time
            total_time += duration

        for idx in range(len(motions) - 1):
            dt = motions[idx+1][0] - motions[idx][0]
            frame_0 = motions[idx][1:] # first element is timestamp
            frame_1 = motions[idx+1][1:]

            velocities[idx, 0] = frame_0[0]
            velocities[idx, 1 + offset_root_joint:] = calc_vel_from_frames(frame_0, frame_1, dt)
    return velocities

def align_rotation(rot):
    q_input = Quaternion(rot[0], rot[1], rot[2], rot[3])
    axis = q_input.axis
    angle = q_input.angle
    q_align_right = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                                [0.0, 0.0, 1.0], 
                                                [0.0, -1.0, 0.0]]))
    q_align_left = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                               [0.0, 0.0, -1.0], 
                                               [0.0, 1.0, 0.0]]))
    q_output = q_align_left * q_input * q_align_right
    return q_output.elements

def read_positions():
    motions = None
    all_states = []

    durations = []

    with open('humanoid3d_backflip.txt') as fin:
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
            state['root'] = each_frame[curr_idx:offset_idx]
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

states, durations = read_positions()

from time import sleep

while True:
    for k in range(len(states)):
        state = states[k]
        dura = durations[k]
        sim_state = sim.get_state()

        sim_state.qpos[:7] = state['root']
        tmp = sim_state.qpos[1]
        sim_state.qpos[1] = -sim_state.qpos[2]
        sim_state.qpos[2] = tmp
        sim_state.qpos[3:7] = align_rotation(sim_state.qpos[3:7])

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

        # sleep(dura)

    if os.getenv('TESTING') is not None:
        break