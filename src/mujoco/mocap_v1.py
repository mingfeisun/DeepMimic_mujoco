#!/usr/bin/env python3
import os
import json
import math
import copy
import numpy as np
from os import getcwd
from pyquaternion import Quaternion
from mujoco.mocap_util import align_position, align_rotation
from mujoco.mocap_util import BODY_JOINTS, BODY_JOINTS_IN_DP_ORDER, DOF_DEF, BODY_DEFS

def quat2euler(elements):
    q0, q1, q2, q3 = elements[0], elements[1], elements[2], elements[3]
    phi = math.atan2(2.0*(q0*q1+q2*q3), 1.0-2.0*(q1*q1+q2*q2))
    # phi = math.atan(2.0*(q0*q1+q2*q3) / (1.0-2.0*(q1*q1+q2*q2)))
    theta = math.asin(2.0*(q0*q2-q3*q1))
    psi = math.atan2(2.0*(q0*q3+q1*q2), 1.0-2.0*(q2*q2+q3*q3))
    # psi = math.atan(2.0*(q0*q3+q1*q2) / (1.0-2.0*(q2*q2+q3*q3)))

    tmp_q0 = math.cos(phi/2)*math.cos(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.sin(theta/2)*math.sin(psi/2)
    tmp_q1 = math.sin(phi/2)*math.cos(theta/2)*math.cos(psi/2) - math.cos(phi/2)*math.sin(theta/2)*math.sin(psi/2)
    tmp_q2 = math.cos(phi/2)*math.sin(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.cos(theta/2)*math.sin(psi/2)
    tmp_q3 = math.cos(phi/2)*math.cos(theta/2)*math.sin(psi/2) - math.sin(phi/2)*math.sin(theta/2)*math.cos(psi/2)
    print("Origin: ", q0, q1, q2, q3)
    print("Converted: ", tmp_q0, tmp_q1, tmp_q2, tmp_q3)
    return [phi, theta, psi]
    # return [psi, theta, phi]

class MocapDM(object):
    def __init__(self):
        self.num_bodies = len(BODY_DEFS)
        self.pos_dim = 3
        self.rot_dim = 4

    def load_mocap(self, filepath):
        self.read_raw_data(filepath)
        self.convert_raw_data()

    def read_raw_data(self, filepath):
        motions = None
        all_states = []

        durations = []

        with open(filepath, 'r') as fin:
            data = json.load(fin)
            motions = np.array(data["Frames"])
            m_shape = np.shape(motions)
            self.data = np.full(m_shape, np.nan)

            total_time = 0.0
            self.dt = motions[0][0]
            for each_frame in motions:
                duration = each_frame[0]
                each_frame[0] = total_time
                total_time += duration
                durations.append(duration)

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

        self.all_states = all_states
        self.durations = durations

    def calc_rot_vel(self, seg_0, seg_1, dura):
        q_0 = Quaternion(seg_0[0], seg_0[1], seg_0[2], seg_0[3])
        q_1 = Quaternion(seg_1[0], seg_1[1], seg_1[2], seg_1[3])

        q_diff =  q_0.conjugate * q_1
        # q_diff =  q_1 * q_0.conjugate
        axis = q_diff.axis
        angle = q_diff.angle
        
        tmp_diff = angle/dura * axis
        diff_angular = [tmp_diff[0], tmp_diff[1], tmp_diff[2]]

        return diff_angular

    def convert_raw_data(self):
        self.data_vel = []
        self.data_angle = []

        for k in range(len(self.all_states)):
            tmp_vel = []
            tmp_angle = []
            state = self.all_states[k]
            if k == 0:
                dura = self.durations[k]
            else:
                dura = self.durations[k-1]

            # time duration
            init_idx = 0
            offset_idx = 1
            self.data[k, init_idx:offset_idx] = dura

            # root pos
            init_idx = offset_idx
            offset_idx += 3
            self.data[k, init_idx:offset_idx] = np.array(state['root_pos'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/dura).tolist()
            tmp_angle += state['root_pos'].tolist()

            # root rot
            init_idx = offset_idx
            offset_idx += 4
            self.data[k, init_idx:offset_idx] = np.array(state['root_rot'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                tmp_vel += self.calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], dura)
            tmp_angle += state['root_rot'].tolist()

            for each_joint in BODY_JOINTS:
                init_idx = offset_idx
                tmp_val = state[each_joint]
                if DOF_DEF[each_joint] == 1:
                    assert 1 == len(tmp_val)
                    offset_idx += 1
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0]
                    else:
                        tmp_vel += [(self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/dura]
                    tmp_angle += state[each_joint].tolist()
                elif DOF_DEF[each_joint] == 3:
                    assert 4 == len(tmp_val)
                    offset_idx += 4
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0, 0.0, 0.0]
                    else:
                        tmp_vel += self.calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], dura)
                    tmp_angle += quat2euler(state[each_joint])
            self.data_vel.append(tmp_vel)
            self.data_angle.append(tmp_angle)

    def play(self, mocap_filepath):
        from mujoco_py import load_model_from_xml, MjSim, MjViewer

        curr_path = getcwd()
        xmlpath = '/mujoco/humanoid_deepmimic/envs/asset/dp_env_v1.xml'
        with open(curr_path + xmlpath) as fin:
            MODEL_XML = fin.read()

        model = load_model_from_xml(MODEL_XML)
        sim = MjSim(model)
        viewer = MjViewer(sim)

        self.read_raw_data(mocap_filepath)
        self.convert_raw_data()

        from time import sleep

        phase_offset = np.array([0.0, 0.0, 0.0])

        while True:
            for k in range(len(self.data)):
                tmp_val = self.data[k, 1:]
                sim_state = sim.get_state()
                sim_state.qpos[:] = tmp_val[:]
                # sim_state.qpos[:3] +=  phase_offset[:]
                sim.set_state(sim_state)
                sim.forward()
                viewer.render()

            sim_state = sim.get_state()
            phase_offset = sim_state.qpos[:3]
            phase_offset[2] = 0

if __name__ == "__main__":
    test = MocapDM()
    curr_path = getcwd()
    test.play(curr_path + "/mujoco/motions/humanoid3d_walk.txt")