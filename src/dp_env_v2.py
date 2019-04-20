#!/usr/bin/env python3
import numpy as np
import math
import random
from os import getcwd

from mujoco.mocap_v2 import MocapDM
from mujoco.mujoco_interface import MujocoInterface
from mujoco.mocap_util import JOINT_WEIGHT
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from gym.envs.mujoco import mujoco_env
from gym import utils

from config import Config
from pyquaternion import Quaternion

from transformations import quaternion_from_euler

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_file_path = Config.xml_path

        self.mocap = MocapDM()
        self.interface = MujocoInterface()
        self.load_mocap(Config.mocap_path)

        self.weight_pose = 0.5
        self.weight_vel = 0.05
        self.weight_root = 0.2
        self.weight_end_eff = 0.15
        self.weight_com = 0.1

        self.scale_pose = 2.0
        self.scale_vel = 0.1
        self.scale_end_eff = 40.0
        self.scale_root = 5.0
        self.scale_com = 10.0
        self.scale_err = 1.0

        self.idx_mocap = 0
        self.reference_state_init()
        self.idx_curr = -1

        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 6)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        position = position[2:] # ignore x and y
        return np.concatenate((position, velocity))

    def reference_state_init(self):
        self.idx_init = random.randint(0, self.mocap_data_len-1)
        self.idx_curr = 0

    def early_termination(self):
        pass

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos[3:] # to exclude root joint

    def get_joint_velocities(self):
        data = self.sim.data
        return data.qvel[3:] # to exclude root joint

    def get_root_pos(self):
        data = self.sim.data
        return data.qpos[2:7] # translation & rotation

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)

    def calc_config_errs(self, env_config, mocap_config):
        assert len(env_config) == len(mocap_config)
        return np.sum(np.abs(env_config - mocap_config))

    def calc_vel_errs(self, now_vel, next_vel):
        assert len(now_vel) == len(next_vel)
        err = sum(abs(np.array(now_vel) - np.array(next_vel)))
        return err

    def calc_root_errs(self, curr_root, target_root): # including root joint
        assert len(curr_root) == len(target_root)
        assert len(curr_root) == 5

        z_curr = curr_root[0]
        z_target = target_root[0]

        q_0 = Quaternion(curr_root[1], curr_root[2], curr_root[3], curr_root[4])
        q_1 = Quaternion(target_root[1], target_root[2], target_root[3], target_root[4])

        q_diff =  q_0.conjugate * q_1
        tmp_diff = q_diff.angle

        return abs(z_curr - z_target) + abs(tmp_diff)

    def calc_reward(self):
        assert len(self.mocap.data) != 0
        # self.update_inteval = int(self.mocap_dt // self.dt)
        self.update_inteval = 1

        err_configs = 0.0
        err_vel = 0.0
        err_root = 0.0
        # err_end_eff = 0.0
        # err_com = 0.0

        if self.idx_curr % self.update_inteval != 0:
            return 0.0

        self.idx_mocap = int(self.idx_curr // self.update_inteval) + self.idx_init
        self.idx_mocap = self.idx_mocap % self.mocap_data_len

        target_config = self.mocap.data_config[self.idx_mocap][3:] # to exclude root joint
        self.curr_frame = target_config
        curr_config = self.get_joint_configs()

        err_configs = self.calc_config_errs(curr_config, target_config)
        # print('Error configs: ', err_configs)

        # curr_mocap_vel = self.mocap.data_vel[self.idx_mocap][3:] # to exclude root joint
        # curr_vel = self.get_joint_velocities()

        # err_vel = self.calc_vel_errs(curr_mocap_vel, curr_vel)

        # target_root = self.mocap.data[self.idx_mocap, 1+2: 1+7] # translation & rotation
        # curr_root = self.get_root_pos()

        # err_root = self.calc_root_errs(curr_root, target_root)
        # print('Error root: ', err_root)

        ## TODO
        # err_end_eff =  0.0
        # reward_end_eff  = math.exp(-self.scale_err * self.scale_end_eff * err_end_eff)

        ## TODO
        # err_com = 0.0
        # reward_com      = math.exp(-self.scale_err * self.scale_com * err_com)

        reward_pose = math.exp(-self.scale_err * self.scale_pose * err_configs)
        # reward_vel  = math.exp(-self.scale_err * self.scale_vel * err_vel)
        # reward_root = math.exp(-self.scale_err * self.scale_root * err_root)

        # reward = self.weight_pose * reward_pose + self.weight_vel * reward_vel + \
        #      self.weight_end_eff * reward_end_eff + self.weight_root * reward_root + \
        #          self.weight_com * reward_com

        # reward = self.weight_pose * reward_pose + self.weight_vel * reward_vel + \
        #     self.weight_root * reward_root

        reward = reward_pose

        return reward

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.idx_curr += 1
        data = self.sim.data

        observation = self._get_obs()
        reward_obs = self.calc_reward()
        reward_acs = np.square(data.ctrl).sum()

        reward = reward_obs - 0.1 * reward_acs

        info = dict(reward_obs=reward_obs, reward_acs=reward_acs)
        done = self.is_done()

        return observation, reward, done, info

    def is_done(self):
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 0.7) or (qpos[2] > 2.0))
        return done

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset_model(self):
        self.reference_state_init()
        qpos = self.mocap.data_config[self.idx_init]
        qvel = self.mocap.data_vel[self.idx_init]
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

if __name__ == "__main__":
    env = DPEnv()
    env.reset_model()

    # env.load_mocap("/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt")
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    print(action_size)
    curr_idx = env.idx_init
    while True:
        curr_idx = curr_idx % env.mocap_data_len
        target_config = env.mocap.data_config[curr_idx][:] # to exclude root joint
        env.sim.data.qpos[:] = target_config[:]
        env.sim.forward()
        print(env.calc_reward())
        # env.calc_reward()
        env.render()
        curr_idx +=1
        env.idx_curr += 1