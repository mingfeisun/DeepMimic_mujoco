#!/usr/bin/env python3
import numpy as np
import math
import random
from os import getcwd

from mujoco.mocap_v1 import MocapDM
from mujoco.mujoco_interface import MujocoInterface
from mujoco.mocap_util import JOINT_WEIGHT
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from gym.envs.mujoco import mujoco_env
from gym import utils

from config import Config

# TODO: load mocap data; calc rewards
# TODO: early stop

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
        self.weight_end_eff = 0.15
        self.weight_root = 0.2
        self.weight_com = 0.1

        self.scale_pose = 2.0
        self.scale_vel = 0.1
        self.scale_end_eff = 40.0
        self.scale_root = 5.0
        self.scale_com = 10.0
        self.scale_err = 1.0

        self.idx_mocap = 0
        self.reference_state_init()

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
        return data.qpos[:3]

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)

    def calc_reward(self):
        assert len(self.mocap.data) != 0
        self.update_inteval = int(self.mocap_dt // self.dt)

        err_pose = 0.0
        err_vel = 0.0
        err_end_eff = 0.0
        err_root = 0.0
        err_com = 0.0

        if self.idx_curr % self.update_inteval != 0:
            return 0.0

        self.idx_mocap = int(self.idx_curr // self.update_inteval) + self.idx_init
        self.idx_mocap = self.idx_mocap % self.mocap_data_len

        target_config = self.mocap.data[self.idx_mocap, 1+3:] # to exclude root joint
        self.curr_frame = target_config
        curr_configs = self.get_joint_configs()

        err_pose = self.interface.calc_config_errs(curr_configs, target_config)

        if self.idx_mocap == self.mocap_data_len - 1: # init to last mocap frame
            pos_prev = self.mocap.data[self.idx_mocap-1, 1+3:]
            pos_curr = self.mocap.data[self.idx_mocap, 1+3:]
        else:
            pos_prev = self.mocap.data[self.idx_mocap, 1+3:]
            pos_curr = self.mocap.data[self.idx_mocap+1, 1+3:]

        vel_pos_err = self.interface.calc_config_err_vec(pos_prev, pos_curr)
        curr_mocap_vel = vel_pos_err * 1.0 / self.mocap_dt
        curr_vel = self.get_joint_velocities()

        err_vel = self.interface.calc_vel_errs(curr_mocap_vel, curr_vel)

        target_root = self.mocap.data[self.idx_mocap, 1: 1+3]
        curr_root = self.get_root_pos()

        err_root = self.interface.calc_root_errs(curr_root, target_root)

        # TODO
        err_end_eff =  0.0
        reward_end_eff  = math.exp(-self.scale_err * self.scale_end_eff * err_end_eff)

        # TODO
        err_com = 0.0
        reward_com      = math.exp(-self.scale_err * self.scale_com * err_com)

        reward_pose     = math.exp(-self.scale_err * self.scale_pose * err_pose)
        reward_vel      = math.exp(-self.scale_err * self.scale_vel * err_vel)
        reward_root     = math.exp(-self.scale_err * self.scale_root * err_root)

        # reward = self.weight_pose * reward_pose + self.weight_vel * reward_vel + \
        #      self.weight_end_eff * reward_end_eff + self.weight_root * reward_root + \
        #          self.weight_com * reward_com

        reward = self.weight_pose * reward_pose + self.weight_vel * reward_vel + \
            self.weight_root * reward_root

        return reward

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.idx_curr += 1
        data = self.sim.data

        observation = self._get_obs()
        reward_obs = self.calc_reward()
        reward_acs = np.square(data.ctrl).sum()

        reward = reward_obs - 0.1 * reward_acs

        info = dict()
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
        qpos = self.mocap.data[self.idx_init, 1:]
        if self.idx_init == self.mocap_data_len - 1: # init to last mocap frame
            root_pos_err = np.array(self.mocap.data[self.idx_init, 1:4]) - np.array(self.mocap.data[self.idx_init-1, 1:4])
            qpos_err = self.interface.calc_config_err_vec_with_root(self.mocap.data[self.idx_init-1, 4:], self.mocap.data[self.idx_init, 4:])
        else:
            root_pos_err = np.array(self.mocap.data[self.idx_init+1, 1:4]) - np.array(self.mocap.data[self.idx_init, 1:4])
            qpos_err = self.interface.calc_config_err_vec_with_root(self.mocap.data[self.idx_init, 4:], self.mocap.data[self.idx_init+1, 4:])
        qvel = np.concatenate((root_pos_err, qpos_err), axis=None) * 1.0 / self.mocap_dt
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

    # env.load_mocap("/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt")
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    print(action_size)
    curr_idx = 0
    while True:
        curr_idx = curr_idx % env.mocap_data_len
        target_config = env.mocap.data[curr_idx, 1+2:] # to exclude root joint
        env.sim.data.qpos[2:] = target_config[:]
        env.sim.forward()
        print(env.sim.data.qvel)
        env.render()
        curr_idx +=1