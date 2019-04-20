import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
               "left_shoulder", "left_elbow", "right_hip", "right_knee", 
               "right_ankle", "left_hip", "left_knee", "left_ankle"]

DOF_DEF = {"chest": 3, "neck": 3, "right_shoulder": 3, "right_elbow": 1, 
           "left_shoulder": 3, "left_elbow": 1, "right_hip": 3, "right_knee": 1, 
           "right_ankle": 3, "left_hip": 3, "left_knee": 1, "left_ankle": 3}

class Humanoid3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.num_joint = np.nan
        self.pos_dim = 3
        self.rot_dim = 4

        self.counter = 0
        self.frame_skip = 1

        mujoco_env.MujocoEnv.__init__(self, '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/humanoid_deepmimic.xml', 5)
        utils.EzPickle.__init__(self)

        rand_seed = np.random.randint(np.iinfo(np.int32).max)
        self.seed(rand_seed)

    def _get_joint_index(self):
        all_joint_names = ["worldbody", "root", "joint_waist", "chest", "neck", 
                           "joint_neck", "right_clavicle", "right_shoulder", "joint_right_shoulder", "right_elbow", 
                           "joint_right_elbow", "right_wrist", "left_clavicle", "left_shoulder", "joint_left_shoulder", 
                           "left_elbow", "joint_left_elbow", "left_wrist", "right_hip", "joint_right_hip", 
                           "right_knee", "joint_right_knee", "right_ankle", "joint_right_ankle", "left_hip", 
                           "joint_left_hip", "left_knee", "joint_left_knee", "left_ankle", "joint_left_ankle"]

        valid_joint_names = ["root", "chest", "neck", "right_shoulder", "right_elbow",
                             "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip",
                             "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]

        self.num_joint = len(valid_joint_names)

        idx = [a in valid_joint_names for a in all_joint_names]
        return idx

    def _update_data(self):
        self.data = self.sim.data

    def _get_obs(self):
        self._update_data()
        # Cartesian position of body frame
        xpos = self.data.body_xpos
        xquat = self.data.body_xquat
        cvel = self.data.cvel

        idx = self._get_joint_index()

        valid_xpos = xpos[idx]
        valid_xquat = xquat[idx]
        valid_cvel = cvel[idx]

        root_xpos = valid_xpos[0]

        total_length = 0
        total_length += self.num_joint * (self.pos_dim + self.rot_dim) + 1
        total_length += self.num_joint * (self.pos_dim + self.rot_dim - 1)

        self.state_size = total_length

        obs = np.zeros(total_length)
        obs.fill(np.nan) # fill with nan to avoid any missing data
        
        obs[0] = root_xpos[1]
        curr_idx = 1
        for i in range(self.num_joint):
            obs[curr_idx:curr_idx+3] = valid_xpos[i] - root_xpos
            curr_idx += 3
            obs[curr_idx:curr_idx+4] = valid_xquat[i]
            curr_idx += 4

        for i in range(self.num_joint):
            obs[curr_idx:curr_idx+6] = valid_cvel[i]
            curr_idx += 6
        
        return obs

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        reward = (np.random.rand() - 0.5) * 2
        done = False
        info = dict(reward_linvel=0, 
                    reward_quadctrl=0, 
                    reward_alive=0, 
                    reward_impact=0)
        return self._get_obs(), reward, done, info

    def update(self, timestep):
        self.frame_skip = 5
        act = 4 * (np.random.rand(self.get_action_size()) - 0.5)
        self.step(act)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._ezpickle_args

    def get_time(self):
        return self.data.time
    
    def set_action(self, action):
        self.ac = action
        self.step(self.ac)
    
    def get_state_size(self):
        return self.state_size

    def get_goal_size(self):
        return 1

    def get_action_size(self):
        return 26

    def get_num_actions(self):
        return 0

if __name__ == "__main__":
    env = Humanoid3DEnv()
    width = 480
    height = 480

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

    import json
    from pyquaternion import Quaternion

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
