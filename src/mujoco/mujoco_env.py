import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from mujoco.mocap import MocapDM
from mujoco.mujoco_interface import MujocoInterface

class DeepMimicEnv(object):
    def __init__(self):
        file_path = '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/dp_env_v1.xml'
        with open(file_path, 'r') as fin:
            MODEL_XML = fin.read()

        self.model = load_model_from_xml(MODEL_XML)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        mocap_filepath = '/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_backflip.txt'

        self.mocap = MocapDM()
        self.mocap.load_mocap(mocap_filepath)

        self.interface = MujocoInterface()
        self.interface.init(self.sim, self.mocap.dt)

        total_length = 1 # phase variable
        total_length += self.mocap.num_bodies * (self.mocap.pos_dim + self.mocap.rot_dim) + 1
        total_length += self.mocap.num_bodies * (self.mocap.pos_dim + self.mocap.rot_dim - 1)

        self.state_size = total_length
        self.action_size = self.interface.action_size

    def update(self, timestep):
        self.sim.step()

    def reset(self):
        self.sim.reset()

    def get_time(self):
        return self.sim.data.time

    def get_name(self):
        return 'test'

    # rendering and UI interface
    def draw(self):
        self.viewer.render()

    def keyboard(self, key, x, y):
        pass

    def mouse_click(self, button, state, x, y):
        pass

    def mouse_move(self, x, y):
        pass

    def reshape(self, w, h):
        pass

    def shutdown(self):
        pass

    def is_done(self):
        return False

    def set_playback_speed(self, speed):
        pass

    def set_updates_per_sec(self, updates_per_sec):
        pass

    def get_win_width(self):
        return 640

    def get_win_height(self):
        return 320

    def get_num_update_substeps(self):
        return 32

    # rl interface
    def is_rl_scene(self):
        return True

    def get_num_agents(self):
        return 1

    def need_new_action(self, agent_id):
        return True

    def record_state(self, agent_id):
        self.data = self.sim.data
        # Cartesian position of body frame
        xpos = self.data.body_xpos
        xquat = self.data.body_xquat
        cvel = self.data.cvel

        valid_xpos = self.interface.align_state(xpos)
        valid_xquat = self.interface.align_state(xquat)
        valid_cvel = self.interface.align_state(cvel)

        root_xpos = valid_xpos[0]

        state = np.zeros(self.state_size)
        state.fill(np.nan) # fill with nan to avoid any missing data
        
        curr_idx = 0
        state[curr_idx] = 0
        curr_idx += 1

        state[curr_idx] = root_xpos[1]
        curr_idx += 1

        for i in range(self.mocap.num_bodies):
            state[curr_idx:curr_idx+3] = valid_xpos[i] - root_xpos
            curr_idx += 3
            state[curr_idx:curr_idx+4] = valid_xquat[i]
            curr_idx += 4

        for i in range(self.mocap.num_bodies):
            state[curr_idx:curr_idx+6] = valid_cvel[i]
            curr_idx += 6
        
        return state

    def record_goal(self, agent_id):
        return np.array([1])

    def get_action_space(self, agent_id):
        return 1
    
    def set_action(self, agent_id, action):
        torque = self.interface.action2torque(action)
        self.sim.data.ctrl[:] = torque[:]
        return
    
    def get_state_size(self, agent_id):
        return self.state_size

    def get_goal_size(self, agent_id):
        return 0

    def get_action_size(self, agent_id):
        return self.action_size

    def get_num_actions(self, agent_id):
        return

    def build_state_offset(self, agent_id):
        return np.zeros(self.get_state_size(agent_id))

    def build_state_scale(self, agent_id):
        return np.ones(self.get_state_size(agent_id))
    
    def build_goal_offset(self, agent_id):
        # return np.zeros(1)
        return np.array([])

    def build_goal_scale(self, agent_id):
        # return np.ones(1)
        return np.array([])
    
    def build_action_offset(self, agent_id):
        return np.zeros(self.get_action_size(agent_id))

    def build_action_scale(self, agent_id):
        return np.ones(self.get_action_size(agent_id))

    def build_action_bound_min(self, agent_id):
        return -10 * np.ones(self.get_action_size(agent_id))

    def build_action_bound_max(self, agent_id):
        return 10 * np.ones(self.get_action_size(agent_id))

    def build_state_norm_groups(self, agent_id):
        tmp = np.zeros(self.get_state_size(agent_id))
        tmp[-1] = 1
        return tmp

    def build_goal_norm_groups(self, agent_id):
        # return np.ones(1)
        return np.array([])

    def calc_reward(self, agent_id):
        # TODO
        return np.random.rand() - 0.5

    def is_episode_end(self):
        return False

    def check_terminate(self, agent_id):
        return 2

    def check_valid_episode(self):
        return True

    def log_val(self, agent_id, val):
        pass

    def set_sample_count(self, count):
        pass

    def set_mode(self, mode):
        pass