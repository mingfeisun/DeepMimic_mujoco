import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env
from env.humanoid3d_env import Humanoid3DEnv

class DeepMimicEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.counter = 0
        self.mujoco_env = Humanoid3DEnv()

    def record_state(self, id):
        return self.mujoco_env._get_obs()

    def calc_reward(self, agent_id):
        return np.random.rand() - 0.5
        '''
        pose_w = 0.5
        vel_w = 0.05
        end_eff_w = 0.15
        root_w = 0.2
        com_w = 0.1

        total_w = pose_w + vel_w + end_eff_w + root_w + com_w
        pose_w /= total_w
        vel_w /= total_w
        end_eff_w /= total_w
        root_w /= total_w
        com_w /= total_w

        pose_scale = 2
        vel_scale = 0.1
        end_eff_scale = 40
        root_scale = 5
        com_scale = 10
        err_scale = 1

        const auto& joint_mat = sim_char.GetJointMat();
        const auto& body_defs = sim_char.GetBodyDefs();
        double reward = 0;

        const Eigen::VectorXd& pose0 = sim_char.GetPose();
        const Eigen::VectorXd& vel0 = sim_char.GetVel();
        const Eigen::VectorXd& pose1 = kin_char.GetPose();
        const Eigen::VectorXd& vel1 = kin_char.GetVel();
        tMatrix origin_trans = sim_char.BuildOriginTrans();
        tMatrix kin_origin_trans = kin_char.BuildOriginTrans();

        tVector com0_world = sim_char.CalcCOM();
        tVector com_vel0_world = sim_char.CalcCOMVel();
        tVector com1_world;
        tVector com_vel1_world;
        cRBDUtil::CalcCoM(joint_mat, body_defs, pose1, vel1, com1_world, com_vel1_world);

        int root_id = sim_char.GetRootID();
        tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0);
        tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
        tQuaternion root_rot0 = cKinTree::GetRootRot(joint_mat, pose0);
        tQuaternion root_rot1 = cKinTree::GetRootRot(joint_mat, pose1);
        tVector root_vel0 = cKinTree::GetRootVel(joint_mat, vel0);
        tVector root_vel1 = cKinTree::GetRootVel(joint_mat, vel1);
        tVector root_ang_vel0 = cKinTree::GetRootAngVel(joint_mat, vel0);
        tVector root_ang_vel1 = cKinTree::GetRootAngVel(joint_mat, vel1);

        pose_err = 0
        vel_err = 0
        end_eff_err = 0
        root_err = 0
        com_err = 0
        heading_err = 0

        num_end_effs = 0
        num_joints = sim_char.GetNumJoints();
        assert(num_joints == mJointWeights.size());

        double root_rot_w = mJointWeights[root_id];
        pose_err += root_rot_w * cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);
        vel_err += root_rot_w * cKinTree::CalcRootAngVelErr(joint_mat, vel0, vel1);

        for (int j = root_id + 1; j < num_joints; ++j)
        {
            double w = mJointWeights[j];
            double curr_pose_err = cKinTree::CalcPoseErr(joint_mat, j, pose0, pose1);
            double curr_vel_err = cKinTree::CalcVelErr(joint_mat, j, vel0, vel1);
            pose_err += w * curr_pose_err;
            vel_err += w * curr_vel_err;

            bool is_end_eff = sim_char.IsEndEffector(j);
            if (is_end_eff)
            {
                tVector pos0 = sim_char.CalcJointPos(j);
                tVector pos1 = cKinTree::CalcJointWorldPos(joint_mat, pose1, j);
                double ground_h0 = mGround->SampleHeight(pos0);
                double ground_h1 = kin_char.GetOriginPos()[1];

                tVector pos_rel0 = pos0 - root_pos0;
                tVector pos_rel1 = pos1 - root_pos1;
                pos_rel0[1] = pos0[1] - ground_h0;
                pos_rel1[1] = pos1[1] - ground_h1;

                pos_rel0 = origin_trans * pos_rel0;
                pos_rel1 = kin_origin_trans * pos_rel1;

                double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
                end_eff_err += curr_end_err;
                ++num_end_effs;
            }
        }

        if (num_end_effs > 0)
        {
            end_eff_err /= num_end_effs;
        }

        double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos());
        double root_ground_h1 = kin_char.GetOriginPos()[1];
        root_pos0[1] -= root_ground_h0;
        root_pos1[1] -= root_ground_h1;
        double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
        
        double root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1);
        root_rot_err *= root_rot_err;

        double root_vel_err = (root_vel1 - root_vel0).squaredNorm();
        double root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm();

        root_err = root_pos_err
                + 0.1 * root_rot_err
                + 0.01 * root_vel_err
                + 0.001 * root_ang_vel_err;
        com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm();

        double pose_reward = exp(-err_scale * pose_scale * pose_err);
        double vel_reward = exp(-err_scale * vel_scale * vel_err);
        double end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);
        double root_reward = exp(-err_scale * root_scale * root_err);
        double com_reward = exp(-err_scale * com_scale * com_err);

        reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward
            + root_w * root_reward + com_w * com_reward;

        return reward;
        '''

    def update(self, timestep):
        self.counter += 1

    def reset(self):
        self.mujoco_env.reset_model()

    def get_time(self):
        return self.mujoco_env.get_time()

    def get_name(self):
        return 'test_mujoco'

    def draw(self):
        self.mujoco_env.render()

    def shutdown(self):
        self.mujoco_env.close()

    def is_done(self):
        return False

    def get_num_update_substeps(self):
        return 10

    # rl interface
    def is_rl_scene(self):
        return True

    def get_num_agents(self):
        # 0 for mocap mode
        # return 0
        # 1 for training mode
        return 1

    def need_new_action(self, agent_id):
        return True

    def record_goal(self, agent_id):
        return np.array([1])

    def get_action_space(self, agent_id):
        return 1
    
    def set_action(self, agent_id, action):
        self.ac = action
        self.mujoco_env.step(self.ac)
    
    def get_state_size(self, agent_id):
        return self.mujoco_env.get_state_size()

    def get_goal_size(self, agent_id):
        return self.mujoco_env.get_goal_size()

    def get_action_size(self, agent_id):
        return self.mujoco_env.get_action_size()

    def get_num_actions(self, agent_id):
        return 0

    def build_state_offset(self, agent_id):
        return np.zeros(self.get_state_size(agent_id))

    def build_state_scale(self, agent_id):
        return np.ones(self.get_state_size(agent_id))
    
    def build_goal_offset(self, agent_id):
        return np.zeros(1)

    def build_goal_scale(self, agent_id):
        return np.ones(1)
    
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
        return np.ones(1)

    def is_episode_end(self):
        if self.counter >= 50000:
            self.counter = 0
            return True
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

    def load_mocap(self, file_name):
        import json
        import copy
        with open(file_name) as fin:
            data_json = json.load(fin)
            self.frames_raw = data_json["Frames"]
            self.frames = copy.deepcopy(self.frames_raw)
            curr_time = 0
            for idx in range(len(self.frames)):
                duration = self.frames[idx][0]
                self.frames[idx][0] = curr_time
                curr_time += duration

if __name__ == "__main__":
    env = DeepMimicEnv()
    env.record_state(0)
    while True:
        fps = 60
        update_timestep = 5.0 / fps
        env.update(update_timestep)
        env.draw()
