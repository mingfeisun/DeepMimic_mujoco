import numpy as np
from dp_env_v3 import DPEnv

if __name__ == "__main__":
    env = DPEnv()
    env.reset_model()

    action_size = env.action_space.shape[0]
    ac = np.ones(action_size)
    
    np.set_printoptions(precision=3)

    while True:
        target_config = env.mocap.data_config[env.idx_curr][7:] # to exclude root joint
        target_config_vel = env.mocap.data_vel[env.idx_curr][6:]
        curr_config = env.sim.data.qpos[7:]
        curr_config_vel = env.sim.data.qvel[6:]
        # print("Configs errors: ", np.sum(np.abs(target_config-curr_config)))
        # ac = 0.2 * np.array(target_config - curr_config)
        ac = 0.8 * np.array(target_config - curr_config) # + 0.02 * np.array(target_config_vel - curr_config_vel)
        # ac[-2] += 0.3
        # ac[-9] += 0.3

        # if ac[-4] > 0:
        #     ac[-4]  = 1
        #     ac[-11] = 1
        # print('Left knee: ', ac[-4])
        # print('Right knee: ', ac[-11])

        # print(ac[6:9])

        # if ac[8] > 1.6:
        #     import pdb
        #     pdb.set_trace()

        # env.sim.data.qpos[7:] = target_config[:]
        # env.sim.forward()
        # print(env.calc_config_reward())
        _, rew, done, info = env.step(ac)
        if done:
            env.reset_model()
        # print("Rewards: ", rew)
        env.render()