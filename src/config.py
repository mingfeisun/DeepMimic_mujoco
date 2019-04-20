from os import getcwd

class Config(object):
    all_motions = ['backflip', 'cartwheel', 'crawl', 'dance_a', 'dance_b', 'getup_facedown'
                   'getup_faceup', 'jump', 'kick', 'punch', 'roll', 'run', 'spin', 'spinkick',
                   'walk']
    curr_path = getcwd()
    # motion = 'spinkick'
    motion = 'dance_b'
    env_name = "dp_env_v3"

    motion_folder = '/mujoco/motions'
    xml_folder = '/mujoco/humanoid_deepmimic/envs/asset'
    xml_test_folder = '/mujoco_test/'

    mocap_path = "%s%s/humanoid3d_%s.txt"%(curr_path, motion_folder, motion)
    xml_path = "%s%s/%s.xml"%(curr_path, xml_folder, env_name)
    xml_path_test = "%s%s/%s_test.xml"%(curr_path, xml_test_folder, env_name)