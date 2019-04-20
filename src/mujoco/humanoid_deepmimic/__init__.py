from gym.envs.registration import register

register(
    id='HumanoidDeepMimic-v0',
    entry_point='humanoid_deepmimic.envs:HumanoidDeepMimicEnv',
)
register(
    id='HumanoidDeepMimic-extrahard-v0',
    entry_point='humanoid_deepmimic.envs:HumanoidDeepMimicExtraHardEnv',
)