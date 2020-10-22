from gym.envs.registration import register

register(id='env_3A_3S_9U-v0',
         entry_point = 'multi_agents_mno.envs:Env_3A_3S_9U_0')

register(id='env_3A_3S_9U-v1',
         entry_point = 'multi_agents_mno.envs:Env_3A_3S_9U_1')

register(id='env_3A_3S_9U-v2',
         entry_point = 'multi_agents_mno.envs:Env_3A_3S_9U_2')


register(id='env_2A_2S_2U-v0',
         entry_point = 'multi_agents_mno.envs:Env_2A_2S_2U')


register(id='env_3A_5S_30U-v0',
         entry_point = 'multi_agents_mno.envs:Env_3A_5S_15U')


register(id='env_3A_1S_9U-v0',
         entry_point = 'multi_agents_mno.envs:Env_3A_1S_9U')


register(id='env_3A_3S_18U-v0',
         entry_point = 'multi_agents_mno.envs:Env_3A_3S_18U')

