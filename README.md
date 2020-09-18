How to INSTALL gym environment :

- go in the folder : gym_env_mno
- run in a terminal : pip install -e .



RENDER 
env.render('output.png') displays and save figure of current environement


ACTIONS : 
actions_n is a list of n_agents actions
a action from a agent is a numpy of shape : n_sites, n_sections, n_agents 
Then action_n[i_prov] is the action of agent i_prov and 
action_n[i_prov][i_site, i_section] is the partition chosen by i_prov for his section i_section of his site i_site


To execute an action, run : env.step(actions_n)
It outputs obs_n, r_n, done_n, info
where :
obs_n , r_n and done_n is a list of the n_agents observations, rewards and dones (terminal condition of episode)




ENVIRONMENTS :
This gym env contains:

A : 3 agents, 3 sites, 9 users 
1. env_3A_3S_9U-v1
2. env_3A_3S_9U-v2






