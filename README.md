How to INSTALL gym environment :

- go in the folder : gym_env_mno
- run in a terminal : pip install -e .


ENVIRONMENTS :
This gym env contains:

A : 3 agents, 3 sites, 9 users 
1. env_3A_3S_9U-v1
2. env_3A_3S_9U-v2


TO CREATE AN ENVIRONMENT :
1. in the file multi_agents_mno/envs/custom_env_template.py :
    a. Choose a name for the class instead of Name_Of_Env_Class
        for example : Env_4A_8S_20U
    b. Modify features of the class : positions of users, sites etc
2. in the file multi_agents_mno/envs/__init__.py :
    a. import the class of your environment : 
        In the template line "from multi_agents_mno.envs.custom_env_template import Name_Of_Env_Class", 
        just modify Name_Of_Env_Class with the name chosen in 1.a
3. in the file multi_agents_mno/__init__.py :
    a. Chose a name for your env, with the form name_of_env-vXY ( XY a number )
    for example env_with_3_agents-v0, draft-v12 etc
    b. in the template line : register(id='name_of_env-vXY', entry_point = 'multi_agents_mno.envs:Name_Of_Env_Class')
    modify name_of_env-vXY by the name chosen in 3.a AND Name_Of_Env_Class by the name chosen in 1.a
    




HOW TO WORK WITH GYM:

RENDER 
    env.render('output.png') displays and save figure of current environment

ACTIONS : 
    actions_n is a list of n_agents actions
    a action from a agent is a numpy of shape : n_sites, n_sections, n_agents 
    Then action_n[i_prov] is the action of agent i_prov and 
    action_n[i_prov][i_site, i_section] is the partition chosen by i_prov for his section i_section of his site i_site
    
    To execute an action, run : env.step(actions_n)
    It outputs obs_n, r_n, done_n, info
    where :
    obs_n , r_n and done_n is a list of the n_agents observations, rewards and dones (terminal condition of episode)