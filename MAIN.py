from examples_env import env_3A_3S_9U


'''

transactions_complete_list = []
for i, prov in enumerate(environment.providers):
    for si in range(prov.n_sites):
        for se in range(prov.n_sections):
            transactions_complete_list.append([i,si,se, [0.33,0.33,0.33]])

environment.transactions(transactions_complete_list)



environment.transactions(0,0,0,[0,1,0])
environment.transactions(1,0,1,[0,0,1])
environment.transactions(2,0,2,[0.5,0.5,0])


'''

for t in range(-1):
    environment.move_users(update_matrix_distance=True)
    environment.update_transactions()
    environment.display('./ESSAI_move/'+str(t)+'.png', fig_size=(6,6),label=True, links = True, close = True)


env = env_3A_3S_9U()

env.display('truc.png', fig_size=(6,6),label=True, links = True, close = False)