import math

def basic_utility(env, cost_alloc=1):
    # U(a) = sum of distances sites-users of a linked to sites of a + cost_alloc

    n_agents = env.n_agents
    n_users = env.n_users

    agents_utilities = [0] * n_agents

    initial_affiliations = env.initial_affiliations  # list of ident of provider (n_users x 1), remains same
    affiliations_users = env.affiliations_users  # list of ident of provider (n_users x 1)
    affiliations_users_sites = env.affiliations_users_sites   # list of ident of sites
    distances_users_sites = env.distances_users_sites

    for i_user in range(n_users):
        init_prov = initial_affiliations[i_user]
        prov = affiliations_users[i_user]
        site = affiliations_users_sites[i_user]
        dist = distances_users_sites[i_user][prov][site]

        agents_utilities[init_prov] += dist # if init_prov takes dist for him
        #agents_utilities[prov] += dist # if prov takes dist for him

    return agents_utilities




def basic_utility_V2(env, cost_alloc=1):
    # U(a) = sum of distances sites-users of a linked to sites of a + cost_alloc

    n_agents = env.n_agents
    n_users = env.n_users

    agents_utilities = [0] * n_agents

    initial_affiliations = env.initial_affiliations  # list of ident of provider (n_users x 1), remains same
    affiliations_users = env.affiliations_users  # list of ident of provider (n_users x 1)
    affiliations_users_sites = env.affiliations_users_sites   # list of ident of sites
    distances_users_sites = env.distances_users_sites

    for i_user in range(n_users):
        init_prov = initial_affiliations[i_user]
        prov = affiliations_users[i_user]
        site = affiliations_users_sites[i_user]
        dist = distances_users_sites[i_user][prov][site]

        #agents_utilities[init_prov] += dist # if init_prov takes dist for him
        agents_utilities[prov] += dist # if prov takes dist for him

    return agents_utilities








def basic_exp_utility(env, cost_alloc=0.0, dist_max = 1):
    # U(a) = sum of distances sites-users of a linked to sites of a + cost_alloc

    n_agents = env.n_agents
    n_users = env.n_users

    agents_utilities = [0] * n_agents

    for i_user in range(n_users):
        init_prov = env.initial_affiliations[i_user]
        prov = env.affiliations_users[i_user]
        site = env.affiliations_users_sites[i_user]
        sec = env.affiliations_users_sections[i_user]
        dist = env.distances_users_sites[i_user][prov][site]

        # number of clients of init_prov in section i_sec for i_prov/i_site
        #print()
        #print("USER ", str(i_user))
        #print(env.affiliations_numbers_per_sites[init_prov])
        #print()
        n_SS_u = env.affiliations_numbers_per_sites[init_prov][prov][site][sec]

        max_allocated = env.providers[prov].affiliations[site][sec][init_prov]

        #print(str(i_user) + " is in prov/site/sec " + str((prov,site,sec)) + " with " + str(n_SS_u) + " others and dist = " + str(dist) )

        #agents_utilities[init_prov] += max_allocated*math.exp(-1.0*dist/dist_max)/(n_SS_u) # if init_prov takes dist for him
        if n_SS_u == 0:
            agents_utilities[init_prov] += -100 # penality for no affiliation !
        else:
            agents_utilities[init_prov] += max_allocated * math.exp(-1.0 * dist / dist_max) / (n_SS_u)  # if init_prov takes dist for him

    for (i_prov_from, prov_from) in enumerate(env.providers):
        for site in range(prov_from.n_sites):
            for sec in range(prov_from.n_sections):
                sec_is_shared = False
                for i_prov_to in range(env.n_agents):
                    if i_prov_from != i_prov_to:
                        sec_is_shared = sec_is_shared or (prov_from.affiliations_parts[site][sec][i_prov_to] >= env.min_shared)
                if sec_is_shared:
                    #print("section " + str((i_prov_from,site,sec)) + " is shared")
                    agents_utilities[i_prov_from] -= cost_alloc

        #agents_utilities[prov] += dist # if prov takes dist for him

    return agents_utilities
    #return 0.0
