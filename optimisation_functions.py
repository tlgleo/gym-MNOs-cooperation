import math
#from Environment import Environment
import utility_functions as uf
from pyswarm import pso

def create_optimisation_function(env, utility_function):
    #env = Environment()
    providers = env.providers
    size = 1
    for prov in env.providers:
        pass



def varX_2_transactions_constraints(x, env, min_alloc=0.1):

    #print("value of x", x)

    providers = env.providers
    n_agents = env.n_agents
    n_agents_for_opt = n_agents - 1 # in variable for optimisation : n_agents or (n_agents-1) (because x_N=1-x_1-...-x_(N-1) )
    env_size = env.env_size

    transactions = [] # n_prov x n_sites x n_sec x n_prov
    constraints = []
    ind_cur = 0
    for i_prov, prov in enumerate(providers):
        list_prov = []
        n_sites_prov, n_sec_prov = env_size[i_prov]
        for i_site in range(n_sites_prov):

            for i_sec in range(n_sec_prov):
                ind_beg = ind_cur + i_site*n_sec_prov*n_agents_for_opt + i_sec*n_agents_for_opt
                ind_end = ind_beg + n_agents_for_opt
                #print(ind_beg, ind_end)
                parts = x[ind_beg:ind_end]
                parts = list(parts)
                parts += [1 - sum(x[ind_beg:ind_end])] # adding the last part (sum = 1)
                #print("parts =", parts)
                constraint_sum = 1 - sum(x[ind_beg:ind_end]) # sum needs to be less than 1
                if i_prov == n_agents-1: #if last agent (since we completed the parts with 1-sum()
                    constraint_min = 1 - sum(x[ind_beg:ind_end]) - min_alloc
                else:
                    constraint_min = x[ind_beg + i_prov] - min_alloc

                transa = [i_prov, i_site, i_sec, parts]
                transactions.append(transa)
                constraints.append(constraint_sum)
                constraints.append(constraint_min)

            ind_cur = ind_end

    #print((transactions, constraints))

    return (transactions, constraints)


def global_optimization(env, min_alloc=0.1):

    env_size = env.env_size
    size_x = sum([s[0]*s[1] for s in env_size])*(len(env_size)-1)
    #print(size_x)



    def cost_f(x):
        (transactions, constraints) = varX_2_transactions_constraints(x, env, min_alloc)
        #print('in cost_f : x = ',x)


        constraint_respected = True
        for const in constraints:
            constraint_respected = constraint_respected and const>=0
        if constraint_respected:

            #print('respected')
            env.transactions(transactions)
            env.update_transactions()
            utilities = uf.basic_utility(env)
            return -1.0*sum(utilities)
        else:
            return 0

    def cons(x):
        # function of constraint (sum of parts = 1)
        (transactions, constraints) = varX_2_transactions_constraints(x, env, min_alloc)
        return constraints

    lb = [0.0] * size_x
    ub = [1.0] * size_x

    xopt, fopt = pso(cost_f, lb, ub, f_ieqcons=cons, maxiter=7000)
    #xopt, fopt = (0,0)

    return xopt, fopt