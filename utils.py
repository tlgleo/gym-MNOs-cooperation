import numpy as np

def partition(N, parts):
    # compute a partition of a integer N with parts
    # addresses the rounding values such that the sum is equals to N
    remainders = []
    values = []
    total = 0
    for p in parts:
        q = int(N * p)
        values.append(q)
        remainders.append(N * p - q)
        total += q

    i = len(parts)-1
    indices = np.argsort(remainders)
    while total < N and i >= 0: # distributes the surplus according to the remainders
        ind = indices[i]
        values[ind] += 1
        i -= 1
        total+=1

    return values


def random_users(n_users, sites, r, x_lim, y_lim):
    # generate positions of users randomly not too close to sites (dist min : r)
    final_list = []
    for i in range(n_users):
        a = False
        while not a:
            x = np.random.rand()*(x_lim[1]-x_lim[0]) + x_lim[0]
            y = np.random.rand() * (y_lim[1] - y_lim[0]) + y_lim[0]
            a = True
            for pos_site in sites:
                distance = euclidean((x,y), pos_site)
                a = a and distance > r
        final_list.append((x,y))

    return final_list

