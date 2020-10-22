import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

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



def create_observation(list_users, list_sites, clients, limits, reso_p):
    # convert to an "image" of n_agent x 2 layers (users positions and sites positions)
    # resolution of reso_p (height) and format according to limits_x and limits_y

    (x_min, x_max) = limits[0]
    (y_min, y_max) = limits[1]

    lenX = x_max - x_min
    lenY = y_max - y_min

    reso_H = reso_p
    reso_W = int(reso_p * lenX / lenY)

    n_agents = len(list_sites)
    n_layers = 2 * n_agents

    image = np.zeros([reso_H, reso_W, n_layers])

    #  fill the sites of agents in the first n_agents layers
    for i_agent, list_sites_agent in enumerate(list_sites):
        for (xS, yS) in list_sites_agent:
            iU = int( (xS - x_min) / lenX * reso_W )
            jU = reso_H - int( (yS - y_min) / lenY * reso_H )
            image[jU,iU, i_agent] = 1

    # fill image with clients in the n_agents last layers
    for identU, (xU, yU) in enumerate(list_users):
        iU = int( (xU - x_min) / lenX * reso_W )
        jU = reso_H - int( (yU - y_min) / lenY * reso_H )
        image[jU,iU, n_agents + clients[identU]] = 1

    return image


list_users = [(2,2)]
list_sites = [[(1,1)],[(3,3)],[(5,3)]]
limits = [(0,10),(0,10)]
clients = [0]
reso_p = 10


def separate_images(image):
    image1 = image[:,:,:3]
    image2 = image[:,:,3:]
    return image1, image2


