import gym
from gym import error, spaces, utils
from gym.utils import seeding

import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
from utils import partition, random_users, create_observation, separate_images

from utility_functions import basic_utility, basic_utility_V2, basic_exp_utility
from kinematics import Kinematics, modifiy_position


class Provider:
    def __init__(self, ident, n_sites, positions_sites, n_agents, max_per_section, n_sections = 3, angles_offset_sites=None):
        self.ident = ident
        self.n_sites = n_sites
        self.n_agents = n_agents
        self.n_sections = n_sections # number of angle section (classic MNO -> 3)
        self.max_per_section = max_per_section  # max of users per section
        self.positions_sites = positions_sites #list of tuples
        self.angles_offset_sites = [0 for _ in range(self.n_sites)] if angles_offset_sites == None else angles_offset_sites
        self.affiliations_parts = [[np.identity(self.n_agents)[self.ident] for _ in range(self.n_sections)] for _ in range(self.n_sites)] #list of list of provider_list (nS x nSec x nProv x 1)
        self.affiliations = [[np.zeros(self.n_agents) for _ in range(self.n_sections)] for _ in range(self.n_sites)]  # list of list of ident_provider (nS x nSec x nProv x 1)
        self.number_terminals_per_section = [[ self.max_per_section[i_S] for _ in range(self.n_sections)] for i_S in range(self.n_sites)]  # list of list of ident_provider (nS x nSec x 1)
        self.utility = 0

    def reset(self):
        self.affiliations_parts = [[np.identity(self.n_agents)[self.ident] for _ in range(self.n_sections)] for _ in
                                   range(self.n_sites)]  # list of list of provider_list (nS x nSec x nProv x 1)
        self.affiliations = [[np.zeros(self.n_agents) for _ in range(self.n_sections)] for _ in
                             range(self.n_sites)]  # list of list of ident_provider (nS x nSec x nProv x 1)
        self.number_terminals_per_section = [[self.max_per_section[i_S] for _ in range(self.n_sections)] for i_S in
                                             range(self.n_sites)]  # list of list of ident_provider (nS x nSec x 1)
        self.utility = 0

VERBOSE = []

class Env_Multi_Agent_MNO(gym.Env):
    metadata = {
    }

    def __init__(self, n_agents, n_sites, n_users,
                 providers, positions_users, clients,
                 kinematics_users = None):

        self.n_agents = n_agents  # number of agents = providers
        self.n_sites = n_sites
        self.n_users = n_users
        self.providers = providers  # list of provider Object
        self.positions_users = positions_users  # list of lists of tuple (n_A x n_Clients_A x 2)
        self.kinematics_users = kinematics_users
        self.initial_affiliations = clients  # list of ident of provider (n_users x 1), remains same
        self.affiliations_users = [-1] * self.n_users  # list of ident of provider (n_users x 1)
        self.affiliations_users_sites = [-1] * self.n_users  # list of ident of sites
        self.affiliations_users_sections = [-1] * self.n_users  # list of ident of section
        self.distances_users_sites = self.compute_distances_user_sites()
        self.indices_order_distance_user_site = self.compute_distances_increasing_user_sites()
        self.is_in_section_table = self.compute_is_in_section_table()
        self.affiliations_numbers_per_sites = [[[[0] * prov.n_sections
                                                 for _ in range(prov.n_sites)]
                                                for prov in self.providers]
                                               for init_prov in self.providers]
        self.min_shared = 0.02
        self.env_size = [[prov.n_sites, prov.n_sections] for prov in self.providers]
        self.limits = [(0, 10), (0, 10)]
        self.colors_env = ["#dc5a3d", '#7eaa55', '#2c70ba', '#000000']
        self.utilities = None

        self.update_transactions()


    def move_users(self, update_matrix_distance = False):

        d = 0.2
        xlim = [self.limits[0][0] + d, self.limits[0][1] - d]
        ylim = [self.limits[1][0] + d, self.limits[1][1] - d]

        for i,user_pos in enumerate(self.positions_users):
            kine = self.kinematics_users[i]
            new_pos, new_kine = modifiy_position(user_pos, xlim, ylim, kine)
            self.positions_users[i] = new_pos
            self.kinematics_users[i] = new_kine

        if update_matrix_distance: # need time/CPU !
            self.distances_users_sites = self.compute_distances_user_sites()
            self.indices_order_distance_user_site = self.compute_distances_increasing_user_sites()
            self.is_in_section_table = self.compute_is_in_section_table()

    def compute_distances_user_sites(self):
        # compute a list of lists of lists for distance between sites and users
        distances = []  # list of list of list : n_users x n_sites x n_sections x 1
        for user_pos in self.positions_users:
            dist_list_user = []
            for iA_prov in range(self.n_agents):
                provider = self.providers[iA_prov]
                positions_sites_iA = provider.positions_sites
                dist_list_user_provIA = []
                for i_S, pos_site in enumerate(positions_sites_iA):
                    distance = euclidean(user_pos, pos_site)
                    dist_list_user_provIA.append(distance)
                dist_list_user.append(dist_list_user_provIA)
            distances.append(dist_list_user)
        return distances

    def compute_distances_increasing_user_sites(self):
        # return the list of couple user/site (site = i_prov, i_site)
        # in distance increasing order

        indices = [] # list of lists : [(i_user, i_prov, i_site)]
        values_distances_user = [] # for ordering
        for i_user, dist_user in enumerate(self.distances_users_sites):
            for iA_prov, dist_prov in enumerate(dist_user):
                for i_site, dist in enumerate(dist_prov):
                    index = (i_user, iA_prov, i_site)
                    i=0
                    n = len(indices)
                    while i<n and values_distances_user[i]<dist:
                        i+=1
                    if i==n+1:
                        i=n
                    indices.insert(i, index)
                    values_distances_user.insert(i, dist)
        return indices

    def is_in_section(self, i_user, iA_prov, i_site, i_sec):
        # verifies if i_user is inside the section i_sec of the site i_site of provider iA_prov

        user_pos = self.positions_users[i_user]
        site_pos = self.providers[iA_prov].positions_sites[i_site]
        n_sec = float(self.providers[iA_prov].n_sections)

        # angles limit (radians)
        offset_angle = self.providers[iA_prov].angles_offset_sites[i_site]
        angle0 = offset_angle*math.pi/180 + (i_sec * 2* math.pi) / n_sec
        v0 = [math.cos(angle0), math.sin(angle0)]
        angle1 = offset_angle*math.pi/180 + ( (i_sec+1) * 2*math.pi) / n_sec
        v1 = [math.cos(angle1), math.sin(angle1)]
        dist = self.distances_users_sites[i_user][iA_prov][i_site]

        x = np.array([user_pos[0] - site_pos[0], user_pos[1] - site_pos[1]])

        x= x/dist

        x_v0 = np.dot(x,v0)
        x_v0 = min(1,max(-1, x_v0))

        x_v1 = np.dot(x,v1)
        x_v1 = min(1,max(-1, x_v1))

        v0_v1 = np.dot(v0,v1)
        v0_v1 = min(1, max(-1, v0_v1))

        # verifies with dot product that x (user) is between v0 and v1 (limits of section)
        is_in_section = math.acos(x_v0) <= math.acos(v0_v1) and math.acos(x_v1) <= math.acos(v0_v1)

        return is_in_section

    def compute_is_in_section_table(self):
        result = []
        for i_user in range(self.n_users):
            list_users = []
            for i_prov, provider in enumerate(self.providers):
                list_prov = []
                for i_site in range(provider.n_sites):
                    list_sites = []
                    for i_sec in range(provider.n_sections):
                        boolean = self.is_in_section(i_user, i_prov, i_site, i_sec)
                        list_sites.append(boolean)
                    list_prov.append(list_sites)
                list_users.append(list_prov)
            result.append(list_users)
        return result

    def update_allocations_terminals(self):
        # update the number of terminals max that a provider accepts for each sites and section
        # just a element-wise product : number_max_per_section x affiliations_parts
        for provider in self.providers:
            for i_site in range(provider.n_sites):
                for i_sec in range(provider.n_sections):
                    max_terminals = np.array(provider.number_terminals_per_section)[i_site][i_sec]
                    parts = np.array(provider.affiliations_parts[i_site][i_sec])
                    provider.affiliations[i_site][i_sec] = partition(max_terminals, parts) # partition function from utils

    def update_affiliations(self):
        # after some transactions, it is necessary to update the affiliations of users (which provider)
        # it takes the closest section which accepts the user

        counter = [[[np.zeros(self.n_agents) for _ in range(prov.n_sections)]
                    for _ in range(prov.n_sites)] for prov in self.providers]

        affiliations_users = [-1] * self.n_users
        affiliations_users_sites = [-1] * self.n_users
        affiliations_users_sections = [-1] * self.n_users
        affiliations_numbers_per_sites = [ [ [  [0]*prov.n_sections
                                                        for _ in range(prov.n_sites)]
                                                        for prov in self.providers]
                                                        for init_prov in self.providers]

        indices_order_distance_user_site = self.indices_order_distance_user_site

        for i,index in enumerate(indices_order_distance_user_site):
            # go through user list in increasing order of distance user-site

            i_user, i_prov, i_site = index
            initial_prov = self.initial_affiliations[i_user]  # client "belongs" to that provider
            provider = self.providers[i_prov]
            accepted  = False  #a = i_site and i_sec accept user i_user

            section_accepted = 0
            i_sec = 0

            while not accepted and affiliations_users[i_user] == -1 and i_sec < provider.n_sections:
                # user is inside the section
                is_in_section = self.is_in_section_table[i_user][i_prov][i_site][i_sec]

                max_for_prov = provider.affiliations[i_site][i_sec][initial_prov]
                is_accepted = counter[i_prov][i_site][i_sec][initial_prov] < max_for_prov

                accepted = is_in_section and is_accepted

                if 1 in VERBOSE:
                    print('i_user', i_user)
                    print('i_prov, i_site', (i_prov, i_site))
                    print('i_sec', i_sec)
                    print('max_for_prov', max_for_prov)
                    print('is_accepted', is_accepted)
                    print('is_in_section', is_in_section)
                    print()


                if accepted:
                    section_accepted = i_sec

                i_sec += 1

            if accepted and affiliations_users[i_user] == -1:
                # one section has been found AND i_user has no affiliation

                i_user, ident_prov, i_site = indices_order_distance_user_site[i]
                counter[ident_prov][i_site][section_accepted][initial_prov] += 1
                affiliations_users[i_user] = ident_prov
                affiliations_users_sites[i_user] = i_site
                affiliations_users_sections[i_user] = section_accepted
                affiliations_numbers_per_sites[initial_prov][ident_prov][i_site][section_accepted] += 1

        self.affiliations_users = affiliations_users
        self.affiliations_users_sites = affiliations_users_sites
        self.affiliations_users_sections = affiliations_users_sections
        self.affiliations_numbers_per_sites = affiliations_numbers_per_sites

    def transactions(self, liste_transactions):
        # transaction = list = [ident_provider, i_site, i_sec, list_parts]
        # list_parts = list of size n_agents whose sum = 1 : example = [0.6, 0.2, 0.2]
        # one provider i_prov_offer allocates for other providers a part for site/section i_site/i_sec

        for trans in liste_transactions:
            i_prov_offer = trans[0]
            i_site = trans[1]
            i_sec = trans[2]
            parts = trans[3]
            self.providers[i_prov_offer].affiliations_parts[i_site][i_sec] = parts


    def update_transactions(self):
        self.update_allocations_terminals()
        self.update_affiliations()


    def display(self, output_name = "output.svg", fig_size=(10,10),
                label=False,
                links = False,
                utilities = None,
                close = False,
                axis = False):


        x_lim = self.limits[0]
        y_lim = self.limits[1]

        # display initial environment (without affiliations)
        scale = fig_size[0]/(x_lim[1])

        colors = ['#dc5a3e', '#7faa55', '#2d70bb', 'k']
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect('equal', adjustable='box')

        plt.xlim(x_lim)
        plt.ylim(y_lim)

        # Lines for sections of providers
        for iA_prov in range(self.n_agents):
            provider = self.providers[iA_prov]
            n_sec = provider.n_sections
            positions_sites_iA = provider.positions_sites
            angles_offset_sites = provider.angles_offset_sites

            for iSite, position_site in enumerate(positions_sites_iA):
                offset_angle = angles_offset_sites[iSite]
                rmax = x_lim[1]*2 #point outside the limits for a half line of section

                for k in range(n_sec):
                    angle = offset_angle*math.pi/180 + k*2*math.pi/n_sec
                    x2 = position_site[0] + rmax * math.cos(angle)
                    y2 = position_site[1] + rmax * math.sin(angle)
                    ax.plot([position_site[0],x2], [position_site[1],y2], color = colors[iA_prov], linestyle = "--", linewidth = 0.3)


        # Sites of Providers, color = provider
        for iA_prov in range(self.n_agents):
            provider = self.providers[iA_prov]
            positions_sites_iA = provider.positions_sites
            d = 0.07*scale
            for i_site, position_site in enumerate(positions_sites_iA):
                ax.plot(position_site[0], position_site[1], color = colors[iA_prov], ms=15 * scale, marker='*')
                if label:
                    pass
                    #ax.text(position_site[0], position_site[1]+d, str(iA_prov)+' '+str(i_site),
                            #fontsize = 9*scale, fontstretch = 'ultra-condensed' ,horizontalalignment = 'center')

                if utilities is not None:
                   ax.text(position_site[0]+7* d, position_site[1] -5* d, str(utilities[iA_prov]),
                                fontsize=15 * scale, fontstretch='ultra-condensed', horizontalalignment='center')


        # Affiliations sections, annulus color = provider
        for provider in self.providers:
            list_sites_iA = provider.positions_sites

            for i_site, pos_site in enumerate(list_sites_iA):

                offset_angle = provider.angles_offset_sites[i_site]
                n_sec = provider.n_sections

                for i_sec in range(n_sec):

                    angle0 = offset_angle + i_sec * 360 / n_sec + 3*scale
                    angle1 = offset_angle + (i_sec + 1) * 360 / n_sec - 3*scale

                    r = 0.3/scale
                    for i_prov in range(self.n_agents):

                        wdt = 0.30*scale* provider.affiliations_parts[i_site][i_sec][i_prov] * scale

                        color_prov = colors[i_prov]
                        annulus = matplotlib.patches.Wedge(pos_site, r+wdt, angle0, angle1, width=wdt,
                                                           color=color_prov, alpha =1)

                        if wdt > 0.01:
                            ax.add_patch(annulus)
                            r += wdt

        # Affiliations, rectangle color = new provider
        for iU, user_pos in enumerate(self.positions_users):
            affi_provider = self.affiliations_users[iU]
            provider = self.providers[affi_provider]

            d = 0.1
            (x0, y0) = (user_pos[0] - d, user_pos[1] - d)

            rect = matplotlib.patches.Rectangle((x0, y0),
                                                2 * d, 2 * d,
                                                fill=False,
                                                linewidth= 2*scale,
                                                color=colors[affi_provider])

            ax.add_patch(rect)

            if links: # display links between sites and users
                if affi_provider != -1:
                    i_site = self.affiliations_users_sites[iU]
                    position_site = provider.positions_sites[i_site]

                    ax.plot([position_site[0], user_pos[0]], [position_site[1], user_pos[1]],
                            color=colors[affi_provider], linestyle="--",
                            linewidth=1.1*scale)


        # Clients, color = initial provider
        for iU, user_pos in enumerate(self.positions_users):
            affi_provider = self.initial_affiliations[iU]
            d = 0.09*scale
            ax.plot(user_pos[0], user_pos[1], color = colors[affi_provider], marker = '.', ms=10 * scale, alpha = 1)
            if label:
                ax.text(user_pos[0]+d, user_pos[1]+d, str(iU))


        if not axis:
            plt.axis('off')



        plt.savefig(output_name)
        if not close:
            plt.show(ax)



    def reset(self):
        for prov in self.providers:
            prov.reset()

        self.update_transactions()


    def get_total_obs(self):
        # output an n_agents x 2 layers image (for sites and users positions)
        list_users = self.positions_users
        list_sites = [prov.positions_sites for prov in self.providers]
        clients = self.initial_affiliations
        limits = self.limits
        reso_p = 15
        image = create_observation(list_users, list_sites, clients, limits, reso_p)
        return image

    def get_partial_obs(self):
        # output an n_agents x 2 layers image (for sites and users positions)
        # only a partial observation (for player i_agent)
        image = self.get_total_obs()
        list_obs = []
        for i_player in range(self.n_agents):
            image_i = image.copy()
            for i in range(self.n_agents):
                if i != i_player:
                    image_i[:,:,self.n_agents + i] = 0
            list_obs.append(image_i)
        return list_obs


    def step(self, action_n):
        # actions_n = list of n_agents actions
        # one personal action is a numpy (n_sites x n_section x n_agents)
        list_transactions = []
        for i_prov, action in enumerate(action_n):
            (n_sites, n_sections, n_agents) = action.shape
            for i_site in range(n_sites):
                for i_sec in range(n_sections):
                    list_transactions.append([i_prov, i_site, i_sec, action[i_site,i_sec]])

        self.transactions(list_transactions)
        self.update_transactions()

        obs_n = self.get_total_obs()
        r_n = [ 0 for i_prov in range(self.n_agents)]
        if self.utilities is not None:
            r_n = self.utilities(self)

        done_n = [False for i_prov in range(self.n_agents)]

        return obs_n, r_n, done_n, {}



    def get_utilities(self):
        r_n = [0 for i_prov in range(self.n_agents)]
        if self.utilities is not None:
            r_n = self.utilities(self)
        return r_n

    def render(self, output="output.png", close = False):
        self.display(output, fig_size=(8,8),label=True, links = True)



class Env_Multi_Agent_Sites_Users(Env_Multi_Agent_MNO):

    def __init__(self, positions_sites, positions_users, clients,
                     kinematics_users = None, max_per_sec = [[]], n_sections = [], angles_offset= [[]]):

        n_agents = len(positions_sites)
        providers = []
        n_users = len(positions_users)
        n_sites = sum([len(sites) for sites in positions_sites ])
        #positions_users = random_users(n_users, sites, 0.8, xlim, ylim)
        #clients = [ [i_prov for _ in range(int(n_users/n_agents))] for i_prov in range(n_agents)]

        for i_prov in range(n_agents):
            providers.append(Provider(i_prov, len(positions_sites[i_prov]), positions_sites[i_prov], n_agents,
                                   max_per_section=max_per_sec[i_prov],
                                   angles_offset_sites=angles_offset[i_prov], n_sections=n_sections[i_prov]))

        super(Env_Multi_Agent_Sites_Users, self).__init__(n_agents, n_sites, n_users,
                     providers, positions_users, clients,
                     kinematics_users = kinematics_users)



class Env_3A_5S_15U(Env_Multi_Agent_Sites_Users):

        def __init__(self):

            positions_sites = [ [(1, 1), (7, 6)],  [(4, 3.5)],  [(1, 6), (7, 1)] ]
            n_agents = len(positions_sites)
            sites = []
            for s in positions_sites:
                sites += s

            r = 0.9
            n_users = 30
            positions_users = random_users(n_users, sites, r, (0 + r, 8 - r), (0 + r, 7 - r))
            clients = []
            for ip in range(n_agents):
                clients += [ip for _ in range(int(n_users/n_agents))]

            print(clients)
            max_per_sec = [[10,10],[10],[10,10]]
            n_sections = [3,3,3]
            angles_offset = [[-30,-30],[-30],[-30,30]]


            super(Env_3A_5S_15U, self).__init__(positions_sites, positions_users, clients,
                     kinematics_users=None, max_per_sec=max_per_sec, n_sections=n_sections, angles_offset=angles_offset)

            self.limits = [(0,8),(0,7)]


class Env_3A_3S_9U_0(Env_Multi_Agent_MNO):

    def __init__(self, max_per_sec = 10, n_sections = 3, positions_users = None):

        positions_sites = [ [(2, 2)],  [(6, 2)],  [(4, 2+4*math.sin(math.pi/3))] ]

        n_agents = 3
        n_sites = 3

        sites1 = positions_sites[0]
        sites2 = positions_sites[1]
        sites3 = positions_sites[2]

        mno1 = Provider(0, len(sites1), sites1, n_agents, max_per_section=[max_per_sec] * len(sites1),
                        angles_offset_sites=[-30], n_sections=n_sections)
        mno2 = Provider(1, len(sites2), sites2, n_agents, max_per_section=[max_per_sec] * len(sites2),
                        angles_offset_sites=[-30], n_sections=n_sections)
        mno3 = Provider(2, len(sites3), sites3, n_agents, max_per_section=[max_per_sec] * len(sites3),
                        angles_offset_sites=[-30], n_sections=n_sections)

        providers = [mno1, mno2, mno3]
        if positions_users is None:
            positions_users = []
            n_u = 3
            r = 1
            for site, prov in zip([sites1, sites2, sites3], providers):
                (xS, yS) = site[0]
                for u in range(n_u):
                    #print((math.pi / 180) * prov.angles_offset_sites[0])
                    angle = 2*(u+0.5)*math.pi/n_u + (math.pi / 180) * prov.angles_offset_sites[0]
                    pos_U = ( xS + r*math.cos(angle) , yS + r*math.sin(angle)     )
                    positions_users.append(   pos_U    )

        n_users = len(positions_users)
        n_u = int(n_users/3)

        clients = [0 for _ in range(n_u)] + [1 for _ in range(n_u)] + [2 for _ in range(n_u)]

        super().__init__(n_agents, n_sites, n_users,
                 providers, positions_users, clients,
                 kinematics_users = None)


        self.limits = [(0, 8), (0, 7)]




class Env_2A_2S_2U(Env_Multi_Agent_MNO):

    def __init__(self, max_per_sec = 10, n_sections = 3):


        limits = [(0, 8), (0, 4)]
        positions_sites = [[(2,2)],[(6,2)]]
        positions_users = [(3,2.5),(5,1.5)]
        clients = [1,0]

        n_agents = 2
        n_sites = sum([len(sites) for sites in positions_sites])
        n_users = len(positions_users)

        providers = []
        angles_offset = [-60, 0]

        for i_prov in range(n_agents):
            providers.append(Provider(i_prov, len(positions_sites[i_prov]), positions_sites[i_prov], n_agents, max_per_section=[max_per_sec] * len(positions_sites[i_prov]),
                        angles_offset_sites=[angles_offset[i_prov]], n_sections=n_sections))

        super().__init__(n_agents, n_sites, n_users,
                 providers, positions_users, clients,
                 kinematics_users = None)


        self.limits = limits




class Env_3A_3S_9U_1(Env_3A_3S_9U_0):

    def __init__(self):
        super(Env_3A_3S_9U_1, self).__init__()
        self.initial_affiliations = [1,0,0, 1,2,1, 2,2,0]
        self.utilities = basic_utility
        self.reset()




class Env_3A_3S_9U_2(Env_3A_3S_9U_0):

    def __init__(self):
        super(Env_3A_3S_9U_2, self).__init__()
        self.initial_affiliations =  [2,0,1, 2,0,1, 2,0,1]
        self.utilities = basic_utility
        self.reset()




class Env_3A_3S_xU_0(Env_3A_3S_9U_0):

    def __init__(self):


        super(Env_3A_3S_xU_0, self).__init__()
        if len(positions_users) != len(clients):
            print('error : number of clients and positions_users')
            pass
        self.positions_users = positions_users()
        self.initial_affiliations = clients
        self.n_users = len(clients)
        self.reset()





class Env_3A_1S_9U(Env_Multi_Agent_Sites_Users):

        def __init__(self):

            positions_sites = [ [(3, 3)],  [(-1,-1)],  [(-1,-1)] ]
            n_agents = len(positions_sites)
            sites = []
            for s in positions_sites:
                sites += s

            angles_offset = [[-30], [-30], [-30]]
            positions_users = []
            n_u = 9
            r = 2
            for u in range(n_u):
                # print((math.pi / 180) * prov.angles_offset_sites[0])
                xS, yS = positions_sites[0][0]
                angle = 2 * (u + 0.5) * math.pi / n_u + (math.pi / 180) * angles_offset[0][0]
                pos_U = (xS + r * math.cos(angle), yS + r * math.sin(angle))
                positions_users.append(pos_U)

            n_users = len(positions_users)

            clients = [0,1,2]*3

            max_per_sec = [[10],[0],[0]]
            n_sections = [3,3,3]



            super(Env_3A_1S_9U, self).__init__(positions_sites, positions_users, clients,
                     kinematics_users=None, max_per_sec=max_per_sec, n_sections=n_sections, angles_offset=angles_offset)

            self.limits = [(0,8),(0,7)]




class Env_3A_3S_18U(Env_Multi_Agent_Sites_Users):

        def __init__(self):

            positions_sites = [ [(2, 2)],  [(6, 2)],  [(4, 2+4*math.sin(math.pi/3))] ]
            n_agents = len(positions_sites)

            angles_offset = [[-30], [-30], [-30]]
            positions_users = []
            n_u = 9
            r = 1
            for iP in range(n_agents):
                for u in range(n_u):
                    xS, yS = positions_sites[iP][0]
                    angle = 2 * u * math.pi / n_u + (math.pi / 180) * angles_offset[iP][0]
                    pos_U = (xS + r * math.cos(angle), yS + r * math.sin(angle))
                    if u%3 != 0:
                        positions_users.append(pos_U)

            n_users = len(positions_users)
            clients = [0,0,2,0,0,1,  1,2,1,1,0,1,   1,2,2,0,2,2   ]

            max_per_sec = [[100],[100],[100]]
            n_sections = [3,3,3]



            super(Env_3A_3S_18U, self).__init__(positions_sites, positions_users, clients,
                     kinematics_users=None, max_per_sec=max_per_sec, n_sections=n_sections, angles_offset=angles_offset)

            self.utilities = basic_exp_utility
            self.limits = [(0,8),(0,7)]
