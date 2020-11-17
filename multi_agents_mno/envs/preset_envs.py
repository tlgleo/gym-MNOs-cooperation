from .env_mno_core import Env_Multi_Agent_MNO, Env_Multi_Agent_Sites_Users, Provider
import math
from utils import partition, random_users, create_observation, separate_images
from utility_functions import basic_utility, basic_utility_V2, basic_exp_utility
from kinematics import Kinematics, modifiy_position

class Env_3A_5S_30U(Env_Multi_Agent_Sites_Users):

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


            super(Env_3A_5S_30U, self).__init__(positions_sites, positions_users, clients,
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
