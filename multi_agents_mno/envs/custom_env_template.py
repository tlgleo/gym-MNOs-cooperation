from .env_mno_core import Env_Multi_Agent_MNO, Env_Multi_Agent_Sites_Users
import math
from utils import partition, random_users, create_observation, separate_images
from utility_functions import basic_utility, basic_utility_V2, basic_exp_utility
from kinematics import Kinematics, modifiy_position

positions_sites_env = [ [(1,1)],  [(2,2)], [(3,3)], [], [(5,5),(4,2)] ]
positions_users_env = [ (0.5,0.5),(1,0.5),(1.5,0.5),(2,0.5),(2.5,0.5),(3,0.5) ]
clients_env = [0,1,2,3,4,4]
limits_geo_env_env = [(0, 6), (0, 6)]


class Name_Of_Env_Class(Env_Multi_Agent_Sites_Users):
        def __init__(self):
            positions_sites = positions_sites_env
            positions_users = positions_users_env
            clients = clients_env
            super().__init__(positions_sites, positions_users, clients)

            self.utilities = basic_exp_utility
            self.limits = limits_geo_env_env
