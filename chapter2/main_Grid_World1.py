# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:59:54 2019

@author: oura
"""

import itertools
import os
import time
import matplotlib.pyplot as mpl
import numpy as np
import product_Grid_World1
import Supervisor
import Supervisor_ambiguous
mpl.style.use('seaborn-whitegrid')

#Proposed 禁止事象にコストをかける
#env = product_Grid_World1.Product_Grid1(init_c_state=0, init_m_state=2, init_q=0, init_v = [0,0])
#agent = Supervisor.Supervisor(env)

#Proposed 禁止事象にコストをかける. ambiguity set を用いる -> 許可事象に報酬
env = product_Grid_World1.Product_Grid1(init_c_state=0, init_m_state=2, init_q=0, init_v = [0,0])
agent = Supervisor_ambiguous.Supervisor(env)

# Proposed 許可事象に報酬を与える
# env = product_Grid_World1.Product_Grid1_2(init_c_state=0, init_m_state=2, init_q=0, init_v = [0,0])
# agent = Supervisor.Supervisor3(env)
#agent.print_()
#Abate
#env = product_Grid_World1.Product_Grid2_Abate(init_state=[2,2], init_q=0 )
#agent = Grid_Agent1.Agent_Abate_Grid3(env)


start = time.time()
agent.learn(env)
elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")