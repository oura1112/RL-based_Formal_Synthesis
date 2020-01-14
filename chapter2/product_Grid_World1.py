# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:30:44 2019

@author: oura
"""

import itertools
import os
import copy
import matplotlib.pyplot as mpl
import numpy as np
import Automaton
import AugAutomaton
import Grid_World1
mpl.style.use('seaborn-whitegrid')

#禁止事象にコスト    
class Product_Grid1(AugAutomaton.AugAutomaton, Grid_World1.Grid_World2):
    
    def __init__(self, init_c_state, init_m_state, init_q, init_v):
        AugAutomaton.AugAutomaton.__init__(self)
        Grid_World1.Grid_World2.__init__(self)
        self.init_c_state = init_c_state
        self.init_m_state = init_m_state
        self.init_q = init_q
        self.init_v = init_v
        self.c_state = init_c_state
        self.m_state = init_m_state
        self.automaton_state = init_q
        self.v = init_v 
    
    def automaton_input(self, s_c, s_m, a, s_c_next, s_m_next, q, v):
        sigma = self.def_Label(s_c, s_m, a, s_c_next, s_m_next)   #sigmaはリスト
        q_next, v_next, v_hat = self.delta_bar(q, v, sigma)
        
        return [q, sigma, q_next], v_next, v_hat
    
    def binary_to_int(self, v) :
        v_d=0
        for i, v_i in enumerate(v) :
            if v_i == 1:
                v_d = v_d + 2**i
                
        return v_d
    
    def reward_func(self, automaton_transit, v, v_hat):
        accepting_frag = False
        
        for i, F_i in enumerate( self.F ) :
            if ( automaton_transit in F_i ) and v[i] == 0 and v_hat[i] == 1 :
                accepting_frag = True
                    
        if accepting_frag == True:
            return 10
        else :
            return 0

    def acc_event_reward_func(self, pi):
        acc_reward = 0
        #cat_doors = [6,7,8,9,10,11,12]
        #mouse_doors = [0,1,2,3,4,5]
        
        cat_doors = [4,5,6,7]
        mouse_doors = [0,1,2,3]
        
        for enable_event in pi:
            if enable_event in cat_doors:
                acc_reward += 0.7
            elif enable_event in mouse_doors:
                acc_reward += 0.7
        
        return acc_reward
        
    def prohibit_cost_func(self, prohibit_pi):
        prohibit_cost = 0
        #cat_doors = [6,7,8,9,10,11,12]
        #mouse_doors = [0,1,2,3,4,5]
        
        cat_doors = [4,5,6,7]
        mouse_doors = [0,1,2,3]
        
        for prohibit_event in prohibit_pi:
            if prohibit_event in cat_doors:
                prohibit_cost -= 0.8
            elif prohibit_event in mouse_doors:
                prohibit_cost -= 0.8
        
        return prohibit_cost
    """   
    def event_cost_func(self, event):
        
        if event in cat_door:
            return -10
    """       
    def can_action_at(self, q):
        if q == 0 :
            return True
        elif q == 1 :
            return False
        
    def _move(self, s_c, s_m, q, v, event):
        #if not self.can_action_at(q) :
         #   raise Exception("Can't move!")
        s_c_next, s_m_next = self.def_P(event, s_c, s_m)
        automaton_transit, v_next, v_hat = self.automaton_input(s_c, s_m, event, s_c_next, s_m_next, q, v)
        return s_c_next, s_m_next, automaton_transit, v_next, v_hat
    
    def state_reset(self):
        del self.c_state
        del self.m_state
        del self.automaton_state
        del self.v
        
        self.c_state = copy.copy(self.init_c_state)
        self.m_state = copy.copy(self.init_m_state)
        self.automaton_state = copy.copy(self.init_q)
        self.v = copy.copy(self.init_v)
        v = self.binary_to_int(self.init_v)
        return self.c_state, self.m_state, self.automaton_state, v
    
    def step(self, event, prohibit_pi, enable_pi):
        s_c_next, s_m_next, automaton_transit, v_next, v_hat = self._move(self.c_state, self.m_state, self.automaton_state, self.v, event) #後でs → self.agent_state

        reward = self.reward_func(automaton_transit, self.v, v_hat)
        
        prohibit_cost = self.prohibit_cost_func(prohibit_pi)

        enable_reward = self.acc_event_reward_func(enable_pi)
        
        self.c_state = s_c_next
        self.m_state = s_m_next
        self.automaton_state = automaton_transit[2]
        self.v = v_next
        
        v_next = self.binary_to_int(v_next)
        
        return s_c_next, s_m_next, reward, prohibit_cost, enable_reward, automaton_transit, v_next
    
#許可事象に報酬    
class Product_Grid1_2(AugAutomaton.AugAutomaton, Grid_World1.Grid_World2):
    
    def __init__(self, init_c_state, init_m_state, init_q, init_v):
        AugAutomaton.AugAutomaton.__init__(self)
        Grid_World1.Grid_World2.__init__(self)
        self.init_c_state = init_c_state
        self.init_m_state = init_m_state
        self.init_q = init_q
        self.init_v = init_v
        self.c_state = init_c_state
        self.m_state = init_m_state
        self.automaton_state = init_q
        self.v = init_v 
    
    def automaton_input(self, s_c, s_m, a, s_c_next, s_m_next, q, v):
        sigma = self.def_Label(s_c, s_m, a, s_c_next, s_m_next)   #sigmaはリスト
        q_next, v_next, v_hat = self.delta_bar(q, v, sigma)
        
        return [q, sigma, q_next], v_next, v_hat
    
    def binary_to_int(self, v) :
        v_d=0
        for i, v_i in enumerate(v) :
            if v_i == 1:
                v_d = v_d + 2**i
                
        return v_d
    
    def reward_func(self, automaton_transit, v, v_hat):
        accepting_frag = False
        
        for i, F_i in enumerate( self.F ) :
            if ( automaton_transit in F_i ) and v[i] == 0 and v_hat[i] == 1 :
                accepting_frag = True
                    
        if accepting_frag == True:
            return 10
        else :
            return 0
        
    def acc_event_reward_func(self, pi):
        acc_reward = 0
        #cat_doors = [6,7,8,9,10,11,12]
        #mouse_doors = [0,1,2,3,4,5]
        
        cat_doors = [4,5,6,7]
        mouse_doors = [0,1,2,3]
        
        for prohibit_event in pi:
            if prohibit_event in cat_doors:
                acc_reward += 0.1
            elif prohibit_event in mouse_doors:
                acc_reward += 0.1
        
        return acc_reward
    """   
    def event_cost_func(self, event):
        
        if event in cat_door:
            return -10
    """       
    def can_action_at(self, q):
        if q == 0 :
            return True
        elif q == 1 :
            return False
        
    def _move(self, s_c, s_m, q, v, event):
        #if not self.can_action_at(q) :
         #   raise Exception("Can't move!")
        s_c_next, s_m_next = self.def_P(event, s_c, s_m)
        automaton_transit, v_next, v_hat = self.automaton_input(s_c, s_m, event, s_c_next, s_m_next, q, v)
        return s_c_next, s_m_next, automaton_transit, v_next, v_hat
    
    def state_reset(self):
        del self.c_state
        del self.m_state
        del self.automaton_state
        del self.v
        
        self.c_state = copy.copy(self.init_c_state)
        self.m_state = copy.copy(self.init_m_state)
        self.automaton_state = copy.copy(self.init_q)
        self.v = copy.copy(self.init_v)
        v = self.binary_to_int(self.init_v)
        return self.c_state, self.m_state, self.automaton_state, v
    
    def step(self, event, enable_pi):
        s_c_next, s_m_next, automaton_transit, v_next, v_hat = self._move(self.c_state, self.m_state, self.automaton_state, self.v, event) #後でs → self.agent_state

        reward = self.reward_func(automaton_transit, self.v, v_hat)
        
        acc_reward = self.acc_event_reward_func(enable_pi)
        
        self.c_state = s_c_next
        self.m_state = s_m_next
        self.automaton_state = automaton_transit[2]
        self.v = v_next
        
        v_next = self.binary_to_int(v_next)
        
        return s_c_next, s_m_next, reward, acc_reward, automaton_transit, v_next
    
    
    
class Product_Grid2(AugAutomaton.AugAutomaton, Grid_World1.Grid_World2):
    
    def __init__(self, init_state, init_q, init_v):
        AugAutomaton.AugAutomaton.__init__(self)
        Grid_World1.Grid_World2.__init__(self)
        self.init_state = init_state
        self.init_q = init_q
        self.init_v = init_v
        self.agent_state = init_state
        self.automaton_state = init_q
        self.v = init_v 
    
    def automaton_input(self, s, a, s_next, q, v):
        sigma = self.def_Label(s, a, s_next)   #sigmaはリスト
        q_next, v_next, v_hat = self.delta_bar(q, v, sigma)
        
        return [q, sigma, q_next], v_next, v_hat
    
    def binary_to_int(self, v) :
        v_d=0
        for i, v_i in enumerate(v) :
            if v_i == 1:
                v_d = v_d + 2**i
                
        return v_d
    
    def reward_func(self, automaton_transit, v, v_hat):
        accepting_frag = False
        
        for i, F_i in enumerate( self.F ) :
            if ( automaton_transit in F_i ) and v[i] == 0 and v_hat[i] == 1 :
                accepting_frag = True
                    
        if accepting_frag == True:
            return 2
        else :
            return 0
    
    #使ってない    
    def can_action_at(self, q):
        if q == 0 :
            return True
        elif q == 1 :
            return False
        
    def _move(self, s, q, v, a):
        #if not self.can_action_at(q) :
         #   raise Exception("Can't move!")
        s_next = self.def_P(a, s)
        automaton_transit, v_next, v_hat = self.automaton_input(s, a, s_next, q, v)
        return s_next, automaton_transit, v_next, v_hat
    
    def state_reset(self):
        del self.agent_state
        del self.automaton_state
        del self.v
        
        self.agent_state = copy.copy(self.init_state)
        self.automaton_state = copy.copy(self.init_q)
        self.v = copy.copy(self.init_v)
        v = self.binary_to_int(self.init_v)
        return self.agent_state, self.automaton_state, v
    
    def step(self, a):
        s_next, automaton_transit, v_next, v_hat = self._move(self.agent_state, self.automaton_state, self.v, a) #後でs → self.agent_state

        reward = self.reward_func(automaton_transit, self.v, v_hat)
        #print(reward)
        print(self.agent_state, a, s_next, automaton_transit, self.v, v_next, reward)
        self.agent_state = s_next
        self.automaton_state = automaton_transit[2]
        self.v = v_next
        
        v_next = self.binary_to_int(v_next)
        
        return s_next, reward, automaton_transit, v_next
    
class Product_Grid3(AugAutomaton.AugAutomaton, Grid_World1.Grid_World1):
    
    def __init__(self, init_state, init_q, init_v):
        AugAutomaton.AugAutomaton.__init__(self)
        Grid_World1.Grid_World1.__init__(self)
        self.init_state = init_state
        self.init_q = init_q
        self.init_v = init_v
        self.agent_state = init_state
        self.automaton_state = init_q
        self.v = init_v 
    
    def automaton_input(self, s, a, s_next, q, v):
        sigma = self.def_Label(s, a, s_next)   #sigmaはリスト
        q_next, v_next, v_hat = self.delta_bar(q, v, sigma)
        
        return [q, sigma, q_next], v_next, v_hat
    
    def binary_to_int(self, v) :
        v_d=0
        for i, v_i in enumerate(v) :
            if v_i == 1:
                v_d = v_d + 2**i
                
        return v_d
    
    def reward_func(self, automaton_transit, v, v_hat):
        accepting_frag = False
        
        for i, F_i in enumerate( self.F ) :
            if ( automaton_transit in F_i ) and v[i] == 0 and v_hat[i] == 1 :
                accepting_frag = True
                    
        if accepting_frag == True:
            return 2
        else :
            return 0
        
    def _move(self, s, q, v, a):
        #if not self.can_action_at(q) :
         #   raise Exception("Can't move!")
        s_next = self.def_P(a, s)
        automaton_transit, v_next, v_hat = self.automaton_input(s, a, s_next, q, v)
        return s_next, automaton_transit, v_next, v_hat
    
    def state_reset(self):
        del self.agent_state
        del self.automaton_state
        del self.v
        
        self.agent_state = copy.copy(self.init_state)
        self.automaton_state = copy.copy(self.init_q)
        self.v = copy.copy(self.init_v)
        v = self.binary_to_int(self.init_v)
        return self.agent_state, self.automaton_state, v
    
    def step(self, a):
        s_next, automaton_transit, v_next, v_hat = self._move(self.agent_state, self.automaton_state, self.v, a) #後でs → self.agent_state

        reward = self.reward_func(automaton_transit, self.v, v_hat)
        #print(reward)
        #print(self.agent_state, a, s_next, automaton_transit, self.v, v_next, reward)
        self.agent_state = s_next
        self.automaton_state = automaton_transit[2]
        self.v = v_next
        
        v_next = self.binary_to_int(v_next)
        
        return s_next, reward, automaton_transit, v_next
    
    
