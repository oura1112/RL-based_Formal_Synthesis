# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:34:19 2019

@author: oura
"""

import itertools
import os
import matplotlib.pyplot as mpl
import numpy as np
import Automaton
mpl.style.use('seaborn-whitegrid')

class AugAutomaton(Automaton.Automaton1):
    def __init__(self):
        Automaton.Automaton1.__init__(self)
        
    def visitf(self, transit):
        v = []
        for i in range(len(self.F)):
            if transit in self.F[i]:
                v.append(1)
            else :
                v.append(0)
                
        return v
                
    def reset(self, v_prev):
         if all(v_prev):
             return np.zeros(len(self.F))
         else :
             return v_prev
         
    def Max(self, v, u):
        return np.logical_or(v,u)
    
    def delta_bar(self, q, v, sigma):
        q_next = self.delta(q, sigma)
        v_hat  = self.visitf([q,sigma,q_next])
        v_next = self.reset( self.Max(v,v_hat) )
        
        return q_next, v_next, v_hat
        