# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:51:12 2019

@author: oura
"""

"""
LTL = GFa & GFb & G!c
"""


import itertools
import os
import matplotlib.pyplot as mpl
import numpy as np
mpl.style.use('seaborn-whitegrid')

class Automaton1(): #GFa & GFb & G!c
    def __init__(self):
        # LDGBA
        self.Q = [0,1]
        self.q0 = 0
        self.AP = ["a","b","c"]
        self.Sigma = [["Empty"],["a"],["b"],["c"],["a","b"],["b","c"],["c","a"],self.AP]
        #self.delta = [[0,["Empty"],0], [0,["a"],0], [0,["b"],0], [0,["a","b"],0], [0,["c"],1], [0,["b","c"],1], [0,["c","a"],1], [0,self.AP,1], [1,"T",1]]
        self.F0 = [[0,["a"],0], [0,["a","b"],0]] 
        self.F1 = [[0,["b"],0], [0,["a","b"],0]]
        self.F = [self.F0, self.F1]
        #print(self.F)
    def delta(self, q, sigma):
        if q == 0:
            if sigma in [["Empty"],["a"],["b"],["a","b"]]:
                return 0
            else :
                return 1
        elif q == 1:
            return 1
        
    #def print_(self):
        #print(self.F)
        

class Automaton1_LDBA(): #GFa & GFb & G!c
    def __init__(self):
        # LDGBA
        self.Q = [0,1,2]
        self.q0 = 0
        self.AP = ["a","b","c"]
        self.Sigma = [["Empty"],["a"],["b"],["c"],["a","b"],["b","c"],["c","a"],self.AP]
        #self.delta = [[0,["Empty"],0], [0,["a"],0], [0,["b"],0], [0,["a","b"],0], [0,["c"],1], [0,["b","c"],1], [0,["c","a"],1], [0,self.AP,1], [1,"T",1]]
        self.F = [[1,["b"],0], [0,["a","b"],0]]
        #print(self.F)
    def delta(self, q, sigma):
        if q == 0:
            if sigma in [["Empty"],["b"],["a","b"]]:
                return 0
            elif sigma in [["a"]] :
                return 1
            else :
                return 2
        elif q == 1:
            if sigma in [["b"],["a","b"]] :
                return 0
            elif sigma in [["a"],["Empty"]] :
                return 1
            else :
                return 2

class Automaton2(): #GFa & GFb & G!c & G(a -> X( !a U b )) & G(b -> X( !b U a )) 受理条件などの定義はMDPに合わせて簡略化している
    def __init__(self):
        # LDGBA
        self.Q = [0,1,2,3]
        self.q0 = 0
        self.AP = ["a","b","c"]
        #self.Sigma = 2^AP
        #self.delta = [[0,["Empty"],0], [0,["a"],0], [0,["b"],0], [0,["a","b"],0], [0,["c"],1], [0,["b","c"],1], [0,["c","a"],1], [0,self.AP,1], [1,"T",1]]
        self.F0 = [[2,["a"],1]]
        self.F1 = [[1,["b"],2]] 
        self.F = [self.F0, self.F1]
        
    def delta(self, q, sigma):
        if q == 0:
            if sigma in [["Empty"]]:
                return 0
            elif sigma in [["a"]] :
                return 1
            elif sigma in [["b"]] :
                return 2
            else :
                return 3
        elif q == 1:
            if sigma in [["Empty"]]:
                return 1
            elif sigma in [["b"]] :
                return 2
            else :
                return 3
        elif q == 2 :
            if sigma in [["Empty"]]:
                return 2
            elif sigma in [["a"]] :
                return 1
            else :
                return 3
        elif q == 3 :
            return 3
        
class Automaton3(): #GFa & GFb & GFd & GFe & G!c 受理条件などの定義はMDPに合わせて簡略化している
    def __init__(self):
        # LDGBA
        self.Q = [0,1,2]
        self.q0 = 0
        self.AP = ["a","b","c","d","e"]
        #self.Sigma = 2^AP
        #self.delta = [[0,["Empty"],0], [0,["a"],0], [0,["b"],0], [0,["a","b"],0], [0,["c"],1], [0,["b","c"],1], [0,["c","a"],1], [0,self.AP,1], [1,"T",1]]
        self.F0 = [[0,["a"],0], [1,["a"],1]] 
        self.F1 = [[0,["b"],0], [0,["b"],1], [1,["b"],1]]
        self.F2 = [[1,["d"],1]]
        self.F3 = [[0,["e"],0], [1,["e"],1]]
        self.F = [self.F0, self.F1, self.F2, self.F3]
        
    def delta(self, q, sigma):
        if q == 0:
            if sigma in [["Empty"],["a"],["e"]]:
                return 0
            elif sigma == ["b"] :
                return 1
            elif sigma in [["c"],["d"]] :
                return 2
        elif q == 1:
            if sigma == ["c"] :
                return 2
            else :
                return 1
        elif q == 2 :
            return 2

       
#automaton = Automaton1()
#automaton.print_