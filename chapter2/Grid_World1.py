# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:18:34 2019

@author: oura
"""

import itertools
import os
import matplotlib.pyplot as mpl
import numpy as np
mpl.style.use('seaborn-whitegrid')

class Grid_World1():
    def __init__(self, s_size=5):

        ###### MDP(Grid World 1) ######
        self.def_S(s_size)
        #self.A = self.def_A()
        #self.def_A() #Grid_Agent1.pyで呼び出す
        #self.def_P()
        self.AP = ["a","b","c"]
        #self.def_Label()

    def def_S(self, s_size):
        self.State_c = s_size
        self.State_m = s_size
        
        """
        for x in range(x_size):
            for y in range(y_size):
                self.S.append([x,y])
        """
                
    """
    c1-c7 = 7-12, m1-m6 = 0-6
    """
    def def_A(self, s_c, s_m): 
        self.A = [i for i in range(13)]
        self.A_un = []
        return self.A, self.A_un
            
    def def_P(self, event, s_c, s_m):
        temp = np.random.rand()
        div = 0.05
        
        s_c_next = s_c
        s_m_next = s_m
        
        if s_c == 0:
            if event == 6 and temp > div:
                s_c_next = 1
            elif event == 8 and temp > div:
                s_c_next = 2
            elif event == 9 and temp > div:
                s_c_next = 3
            elif event == 11 and temp > div:
                s_c_next = 4
                
        elif s_c == 1:
            if event == 6 and temp > div:
                s_c_next = 0
            elif event == 7 and temp > div:
                s_c_next = 2
            elif event == 12 and temp > div: #12
                s_c_next = 3
            
        elif s_c == 2:
            if event == 7 and temp > div:
                s_c_next = 1
            elif event == 8 and temp > div:
                s_c_next = 0
                
        elif s_c == 3:
            if event == 9 and temp > div:
                s_c_next = 0
            elif event == 10 and temp > div:
                s_c_next = 4
            elif event == 12 and temp > div: #12
                s_c_next = 1
                
        elif s_c == 4:
            if event == 10 and temp > div:
                s_c_next = 3
            elif event == 11 and temp > div:
                s_c_next = 0
                
        if s_m == 0:
            if event == 0 and temp > div:
                s_m_next = 2
            elif event == 2 and temp > div:
                s_m_next = 1
            elif event == 3 and temp > div:
                s_m_next = 4
            elif event == 5 and temp > div:
                s_m_next = 3
                
        elif s_m == 1:
            if event == 1 and temp > div:
                s_m_next = 2
            elif event == 2 and temp > div:
                s_m_next = 0
            
        elif s_m == 2:
            if event == 0 and temp > div:
                s_m_next = 0
            elif event == 1 and temp > div:
                s_m_next = 1
                
        elif s_m == 3:
            if event == 4 and temp > div:
                s_m_next = 4
            elif event == 5 and temp > div:
                s_m_next = 0
                
        elif s_m == 4:
            if event == 3 and temp > div:
                s_m_next = 0
            elif event == 4 and temp > div:
                s_m_next = 3
                
        return s_c_next, s_m_next
    
    #オートマトンへの入力になる
    def def_Label(self, s_c, s_m, event, s_c_next, s_m_next):
        if ( s_c_next == 1) and (s_c_next != s_m_next):
            return ["a"]
        elif ( s_c_next == 2) and (s_c_next != s_m_next):
            return ["b"]
        elif ( s_m_next == 1) and (s_c_next != s_m_next):
            return ["c"]
        elif ( s_m_next == 4) and (s_c_next != s_m_next):
            return ["d"]
        elif (s_c_next == s_m_next):
            return ["e"]
        else :
            return ["Empty"]
    
class Grid_World2():
    def __init__(self, s_size=4):

        ###### MDP(Grid World 1) ######
        self.def_S(s_size)
        #self.A = self.def_A()
        #self.def_A() #Grid_Agent1.pyで呼び出す
        #self.def_P()
        self.AP = ["a","b","c"]
        #self.def_Label()

    def def_S(self, s_size):
        self.State_c = s_size
        self.State_m = s_size
        
        """
        for x in range(x_size):
            for y in range(y_size):
                self.S.append([x,y])
        """
                
    """
    c1-c7 = 7-12, m1-m6 = 0-6
    """
    def def_A(self, s_c, s_m): 
        self.A = [i for i in range(7)]
        self.A_un = [7]
        return self.A, self.A_un
            
    def def_P(self, event, s_c, s_m):
        temp = np.random.rand()
        div = 0.05
        
        s_c_next = s_c
        s_m_next = s_m
        
        if s_c == 0:
            if event == 4 and temp > div:
                s_c_next = 2
            elif event == 5 and temp > div:
                s_c_next = 1
                
        elif s_c == 1:
            if event == 5 and temp > div:
                s_c_next = 0
            elif event == 6 and temp > div:
                s_c_next = 3
            
        elif s_c == 2:
            if event == 4 and temp > div:
                s_c_next = 0
            elif event == 7 and temp > div:
                s_c_next = 3
                
        elif s_c == 3:
            if event == 6 and temp > div:
                s_c_next = 1
            elif event == 7 and temp > div: #
                s_c_next = 2

                
        if s_m == 0:
            if event == 0 and temp > div:
                s_m_next = 2
            elif event == 1 and temp > div:
                s_m_next = 1
                
        elif s_m == 1:
            if event == 1 and temp > div:
                s_m_next = 0
            elif event == 2 and temp > div:
                s_m_next = 3
            
        elif s_m == 2:
            if event == 0 and temp > div:
                s_m_next = 0
            elif event == 3 and temp > div:
                s_m_next = 3
                
        elif s_m == 3:
            if event == 2 and temp > div:
                s_m_next = 1
            elif event == 3 and temp > div:
                s_m_next = 2
                
        return s_c_next, s_m_next
    
    #オートマトンへの入力になる
    def def_Label(self, s_c, s_m, event, s_c_next, s_m_next):
        if ( s_c_next == 1) and (s_c_next != s_m_next):
            return ["a"]
        #elif ( s_c_next == 1) and (s_c_next != s_m_next):
         #   return ["b"]
        #elif ( s_m_next == 2) and (s_c_next != s_m_next):
          #  return ["c"]
        elif ( s_m_next == 1) and (s_c_next != s_m_next):
            return ["b"]
        elif (s_c_next == s_m_next):
            return ["e"]
        else :
            return ["Empty"]
