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
    def __init__(self, x_size=4,  y_size=3):

        ###### MDP(Grid World 1) ######
        self.S = []
        self.A = []
        self.def_S(x_size,  y_size)
        #self.def_A() #Grid_Agent1.pyで呼び出す
        #self.def_P()
        self.def_s0()
        self.AP = ["a","b","c"]
        #self.def_Label()

    def def_S(self, x_size,  y_size):
        self.X = x_size
        self.Y = y_size
        """
        for x in range(x_size):
            for y in range(y_size):
                self.S.append([x,y])
        """
                
    """
    Right = 0, Down = 1, Left = 2, Up = 3
    """
    def def_A(self, actions = [0,1,2,3]): 
        for a in actions:
            self.A.append(a)
        print(self.A)
        return self.A
            
    def def_P(self, a, s):
        temp = np.random.rand()
        div = 0.2
        s_next = [0,0]  #s_next = sとするとアドレスごとコピーされてしまう
        # Right
        if a  == 0:
            # left
            if (0.0 <= temp) and (temp < div):
                if s[0]  > 0:
                     s_next[0]  = s[0]  - 1
                     s_next[1]  = s[1] 
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
            # right
            else:
                if s[0]  < self.X - 1:
                     s_next[0]  = s[0]  + 1
                     s_next[1]  = s[1] 
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
        # Down
        elif a  == 1:
            # down
            if (0.0 <= temp) and (temp < 1-div):
                if  s[1]  > 0:
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  - 1
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
            # up
            else:
                if  s[1]  < self.Y - 1:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1]  + 1
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
        # Left
        elif a  == 2:
            # right
            if (0.0 <= temp) and (temp < div):
                if s[0]  < self.X - 1:
                     s_next[0]  = s[0]  + 1
                     s_next[1]  = s[1] 
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
            # left
            else:
                if s[0] > 0:
                     s_next[0]  = s[0]  - 1
                     s_next[1]  = s[1] 
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
        # Up
        elif a  == 3:
            # up
            if (0.0 <= temp) and (temp < 1-div):
                if  s[1]  < self.Y - 1:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1]  + 1
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
            # down
            else:
                if  s[1]  > 0:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1]  - 1
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
        return s_next

    def def_s0(self):
        self.x0 = [0]
        self.y0 = [0]
    
    #オートマトンへの入力になる
    """
    # 3*2 ex2
    def def_Label(self, s, a, s_next):
        if ( s == [1,0] and a==2 and s_next == [0,0]):
            return ["a"]
        elif ( s == [2,0] and a==3 and s_next == [2,1]):
            return ["b"]
        elif ( s_next == [0,1]):
            return ["c"]
        else :
            return ["Empty"]
    """
    
    #4*3 ex3
    def def_Label(self, s, a, s_next):
        if ( s == [1,0] and a==2 and s_next == [0,0]):
            return ["a"]
        elif ( s == [3,1] and a==3 and s_next == [3,2]):
            return ["b"]
        elif ( s_next == [0,2] ):
            return ["c"]
        else :
            return ["Empty"]
    
    """
    # 4*3 ex4
    def def_Label(self, s, a, s_next):
        if ( s == [1,0] and a==2 and s_next == [0,0]):
            return ["a"]
        elif ( s == [3,1] and a==3 and s_next == [3,2]):
            return ["b"]
        elif ( (s == [0,0] and a==3 and s_next==[0,1]) or (s == [1,0] and a==3 and s_next==[1,1]) or (s == [2,1] and ((a==2 and s_next==[1,1]) or (a==3 and s_next==[2,2])) ) or (s == [3,2] and a==2 and s_next==[2,2]) ):
            return ["c"]
        else :
            return ["Empty"]
    """
    """ 5*3 ex1
    def def_Label(self, s,  s_next):
        if ( (s == [0,1] or s == [1,0]) and s_next == [0,0]):
            return ["a"]
        elif ( (s == [4,1] or s == [3,2]) and s_next == [4,2]):
            return ["b"]
        elif ( ((s == [3,1] or s == [4,0]) and s_next == [3,0]) or (s == [2,2] and s_next == [3,2]) ):
            return ["a","b"]
        elif( (s == [3,0] and (s_next == [3,1] or s_next == [4,0])) and ((s == [0,2] or s == [1,1]) and s_next == [1,2]) or (s == [3,2] and s_next == [2,2])):    #
            return ["c"]
        else :
            return ["Empty"]
    """
    
    """
    # 5*5 ex5
    def def_Label(self, s, a, s_next):
        if ( s == [1,0] and a==2 and s_next == [0,0]):
            return ["a"]
        elif ( (s == [3,0] and a==0 and s_next == [4,0]) or (s == [4,1] and a==1 and s_next == [4,0]) ):
            return ["b"]
        elif ( (s == [0,1] and a==3 and s_next==[0,2]) or (s == [1,1] and a==3 and s_next==[1,2]) or (s == [2,1] and a==3 and s_next==[2,2])\
               or (s == [0,4] and a==1 and s_next==[0,3]) or (s == [1,4] and a==1 and s_next==[1,3])\
               or (((s == [2,1] and a == 3) or (s == [3,2] and a == 2)) and s_next == [2,2])\
               or (((s == [2,4] and a == 1) or (s == [3,3] and a == 2)) and s_next == [2,3]) ) :
            return ["c"]
        elif ( (s == [4,3] and a==3 and s_next == [4,4]) or (s == [3,4] and a==0 and s_next == [4,3]) ):
            return ["d"]
        elif (s == [3,4] and a==2 and s_next == [2,4]) :
            return ["e"]
        else :
            return ["Empty"]
        """

class Grid_World2():
    def __init__(self, x_size=3,  y_size=2):

        ###### MDP(Grid World 1) ######
        self.S = []
        self.A = []
        self.def_S(x_size,  y_size)
        #self.def_A() #Grid_Agent1.pyで呼び出す
        #self.def_P()
        self.def_s0()
        self.AP = ["a","b","c"]
        #self.def_Label()

    def def_S(self, x_size,  y_size):
        self.X = x_size
        self.Y = y_size
        """
        for x in range(x_size):
            for y in range(y_size):
                self.S.append([x,y])
        """
                
    """
    Right = 0, Down = 1, Left = 2, Up = 3
    """
    def def_A(self, actions = [0,1,2,3]): 
        self.A = actions
        return self.A
            
    def def_P(self, a, s):
        temp = np.random.rand()
        div = 0.1
        s_next = [0,0]  #s_next = sとするとアドレスごとコピーされてしまう
        # Right        
        if s != [1,0] :
            if a  == 0:
                # left
                if (0.0 <= temp) and (temp < div):
                    if s[0]  > 0:
                         s_next[0]  = s[0]  - 1
                         s_next[1]  = s[1] 
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
                # right
                else:
                    if s[0]  < self.X - 1:
                         s_next[0]  = s[0]  + 1
                         s_next[1]  = s[1] 
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
            # Down
            elif a  == 1:
                # down
                if (0.0 <= temp) and (temp < 1-div):
                    if  s[1]  > 0:
                         s_next[0]  = s[0]
                         s_next[1]  = s[1]  - 1
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
                # up
                else:
                    if  s[1]  < self.Y - 1:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1]  + 1
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
            # Left
            elif a  == 2:
                # right
                if (0.0 <= temp) and (temp < div):
                    if s[0]  < self.X - 1:
                         s_next[0]  = s[0]  + 1
                         s_next[1]  = s[1] 
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
                # left
                else:
                    if s[0] > 0:
                         s_next[0]  = s[0]  - 1
                         s_next[1]  = s[1] 
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
            # Up
            elif a  == 3:
                # up
                if (0.0 <= temp) and (temp < 1-div):
                    if  s[1]  < self.Y - 1:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1]  + 1
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
                # down
                else:
                    if  s[1]  > 0:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1]  - 1
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1]                          
        elif s == [1,0] :
            #UpRight
            if a  == 0:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # UpRight
                else:
                     s_next[0]  = s[0] + 1
                     s_next[1]  = s[1] + 1
            #Right
            elif a  == 1:
                # Right
                if (0.0 <= temp) and (temp < 1-div):
                     s_next[0]  = s[0] + 1
                     s_next[1]  = s[1] 
                # stay
                else:
                     s_next[0]  = s[0]  
                     s_next[1]  = s[1] 
            # Left
            elif a  == 2:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1] 
                    
                # left
                else:
                     s_next[0]  = s[0] - 1
                     s_next[1]  = s[1]
            # UpLeft
            elif a  == 3:
                # UpLeft
                if (0.0 <= temp) and (temp < 1-div):
                     s_next[0]  = s[0] - 1
                     s_next[1]  = s[1] + 1 
                # stay
                else:
                     s_next[0]  = s[0] 
                     s_next[1]  = s[1]

        if s_next == [1,1]:
            s_next = [1,0]
        return s_next

    def def_s0(self):
        self.x0 = [0]
        self.y0 = [0]
    
    #オートマトンへの入力になる
    # 3*2 ex2 (s[0] = 1 は廊下)
    def def_Label(self, s, a, s_next):
        if ( s == [1,0] and a == 2 and s_next == [0,0]):
            return ["a"]
        elif ( s == [2,0] and a == 3 and s_next == [2,1]):
            return ["b"]
        elif ( s_next == [0,1]):
            return ["c"]
        else :
            return ["Empty"]   
        
class Grid_World3():
    def __init__(self, x_size=3, y_size=4, s_size=9):

        ###### MDP(Grid World 1) ######
        self.S = []
        self.A = []
        self.def_S(x_size, y_size, s_size)
        #self.def_A() #Grid_Agent1.pyで呼び出す
        #self.def_P()
        self.AP = ["a","b","c"]
        #self.def_Label()

    def def_S(self, x_size, y_size, s_size):
        self.State = s_size
        self.X = x_size
        self.Y = y_size
        """
        for x in range(x_size):
            for y in range(y_size):
                self.S.append([x,y])
        """
                
    """
    Right = 0, Down = 1, Left = 2, Up = 3
    """
    def def_A(self, s):
        
        if s != 4:
            self.A = [0,1,2,3]
        elif s == 4 :
            self.A = [0,1,2,3,4,5,6,7]
        return self.A
            
    def def_P(self, a, s):
        temp = np.random.rand()
        div = 0.1
        s_next = [0,0]  #s_next = sとするとアドレスごとコピーされてしまう
        # Right        
        if s != [1,0] :
            if a  == 0:
                # left
                if (0.0 <= temp) and (temp < div):
                    if s[0]  > 0:
                         s_next[0]  = s[0]  - 1
                         s_next[1]  = s[1] 
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
                # right
                else:
                    if s[0]  < self.X - 1:
                         s_next[0]  = s[0]  + 1
                         s_next[1]  = s[1] 
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
            # Down
            elif a  == 1:
                # down
                if (0.0 <= temp) and (temp < 1-div):
                    if  s[1]  > 0:
                         s_next[0]  = s[0]
                         s_next[1]  = s[1]  - 1
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
                # up
                else:
                    if  s[1]  < self.Y - 1:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1]  + 1
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
            # Left
            elif a  == 2:
                # right
                if (0.0 <= temp) and (temp < div):
                    if s[0]  < self.X - 1:
                         s_next[0]  = s[0]  + 1
                         s_next[1]  = s[1] 
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
                # left
                else:
                    if s[0] > 0:
                         s_next[0]  = s[0]  - 1
                         s_next[1]  = s[1] 
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
            # Up
            elif a  == 3:
                # up
                if (0.0 <= temp) and (temp < 1-div):
                    if  s[1]  < self.Y - 1:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1]  + 1
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1] 
                # down
                else:
                    if  s[1]  > 0:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1]  - 1
                    else:
                         s_next[0]  = s[0] 
                         s_next[1]  = s[1]     
                         
            else :
                print("Error!")
                
        elif s == [1,0] :
            #to_8
            if a  == 0:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # to_8
                else:
                     s_next[0]  = s[0] + 1
                     s_next[1]  = s[1] + 3
            #to_7
            if a  == 1:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # to_7
                else:
                     s_next[0]  = s[0] + 1
                     s_next[1]  = s[1] + 2                     
            #to_6
            if a  == 2:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # to_6
                else:
                     s_next[0]  = s[0] + 1
                     s_next[1]  = s[1] + 1
            #to5         
            if a  == 3:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # to_5
                else:
                     s_next[0]  = s[0] + 1
                     s_next[1]  = s[1]
            #to_3
            if a  == 4:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # to_3
                else:
                     s_next[0]  = s[0] - 1
                     s_next[1]  = s[1] + 3
            #to_2         
            if a  == 5:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # to_2
                else:
                     s_next[0]  = s[0] - 1
                     s_next[1]  = s[1] + 2
            #to_1         
            if a  == 6:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # to_1
                else:
                     s_next[0]  = s[0] - 1
                     s_next[1]  = s[1] + 1
            #to_0         
            if a  == 7:
                # stay
                if (0.0 <= temp) and (temp < div):
                     s_next[0]  = s[0]
                     s_next[1]  = s[1]  
                # to_0
                else:
                     s_next[0]  = s[0] - 1
                     s_next[1]  = s[1] 
                     

        if (s_next == [1,3] or s_next == [1,2] or s_next == [1,1]):
            s_next = [1,0]
        return s_next
    
    #オートマトンへの入力になる
    # 3*2 ex2 (s[0] = 1 は廊下)
    def def_Label(self, s, a, s_next):
        if ( s == [1,0] and a == 7 and s_next == [0,0]):
            return ["a"]
        elif ( s == [1,0] and a == 0 and s_next == [2,3]):
            return ["b"]
        elif ( s_next == [0,2] or s_next == [0,3] or s_next == [2,0] or s_next == [2,1] ):
            return ["c"]
        else :
            return ["Empty"]     
        