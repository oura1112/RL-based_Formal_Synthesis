# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:21:30 2019

@author: oura
"""

from collections import defaultdict
import random
import itertools
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statistics import mean, median,variance,stdev

class Agent():
    
    def __init__(self, env, epsilon_inv = 1):
        self.actions = env.def_A()
        self.epsilon_inv = epsilon_inv
        self.Q = [[[[[2] * len(self.actions) for l in range(2**len(env.v)-1)] for i in range(len(env.Q))] for j in range(env.Y)] for k in range(env.X)]
        self.N = [[[[[0] * len(self.actions) for l in range(2**len(env.v)-1)] for i in range(len(env.Q))] for j in range(env.Y)] for k in range(env.X)]
        #without v
        #self.Q = [[[[10] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        #self.N = [[[[0] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        
        #print(self.Q)
        #self.Q[4][2][0] =10 
        #self.Q = np.array(self.Q)
        
    def policy(self, s, q, v, actions, episode_count):
        #print(self.Q)
        if np.random.random() < 1/(self.epsilon_inv + episode_count) :
            return random.choice(self.actions)
        else:
            return np.argmax(self.Q[s[0]][s[1]][q][v])
            #without v
            #return np.argmax(self.Q[s[0]][s[1]][q])
            
    def learn(self, env, episode_count=3000, step_count=10000, gamma=0.9, learning_rate_inv=1):
        #print(self.Q)
        count = 0
        total_reward = 0
        total_reward_mean = []
        for e in range(episode_count):
            print("epispode:{}\n".format(e))
            s, q, v = env.state_reset()
            #print(v)
            for step in range(step_count):
                a = self.policy(s, q, v, self.actions, e)
                s_next, reward, automaton_transit, v_next = env.step(a) #環境内での状態も変化する
                #print(s, s_next, automaton_transit, reward)
                if automaton_transit[2] == env.Q[-1]:
                    print("break")
                    #print(count)
                    #print("\n")
                    break

                
                gain = reward + gamma * max(self.Q[s_next[0]][s_next[1]][automaton_transit[2]][v_next])
                estimated = self.Q[s[0]][s[1]][automaton_transit[0]][v][a]
                self.N[s[0]][s[1]][automaton_transit[0]][v][a] += 1
                self.Q[s[0]][s[1]][automaton_transit[0]][v][a] += 1/(learning_rate_inv + self.N[s[0]][s[1]][automaton_transit[0]][v][a]) * (gain - estimated)
                #print(s, a, s_next, automaton_transit, reward)
                
                """
                #without v
                gain = reward + gamma * max(self.Q[s_next[0]][s_next[1]][automaton_transit[2]])
                estimated = self.Q[s[0]][s[1]][automaton_transit[0]][a]
                self.N[s[0]][s[1]][automaton_transit[0]][a] += 1
                self.Q[s[0]][s[1]][automaton_transit[0]][a] += ( 1/(learning_rate_inv + self.N[s[0]][s[1]][automaton_transit[0]][a]) ) * (gain - estimated)
                """
                
                #エージェントの持つ状態の更新
                s = s_next
                q = automaton_transit[2]
                v = v_next
                
                total_reward += reward
                count += 1
                #print("reward = {}, V = {}".format(reward, v))
            total_reward_mean.append(total_reward/step_count)
            total_reward = 0
            count = 0
            
        #print(total_reward_mean[:10])        
        #print(total_reward_mean[299000:])
        
        total_reward_mean_ = []
        x_axis = []
        
        for i, x in enumerate(total_reward_mean):
            if i%10 == 0:
                x_axis.append(i)
                total_reward_mean_.append(x)
        
        opt_a = []
        opt_a_ = []
        opt_a_q = []
        
        for q in env.Q :
            for x in range(env.X):
                for y in range(env.Y):
                    for v in  range(2**len(env.v) - 1):
                        opt_a.append(np.argmax(self.Q[x][y][q][v]))
                        #print(opt_a)
                    opt_a_.append(opt_a)
                    opt_a = []
            
            opt_a_q.append(opt_a_)
            opt_a_ = []
                    
        opt_a_q = np.array(opt_a_q)
        print(opt_a_q)
        #print(2**len(env.v))
                
        plt.plot(x_axis, total_reward_mean_) 
        plt.show()
        
        #print(self.Q.shape)
        #print(self.Q)
        #print(env.Q)
        
class Agent_Grid3():
    
    def __init__(self, env, epsilon_inv = 1):
        self.actions = [0,1,2,3]
        self.epsilon_inv = epsilon_inv
        self.Q = self.initialize_Q(self.actions, env.v, env.Q, env.State)
        self.N = self.initialize_N(self.actions, env.v, env.Q, env.State)
        self.Ns = self.initialize_Ns(env.v, env.Q, env.State)
        #without v
        #self.Q = [[[[10] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        #self.N = [[[[0] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        
        #print(self.Q)
        #self.Q[4][2][0] =10 
        #self.Q = np.array(self.Q)
        #print(self.Q)
        
    def initialize_Q(self, actions, v, Q_num, S_num) :
        Q = [[[[2] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num - 1)]
        Q_ = [[[2] * (len(actions) + 4) for l in range(2**len(v)-1)] for i in range(len(Q_num))]
        Q.insert(4,Q_)
        
        return Q
    
    def initialize_N(self, actions, v, Q_num, S_num) :
        N = [[[[0] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num - 1)]
        N_ = [[[0] * (len(actions) + 4) for l in range(2**len(v)-1)] for i in range(len(Q_num))]
        N.insert(4,N_)
        
        return N
    
    def initialize_Ns(self, v, Q_num, S_num) :
        Ns = [[[0] * (2**len(v)-1) for i in range(len(Q_num))] for j in range(S_num - 1)]
        Ns_ = [[0] * (2**len(v)-1) for i in range(len(Q_num))]
        Ns.insert(4,Ns_)
        
        return Ns
     
    def coord_to_snum (self, s):
        if s == [0,0]:
            return 0
        elif s == [0,1]:
            return 1
        elif s == [0,2]:
            return 2
        elif s == [0,3]:
            return 3
        elif s == [1,0]:
            return 4
        elif s == [2,0]:
            return 5
        elif s == [2,1]:
            return 6
        elif s == [2,2]:
            return 7
        elif s == [2,3]:
            return 8
        
    def policy(self, s, q, v, Ns, actions, episode_count):
        #print(self.Q)
        #if np.random.random() < 1/(self.epsilon_inv + 15*episode_count) :
         #   return random.choice(self.actions)
        #else:
         #   return np.argmax(self.Q[s][q][v])
            
        if np.random.random() < 0.95/Ns :
            return random.choice(actions)
        else:
            return np.argmax(self.Q[s][q][v])
            
    def learn(self, env, episode_count=1001, step_count=10000, gamma=0.95, learning_rate_inv=1, session=20):
        
        total_reward_mean_m = []
        total_reward_mean_std = []
        
        total_reward_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        
        for i in range(session) :
            count = 0
            total_reward = 0
            total_reward_mean = []
            self.actions = [0,1,2,3]
            
            del self.Q
            del self.N
            del self.Ns
            self.Q = self.initialize_Q(self.actions, env.Q, env.v, env.State)
            self.N = self.initialize_N(self.actions, env.Q, env.v, env.State)
            self.Ns = self.initialize_Ns(env.v, env.Q, env.State)
            #print(self.Q)
            print("Session : {}\n".format(i))
            
            for e in range(episode_count):
                print("epispode:{}\n".format(e))
                s, q, v = env.state_reset()
                s = self.coord_to_snum(s)
                #print(v)
                for step in range(step_count):
                    self.actions = env.def_A(s)                    
                    self.Ns[s][q][v] += 1
                    a = self.policy(s, q, v, self.Ns[s][q][v], self.actions, e)
                    s_next, reward, automaton_transit, v_next = env.step(a) #環境内での状態も変化する
                    #print("current state:({0},{1},{2}); action:{3}; next state:({4},{5},{6}); reward:{7}".format(s,v,automaton_transit[0], a, s_next,v_next,automaton_transit[2], reward))
                    
                    if automaton_transit[2] == env.Q[-1]:
                        print("break")
                        #print(count)
                        #print("\n")
                        break
    
                    s_next = self.coord_to_snum(s_next)
                    
                    gain = reward + gamma * max(self.Q[s_next][automaton_transit[2]][v_next])
                    estimated = self.Q[s][automaton_transit[0]][v][a]
                    self.N[s][automaton_transit[0]][v][a] += 1
                    self.Q[s][automaton_transit[0]][v][a] += 1/(learning_rate_inv + self.N[s][automaton_transit[0]][v][a]) * (gain - estimated)
                    #print(s, a, s_next, automaton_transit, reward)
                    
                    """
                    #without v
                    gain = reward + gamma * max(self.Q[s_next[0]][s_next[1]][automaton_transit[2]])
                    estimated = self.Q[s[0]][s[1]][automaton_transit[0]][a]
                    self.N[s[0]][s[1]][automaton_transit[0]][a] += 1
                    self.Q[s[0]][s[1]][automaton_transit[0]][a] += ( 1/(learning_rate_inv + self.N[s[0]][s[1]][automaton_transit[0]][a]) ) * (gain - estimated)
                    """
                    
                    
                    
                    #エージェントの持つ状態の更新
                    s = s_next
                    q = automaton_transit[2]
                    v = v_next
                    
                    total_reward += reward
                    count += 1
                    #print("reward = {}, V = {}".format(reward, v))
                total_reward_mean.append(total_reward/step_count)
                total_reward = 0
                count = 0
            
            total_reward_mean_all = np.append( total_reward_mean_all, [total_reward_mean], axis=0 )
                
            #print(total_reward_mean[:10])        
            #print(total_reward_mean[299000:])
            
        total_reward_mean_ = []
        total_reward_std_ = []
        x_axis = []
        
        total_reward_mean_all = np.delete(total_reward_mean_all, 0, 0)
        
        total_reward_mean_m = np.mean(total_reward_mean_all, axis=0)
        total_reward_mean_std = np.std(total_reward_mean_all, axis=0)
            
        for i, (x_mean, x_std) in enumerate(zip(total_reward_mean_m, total_reward_mean_std)):
            if i%100 == 0:
                x_axis.append(i)
                total_reward_mean_.append(x_mean)
                total_reward_std_.append(x_std)
        
        opt_a = []
        opt_a_ = []
        opt_a_q = []
        
        print(opt_a_q)
        for q in env.Q :
            for s in range(env.State):
                for v in  range(2**len(env.v) - 1):
                    opt_a.append(np.argmax(self.Q[s][q][v]))
                    #print(opt_a)
                opt_a_.append(opt_a)
                opt_a = []
            
            opt_a_q.append(opt_a_)
            opt_a_ = []
                    
        opt_a_q = np.array(opt_a_q)
        print(opt_a_q)
        #print(2**len(env.v))
        
        total_reward_mean_ = np.array(total_reward_mean_)
        total_reward_std_ = np.array(total_reward_std_)
                
        plt.style.use('default')
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        ax.plot(x_axis, total_reward_mean_, color="blue")
        #ax.errorbar(x_axis, total_reward_mean_, yerr=total_reward_std_, marker='o', capthick=1, capsize=10, lw=1)
        ax.fill_between(x_axis, total_reward_mean_ - total_reward_std_, total_reward_mean_ + total_reward_std_, alpha = 0.2, color = "g")
        
        ax.set_xlabel("Episode Count")
        ax.set_ylabel("Average Reward")
        
        plt.ylim(0,1)

        plt.show()
        
        self.Q = np.array(self.Q)
        print(self.Q.shape)
        print(self.Q)
        #print(env.Q)
        
        def print_(self):
            print(self.Q.shape)

class Agent_Abate():
    
    def __init__(self, env, epsilon_inv = 1):
        self.actions = env.def_A()
        self.epsilon_inv = epsilon_inv
        self.Q = [[[[2] * len(self.actions) for i in range(len(env.Q))] for j in range(env.Y)] for k in range(env.X)]
        self.N = [[[[0] * len(self.actions) for i in range(len(env.Q))] for j in range(env.Y)] for k in range(env.X)]
        
        #print(self.Q)
        #self.Q[4][2][0] =10 
        #self.Q = np.array(self.Q)
        
        
    def policy(self, s, q, actions, episode_count):
        #print(self.Q)
        if np.random.random() < 1/(self.epsilon_inv + episode_count) :
            return random.choice(self.actions)
        else:
            return np.argmax(self.Q[s[0]][s[1]][q])
            
    def learn(self, env, episode_count=3000, step_count=10000, gamma=0.9, learning_rate_inv=1):
        #print(self.Q)
        count = 0
        total_reward = 0
        total_reward_mean = []
        for e in range(episode_count):
            print("epispode:{}\n".format(e))
            s, q = env.state_reset()
            #print(v)
            for step in range(step_count):
                a = self.policy(s, q, self.actions, e)
                s_next, reward, automaton_transit = env.step(a) #環境内での状態も変化する
                #print(s, a, s_next, automaton_transit, reward)
                if automaton_transit[2] == env.Q[-1]:
                    print("break")
                    #print(count)
                    #print("\n")
                    break

                gain = reward + gamma * max(self.Q[s_next[0]][s_next[1]][automaton_transit[2]])
                estimated = self.Q[s[0]][s[1]][automaton_transit[0]][a]
                self.N[s[0]][s[1]][automaton_transit[0]][a] += 1
                self.Q[s[0]][s[1]][automaton_transit[0]][a] += ( 1/(learning_rate_inv + self.N[s[0]][s[1]][automaton_transit[0]][a]) ) * (gain - estimated)
                
                
                #エージェントの持つ状態の更新
                s = s_next
                q = automaton_transit[2]
                
                total_reward += reward
                count += 1
                #print("reward = {}, V = {}".format(reward, v))
            total_reward_mean.append(total_reward/step_count)
            total_reward = 0
            count = 0
            
        #print(total_reward_mean[:10])        
        #print(total_reward_mean[299000:])
        
        total_reward_mean_ = []
        x_axis = []
        
        for i, x in enumerate(total_reward_mean):
            if i%10 == 0:
                x_axis.append(i)
                total_reward_mean_.append(x)
                
        #print(self.Q)
        
        opt_a = []
        opt_a_q = []
        
        for q in env.Q :
            for x in range(env.X):
                for y in range(env.Y):
                    opt_a.append(np.argmax(self.Q[x][y][q]))
            
            opt_a_q.append(opt_a)
            opt_a = []

        opt_a_q = np.array(opt_a_q)
        print(opt_a_q)
                
        plt.plot(x_axis, total_reward_mean_) 
        plt.show()
        
        #print(self.Q.shape)
        #print(self.Q)
        
class Agent_Abate_Grid3():
    
    def __init__(self, env, epsilon_inv = 1):
        self.actions = [0,1,2,3]
        self.epsilon_inv = epsilon_inv
        self.Q = self.initialize_Q(self.actions, env.Q, env.State)
        self.N = self.initialize_N(self.actions, env.Q, env.State)
        self.Ns = self.initialize_Ns(env.Q, env.State)
        
        #print(self.Q)
        #self.Q[4][2][0] =10 
        #self.Q = np.array(self.Q)
        
        Q_ = [[2] * (len(self.actions) + 4) for i in range(len(env.Q))]
        N_ = [[0] * (len(self.actions) + 4) for i in range(len(env.Q))]

        self.Q.insert(4,Q_)
        self.N.insert(4,N_)
        #without v
        #self.Q = [[[[10] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        #self.N = [[[[0] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        
        #print(self.Q)
        #self.Q[4][2][0] =10 
        self.Q = np.array(self.Q)
        #print(self.Q)
        
    def initialize_Q(self, actions, Q_num, S_num) :
        Q = [[[2] * len(actions) for i in range(len(Q_num))] for j in range(S_num - 1)]
        Q_ = [[2] * (len(actions) + 4) for i in range(len(Q_num))]
        Q.insert(4,Q_)
        
        return Q
    
    def initialize_N(self, actions, Q_num, S_num) :
        N = [[[0] * len(actions) for i in range(len(Q_num))] for j in range(S_num - 1)]
        N_ = [[0] * (len(actions) + 4) for i in range(len(Q_num))]
        N.insert(4,N_)
        
        return N
    
    def initialize_Ns(self, Q_num, S_num) :
        Ns = [[0] * len(Q_num) for j in range(S_num - 1)]
        Ns_ = [0] * len(Q_num)
        Ns.insert(4,Ns_)
        
        return Ns
     
    def coord_to_snum (self, s):
        if s == [0,0]:
            return 0
        elif s == [0,1]:
            return 1
        elif s == [0,2]:
            return 2
        elif s == [0,3]:
            return 3
        elif s == [1,0]:
            return 4
        elif s == [2,0]:
            return 5
        elif s == [2,1]:
            return 6
        elif s == [2,2]:
            return 7
        elif s == [2,3]:
            return 8
        
    def policy(self, s, q, Ns, actions, episode_count):
        #print(self.Q)
        #if np.random.random() < 1/(self.epsilon_inv + 15*episode_count) :
         #   return random.choice(self.actions)
        #else:
         #   return np.argmax(self.Q[s][q])
            
        if np.random.random() < 0.95/Ns :
            return random.choice(actions)
        else:
            return np.argmax(self.Q[s][q])
    def learn(self, env, episode_count=1001, step_count=10000, gamma=0.95, learning_rate_inv=1, session=20):
        #print(self.Q)
        total_reward_mean_m = []
        total_reward_mean_std = []
        
        total_reward_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        
        for i in range(session) :
            count = 0
            total_reward = 0
            total_reward_mean = []
            self.actions = [0,1,2,3]
            
            del self.Q
            del self.N
            del self.Ns
            self.Q = self.initialize_Q(self.actions, env.Q, env.State)
            self.N = self.initialize_N(self.actions, env.Q, env.State)
            self.Ns = self.initialize_Ns(env.Q, env.State)
            print("Session : {}\n".format(i))
            
            for e in range(episode_count):
                print("epispode:{}\n".format(e))
                s, q = env.state_reset()
                s = self.coord_to_snum(s)
                #print(v)
                for step in range(step_count):
                    self.actions = env.def_A(s)                    
                    self.Ns[s][q] += 1
                    a = self.policy(s, q, self.Ns[s][q], self.actions, e)
                    s_next, reward, automaton_transit = env.step(a) #環境内での状態も変化する
                    #print(s, a, s_next, automaton_transit, reward)
                    if automaton_transit[2] == env.Q[-1]:
                        print("break")
                        #print(count)
                        #print("\n")
                        break
                    
                    s_next = self.coord_to_snum(s_next)
    
                    gain = reward + gamma * max(self.Q[s_next][automaton_transit[2]])
                    estimated = self.Q[s][automaton_transit[0]][a]
                    self.N[s][automaton_transit[0]][a] += 1
                    self.Q[s][automaton_transit[0]][a] += ( 1/(learning_rate_inv + self.N[s][automaton_transit[0]][a]) ) * (gain - estimated)
                    
                    
                    #エージェントの持つ状態の更新
                    s = s_next
                    q = automaton_transit[2]
                    
                    total_reward += reward
                    count += 1
                    #print("reward = {}, V = {}".format(reward, v))
                total_reward_mean.append(total_reward/step_count)    
                total_reward = 0
                count = 0
                
            total_reward_mean_all = np.append( total_reward_mean_all, [total_reward_mean], axis=0 )
            self.Q = self.initialize_Q(self.actions, env.Q, env.State)
            self.N = self.initialize_N(self.actions, env.Q, env.State)
                
            #print(total_reward_mean[:10])        
            #print(total_reward_mean[299000:])
            
        total_reward_mean_ = []
        total_reward_std_ = []
        x_axis = []
        
        total_reward_mean_all = np.delete(total_reward_mean_all, 0, 0)
        
        total_reward_mean_m = np.mean(total_reward_mean_all, axis=0)
        total_reward_mean_std = np.std(total_reward_mean_all, axis=0)
            
        for i, (x_mean, x_std) in enumerate(zip(total_reward_mean_m, total_reward_mean_std)):
            if i%100 == 0:
                x_axis.append(i)
                total_reward_mean_.append(x_mean)
                total_reward_std_.append(x_std)
                
        #print(self.Q)
        opt_a = []
        opt_a_q = []
        
        for q in env.Q :
            for s in range(env.State):
                opt_a.append(np.argmax(self.Q[s][q]))
            
            opt_a_q.append(opt_a)
            opt_a = []
            
        total_reward_mean_ = np.array(total_reward_mean_)
        total_reward_std_ = np.array(total_reward_std_)

        plt.style.use('default')
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        ax.plot(x_axis, total_reward_mean_, color="blue")
        #ax.errorbar(x_axis, total_reward_mean_, yerr=total_reward_std_, marker='o', capthick=1, capsize=10, lw=1)
        ax.fill_between(x_axis, total_reward_mean_ - total_reward_std_, total_reward_mean_ + total_reward_std_, alpha = 0.2, color = "g")
        
        ax.set_xlabel("Episode Count")
        ax.set_ylabel("Average Reward")
        
        plt.ylim(0,1)

        plt.show()