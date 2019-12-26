# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:03:45 2019

@author: oura
"""

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

class Supervisor_():
    
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

#山崎さんの方法＋時相論理        
class Supervisor():
    
    def __init__(self, env, epsilon_inv = 1):
        self.actions_con = [0,1,2,3,4,5,6]
        self.actions_un = [7]
        self.actions = self.actions_con + self.actions_un
        self.event_probs = [1/(len(self.actions)+len(self.actions_un))] * (len(self.actions)+len(self.actions_un))
        self.epsilon_inv = epsilon_inv
        self.T = self.initialize_T(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.Q = self.initialize_Q(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
        self.Nt = self.initialize_Nsigma(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.Ns = self.initialize_Ns(env.v, env.Q, env.State_c, env.State_m)
        self.Nsigma = self.initialize_Nsigma(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.eta = self.initialize_eta(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.prohibit_cost = self.initialize_prohibit_cost(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
        self.alpha = 0.1
        self.delta = 0.1
        #without v
        #self.Q = [[[[10] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        #self.N = [[[[0] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        
        #print(self.Q)
        #self.Q[4][2][0] =10 
        #self.Q = np.array(self.Q)
        #print(self.Q)
    
    def initialize_T(self, actions, v, Q_num, S_num_c, S_num_m) :
        T = [[[[[0] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return T
    
    def initialize_Q(self, actions, v, Q_num, S_num_c, S_num_m) :
        Q = [[[[[0] * 2**len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Q
    
    def initialize_Nsigma(self, actions, v, Q_num, S_num_c, S_num_m) :
        Npi = [[[[[10] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Npi
    
    def initialize_eta(self, actions, v, Q_num, S_num_c, S_num_m) :
        Npi = [[[[[1/len(actions)] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Npi
    
    def initialize_prohibit_cost(self, actions, v, Q_num, S_num_c, S_num_m) :
        prohibit_cost = [[[[[0] * 2**len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return prohibit_cost
    
    #def initialize_Nsigma(self, actions, v, Q_num, S_num_c, S_num_m) :
       # Nsigma = [[[[[[0] * len(actions) for m in  range(2**len(actions))] for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
       # return Nsigma
    
    def initialize_Ns(self, v, Q_num, S_num_c, S_num_m) :
        Ns = [[[[0] * (2**len(v)-1) for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Ns
    
    def Event_prob(self, pi):
        
        event_prob = []
        
        for i in pi:
            event_prob.append(self.event_probs[i])
        
        #print(pi)
        #print(event_prob)
        return event_prob / sum(event_prob)
        #return [1/len(pi)] * len(pi)
        
    def estimate_event_Prob(self, s_m, s_c, q, v, pi):
        
        event_probs = []
        
        event_probs_all = np.random.dirichlet(self.Nsigma[s_m][s_c][q][v])        
        sum_probs_pi = sum(self.Nsigma[s_m][s_c][q][v][event] for event in pi)
        
        for event_prob, event in enumerate(event_probs_all):
            if event in pi:
                event_probs.append(event_prob)
                
        event_probs = event_probs / sum_probs_pi
        
        return event_probs
    
    def event_occur(self, s_m, s_c, q, v, pi):
        
        if len(pi) != 0 :
            event_probs = self.Event_prob(pi)       
            event =np.random.choice(pi, 1, list(event_probs))
        else :
             event = -1
        #print(type(int(event)))
        return int(event)
    
    def action_to_pi(self, control_pat, actions):

        a = bin(control_pat)
        a = a.lstrip("0b")
        a = a.rjust(len(actions), '0')
        
        pi = []
        for i in range(len(a)):
            if a[len(actions)-1-i] == '1':
                #print(i)
                pi.append(actions[i])        
        return pi
    
    def prohibit_pi(self, control_pat, actions):
        a = bin(control_pat)
        a = a.lstrip("0b")
        a = a.rjust(len(actions), '0')
        
        prohibit_pi = []  
        for i in range(len(a)):
            if a[len(actions)-1-i] == '0':
                #print(i)
                prohibit_pi.append(actions[i])
        return prohibit_pi
    
    def action_count(self, a):
        a = bin(a)
        count = 0
        
        for i in range(len(a)):
            if a[i] == '1':
                count += 1           
        return count
    
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
        
    def policy(self, s_m, s_c, q, v, Ns, control_pats, episode_count):
        #print(self.Q)
        if np.random.random() < 1/(self.epsilon_inv + episode_count) :
            return random.choice(control_pats)
        else:
            return np.argmax(self.Q[s_m][s_c][q][v])
            
        #if np.random.random() < 0.95/Ns :
         #   return random.choice(control_pats)
       #else:
        #    return np.argmax(self.Q[s_m][s_c][q][v])
            
    def learn(self, env, episode_count=20000, step_count=20000, gamma=0.99, learning_rate_inv=1, session=1):
        
        total_reward_mean_m = []
        total_reward_mean_std = []
        total_cost_mean_m = []
        total_cost_mean_std = []
        
        total_reward_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        total_cost_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        
        for i in range(session) :
            count = 0
            total_reward = 0
            total_cost = 0
            total_reward_mean = []
            total_cost_mean = []
            self.actions_con = [0,1,2,3,4,5,6]
            self.actions_un = [7]
            
            self.actions = self.actions_con + self.actions_un
            
            self.event_probs = [np.random.uniform(2,5) for i in range(len(self.actions_con + self.actions_un))]
            self.event_probs = np.array(self.event_probs) / sum(self.event_probs)
            
            del self.T
            del self.Q
            del self.Nt
            del self.Nsigma
            del self.eta
            del self.Ns
            del self.prohibit_cost
            self.T = self.initialize_T(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Q = self.initialize_Q(self.actions_con, env.v, env.Q,  env.State_c, env.State_m)
            self.Nt = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Nsigma = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Ns = self.initialize_Ns(env.v, env.Q, env.State_c, env.State_m)
            self.eta = self.initialize_eta(self.actions, env.v, env.Q, env.State_c, env.State_m)
            self.prohibit_cost = self.initialize_prohibit_cost(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
            #print(self.Q)
            print("Session : {}\n".format(i))
            
            for e in range(episode_count):
                print("epispode:{}\n".format(e))
                s_c, s_m, q, v = env.state_reset()
                #s = self.coord_to_snum(s)
                #print(v)
                break_point = 0
                for step in range(step_count):
                    #可制御事象と負可制御事象の集合
                    self.actions_con, self.actions_un = env.def_A(s_m, s_c)                    
                    self.Ns[s_m][s_c][q][v] += 1
                    
                    #可制御事象内のコントロールパターンの選択．control_patはリストではなく整数で表現．piは可制御事象内のコントロールパターン（集合）
                    control_pats = [i for i in range(2**len(self.actions_con))]
                    control_pat = self.policy(s_m, s_c, q, v, self.Ns[s_m][s_c][q][v], control_pats, e)
                    pi = self.action_to_pi(control_pat, self.actions_con)
                    #print("control")
                    #prohibit_cost_count = self.action_count(len(self.actions)) - self.action_count(control_pat)
                    #禁止パターンの取得
                    prohibit_pi = self.prohibit_pi(control_pat, self.actions_con)
                    #print("prohibit")
                    #システムに提示するコントロールパターン
                    pi.extend(self.actions_un)
                    #Product MDP内で確率的に生起する事象
                    event = self.event_occur(s_m, s_c, q, v, pi)
                    
                    #Product MDP での状態遷移と報酬の取得
                    s_c_next, s_m_next, reward, prohibit_cost, automaton_transit, v_next = env.step(event, prohibit_pi) #環境内での状態も変化する
                    #print("current state:({0},{1},{2}); action:{3}; next state:({4},{5},{6}); reward:{7}".format(s,v,automaton_transit[0], a, s_next,v_next,automaton_transit[2], reward))
                    #print("current state:({0},{1},{2},{3}); event:{4}; next state:({5},{6},{7},{8}); reward:{9}; cost:{10}".format(s_c,s_m,v,automaton_transit[0], event, s_c_next,s_m_next,v_next,automaton_transit[2], reward, prohibit_cost))
                    #print(pi)
                    #print(prohibit_pi)
                    self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat] = prohibit_cost
                    
                    if automaton_transit[2] == env.Q[-1]:
                        #print("break")
                        #print("current state:({0},{1},{2},{3}); event:{4}; next state:({5},{6},{7},{8}); reward:{9}; cost:{10}".format(s_c,s_m,v,automaton_transit[0], event, s_c_next,s_m_next,v_next,automaton_transit[2], reward, prohibit_cost))
                        #print("step_count:{}".format(count))
                        #print("\n")
                        #break
                        
                        if break_point == 0:
                            break_count = count
                            break_state = [s_c,s_m]
                            break_pi = pi
                            break_event = event
                            break_n_state = [s_c_next, s_m_next]
                        break_point = 1
                        reward = -100
                        """
                        #価値関数やパラメータの更新 T -> Q
                        gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                        estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                        self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                        self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                        
                        #事象の推定生起確率の更新                    
                        #sum_eta = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                        self.Nsigma[s_m][s_c][automaton_transit[0]][v][event] += 1 
                        sum_eta_pi = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                        
                        for sigma in pi :
                            if sigma == event :
                                self.eta[s_m][s_c][automaton_transit[0]][v][sigma] +=  0.1*(sum_eta_pi - self.eta[s_m][s_c][automaton_transit[0]][v][sigma])
                            else :
                                self.eta[s_m][s_c][automaton_transit[0]][v][sigma] = (1-0.1)*self.eta[s_m][s_c][automaton_transit[0]][v][sigma]
                            
                            
                        #今回生起した事象を含むコントロールパターンに対するQを更新する．
                        for control_pat_d in control_pats:
                            pi_d = self.action_to_pi(control_pat_d, self.actions)
                            pi_d = pi_d + self.actions_un
                            if (event in pi_d) and (self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] != 0):
                                estimated = 0
                                sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                                sum_eta_pid = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                                if sum_eta_pid == 0: print(pi_d)
                                for sigma in pi_d:
                                    #estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                                    estimated += (self.eta[s_m][s_c][automaton_transit[0]][v][sigma] / sum_eta_pid)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                                self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + estimated
                                #self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + min([self.T[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d])
                        #print(s, a, s_next, automaton_transit, reward)
                    
                        break
                        """
                    #s_next = self.coord_to_snum(s_next)
                    
                    #価値関数やパラメータの更新 T -> Q
                    gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                    estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                    self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                    self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                    
                    #事象の推定生起確率の更新                    
                    #sum_eta = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    self.Nsigma[s_m][s_c][automaton_transit[0]][v][event] += 1 
                    sum_eta_pi = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                
                    """
                    for sigma in pi :
                        if sigma == event :
                            self.eta[s_m][s_c][automaton_transit[0]][v][sigma] +=  0.1*(sum_eta_pi - self.eta[s_m][s_c][automaton_transit[0]][v][sigma])
                        else :
                            self.eta[s_m][s_c][automaton_transit[0]][v][sigma] = (1-0.1)*self.eta[s_m][s_c][automaton_transit[0]][v][sigma]
                    """    
                        
                    #今回生起した事象を含むコントロールパターンに対するQを更新する．
                    """
                    for control_pat_d in control_pats:
                        pi_d = self.action_to_pi(control_pat_d, self.actions)
                        pi_d = pi_d + self.actions_un
                        if (event in pi_d) and (self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] != 0):
                            estimated = 0
                            #sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                            #sum_eta_pid = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                            if sum_eta_pid == 0: print(pi_d)
                            for sigma in pi_d:
                                #estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                                estimated += (self.eta[s_m][s_c][automaton_transit[0]][v][sigma] / sum_eta_pid)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                            self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + estimated
                            #self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + min([self.T[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d])
                    #print(s, a, s_next, automaton_transit, reward)
                    """
                    
                    for control_pat_d in control_pats:
                        pi_d = self.action_to_pi(control_pat_d, self.actions_con)
                        pi_d = pi_d + self.actions_un
                        if (event in pi_d) and (self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] != 0):
                            estimated = 0
                            event_probs = self.estimate_event_Prob(s_m, s_c, automaton_transit[0], v, pi_d)
                            for event_prob, sigma in zip(event_probs,pi_d):
                                estimated += event_prob*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                            self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + estimated
                    
                    """
                    #価値関数の更新　Q -> T
                    estimated = 0
                    sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    for sigma in pi:
                        estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                    self.Q[s_m][s_c][automaton_transit[0]][v][control_pat] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat] + estimated
            
                    gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                    estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                    self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                    self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                    """
                    
                    if break_point == 1:
                        break
                    
                    #エージェントの持つ状態の更新
                    s_c = s_c_next
                    s_m = s_m_next
                    q = automaton_transit[2]
                    v = v_next
                    
                    total_reward += reward
                    total_cost += prohibit_cost
                    count += 1
                    #print("reward = {}, V = {}".format(reward, ))
                total_reward_mean.append(total_reward/step_count)
                total_cost_mean.append(total_cost/step_count)
                #print("total_reward_mean = {}, total_cost_mean = {}".format(total_reward/step_count, total_cost/step_count))
                if break_point == 1:
                    print("break:{0}, count:{1}".format(break_count,count))
                    print(break_pi)
                    print("break_state:[{0},{1}], break_event:{2}, break_next_state:[{3},{4}]".format(break_state[0],break_state[1],break_event,break_n_state[0],break_n_state[1]))
                total_reward = 0
                total_cost = 0
                count = 0
            
            total_reward_mean_all = np.append( total_reward_mean_all, [total_reward_mean], axis=0 )
            total_cost_mean_all = np.append( total_cost_mean_all, [total_cost_mean], axis=0 )    
            #print(total_reward_mean[:10])        
            #print(total_reward_mean[299000:])
            
        total_reward_mean_ = []
        total_reward_std_ = []
        total_cost_mean_ = []
        total_cost_std_ = []
        x_axis = []
        
        total_reward_mean_all = np.delete(total_reward_mean_all, 0, 0)
        total_cost_mean_all = np.delete(total_cost_mean_all, 0, 0)
        
        total_reward_mean_m = np.mean(total_reward_mean_all, axis=0)
        total_reward_mean_std = np.std(total_reward_mean_all, axis=0)
        total_cost_mean_m = np.mean(total_cost_mean_all, axis=0)
        total_cost_mean_std = np.std(total_cost_mean_all, axis=0)
            
        for i, (x_r_mean, x_r_std, x_c_mean, x_c_std) in enumerate(zip(total_reward_mean_m, total_reward_mean_std, total_cost_mean_m, total_cost_mean_std)):
            if i%10 == 0:
                x_axis.append(i)
                total_reward_mean_.append(x_r_mean)
                total_reward_std_.append(x_r_std)
                total_cost_mean_.append(x_c_mean)
                total_cost_std_.append(x_c_std)
        
        opt_a = []
        opt_a_ = []
        opt_a_q = []
       
        print(opt_a_q)
        for q in env.Q :
            for s_c in range(env.State_c):
                for s_m in range(env.State_m):
                    for v in  range(2**len(env.v) - 1):
                        opt_a.append(np.argmax(self.Q[s_m][s_c][q][v]))
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
        total_cost_mean_ = np.array(total_cost_mean_)
        total_cost_std_ = np.array(total_cost_std_)
                
        plt.style.use('default')
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        
        ax1.plot(x_axis, total_reward_mean_, color="blue")
        ax2.plot(x_axis, total_cost_mean_, color="red")
        #ax.errorbar(x_axis, total_reward_mean_, yerr=total_reward_std_, marker='o', capthick=1, capsize=10, lw=1)
        ax1.fill_between(x_axis, total_reward_mean_ - total_reward_std_, total_reward_mean_ + total_reward_std_, alpha = 0.2, color = "g")
        ax2.fill_between(x_axis, total_cost_mean_ - total_cost_std_, total_cost_mean_ + total_cost_std_, alpha = 0.2, color = "g")
        
        ax1.set_ylabel("Average Reward")
        ax2.set_xlabel("Episode Count")
        ax2.set_ylabel("Average Cost")

        plt.show()
        
        self.Q = np.array(self.Q)
        print(self.Q.shape)
        #print(self.Q)
        #print(env.Q)
        
        def print_(self):
            print(self.Q.shape)
            

#Q->Tの順で更新           
class Supervisor2():
    
    def __init__(self, env, epsilon_inv = 1):
        self.actions_con = [0,1,2,3,4,5,6,7,8,9,10,11]
        self.actions_un = [12]
        self.actions = self.actions_con + self.actions_un
        self.event_probs = [1/(len(self.actions)+len(self.actions_un))] * (len(self.actions)+len(self.actions_un))
        self.epsilon_inv = epsilon_inv
        self.T = self.initialize_T(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.Q = self.initialize_Q(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
        self.Nt = self.initialize_Nsigma(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.Ns = self.initialize_Ns(env.v, env.Q, env.State_c, env.State_m)
        self.Nsigma = self.initialize_Nsigma(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.eta = self.initialize_eta(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.prohibit_cost = self.initialize_prohibit_cost(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
        self.alpha = 0.1
        self.delta = 0.1
        #without v
        #self.Q = [[[[10] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        #self.N = [[[[0] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        
        #print(self.Q)
        #self.Q[4][2][0] =10 
        #self.Q = np.array(self.Q)
        #print(self.Q)
    
    def initialize_T(self, actions, v, Q_num, S_num_c, S_num_m) :
        T = [[[[[0] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return T
    
    def initialize_Q(self, actions, v, Q_num, S_num_c, S_num_m) :
        Q = [[[[[0] * 2**len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Q
    
    def initialize_Nsigma(self, actions, v, Q_num, S_num_c, S_num_m) :
        Npi = [[[[[10] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Npi
    
    def initialize_eta(self, actions, v, Q_num, S_num_c, S_num_m) :
        Npi = [[[[[1/len(actions)] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Npi
    
    def initialize_prohibit_cost(self, actions, v, Q_num, S_num_c, S_num_m) :
        prohibit_cost = [[[[[0] * 2**len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return prohibit_cost
    
    #def initialize_Nsigma(self, actions, v, Q_num, S_num_c, S_num_m) :
       # Nsigma = [[[[[[0] * len(actions) for m in  range(2**len(actions))] for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
       # return Nsigma
    
    def initialize_Ns(self, v, Q_num, S_num_c, S_num_m) :
        Ns = [[[[0] * (2**len(v)-1) for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Ns
    
    def Event_prob(self, pi):
        event_prob = []
        
        for i in pi:
            event_prob.append(self.event_probs[i])
        
        #print(pi)
        #print(event_prob)
        return event_prob / sum(event_prob)
        #print([1/len(pi)] * len(pi))
        #return [1/len(pi)] * len(pi)
        
    def estimate_event_Prob(self, s_m, s_c, q, v, pi):
        
        event_probs = []
        sum_eta = sum(self.eta[s_m][s_c][q][v][sigma_] for sigma_ in pi)
        
        for sigma in pi:
            event_probs.append(self.eta[s_m][s_c][q][v][sigma]/sum_eta)
            
        return event_probs
    
    def event_occur(self, s_m, s_c, q, v, pi):

        if len(pi) != 0 :
            event_probs = self.Event_prob(pi)       
            event =np.random.choice(pi, 1, list(event_probs))
        else :
            event = -1
        #print(type(int(event)))
        return int(event)
    
    def action_to_pi(self, control_pat, actions):

        a = bin(control_pat)
        a = a.lstrip("0b")
        a = a.rjust(len(actions), '0')
        
        pi = []
        for i in range(len(a)):
            if a[len(actions)-1-i] == '1':
                #print(i)
                pi.append(actions[i])        
        return pi
    
    def prohibit_pi(self, control_pat, actions):
        a = bin(control_pat)
        a = a.lstrip("0b")
        a = a.rjust(len(actions), '0')
        
        prohibit_pi = []  
        for i in range(len(a)):
            if a[len(actions)-1-i] == '0':
                #print(i)
                prohibit_pi.append(actions[i])
        return prohibit_pi
    
    def action_count(self, a):
        a = bin(a)
        count = 0
        
        for i in range(len(a)):
            if a[i] == '1':
                count += 1           
        return count
    
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
        
    def policy(self, s_m, s_c, q, v, Ns, control_pats, episode_count):
        #print(self.Q)
        if np.random.random() < 1/(self.epsilon_inv + episode_count) :
            return random.choice(control_pats)
        else:
            return np.argmax(self.Q[s_m][s_c][q][v])
            
        #if np.random.random() < 0.95/Ns :
         #   return random.choice(control_pats)
       #else:
        #    return np.argmax(self.Q[s_m][s_c][q][v])
            
    def learn(self, env, episode_count=10000, step_count=20000, gamma=0.99, learning_rate_inv=1, session=1):
        
        total_reward_mean_m = []
        total_reward_mean_std = []
        total_cost_mean_m = []
        total_cost_mean_std = []
        
        total_reward_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        total_cost_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        
        for i in range(session) :
            count = 0
            total_reward = 0
            total_cost = 0
            total_reward_mean = []
            total_cost_mean = []
            self.actions_con = [0,1,2,3,4,5,6]
            self.actions_un = [7]
            
            self.event_probs = [np.random.uniform(2,5) for i in range(len(self.actions_con + self.actions_un))]
            self.event_probs = np.array(self.event_probs) / sum(self.event_probs)
            
            self.actions = self.actions_con + self.actions_un
            
            del self.T
            del self.Q
            del self.Nt
            del self.Nsigma
            del self.eta
            del self.Ns
            del self.prohibit_cost
            self.T = self.initialize_T(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Q = self.initialize_Q(self.actions_con, env.v, env.Q,  env.State_c, env.State_m)
            self.Nt = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Nsigma = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Ns = self.initialize_Ns(env.v, env.Q, env.State_c, env.State_m)
            self.eta = self.initialize_eta(self.actions, env.v, env.Q, env.State_c, env.State_m)
            self.prohibit_cost = self.initialize_prohibit_cost(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
            #print(self.Q)
            print("Session : {}\n".format(i))
            
            for e in range(episode_count):
                print("epispode:{}\n".format(e))
                s_c, s_m, q, v = env.state_reset()
                #s = self.coord_to_snum(s)
                #print(v)
                break_point = 0
                for step in range(step_count):
                    #可制御事象と負可制御事象の集合
                    self.actions_con, self.actions_un = env.def_A(s_m, s_c)                    
                    self.Ns[s_m][s_c][q][v] += 1
                    
                    #可制御事象内のコントロールパターンの選択．control_patはリストではなく整数で表現．piは可制御事象内のコントロールパターン（集合）
                    control_pats = [i for i in range(2**len(self.actions_con))]
                    control_pat = self.policy(s_m, s_c, q, v, self.Ns[s_m][s_c][q][v], control_pats, e)
                    pi = self.action_to_pi(control_pat, self.actions_con)
                    #print("control")
                    #prohibit_cost_count = self.action_count(len(self.actions)) - self.action_count(control_pat)
                    #禁止パターンの取得
                    prohibit_pi = self.prohibit_pi(control_pat, self.actions_con)
                    #print("prohibit")
                    #システムに提示するコントロールパターン
                    pi.extend(self.actions_un)
                    #Product MDP内で確率的に生起する事象
                    event = self.event_occur(s_m, s_c, q, v, pi)
                    
                    #Product MDP での状態遷移と報酬の取得
                    s_c_next, s_m_next, reward, prohibit_cost, automaton_transit, v_next = env.step(event, prohibit_pi) #環境内での状態も変化する
                    #print("current state:({0},{1},{2}); action:{3}; next state:({4},{5},{6}); reward:{7}".format(s,v,automaton_transit[0], a, s_next,v_next,automaton_transit[2], reward))
                    #print("current state:({0},{1},{2},{3}); event:{4}; next state:({5},{6},{7},{8}); reward:{9}; cost:{10}".format(s_c,s_m,v,automaton_transit[0], event, s_c_next,s_m_next,v_next,automaton_transit[2], reward, prohibit_cost))
                    #print(pi)
                    #print(prohibit_pi)
                    self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat] = prohibit_cost
                    
                    if automaton_transit[2] == env.Q[-1]:
                        #print("break")
                        #print("current state:({0},{1},{2},{3}); event:{4}; next state:({5},{6},{7},{8}); reward:{9}; cost:{10}".format(s_c,s_m,v,automaton_transit[0], event, s_c_next,s_m_next,v_next,automaton_transit[2], reward, prohibit_cost))
                        #print("step_count:{}".format(count))
                        #print("\n")
                        #break
                        
                        if break_point == 0:
                            break_count = count
                            break_state = [s_c,s_m]
                            break_pi = pi
                            break_event = event
                            break_n_state = [s_c_next, s_m_next]
                        break_point = 1
                        reward = -10
                        """
                        #価値関数やパラメータの更新 T -> Q
                        gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                        estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                        self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                        self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                        """
                        
                        #事象の推定生起確率の更新                    
                        #sum_eta = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                        self.Nsigma[s_m][s_c][automaton_transit[0]][v][event] += 1 
                        sum_eta_pi = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                        
                        for sigma in pi :
                            if sigma == event :
                                self.eta[s_m][s_c][automaton_transit[0]][v][sigma] +=  0.1*(sum_eta_pi - self.eta[s_m][s_c][automaton_transit[0]][v][sigma])
                            else :
                                self.eta[s_m][s_c][automaton_transit[0]][v][sigma] = (1-0.1)*self.eta[s_m][s_c][automaton_transit[0]][v][sigma]
                            
                        """   
                        #今回生起した事象を含むコントロールパターンに対するQを更新する．
                        for control_pat_d in control_pats:
                            pi_d = self.action_to_pi(control_pat_d, self.actions)
                            pi_d = pi_d + self.actions_un
                            if (event in pi_d) and (self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] != 0):
                                estimated = 0
                                sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                                sum_eta_pid = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                                if sum_eta_pid == 0: print(pi_d)
                                for sigma in pi_d:
                                    #estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                                    estimated += (self.eta[s_m][s_c][automaton_transit[0]][v][sigma] / sum_eta_pid)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                                self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + estimated
                                #self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + min([self.T[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d])
                        #print(s, a, s_next, automaton_transit, reward)
                        """
                        
                        #価値関数の更新　Q -> T
                        estimated = 0
                        sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                        sum_eta_pi = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                        for sigma in pi:
                            estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                            #estimated += (self.eta[s_m][s_c][automaton_transit[0]][v][sigma] / sum_eta_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                        self.Q[s_m][s_c][automaton_transit[0]][v][control_pat] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat] + estimated
                
                        gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                        estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                        self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                        self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                        
                        #break
                        
                    #s_next = self.coord_to_snum(s_next)
                    """
                    #価値関数やパラメータの更新 T -> Q
                    gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                    estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                    self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                    self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                    """
                    #事象の推定生起確率の更新                    
                    #sum_eta = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    self.Nsigma[s_m][s_c][automaton_transit[0]][v][event] += 1 
                    sum_eta_pi = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    
                    for sigma in pi :
                        if sigma == event :
                            self.eta[s_m][s_c][automaton_transit[0]][v][sigma] +=  0.1*(sum_eta_pi - self.eta[s_m][s_c][automaton_transit[0]][v][sigma])
                        else :
                            self.eta[s_m][s_c][automaton_transit[0]][v][sigma] = (1-0.1)*self.eta[s_m][s_c][automaton_transit[0]][v][sigma]
                        
                    """    
                    #今回生起した事象を含むコントロールパターンに対するQを更新する．
                    for control_pat_d in control_pats:
                        pi_d = self.action_to_pi(control_pat_d, self.actions)
                        pi_d = pi_d + self.actions_un
                        if (event in pi_d) and (self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] != 0):
                            estimated = 0
                            sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                            sum_eta_pid = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                            if sum_eta_pid == 0: print(pi_d)
                            for sigma in pi_d:
                                estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                                #estimated += (self.eta[s_m][s_c][automaton_transit[0]][v][sigma] / sum_eta_pid)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                            self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + estimated
                            #self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + min([self.T[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d])
                    #print(s, a, s_next, automaton_transit, reward)
                    """
                    
                    
                    #価値関数の更新　Q -> T
                    estimated = 0
                    sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    sum_eta_pi = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    for sigma in pi:
                        estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                        #estimated += (self.eta[s_m][s_c][automaton_transit[0]][v][sigma] / sum_eta_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                    self.Q[s_m][s_c][automaton_transit[0]][v][control_pat] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat] + estimated
            
                    gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                    estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                    self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                    self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                    
                    
                    
                    #エージェントの持つ状態の更新
                    s_c = s_c_next
                    s_m = s_m_next
                    q = automaton_transit[2]
                    v = v_next
                    
                    total_reward += reward
                    total_cost += prohibit_cost
                    count += 1
                    #print("reward = {}, V = {}".format(reward, ))
                total_reward_mean.append(total_reward/step_count)
                total_cost_mean.append(total_cost/step_count)
                print("total_reward_mean = {}, total_cost_mean = {}".format(total_reward/step_count, total_cost/step_count))
                if break_point == 1:
                    print("break:{0}, count:{1}".format(break_count,count))
                    print(break_pi)
                    print("break_state:[{0},{1}], break_event:{2}, break_next_state:[{3},{4}]".format(break_state[0],break_state[1],break_event,break_n_state[0],break_n_state[1]))
                total_reward = 0
                total_cost = 0
                count = 0
            
            total_reward_mean_all = np.append( total_reward_mean_all, [total_reward_mean], axis=0 )
            total_cost_mean_all = np.append( total_cost_mean_all, [total_cost_mean], axis=0 )    
            #print(total_reward_mean[:10])        
            #print(total_reward_mean[299000:])
            
        total_reward_mean_ = []
        total_reward_std_ = []
        total_cost_mean_ = []
        total_cost_std_ = []
        x_axis = []
        
        total_reward_mean_all = np.delete(total_reward_mean_all, 0, 0)
        total_cost_mean_all = np.delete(total_cost_mean_all, 0, 0)
        
        total_reward_mean_m = np.mean(total_reward_mean_all, axis=0)
        total_reward_mean_std = np.std(total_reward_mean_all, axis=0)
        total_cost_mean_m = np.mean(total_cost_mean_all, axis=0)
        total_cost_mean_std = np.std(total_cost_mean_all, axis=0)
            
        for i, (x_r_mean, x_r_std, x_c_mean, x_c_std) in enumerate(zip(total_reward_mean_m, total_reward_mean_std, total_cost_mean_m, total_cost_mean_std)):
            if i%10 == 0:
                x_axis.append(i)
                total_reward_mean_.append(x_r_mean)
                total_reward_std_.append(x_r_std)
                total_cost_mean_.append(x_c_mean)
                total_cost_std_.append(x_c_std)
        
        opt_a = []
        opt_a_ = []
        opt_a_q = []
       
        print(opt_a_q)
        for q in env.Q :
            for s_c in range(env.State_c):
                for s_m in range(env.State_m):
                    for v in  range(2**len(env.v) - 1):
                        opt_a.append(np.argmax(self.Q[s_m][s_c][q][v]))
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
        total_cost_mean_ = np.array(total_cost_mean_)
        total_cost_std_ = np.array(total_cost_std_)
                
        plt.style.use('default')
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        
        ax1.plot(x_axis, total_reward_mean_, color="blue")
        ax2.plot(x_axis, total_cost_mean_, color="red")
        #ax.errorbar(x_axis, total_reward_mean_, yerr=total_reward_std_, marker='o', capthick=1, capsize=10, lw=1)
        ax1.fill_between(x_axis, total_reward_mean_ - total_reward_std_, total_reward_mean_ + total_reward_std_, alpha = 0.2, color = "g")
        ax2.fill_between(x_axis, total_cost_mean_ - total_cost_std_, total_cost_mean_ + total_cost_std_, alpha = 0.2, color = "g")
        
        ax1.set_ylabel("Average Reward")
        ax2.set_xlabel("Episode Count")
        ax2.set_ylabel("Average Cost")

        plt.show()
        
        self.Q = np.array(self.Q)
        print(self.Q.shape)
        #print(self.Q)
        #print(env.Q)
        
        def print_(self):
            print(self.Q.shape)

#禁止事象にコストを与えるのではなく許可事象に報酬を与える            
class Supervisor3():
    
    def __init__(self, env, epsilon_inv = 1):
        self.actions_con = [0,1,2,3,4,5,6]
        self.actions_un = [7]
        self.actions = self.actions_con + self.actions_un
        self.event_probs = [1/(len(self.actions)+len(self.actions_un))] * (len(self.actions)+len(self.actions_un))
        self.epsilon_inv = epsilon_inv
        self.T = self.initialize_T(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.Q = self.initialize_Q(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
        self.Nt = self.initialize_Nsigma(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.Ns = self.initialize_Ns(env.v, env.Q, env.State_c, env.State_m)
        self.Nsigma = self.initialize_Nsigma(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.eta = self.initialize_eta(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.prohibit_cost = self.initialize_prohibit_cost(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
        self.alpha = 0.1
        self.delta = 0.1
        #without v
        #self.Q = [[[[10] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        #self.N = [[[[0] * len(self.actions) for i in range(2)] for j in range(env.Y)] for k in range(env.X)]
        
        #print(self.Q)
        #self.Q[4][2][0] =10 
        #self.Q = np.array(self.Q)
        #print(self.Q)
    
    def initialize_T(self, actions, v, Q_num, S_num_c, S_num_m) :
        T = [[[[[0] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return T
    
    def initialize_Q(self, actions, v, Q_num, S_num_c, S_num_m) :
        Q = [[[[[0] * 2**len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Q
    
    def initialize_Nsigma(self, actions, v, Q_num, S_num_c, S_num_m) :
        Npi = [[[[[20] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Npi
    
    def initialize_eta(self, actions, v, Q_num, S_num_c, S_num_m) :
        Npi = [[[[[1/len(actions)] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Npi
    
    def initialize_prohibit_cost(self, actions, v, Q_num, S_num_c, S_num_m) :
        prohibit_cost = [[[[[0] * 2**len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return prohibit_cost
    
    #def initialize_Nsigma(self, actions, v, Q_num, S_num_c, S_num_m) :
       # Nsigma = [[[[[[0] * len(actions) for m in  range(2**len(actions))] for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
       # return Nsigma
    
    def initialize_Ns(self, v, Q_num, S_num_c, S_num_m) :
        Ns = [[[[0] * (2**len(v)-1) for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Ns
    
    def Event_prob(self, pi):
        
        return [1/len(pi)] * len(pi)
        
    def estimate_event_Prob(self, s_m, s_c, q, v, pi):
        
        event_probs = []
        sum_eta = sum(self.eta[s_m][s_c][q][v][sigma_] for sigma_ in pi)
        
        for sigma in pi:
            event_probs.append(self.eta[s_m][s_c][q][v][sigma]/sum_eta)
            
        return event_probs
    
    def event_occur(self, s_m, s_c, q, v, pi):

        event_probs = self.Event_prob(pi)       
        event =np.random.choice(pi, 1, event_probs)
        #print(type(int(event)))
        return int(event)
    
    def action_to_pi(self, control_pat, actions):

        a = bin(control_pat)
        a = a.lstrip("0b")
        a = a.rjust(len(actions), '0')
        
        pi = []
        for i in range(len(a)):
            if a[len(actions)-1-i] == '1':
                #print(i)
                pi.append(actions[i])        
        return pi
    
    def prohibit_pi(self, control_pat, actions):
        a = bin(control_pat)
        a = a.lstrip("0b")
        a = a.rjust(len(actions), '0')
        
        prohibit_pi = []  
        for i in range(len(a)):
            if a[len(actions)-1-i] == '0':
                #print(i)
                prohibit_pi.append(actions[i])
        return prohibit_pi
    
    def action_count(self, a):
        a = bin(a)
        count = 0
        
        for i in range(len(a)):
            if a[i] == '1':
                count += 1           
        return count
    
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
        
    def policy(self, s_m, s_c, q, v, Ns, control_pats, episode_count):
        #print(self.Q)
        if np.random.random() < 1/(self.epsilon_inv + 10*episode_count) :
            return random.choice(control_pats)
        else:
            return np.argmax(self.Q[s_m][s_c][q][v])
            
        #if np.random.random() < 0.95/Ns :
         #   return random.choice(control_pats)
       #else:
        #    return np.argmax(self.Q[s_m][s_c][q][v])
            
    def learn(self, env, episode_count=20000, step_count=200000, gamma=0.99, learning_rate_inv=1, session=1):
        
        total_reward_mean_m = []
        total_reward_mean_std = []
        total_cost_mean_m = []
        total_cost_mean_std = []
        
        total_reward_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        total_cost_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        
        for i in range(session) :
            count = 0
            total_reward = 0
            total_cost = 0
            total_reward_mean = []
            total_cost_mean = []
            self.actions_con = [0,1,2,3,4,5,6]
            self.actions_un = [7]
            
            self.actions = self.actions_con + self.actions_un
            
            del self.T
            del self.Q
            del self.Nt
            del self.Nsigma
            del self.eta
            del self.Ns
            del self.prohibit_cost
            self.T = self.initialize_T(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Q = self.initialize_Q(self.actions_con, env.v, env.Q,  env.State_c, env.State_m)
            self.Nt = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Nsigma = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Ns = self.initialize_Ns(env.v, env.Q, env.State_c, env.State_m)
            self.eta = self.initialize_eta(self.actions, env.v, env.Q, env.State_c, env.State_m)
            self.prohibit_cost = self.initialize_prohibit_cost(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
            #print(self.Q)
            print("Session : {}\n".format(i))
            
            for e in range(episode_count):
                print("epispode:{}\n".format(e))
                s_c, s_m, q, v = env.state_reset()
                #s = self.coord_to_snum(s)
                #print(v)
                break_point = 0
                for step in range(step_count):
                    #可制御事象と負可制御事象の集合
                    self.actions, self.actions_un = env.def_A(s_m, s_c)                    
                    self.Ns[s_m][s_c][q][v] += 1
                    
                    #可制御事象内のコントロールパターンの選択．control_patはリストではなく整数で表現．piは可制御事象内のコントロールパターン（集合）
                    control_pats = [i for i in range(2**len(self.actions_con))]
                    control_pat = self.policy(s_m, s_c, q, v, self.Ns[s_m][s_c][q][v], control_pats, e)
                    pi = self.action_to_pi(control_pat, self.actions_con)
                    #print("control")
                    #prohibit_cost_count = self.action_count(len(self.actions)) - self.action_count(control_pat)
                    #禁止パターンの取得
                    #prohibit_pi = self.prohibit_pi(control_pat, self.actions_con)
                    #print("prohibit")
                    #システムに提示するコントロールパターン
                    pi_ = pi
                    pi.extend(self.actions_un)
                    #Product MDP内で確率的に生起する事象
                    event = self.event_occur(s_m, s_c, q, v, pi)
                    
                    #Product MDP での状態遷移と報酬の取得
                    s_c_next, s_m_next, reward, prohibit_cost, automaton_transit, v_next = env.step(event, pi_) #環境内での状態も変化する
                    #print("current state:({0},{1},{2}); action:{3}; next state:({4},{5},{6}); reward:{7}".format(s,v,automaton_transit[0], a, s_next,v_next,automaton_transit[2], reward))
                    #print("current state:({0},{1},{2},{3}); event:{4}; next state:({5},{6},{7},{8}); reward:{9}; cost:{10}".format(s_c,s_m,v,automaton_transit[0], event, s_c_next,s_m_next,v_next,automaton_transit[2], reward, prohibit_cost))
                    #print(pi)
                    #print(prohibit_pi)
                    self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat] = prohibit_cost
                    
                    if automaton_transit[2] == env.Q[-1]:
                        #print("break")
                        #print("current state:({0},{1},{2},{3}); event:{4}; next state:({5},{6},{7},{8}); reward:{9}; cost:{10}".format(s_c,s_m,v,automaton_transit[0], event, s_c_next,s_m_next,v_next,automaton_transit[2], reward, prohibit_cost))
                        #print("step_count:{}".format(count))
                        #print("\n")
                        
                        if break_point == 0:
                            break_count = count
                            break_state = [s_c,s_m]
                            break_pi = pi
                            break_event = event
                            break_n_state = [s_c_next, s_m_next]
                        break_point = 1
                        
                        break
    
                    #s_next = self.coord_to_snum(s_next)
                    """
                    #価値関数やパラメータの更新 T -> Q
                    gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                    estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                    self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                    self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                    """
                    #事象の推定生起確率の更新                    
                    #sum_eta = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    self.Nsigma[s_m][s_c][automaton_transit[0]][v][event] += 1 
                    sum_eta_pi = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    
                    for sigma in pi :
                        if sigma == event :
                            self.eta[s_m][s_c][automaton_transit[0]][v][sigma] +=  0.1*(sum_eta_pi - self.eta[s_m][s_c][automaton_transit[0]][v][sigma])
                        else :
                            self.eta[s_m][s_c][automaton_transit[0]][v][sigma] = (1-0.1)*self.eta[s_m][s_c][automaton_transit[0]][v][sigma]
                        
                    """    
                    #今回生起した事象を含むコントロールパターンに対するQを更新する．
                    for control_pat_d in control_pats:
                        pi_d = self.action_to_pi(control_pat_d, self.actions)
                        pi_d = pi_d + self.actions_un
                        if (event in pi_d) and (self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] != 0):
                            estimated = 0
                            sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                            sum_eta_pid = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d)
                            if sum_eta_pid == 0: print(pi_d)
                            for sigma in pi_d:
                                estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                                #estimated += (self.eta[s_m][s_c][automaton_transit[0]][v][sigma] / sum_eta_pid)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                            self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + estimated
                            #self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat_d] + min([self.T[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi_d])
                    #print(s, a, s_next, automaton_transit, reward)
                    """
                    
                    
                    #価値関数の更新　Q -> T
                    estimated = 0
                    sum_sigma_pi = sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    sum_eta_pi = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    for sigma in pi:
                        estimated += (self.Nsigma[s_m][s_c][automaton_transit[0]][v][sigma] / sum_sigma_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                        #estimated += (self.eta[s_m][s_c][automaton_transit[0]][v][sigma] / sum_eta_pi)*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                    self.Q[s_m][s_c][automaton_transit[0]][v][control_pat] = self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat] + estimated
            
                    gain_T = reward + gamma * max(self.Q[s_m_next][s_c_next][automaton_transit[2]][v_next])
                    estimated_T = self.T[s_m][s_c][automaton_transit[0]][v][event]
                    self.Nt[s_m][s_c][automaton_transit[0]][v][event] += 1
                    self.T[s_m][s_c][automaton_transit[0]][v][event] += 1/(self.Nt[s_m][s_c][automaton_transit[0]][v][event])*(gain_T - estimated_T)
                    
                    
                    
                    #エージェントの持つ状態の更新
                    s_c = s_c_next
                    s_m = s_m_next
                    q = automaton_transit[2]
                    v = v_next
                    
                    total_reward += reward
                    total_cost += prohibit_cost
                    count += 1
                    #print("reward = {}, V = {}".format(reward, ))
                total_reward_mean.append(total_reward/step_count)
                total_cost_mean.append(total_cost/step_count)
                print("total_reward_mean = {}, total_cost_mean = {}".format(total_reward/step_count, total_cost/step_count))
                if break_point == 1:
                    print("break:{0}, count:{1}".format(break_count,count))
                    print(break_pi)
                    print("break_state:[{0},{1}], break_event:{2}, break_next_state:[{3},{4}]".format(break_state[0],break_state[1],break_event,break_n_state[0],break_n_state[1]))
                total_reward = 0
                total_cost = 0
                count = 0
            
            total_reward_mean_all = np.append( total_reward_mean_all, [total_reward_mean], axis=0 )
            total_cost_mean_all = np.append( total_cost_mean_all, [total_cost_mean], axis=0 )    
            #print(total_reward_mean[:10])        
            #print(total_reward_mean[299000:])
            
        total_reward_mean_ = []
        total_reward_std_ = []
        total_cost_mean_ = []
        total_cost_std_ = []
        x_axis = []
        
        total_reward_mean_all = np.delete(total_reward_mean_all, 0, 0)
        total_cost_mean_all = np.delete(total_cost_mean_all, 0, 0)
        
        total_reward_mean_m = np.mean(total_reward_mean_all, axis=0)
        total_reward_mean_std = np.std(total_reward_mean_all, axis=0)
        total_cost_mean_m = np.mean(total_cost_mean_all, axis=0)
        total_cost_mean_std = np.std(total_cost_mean_all, axis=0)
            
        for i, (x_r_mean, x_r_std, x_c_mean, x_c_std) in enumerate(zip(total_reward_mean_m, total_reward_mean_std, total_cost_mean_m, total_cost_mean_std)):
            if i%10 == 0:
                x_axis.append(i)
                total_reward_mean_.append(x_r_mean)
                total_reward_std_.append(x_r_std)
                total_cost_mean_.append(x_c_mean)
                total_cost_std_.append(x_c_std)
        
        opt_a = []
        opt_a_ = []
        opt_a_q = []
       
        print(opt_a_q)
        for q in env.Q :
            for s_c in range(env.State_c):
                for s_m in range(env.State_m):
                    for v in  range(2**len(env.v) - 1):
                        opt_a.append(np.argmax(self.Q[s_m][s_c][q][v]))
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
        total_cost_mean_ = np.array(total_cost_mean_)
        total_cost_std_ = np.array(total_cost_std_)
                
        plt.style.use('default')
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        
        ax1.plot(x_axis, total_reward_mean_, color="blue")
        ax2.plot(x_axis, total_cost_mean_, color="red")
        #ax.errorbar(x_axis, total_reward_mean_, yerr=total_reward_std_, marker='o', capthick=1, capsize=10, lw=1)
        ax1.fill_between(x_axis, total_reward_mean_ - total_reward_std_, total_reward_mean_ + total_reward_std_, alpha = 0.2, color = "g")
        ax2.fill_between(x_axis, total_cost_mean_ - total_cost_std_, total_cost_mean_ + total_cost_std_, alpha = 0.2, color = "g")
        
        ax1.set_ylabel("Average Reward")
        ax2.set_xlabel("Episode Count")
        ax2.set_ylabel("Average Cost")

        plt.show()
        
        self.Q = np.array(self.Q)
        print(self.Q.shape)
        #print(self.Q)
        #print(env.Q)
        
        def print_(self):
            print(self.Q.shape)
            
                        