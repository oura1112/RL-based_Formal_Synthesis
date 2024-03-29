from collections import defaultdict
import random
import itertools
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statistics import mean, median,variance,stdev

class Supervisor():
    
    def __init__(self, env, epsilon_inv = 1):
        self.actions_con = [0,1,2,3,4,5,6]
        self.actions_un = [7]
        self.actions = self.actions_con + self.actions_un
        self.event_probs = [1/(len(self.actions)+len(self.actions_un))] * (len(self.actions)+len(self.actions_un))
        self.epsilon_inv = epsilon_inv
        self.T = self.initialize_T(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.Q = self.initialize_Q(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
        self.Q_var = self.initialize_Q_var(self.actions_con, env.v, env.Q,  env.State_c, env.State_m)
        self.Nt = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
        self.prob_list = self.initialize_prob_list(self.actions, env.v, env.Q,  env.State_c, env.State_m)
        self.prob_mean = self.initialize_prob_mean(self.actions, env.v, env.Q,  env.State_c, env.State_m)
        self.Nsigma = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
        self.Ns = self.initialize_Ns(env.v, env.Q, env.State_c, env.State_m)
        self.eta = self.initialize_eta(self.actions, env.v, env.Q, env.State_c, env.State_m)
        self.prohibit_cost = self.initialize_prohibit_cost(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
        self.alpha = 0.1
        self.delta = 0.1
    
    def initialize_T(self, actions, v, Q_num, S_num_c, S_num_m) :
        T = [[[[[0] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return T
    
    def initialize_Q(self, actions, v, Q_num, S_num_c, S_num_m) :
        Q = [[[[[10] * 2**len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Q

    def initialize_Q_var(self, actions, v, Q_num, S_num_c, S_num_m) :
        Q_var = [[[[[0] * 2**len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Q_var
    
    def initialize_Nsigma(self, actions, v, Q_num, S_num_c, S_num_m) :
        Npi = [[[[[[[[[10] * 2**len(v)-1 for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)] for e in range(2**len(actions))] for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Npi

    def initialize_prob_list(self, actions, v, Q_num, S_num_c, S_num_m) :
        prob_list = [[[] for i in range(S_num_c)] for k in range(S_num_m)]
        
        return prob_list

    def initialize_prob_mean(self, actions, v, Q_num, S_num_c, S_num_m) :
        prob_mean = [[[] for i in range(S_num_c)] for k in range(S_num_m)]
        
        return prob_mean
    
    def initialize_eta(self, actions, v, Q_num, S_num_c, S_num_m) :
        Npi = [[[[[1/len(actions)] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Npi
    
    def initialize_prohibit_cost(self, actions, v, Q_num, S_num_c, S_num_m) :
        prohibit_cost = [0] * (2**len(actions))

        return prohibit_cost
    
    #def initialize_Nsigma(self, actions, v, Q_num, S_num_c, S_num_m) :
       # Nsigma = [[[[[[0] * len(actions) for m in  range(2**len(actions))] for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
       # return Nsigma
    
    def initialize_Ns(self, v, Q_num, S_num_c, S_num_m) :
        Ns = [[[[0] * (2**len(v)-1) for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return Ns

    def initialize_psi(self, actions, v, Q_num, S_num_c, S_num_m) :
        psi = [[[[[1000] * len(actions) for l in range(2**len(v)-1)] for i in range(len(Q_num))] for j in range(S_num_c)] for k in range(S_num_m)]
        
        return psi
    
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
        
        event_probs_all = np.random.dirichlet(list(self.flatten(self.Nsigma[s_m][s_c][q][v][pi])))        
        sum_probs_pi = 0

        #print(pi)
        for event, event_prob in enumerate(event_probs_all):
            #print("event_num : {}".format(event))
            #print("event_prob : {}".format(event_prob))
            if event in pi:
                event_probs.append(event_prob)
                sum_probs_pi += event_prob
            #else :
             #   print("Error!")
        event_probs = np.array(event_probs)        
        event_probs = event_probs / sum_probs_pi
        return event_probs

    def eta_to_probs(self,eta,pi):
        event_probs = []
        sum_probs_pi = 0

        for event, event_prob in enumerate(eta):
            #print("event_num : {}".format(event))
            #print("event_prob : {}".format(event_prob))
            if event in pi:
                event_probs.append(event_prob)
                sum_probs_pi += event_prob
            #else :
             #   print("Error!")
        event_probs = np.array(event_probs)        
        event_probs = event_probs / sum_probs_pi
        return event_probs 
    
    def event_occur(self, s_m, s_c, q, v, pi):
        
        if len(pi) != 0 :
            event_probs = self.Event_prob(pi)       
            event =np.random.choice(pi, 1, list(event_probs))
        else :
             event = -1
        #print(type(int(event)))
        #print(event_probs)
        return int(event), event_probs

    def mu_spi(self, s_m, s_c, q, v, episode, pi):

        var_mean = 0

        second_term = 0

        for s_m_next in range(env.State_m) :
            for s_c_next in range(env.State_c) :
                for q_next in range(len(env.Q)) :
                    for v_next in range(2**len(env.v)-1) :
                        Nsigma = np.array(self.Nsigma[s_m][s_c][q][v][pi][s_m_next][s_c_next][q_next][v_next])
                        sum_Nsigma = Nsigma.sum()

                        dir_mean = Nsigma/sum_Nsigma
                        dir_var = Nsigma*(sum_Nsigma - Nsigma) / sum_Nsigma**2 * (sum_Nsigma + 1)
                        var_mean = dir_var / dir_mean

                        var_mean += var_mean

                        for pi in 2**len(self.actions_con) :
                            if pi == np.argmax(self.Q[s_m_next][s_c_next][q_next][v_next]):
                                second_term += (1-1/episode) * dir_mean * self.Q_var[s_m_next][s_c_next][q_next][v_next]
                            else :
                                second_term += 1/episode * dir_mean * self.Q_var[s_m_next][s_c_next][q_next][v_next]


        mu_s_pi = 1/(1-0.99) * 10 * var_mean

        return mu_s_pi + (0.99)**2 * second_term                       
    
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
        gzi = np.random.normal(loc = 0, scale = 1, size = 1)

        if np.random.random() < 0.1 : #1/(self.epsilon_inv + episode_count) :
            return random.choice(control_pats)
        else:
            return np.argmax(np.array(self.Q[s_m][s_c][q][v]) + gzi*np.sqrt(self.Q_var[s_m][s_c][q][v]))
            
        #if np.random.random() < 0.95/Ns :
         #   return random.choice(control_pats)
       #else:
        #    return np.argmax(self.Q[s_m][s_c][q][v])

    import collections

    def flatten(self,l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
            
    def learn(self, env, episode_count=3000, step_count=10000, gamma=0.99, learning_rate_inv=1, session=1):
        
        total_reward_mean_m = []
        total_reward_mean_std = []
        total_cost_mean_m = []
        total_cost_mean_std = []
        
        total_reward_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        total_cost_mean_all = np.zeros(episode_count).reshape(1, episode_count)
        total_max_Q = np.zeros(episode_count).reshape(1, episode_count)
        
        for i in range(session) :
            count = 0
            total_reward = 0
            total_cost = 0
            total_reward_mean = []
            total_cost_mean = []
            max_Q_list = []
            self.actions_con = [0,1,2,3,4,5,6]
            self.actions_un = [7]
            
            self.actions = self.actions_con + self.actions_un
            
            self.event_probs = [np.random.uniform(2,5) for i in range(len(self.actions_con + self.actions_un))]
            self.event_probs = np.array(self.event_probs) / sum(self.event_probs)
            
            del self.T
            del self.Q
            del self.Q_var
            del self.Nt
            del self.Nsigma
            del self.prob_list
            del self.prob_mean
            del self.eta
            del self.Ns
            del self.prohibit_cost

            self.T = self.initialize_T(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Q = self.initialize_Q(self.actions_con, env.v, env.Q,  env.State_c, env.State_m)
            self.Q_var = self.initialize_Q_var(self.actions_con, env.v, env.Q,  env.State_c, env.State_m)
            self.Nt = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.prob_list = self.initialize_prob_list(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.prob_mean = self.initialize_prob_mean(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Nsigma = self.initialize_Nsigma(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            self.Ns = self.initialize_Ns(env.v, env.Q, env.State_c, env.State_m)
            self.eta = self.initialize_eta(self.actions, env.v, env.Q, env.State_c, env.State_m)
            self.prohibit_cost = self.initialize_prohibit_cost(self.actions_con, env.v, env.Q, env.State_c, env.State_m)
            #self.psi = self.initialize_T(self.actions, env.v, env.Q,  env.State_c, env.State_m)
            #print(self.Q)
            print("Session : {}\n".format(i))

            for control_pat in range(2**len(self.actions_con)):
                prohibit_pat = self.prohibit_pi(control_pat, self.actions_con)

                self.prohibit_cost[control_pat] = env.prohibit_cost_func(prohibit_pat)
            
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
                    event, event_probs_true = self.event_occur(s_m, s_c, q, v, pi)
                    
                    #Product MDP での状態遷移と報酬の取得
                    s_c_next, s_m_next, reward, prohibit_cost, automaton_transit, v_next = env.step(event, prohibit_pi) #環境内での状態も変化する
                    #print("current state:({0},{1},{2}); action:{3}; next state:({4},{5},{6}); reward:{7}".format(s,v,automaton_transit[0], a, s_next,v_next,automaton_transit[2], reward))
                    #print("current state:({0},{1},{2},{3}); event:{4}; next state:({5},{6},{7},{8}); reward:{9}; cost:{10}".format(s_c,s_m,v,automaton_transit[0], event, s_c_next,s_m_next,v_next,automaton_transit[2], reward, prohibit_cost))
                    #print(pi)
                    #print(prohibit_pi)
                    #self.prohibit_cost[s_m][s_c][automaton_transit[0]][v][control_pat] = prohibit_cost
                    
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
                        
                    #s_next = self.coord_to_snum(s_next)
                    
                    #価値関数やパラメータの更新 Q
                    #事象の推定生起回数の更新                    
                    #sum_eta = sum(self.eta[s_m][s_c][automaton_transit[0]][v][sigma_] for sigma_ in pi)
                    self.Nsigma[s_m][s_c][automaton_transit[0]][v][control_pat] += 1 

                    
                    #ambiuity setに基づくQの更新                
                    for control_pat_d in [control_pat]:
                        pi_d = self.action_to_pi(control_pat_d, self.actions_con)
                        pi_d = pi_d + self.actions_un
                        if event in pi_d :
                            ambiguity_set = []
                            estimated_set = []
                            estimated = 0
                            #count = 0

                            while(len(ambiguity_set) < 500) :
                                del self.prob_list
                                del self.prob_mean
                                self.prob_list = self.initialize_prob_list(self.actions, env.v, env.Q,  env.State_c, env.State_m)
                                self.prob_mean = self.initialize_prob_mean(self.actions, env.v, env.Q,  env.State_c, env.State_m)
                                for i in range(10) :
                                    #count += 1
                                    #print(s_c,s_m)
                                    #print(count)
                                    #etaのサンプリング
                                    diri = np.random.dirichlet(self.Nsigma[s_m][s_c][automaton_transit[0]][v][pi])
                                    self.prob_list[s_m][s_c][pi].append(diri)
                                    #print(len(self.prob_list[s_m][s_c]))
                                #self.prob_mean[s_m][s_c] = sum(self.prob_list[s_m][s_c])/10
                                self.prob_mean[s_m][s_c][pi] = np.array(self.Nsigma[s_m][s_c][automaton_transit[0]][v][pi]) / sum(self.Nsigma[s_m][s_c][automaton_transit[0]][v][pi])
                                #print(len(self.prob_mean[s_m][s_c]))
                                #print(len(self.prob_list[s_m][s_c][event]))
                                #confidence_level = np.sqrt(2*math.log(env.State_c*env.State_m*2**(env.State_m*env.State_c)/0.05)/ self.Nsigma[s_m][s_c][automaton_transit[0]][v][event])
                                
                                if e < episode_count/3 :
                                    confidence_level = 1
                                elif e > episode_count/3 and e < 2*episode_count/3 :
                                    confidence_level = 0.6
                                elif e > 2*episode_count/3 :
                                    confidence_level = 0.3
                                for prob in self.prob_list[s_m][s_c] :
                                    if (np.linalg.norm(prob - self.prob_mean[s_m][s_c]) < confidence_level ) :
                                        ambiguity_set.append(prob)
                            
                            
                            #print("estimated_event_probs : {}".format(event_probs))
                            #print("true_event_probs : {}".format(event_probs_true))

                            for eta in ambiguity_set :
                                event_probs = self.eta_to_probs(eta,pi_d)
                                for event_prob, sigma in zip(event_probs,pi_d) :
                                    #print(self.T[s_m][s_c][automaton_transit[0]][v][sigma])
                                    estimated += event_prob*self.T[s_m][s_c][automaton_transit[0]][v][sigma]
                                #print(estimated)
                                estimated_set.append(estimated)
                                estimated = 0
                            #print(len(estimated_set))
                            self.Q[s_m][s_c][automaton_transit[0]][v][control_pat_d] = self.prohibit_cost[control_pat_d] + min(estimated_set)
                            #print("comp")
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
                max_Q_list.append(self.Q[2][0][0][0][120])
                print("total_reward_mean = {}, total_cost_mean = {}".format(total_reward/step_count, total_cost/step_count))
                print("max Q value {}".format(self.Q[2][0][0][0][120]))
                if break_point == 1:
                    print("break:{0}, count:{1}".format(break_count,count))
                    print(break_pi)
                    print("break_state:[{0},{1}], break_event:{2}, break_next_state:[{3},{4}]".format(break_state[0],break_state[1],break_event,break_n_state[0],break_n_state[1]))
                total_reward = 0
                total_cost = 0
                count = 0
            
            total_reward_mean_all = np.append( total_reward_mean_all, [total_reward_mean], axis=0 )
            total_cost_mean_all = np.append( total_cost_mean_all, [total_cost_mean], axis=0 ) 
            total_max_Q = np.append( total_max_Q, [max_Q_list], axis=0 )   
            #print(total_reward_mean[:10])        
            #print(total_reward_mean[299000:])
            
        total_reward_mean_ = []
        total_reward_std_ = []
        total_cost_mean_ = []
        total_cost_std_ = []
        x_axis = []

        max_Q_list_ = []
        x_q_axis = []
        
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

        for i, max_q in enumerate(max_Q_list) :
            if i%10 == 0 :
                x_q_axis.append(i)
                max_Q_list_.append(max_q)
        
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

        print(np.array(self.event_probs))
        print(np.array(self.Nsigma[2][0][0][0])/sum(self.Nsigma[2][0][0][0]))
        
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

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(x_q_axis, max_Q_list_, color="blue")
        ax1.set_ylabel("Max Q value")
        ax1.set_xlabel("Episode Count")    

        plt.show()