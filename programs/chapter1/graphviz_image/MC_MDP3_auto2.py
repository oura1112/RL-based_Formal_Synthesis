# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:29:54 2019

@author: oura
"""

from graphviz import Digraph

# formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
G = Digraph(format='pdf')
G.attr(rankdir='LR')
G.attr('node', shape='circle')

#Transient
G.edge("(s7,q0,(0,0)","(s7,q0,(0,0)",label="L/0.1")
G.edge("(s7,q0,(0,0)","(s4,q0,(0,0)",label="L/0.9")

#Recurrent
G.edge("(s4,q0,(0,0)","(s8,q0,(0,1)",label="to8/1")

G.edge("(s8,q0,(0,1)","(s8,q0,(0,1)",label="L/0.1")
G.edge("(s8,q0,(0,1)","(s4,q0,(0,1)",label="L/0.9")

G.edge("(s4,q0,(0,1)","(s0,q0,(0,0)",label="to0/1")

G.edge("(s0,q0,(0,0)","(s4,q0,(0,0)",label="R/0.9")
G.edge("(s0,q0,(0,0)","(s0,q0,(0,0)",label="R/0.1")


# print()するとdot形式で出力される
print(G)

# binary_tree.pngで保存
G.render('MC_MDP3_AUTO2')