# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:21:23 2019

@author: oura
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:57:14 2019

@author: oura
"""

from graphviz import Digraph

G = Digraph(format='pdf')
G.attr(rankdir='LR')

G.attr('node', shape='circle')
G.node("s0")
G.node("s1")
G.node("s2")

G.attr('node', shape='polygon')
G.node("a00", label="a0")
G.node("a01", label="a0")
G.node("a10", label="a1")
G.node("a12", label="a1")

G.edge("s0","a00")
G.edge("s0","a10")

G.edge("a00","s0",label="0.2")
G.edge("a00","s1",label="0.8")
G.edge("a10","s1",label="0.3")
G.edge("a10","s2",label="0.7")

G.edge("s1","a01")

G.edge("a01","s0",label="0.5")
G.edge("a01","s1",label="0.5")

G.edge("s2","a12")

G.edge("a12","s0",label="1")

# print()するとdot形式で出力される
print(G)

# pdfで保存
G.render('MDP_sample')