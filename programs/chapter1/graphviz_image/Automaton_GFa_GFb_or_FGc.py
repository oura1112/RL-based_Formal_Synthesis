# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:55:07 2019

@author: oura
"""

from graphviz import Digraph

# formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
G = Digraph(format='pdf')
G.attr('node', shape='circle')

G.edge("q0","q0", label="T")

G.edge("q0","q1", label="a|b")

G.edge("q0","q2", label="c")

G.edge("q2","q2", label="c", color="purple")

G.edge("q1","q1",label="!a&!b", headport = "sw" , tailport = "sw")
G.edge("q1","q1", headlabel="a&!b", color="red", headport = "nw" , tailport = "nw")
G.edge("q1","q1",headlabel="!a&b", color="blue", headport = "ne" , tailport = "ne")
G.edge("q1","q1",label="a&b", color="purple", headport = "se" , tailport = "se")

G.edge("q2","q3", label="!c")

G.edge("q3","q3", label="T")

# print()するとdot形式で出力される
print(G)

# binary_tree.pngで保存
G.render('Automaton3')