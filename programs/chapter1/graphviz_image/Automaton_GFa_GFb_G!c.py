# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:57:14 2019

@author: oura
"""

from graphviz import Digraph

G = Digraph(format='pdf')
G.attr('node', shape='circle')

G.edge("x0","x0",headlabel="&#172;a&&#172;b&&#172;c", labeldistance = "6", headport = "sw" , tailport = "sw")
G.edge("x0","x0", label="a&&#172;b&&#172;c", color="red", headport = "ne" , tailport = "ne")
G.edge("x0","x0",label="&#172;a&b&&#172;c", color="green", headport = "nw" , tailport = "nw")
G.edge("x0","x0",label="a&b&&#172;c", color="yellow", headport = "se" , tailport = "e")
G.edge("x0","x1",label="c", color="blue")

G.edge("x1","x1",label="T")

# print()するとdot形式で出力される
print(G)

# binary_tree.pngで保存
G.render('Automaton1')