# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:31:34 2019

@author: oura
"""

from graphviz import Digraph

# formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
G = Digraph(format='png')
G.attr('node', shape='circle')

G.edge("q0","q0",label="!a&!b&!c")
G.edge("q0", "q1", label="a&!b&!c")
G.edge("q0","q2",label="!a&b&!c")
G.edge("q0","q3",label="a&b&!c")
G.edge("q0","q4",label="c")

G.edge("q1","q1",label="!a&!b&!c")
G.edge("q1","q2",label="!a&b&!c", color="red")
G.edge("q1","q3",label="a&b&!c")
G.edge("q1","q4",label="(a&!b&!c)|c")

G.edge("q2","q2",label="!a&!b&!c")
G.edge("q2","q1",label="a&!b&!c", color="red")
G.edge("q2","q3",label="a&b&!c")
G.edge("q2","q4",label="(!a&b&!c)|c")

G.edge("q3","q3",label="(!a&!b&!c)|(a&b&!c)")
G.edge("q3","q4",label="(a&!b)|(!a&b)|c")

G.edge("q4","q4",label="True")

# print()するとdot形式で出力される
print(G)

# binary_tree.pngで保存
G.render('Automaton2')