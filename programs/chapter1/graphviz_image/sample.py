# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:15:55 2019

@author: oura
"""

from graphviz import Digraph

# formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
G = Digraph(format='png')
G.attr('node', shape='circle')

N = 15    # ノード数

# ノードの追加
for i in range(N):
    G.node(str(i), str(i))

# 辺の追加
for i in range(N):
    if (i - 1) // 2 >= 0:
        G.edge(str((i - 1) // 2), str(i))

# print()するとdot形式で出力される
print(G)

# binary_tree.pngで保存
G.render('binary_tree')