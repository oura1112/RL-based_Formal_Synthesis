# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:29:36 2019

@author: oura
"""
import numpy as np

#for i in range(3):
    #print(i)
    
a = [1]

a = [[2], [3]]
#print(type(a[0][0]))
#print(np.shape(np.array(a)))

a = [1,2]
b = [3,4]
c = a + b
print(c)

a = bin(5)
a = a.lstrip("0b")
a = a.rjust(8,"0")
print(a)
#for i in range(len(a)):
    #print(i)
    #if a[i] == '1' :
        #print(i)
    
#print(a)

b = [i for i in range(12)]
print(bin(4))

c = [2*i for i in range(6)]
#print(b)
sum_ = sum(b[i] for i in c )
#print(sum_)

d = [np.random.uniform(2,10) for i in range(len(b))]
print(sum(d))
d = np.array(d) / sum(d)
print(sum(d))

p = np.random.dirichlet([500,1000,1500])
print(p)

#print(d)