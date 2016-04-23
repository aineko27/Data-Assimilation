# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 20:14:42 2016

@author: seis
"""

import numpy as np
from function import initArray, draw, Lorenz96, RungeKutta4, calcLyapunov1, calcLyapunov2
np.seterr(all="ignore")
np.seterr(all="raise")

#各定数の定義を行う
T = 0
dt = 0.05
F = 8.
J = 40
x = np.zeros(J)

initArray(x, 1*F)
sum1 = []
sum2 = []
for i in range(3100):
    x  = RungeKutta4(Lorenz96, x, F, dt)
    if i > 2980 and i%100==0:
        calcLyapunov1(Lorenz96, x, F, dt)
        calcLyapunov2(Lorenz96, x, F, dt)
        #sum1.append(A)
        #sum2.append(B)
        #sum1.append(calcLyapunov(Lorenz96, x, F, dt))
#sum1 = np.array(sum1)
#L = sum1.sum()/len(sum1)
#print(L, np.log(2)/L)
#draw(sum2)