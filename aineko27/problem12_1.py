# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 05:17:03 2016

@author: seis
"""

import numpy as np
import matplotlib.pyplot as plt
from function import *
import time

dt = 0.05
F = 8
J = 40
H = np.eye(J)
R = np.eye(J)

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")

X_a = (np.loadtxt("X_init.txt", delimiter=",")).T
m = X_a.shape[1]

gauss = np.exp(-np.min([np.arange(J), J-np.arange(J)], axis= 0)**2)
a = np.arange(J)
gauss_matrix = np.zeros([J, J])
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]
    
#σの刻み幅とレンジを決めておく
sig_init = 1
sig_gap = 0.02
sig_num = 300
a = np.arange(J)
Fig1 = []
RMSE = []
delta = 3

#アダプティブ法で計算してみる
for i in range(sig_num):
    print(delta)
    sigma = (sig_init + i* sig_gap)
    loc = gauss_matrix**(1/(2*sigma*sigma))
    X_a = (np.loadtxt("X_init.txt", delimiter=",")).T
    for j in range(1, 1460):
        x_t = data1[j]
        X_f = RungeKutta4(Lorenz96, X_a, F, dt)
        y = data2[j]
        X_a, delta = EnKF3(X_f, X_a, y, m, R, H, loc, delta)
        RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
    Fig1.append(np.array(RMSE).mean())
    if sig_num==1: print(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
    RMSE = []
    
#%%
#plt.plot(RMSE)
plt.title("PO method(Adapative): m=40")
plt.xlabel("sigma")
plt.ylabel("RMSE")
#plt.xlim(1, 13)
plt.plot(np.arange(sig_init, sig_init+sig_gap*sig_num-0.001, sig_gap), Fig1)
    