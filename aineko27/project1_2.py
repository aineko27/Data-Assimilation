# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 02:46:57 2016

@author: seis
"""

import numpy as np
import matplotlib.pyplot as plt
from function import *
import time

dt = 0.05
n = 10
F = 8
J = 40
isObserved = np.in1d(np.arange(J), np.round(np.arange(0, J, J/(J-n))))
isObserved = np.in1d(np.arange(J), np.arange(n, J))
H = np.eye(J)[isObserved]
R = np.eye(J-n)*1

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")
Fig1 = []
RMSE = np.zeros(1459)
sigma_list = []
X_init = (np.loadtxt("X_init.txt", delimiter=",")[:40]).T.copy()
X_a = X_init.copy()
X_a_true = X_init.copy()
m = X_a.shape[1]

#gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
#gauss_matrix = np.zeros([J, J])
#a = np.arange(J)
#for i in range(J):
#    gauss_matrix[a, a-i] = gauss[i]

for l in range(200):
    if l%20==0: print(l)
    sigma = 0.6 + 0.03* l
    loc = gauss_matrix**(1/(2*sigma*sigma))
    sigma_list.append(sigma)
    X_a = X_init.copy()
    delta_mean = 1
    
    for i in range(1, 1460):
        x_t = data1[i]
        y = data2[i][isObserved]
        X_f = RungeKutta4(Lorenz96, X_a, F, dt)
        X_a, delta, P_f, dX = EnKF5(X_f, y, m, R, H, loc, delta_mean)
        delta_mean = 0.97*delta_mean + 0.03* delta
        RMSE[i-1] = np.linalg.norm(x_t- X_a.mean(axis=1))/ np.sqrt(J)
    
    Fig1.append(RMSE.mean())
#%%
Fig1 = np.array(Fig1)
sigma = np.array(sigma_list)

plt.xlabel("sigma")
plt.ylabel("RMSE")
#plt.ylim(np.min(Fig1)-0.1, np.min(Fig1)+0.1)
plt.plot(sigma, Fig1)
