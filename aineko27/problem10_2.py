# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 08:25:12 2016

@author: seis
"""
import matplotlib.pyplot as plt
import numpy as np
from function import *
import time
plt.show()

first_time = time.time()
dt = 0.05
F = 8
J = 40
R = np.eye(J)
R_loc = np.zeros([J, J, J])
P_a = R*10
H = np.eye(J)

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")

Fig1 = []
Fig2 = []
RMSE = []

X_a = np.loadtxt("X_init.txt", delimiter=",")
X_a = X_a.T
m = X_a.shape[1]

#ガウス分布の形をここで計算しておく。後でこの関数を1/(2*sigma*simga)乗してρを計算する
gauss = np.exp(-np.min([np.arange(J), J-np.arange(J)], axis=0)**2)
inf = 1.0

gauss = gauss**(1/200)
a = np.arange(J)
for i in range(J):
    R_loc[i] = R/gauss[a-i]
    
for i in range(1, 1460):
    x_t = data1[i]
    X_f = RungeKutta4(Lorenz96, X_a, F, dt)
    y = data2[i]
    X_a = LETKF2(X_f, y, m, R_loc, H, inf)
    Fig1.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
    
Fig1 = np.array(Fig1)
print(Fig1.mean(), Fig1[0])
plt.xlim(0, 1460)
plt.xlabel("Time Steps")
plt.ylabel("RMSE")
plt.plot(Fig1)
plt.show()

last_time =  time.time()
print(last_time-first_time)