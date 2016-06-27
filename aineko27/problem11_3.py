# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 20:23:42 2016

@author: seis
"""

import matplotlib.pyplot as plt
import numpy as np
from function import *
plt.show()

n = 5
dt = 0.05/n
F = 8
J = 40
H = np.eye(J)
R = np.eye(J)

observe_num = np.floor(np.arange(0, n, n/J))
observe_num = np.arange(J)%n

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")
data3 = np.loadtxt("data03.txt", delimiter=", ")

X_a = np.loadtxt("X_init.txt", delimiter=",")
X_a = X_a.T
m = X_a.shape[1]

gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
a = np.arange(J)
gauss_matrix = np.zeros([J, J])
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]

Fig1 = []
Fig2 = []
RMSE = []

sigma = 5.2
inf = 1.15

loc = gauss_matrix**(1/(2*sigma*sigma))

#ばらばらのタイミングの観測データに対応した手法で計算を行う
for i in range(1, 1460):
    x_t = data1[i]
    X_f = RungeKutta4(Lorenz96, X_a, F, dt)
    X_f_temp = X_f
    for j in range(n-1):
        X_f = RungeKutta4(Lorenz96, X_f, F, dt)
        X_f_temp[observe_num==j+1] = X_f[observe_num==j+1]
    y = data3[i]
    X_a = EnKF2(X_f_temp, X_f, y, m, R, H, loc, inf)
    Fig1.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))

print(np.array(Fig1).mean())
plt.xlim(0, 1460)
plt.ylim(0)
plt.xlabel("Time Steps")
plt.ylabel("RMSE")
plt.plot(Fig1)
plt.show()

X_a = np.loadtxt("X_init.txt", delimiter=",")
X_a = X_a.T
dt = 0.05

#こっちはばらばらのタイミングのものを一緒のタイミングとみなして計算したらどうなるのかを計算してみたもの。上より精度は悪くなる
for i in range(1, 1460):
    x_t = data1[i]
    X_f = RungeKutta4(Lorenz96, X_a, F, dt)
    y = data3[i]
    X_a = EnKF(X_f, y, m, R, H, loc, inf)
    Fig2.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
    
print(np.array(Fig2).mean())
plt.xlim(0, 1460)
plt.ylim(0)
plt.xlabel("Time Steps")
plt.ylabel("RMSE")
plt.plot(Fig2)
plt.show()
