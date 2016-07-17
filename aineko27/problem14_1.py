# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:33:29 2016

@author: seis
"""

import numpy as np
import matplotlib.pyplot as plt
from function import *
import time
#いろいろな形のRの推定を行っていくプログラム
#このプログラムを回す前にまず分散、共分散の異なる観測データを作る必要がある。problem14_3で計算できるのでそっちを先に回す必要がある

dt = 0.05
F = 8
J = 40
H = np.eye(J)

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")
RMSE = []
Fig1 = []
Fig2 = []
#Rの初期値を決める
R_mean = np.eye(J)*10

X_a = (np.loadtxt("X_init.txt", delimiter=",")).T
m = X_a.shape[1]

gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
a = np.arange(J)
gauss_matrix = np.zeros([J, J])
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]

#シグマとインフレーションの値を決めておく
sigma = 5.6
inf = 1.2
loc = gauss_matrix**(1/(2*sigma*sigma))

for i in range(1, 14600):
    x_t = data1[i]
    X_f = RungeKutta4(Lorenz96, X_a, F, dt)
    y = data2[i]
    X_a, R_temp = EnKF(X_f, y, m, R_mean, H, loc, inf)
    R_mean = R_mean* (i/(i+1)) + R_temp* (1/(i+1))
    if i%1460==0:
        plt.xlabel("row")
        plt.ylabel("column")
        imshow(R_mean)
        print(np.diag(R_mean))
        print(R_mean[a, a-1])
    RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
 
#%%   
#R_mean /= (1460)
plt.xlabel("row")
plt.ylabel("column")
imshow(R_mean)
print(R_mean[a, a])
print(R_mean[a, a-1])
plt.xlabel("Time Steps")
plt.ylabel("RMSE")
plt.xlim(0, 1460)
plt.plot(RMSE)
plt.show()
print(np.array(RMSE).mean())