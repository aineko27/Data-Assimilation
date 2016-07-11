# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 10:37:32 2016

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
R = np.random.normal(0, 100, [J, J])
R = np.eye(J)

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")
RMSE = []
Fig1 = []
Fig2 = []
sum = np.eye(J)*50
sum = np.random.normal(0, 1, [J, J])

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

for i in range(1, 1460):
    x_t = data1[i]
    X_f = RungeKutta4(Lorenz96, X_a, F, dt)
    y = data2[i]
    X_a, R_temp = EnKF(X_f, y, m, sum/i, H, loc, inf)
    for j in range(J):
        R_temp[a, a-j] = R_temp[a, a-j].mean()/40
    sum += R_temp
    if i == 1: imshow(R_temp)
    RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
    Fig1.append(np.trace(sum)/i)
 
#%%   
sum /= (1460)
plt.xlabel("row")
plt.ylabel("column")
imshow(sum)
print(np.diag(sum).mean())
plt.xlabel("row")
plt.ylabel("column")
imshow(R_temp)
print(R_temp[a, a]/(40), R_temp[a, a].mean()/40)
print(np.array(RMSE).mean())
plt.xlabel("Timse Steps")
plt.ylabel("RMSE")
plt.xlim(0,1460)
plt.plot(RMSE)
plt.show()
plt.plot(Fig1)