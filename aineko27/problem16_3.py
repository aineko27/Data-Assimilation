# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 22:24:19 2016

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
isObserved = np.in1d(np.arange(J), np.arange(n, J))
isObserved = np.in1d(np.arange(J), np.round(np.arange(0, J, J/(J-n))))
H = np.eye(J)[isObserved]
R = np.eye(J-n)*1

Fig1 = []
Fig2 = []
Fig3 = []
Fig4 = []
Fig5 = []
Fig6 = []
test1 = np.zeros(1459)
test2 = np.zeros(1459)
data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")
RMSE = np.zeros(1459)
diag_list = np.zeros(1459)
nondiag_list = np.zeros(1459)
sigma_list = []
X_init = (np.loadtxt("X_init.txt", delimiter=",")).T.copy()
X_a = X_init
m = X_a.shape[1]

gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
gauss_matrix = np.zeros([J, J])
diag = np.zeros([J, J])
diag[10:30] = 1
diag_matrix = np.zeros([J, J])
a = np.arange(J)
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]
    diag_matrix[a, a-i] = diag[i]

for l in range(100):
    print(l)
    sigma = 0.1 + 0.1* l
    loc = gauss_matrix**(1/(2*sigma*sigma))
    sigma_list.append(sigma)
    X_a = X_init
    delta_mean = 1
    
    for i in range(1, 1460):
        x_t = data1[i]
        y = data2[i][isObserved]
        X_f = RungeKutta4(Lorenz96, X_a, F, dt)
        dX = (X_f- X_f.mean(axis=1, keepdims=True))
        P_f = dX@ dX.T/(m-1)
        diag_list[i-1] = (np.linalg.norm(P_f* diag_matrix)- np.linalg.norm(P_f* diag_matrix* loc))/ (np.linalg.norm(P_f*diag_matrix))
        nondiag_list[i-1] = (np.linalg.norm(P_f* (1-diag_matrix))- np.linalg.norm(P_f* (1- diag_matrix)* loc))/ (np.linalg.norm(P_f)- np.linalg.norm(P_f*loc))
        X_a, delta, R_temp, HBH, HBH_R, P_f, K = EnKF5(X_f, y, m, R, H, loc, delta_mean)
        delta_mean = 0.97*delta_mean + 0.03* delta
        RMSE[i-1] = np.linalg.norm(x_t- X_a.mean(axis=1))/ np.sqrt(J)
        test1[i-1] = np.linalg.norm(P_f* diag_matrix*loc)
        test2[i-1] = np.linalg.norm(P_f/(m-1)* (1- diag_matrix))
    
    Fig1.append(RMSE.mean())
    Fig2.append(diag_list.mean())
    Fig3.append(nondiag_list.mean())
    Fig4.append(delta_mean)
    Fig5.append(test1.mean())
    Fig6.append(test2.mean())
#%%
Fig1 = np.array(Fig1)
Fig2 = np.array(Fig2)
Fig3 = np.array(Fig3)
Fig5 = np.array(Fig5)
Fig6 = np.array(Fig6)

plt.xlabel("sigma")
plt.ylabel("RMSE")
plt.plot(sigma_list, Fig1)
plt.show()
plt.title("nondiag")
plt.plot(sigma_list, Fig2)
plt.show()
plt.title("diag")
plt.plot(sigma_list, Fig3)
plt.show()
plt.plot(sigma_list, (Fig2- Fig3))
plt.show()
plt.plot(sigma_list, Fig2/Fig3)
plt.show()
plt.title("delta")
plt.plot(sigma_list, Fig4)
plt.show()
plt.plot(sigma_list, Fig5)
plt.show()
plt.plot(sigma_list, Fig6)
plt.show()
plt.plot(sigma_list, Fig5/Fig6)