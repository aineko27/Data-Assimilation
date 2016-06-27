# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:45:35 2016

@author: seis
"""

import matplotlib.pyplot as plt
import numpy as np
from function import *
import time
start = time.time()

dt = 0.05
F = 8
J = 40

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")

X_a = np.loadtxt("X_init.txt", delimiter=",")
X_a = X_a.T
m = X_a.shape[1]

gauss = np.exp(-np.min([np.arange(J), J-np.arange(J)], axis=0)**2)
a = np.arange(J)
gauss_matrix = np.zeros([J, J])
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]

#どこの観測点を除くかを決める
n = 0
isExist = np.in1d(np.arange(J), np.round(np.arange(0, J, J/(J-n))))
isExist = np.in1d(np.arange(J), np.arange(n, J))
H = np.eye(J)[isExist]
R = np.eye(J-n)

Fig1 = []
Fig2 = []
RMSE = []

#変数二つの刻み幅とレンジをここで決めておく
sig_init = 2
sig_gap = 0.2
sig_num = 40
inf_init = 1.0
inf_gap = 0.02
inf_num = 45
a = np.arange(J)

#アンサンブル数と観測点の場所を固定してσとρを変えて誤差がどのように変化するのかを調べる
for i in range(sig_num):
    print(i)
    sigma = (sig_init + i* sig_gap)
    loc = gauss_matrix**(1/(2*sigma*sigma))
    
    for j in range(inf_num):
        inf = inf_init + j* inf_gap
        X_a = np.loadtxt("X_init.txt", delimiter=",")
        X_a = X_a.T
        
        for k in range(1, 1460):
            x_t = data1[k]
            X_f = RungeKutta4(Lorenz96, X_a, F, dt)
            y = data2[k][isExist]
            X_a = EnKF(X_f, y, m, R, H, loc, inf)
            RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
        Fig1.append(np.array(RMSE).mean())
        RMSE = []

#%%
Fig1 = np.array(Fig1)
Fig1 = Fig1.reshape(sig_num, inf_num)
Fig1 = Fig1.T
#%%
#グラフの目盛りは適当に書いているので書き直す必要があったりする
plt.title("RMSE (PO method) m=" + str(m) + ", n = " + str(n))
plt.xticks(np.arange(0, sig_num, 10), sig_init + sig_gap*np.arange(0, sig_num, 10))
plt.yticks(np.arange(0, inf_num, 10), inf_init + inf_gap*np.arange(0, inf_num, 10))
#plt.xticks([0, 10, 20, 30, 40, 50, 60], (0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1))
#plt.yticks([0, 10, 20, 30, 40, 50], (0.70, 0.80, 0.90, 1.00, 1.10, 1.20))
plt.xlabel("sigma")
plt.ylabel("inflation value")
plt.colorbar(plt.imshow(Fig1, interpolation="nearest", vmax=0.7))


print(time.time()- start)












