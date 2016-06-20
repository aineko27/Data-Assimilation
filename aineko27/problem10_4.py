# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:39:47 2016

@author: seis
"""

import matplotlib.pyplot as plt
import numpy as np
from function import *
plt.show()

dt = 0.05
F = 8
J = 40
R = np.eye(J)
R_loc = np.zeros([J, J, J])
P_a = R* 10
H = np.eye(J)

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")

Fig1 = []
Fig2 = []
RMSE = []

X_a = np.loadtxt("X_init.txt", delimiter=",")
X_a = X_a.T
m = X_a.shape[1]

#ガウス分布の形をここで計算しておく。
gauss = np.exp(-np.min([np.arange(J), J-np.arange(J)], axis=0)**2)

#変数二つの刻み幅とレンジをここで決めておく。
sig_gap = 0.1
sig_num = 50
inf_gap = 0.01
inf_num = 30
a = np.arange(J)

#局所化の分散σとインフレーションの値をかえてRMSEがどのように変化するのかを調べる。LETKFで計算するのでものすごい時間がかかる
for i in range(sig_num):
    print(i)
    sigma = (0.6+i*sig_gap)/1
    #分散σからRの具体的な形を求めてそれをR_locとして使う
    rho = gauss**(1/(2*sigma*sigma))
    for j in range(J):
        R_loc[j] = R/rho[a-j]
        
    #インフレーションの値を変えていく
    for j in range(inf_num):
        inf = 1.0+ j* inf_gap
        X_a = np.loadtxt("X_init.txt", delimiter=",")       
        X_a = X_a.T
        
        for k in range(1, 1460):
            x_t = data1[k]
            X_f = RungeKutta4(Lorenz96, X_a, F, dt)
            y = data2[k]
            X_a = LETKF2(X_f, y, m, R_loc, H, inf)
            RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
        Fig1.append(np.array(RMSE).mean())
        RMSE = []
#%%        
Fig1 = np.array(Fig1)
AAAAAA = Fig1.copy()
Fig1 = Fig1.reshape(sig_num, inf_num)
Fig1 = Fig1.T
#%%
#グラフの目盛りは適当に書いているので書き直す必要があったりする
plt.title("RMSE (LETKF method)")
plt.xticks([0, 5, 10, 15], (0.1, 0.6, 1.1, 1.6))
plt.yticks([0, 5, 10, 15], (1.0, 1.05, 1.10, 1.15))
plt.xlabel("sigma")
plt.ylabel("inflation value")
imshow(Fig1)

