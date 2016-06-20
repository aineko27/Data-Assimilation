# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:10:24 2016

@author: seis
"""

import matplotlib.pyplot as plt
import numpy as np
from function import *
plt.show()

dt = 0.05
F = 8
J = 40
m = 40

R = np.eye(J)
H = np.eye(J)

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")

Fig1 = []
Fig2 = []
RMSE = []

gauss = np.exp(-np.min([np.arange(J), J-np.arange(J)], axis=0)**2)
#ガウス分布の形からPDF:Covariance　localizationのページのρをもとめる。plt.imshowでρの形を見たらわかりやすいかもしれない
a = np.arange(J)
rho = np.zeros([J, J])
for i in range(J):
    rho[a, a-i] = gauss[i]

#変数二つの刻み幅とレンジをここで決めておく
sig_gap = 0.1
sig_num = 165
inf_gap = 0.01
inf_num = 155
a = np.arange(J)

#局所化の分散σとインフレーションの値をかえてRMSEがどのように変化するのかを調べる。sigma_gapは結局使わなかった
for i in range(sig_num):
    print(i)
    sigma = (1*i+1)/10
    rho_loc = rho**(1/(2*sigma*sigma))
    
    for j in range(inf_num):
        inf = 0.9+ j* inf_gap
        X_a = np.loadtxt("X_init.txt", delimiter=",")
        X_a = X_a.T
        
        for k in range(1, 1460):
            x_t = data1[k]
            X_f = RungeKutta4(Lorenz96, X_a, F, dt)
            y = data2[k]
            X_a = EnKF(X_f, y, m, R, H, rho_loc, inf)
            RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/ np.sqrt(J))
        Fig1.append(np.array(RMSE).mean())
        RMSE = []
#%%
Fig1 = np.array(Fig1)
Fig1 = Fig1.reshape(sig_num, inf_num)
Fig1 = Fig1.T
#%%
#グラフの目盛りは適当に書いているので書き直す必要があったりする
plt.title("RMSE (PO method)")
plt.xticks([0, 10, 20, 30, 40, 50, 60], (0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1))
plt.yticks([0, 10, 20, 30, 40, 50], (0.70, 0.80, 0.90, 1.00, 1.10, 1.20))
plt.xlabel("sigma")
plt.ylabel("inflation value")
plt.colorbar(plt.imshow(Fig1, interpolation="nearest", vmax=0.3))
        
        
        
        
        
        
        
        
        
        
        
        
        