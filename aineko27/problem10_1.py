# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:15:39 2016

@author: seis
"""

import matplotlib.pyplot as plt
import numpy as np
from function import *
plt.show()

#各定数の定義を行う。mはアンサンブルする個数
dt = 0.05
F = 8
J = 40
m = 40

R = np.eye(J)
H = np.eye(J)

#データの読み込みを行う
data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")

Fig1 = []
Fig2 = []
RMSE = []

#ガウス分布の形をここで計算しておく。後でこの関数を1/(2*σ*σ)乗してρを計算する
gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
#ガウス分布の形からPDF:Covariance　localizationのページのρをもとめる。plt.imshowでρの形を見たらわかりやすいかもしれない
a = np.arange(J)
rho = np.zeros([J, J])
for i in range(J):
    rho[a, a-i] = gauss[i]

#アンサンブルカルマンフィルターで計算する
for i in range(1,100):
    print(i)
    sigma = (i/10)
    X_a = np.loadtxt("X_init.txt", delimiter=",")
    X_a = X_a.T
    m = X_a.shape[1]
    for j in range(1, 1460):
        x_t = data1[j]
        X_f = RungeKutta4(Lorenz96, X_a, F, dt)
        y = data2[j]
        X_a = EnKF(X_f, y, m, R, H, rho**(1/(2*sigma*sigma)))
        RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
    RMSE = np.array(RMSE)
    Fig1.append(RMSE.mean())
    RMSE = []
    
#%%
Fig1 = np.array(Fig1)
print(Fig1.mean(), np.min(Fig1))
plt.xlabel("sigma")
plt.ylabel("RMSE")
plt.xlim(0, 100/100)
plt.plot(np.arange(99)/100, Fig1)
plt.show()
    