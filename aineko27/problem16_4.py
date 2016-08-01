# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 07:20:18 2016

@author: seis
"""

#観測データ、アンサンブル数をいじりたい場合はproblem3&4をはしらせること。一番最初にこのプログラムを走らせる場合は下の方でコメントアウトされているlocを解除する必要がある
import numpy as np
import matplotlib.pyplot as plt
from function import *
import time

dt = 0.05
F = 8
J = 40
n = 0
isObserved = np.in1d(np.arange(J), np.arange(n, J))
isObserved = np.in1d(np.arange(J), np.round(np.arange(0, J, J/(J-n))))
H = np.eye(J)[isObserved]
R = np.eye(J-n)*1
RMSE = []

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")
norm = np.zeros(40)
sigma_list = []

X_a = (np.loadtxt("X_init.txt", delimiter=",")).T
m = X_a.shape[1]

gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
a = np.arange(J)
gauss_matrix = np.zeros([J, J])
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]

sigma = 60000.3
inf = 1.18
#下のコメントアウトを解除することで適当な値(localizationがない場合)から推定を始めることが出きる。コメントアウトをしないと推定した値が上書きされてしまうので、推定したlocを使いたい場合はコメントアウトすること
#loc = gauss_matrix**(1/(2*sigma*sigma))*1
delta_mean = 1
for i in range(1, 1460*10):
    x_t = data1[i]
    y = data2[i][isObserved]
    X_f = RungeKutta4(Lorenz96, X_a, F, dt)
    X_a, delta, R_temp, HBH, HBH_R, P_f, K = EnKF5(X_f, y, m, R, H, loc, delta_mean)
    #ここで対角方向のnormをけいさんする。個々の計算はもしかしたらnp.stdでも行けるかもしれないがいまのところnormを使う方が精度が良い
    for j in range(40):
        norm[j] += np.linalg.norm(P_f[a, a-j]/ (m-1))
    delta_mean = 0.97*delta_mean + 0.03* delta
    RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))

#%%
#ここから下は平均の計算とかグラフの表示とか
RMSE = np.array(RMSE)
print(RMSE.mean())
plt.xlabel("Time Steps")
plt.ylabel("RMSE")
plt.plot(RMSE)
plt.xlim(0, 1460)
plt.show()
#norm /= 1459
print(norm[20])
plt.xlabel("row")
plt.ylabel("column")
imshow(loc)
test = np.zeros(J)
test[a] = (norm[a]**2- norm[20]**2)/norm[a]**2
for i in range(40):
    loc[a, a-i] = test[i]