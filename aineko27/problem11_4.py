# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 19:39:44 2016

@author: seis
"""

import numpy as np
import matplotlib.pyplot as plt
plt.show()
from function import *
import time
start = time.time()

dt = 0.05
F = 8.
J = 40
H = np.eye(J)
R = np.eye(J)
x = np.zeros(J)
initArray(x, F)
x2 = x.copy()
x3 = x.copy()

gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
a = np.arange(J)
gauss_matrix = np.zeros([J, J])
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]

Fig1 = []
Fig2 = []
RMSE = []

sigma = 4.5
inf = 1.15

loc = gauss_matrix**(1/(2*sigma*sigma))

for i in range(1460):
    x = RungeKutta4(Lorenz96, x, F, dt)

#観測データをn回に分けて保存し、それぞれの場合においてRMSEがどのように変化するかを調べる。problem11_2,3 で行ったものを一気に計算している
for n in range(1, J):
    print(n)
    observe_time = np.floor(np.arange(0, n, n/J))
    dt = 0.05/n
    #個の回数分計算を行い、その平均をとることでより確からしい値を求める
    for kaisuu in range(100):
        f1 = open("data04.txt", "w")
        f2 = open("data05.txt", "w")
        f3 = open("data06.txt", "w")
        for i in range(1460):
            for j in range(n):
                x = RungeKutta4(Lorenz96, x, F, dt)
                x3[observe_time==j] = x[observe_time==j]
            x2 = x + np.random.normal(0, 1, J)
            x3 += np.random.normal(0, 1, J)
            string1 = str(x[0])
            string2 = str(x2[0])
            string3 = str(x3[0])
            for j in range(1, J):
                string1 += ", " + str(x[j])
                string2 += ", " + str(x2[j])
                string3 += ", " + str(x3[j])
            f1.write(string1 + "\n")
            f2.write(string2 + "\n")
            f3.write(string3 + "\n")
        f1.close()
        f2.close()
        f3.close()
        
        data1 = np.loadtxt("data04.txt", delimiter=", ")
        data2 = np.loadtxt("data05.txt", delimiter=", ")
        data3 = np.loadtxt("data06.txt", delimiter=", ")
        
        X_a = (np.loadtxt("X_init.txt", delimiter=",")).T
        m = X_a.shape[1]
        
        for i in range(1460):
            x_t = data1[i]
            X_f = RungeKutta4(Lorenz96, X_a, F, dt)
            X_f_temp = X_f
            for j in range(n-1):
                X_f = RungeKutta4(Lorenz96, X_f, F, dt)
                X_f_temp[observe_time==j+1] = X_f[observe_time==j+1]
            y = data3[i]
            X_a = EnKF2(X_f_temp, X_f, y, m, R, H, loc, inf)
            RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))
        #たまに大きい値が出るのをはじいている。これも値に含めるべきなのかもしれない
        if np.array(RMSE).mean() < 0.3:
            Fig1.append(np.array(RMSE).mean())
        RMSE = []
    Fig2.append(np.array(Fig1).mean())
    plt.plot(Fig1)
    plt.show()
    Fig1 = []
    
plt.plot(Fig2)
print(Fig2)
print(time.time()- start)





















       