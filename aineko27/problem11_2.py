# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 21:23:17 2016

@author: seis
"""

import numpy as np
import matplotlib.pyplot as plt
plt.show()
from function import *

T = 0
dt = 0.05
F = 8.
J = 40
x = np.zeros(J)
n = 5

initArray(x, F)
x2 = x.copy()
x3 = x.copy()

#ばらばらのタイミングで観測点の情報を保存できるように配列で順番を入れたものをつくっておく
observe_num = np.floor(np.arange(0, n, n/J))
observe_num = np.arange(J)%n

for i in range(1460):
    x = RungeKutta4(Lorenz96, x, F, dt)

dt = 0.05/n
f1 = open("data01.txt", "w")
f2 = open("data02.txt", "w")
f3 = open("data03.txt", "w")

#ばらばらのタイミングで観測点の情報を保存するdata03の計算をする
for i in range(1460):
    for j in range(n):
        x = RungeKutta4(Lorenz96, x, F, dt)
        #事前に作っておいた観測点を保存する配列に従って個別のタイミングでデータを保存する
        x3[observe_num==j] = x[observe_num==j]
    x2 = x + np.random.normal(0, 1, J)
    x3 += np.random.normal(0, 1, J)
    #真値とばらばらのタイミングの観測データと0.5秒ごとの観測データをそれぞれ保存する
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

m = 40
dt = 0.05
f = open("X_init.txt", "w")
for i in range(100*m):
    x = RungeKutta4(Lorenz96, x, F, dt)
    if i%100==0:
        string = str(x[0])
        for j in range(1, J):
            string += "," + str(x[j])
        f.write(string+ "\n")
f.close()
#%%
plt.xlabel("n")
plt.ylabel("RMSE")
plt.plot(Fig2[0:31])


