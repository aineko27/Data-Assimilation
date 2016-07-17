# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:15:17 2016

@author: seis
"""
import numpy as np
import matplotlib.pyplot as plt
plt.show()
import time
from function import initArray, RungeKutta4, Lorenz96

start = time.time()

#各定数の定義を行う
T = 0
dt = 0.05
F = 8.
J = 40
x = np.zeros(J)
k = 0.8
k = 1/k**2-1
a = np.arange(J)

initArray(x, F)

test = np.zeros(40)
test[20] = 1
test[np.arange(20)*2] = 1
#test1 = np.zeros(40)
#test1[np.arange(13)] = 1
#test2 = np.zeros(40)
#test2[np.arange(13)+13] = 1
#data2の計算
data1 = np.loadtxt("data01.txt", delimiter=", ")
f = open("data02.txt", "w")
for i in range(len(data1)):
    line = data1[i] + np.random.normal(0, 0.8, J)*test1 + np.random.normal(0, 0.9, J)*test2 + np.random.normal(0, 1, J)*(1- test1- test2)
#    ransuu = np.random.normal(0, 1, J)
#    line = data1[i] + ransuu* test + ((-ransuu[a-1] + np.random.normal(0, 1, J)*np.sqrt(k))/np.sqrt(1+k))* (1- test)
    string = str(line[0])
    for j in range(1, J):
        string += ", " + str(line[j])
    f.write(string+ "\n")
f.close()

#%%
#アトラクタ上からm個のデータを取ってきて保存する
m = 40
f = open("X_init.txt", "w")
for i in range(100*m):
    x = RungeKutta4(Lorenz96, x, F, dt)
    if i%100==0:
        string =str(x[0])
        for j in range(1, J):
            string += "," + str(x[j])
        f.write(string+ "\n")
f.close()

print(time.time()- start)
