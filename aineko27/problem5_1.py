# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:15:17 2016

@author: seis
"""
import numpy as np
import matplotlib.pyplot as plt
from function import Lorenz96, RungeKutta4, KF, draw

#各定数の定義を行う
T = 0
dt = 0.05
F = 8.
J = 40

P_a = np.eye(J)
R = np.eye(J)
H = np.eye(J)

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")

#カルマンフィルターを適用した場合
Fig1 = []
x_a = data2[0]
for i in range(1, len(data2)):
    x_t = data1[i]
    x_f = RungeKutta4(Lorenz96, x_a, F, dt)
    y = data2[i]
    x_a, P_a = KF(x_f, y, dt, P_a, H, R)
    Fig1.append(np.linalg.norm(x_t- x_a))

#最初に観測値だけを代入してその後は観測値を全く使わずに計算した場合
Fig2 = []
x_a = data2[0]
for i in range(1, len(data2)):
    x_t = data1[i]
    x_f = RungeKutta4(Lorenz96, x_a, F, dt)
    y = data2[i]
    x_a = x_f.copy()
    Fig2.append(np.linalg.norm(x_t- x_a)) 

#毎回観測値を代入して計算した場合
Fig3 = []
x_a = data2[0]
for i in range(1, len(data2)):
    x_t = data1[i]
    x_f = RungeKutta4(Lorenz96, y, F, dt)
    y = data2[i]
    x_a = x_f.copy()
    Fig3.append(np.linalg.norm(x_t- x_a))

#毎回観測値と予測値を1/2ずつ足しあわせたものを解析値とした場合
Fig4 = []
x_a = data2[0]
for i in range(1, len(data2)):
    x_t = data1[i]
    x_f = RungeKutta4(Lorenz96, x_a, F, dt)
    y = data2[i]
    x_a = (x_f + y)/2
    Fig4.append(np.linalg.norm(x_t- x_a))
    
#その他の場合
Fig5 = []
x_a = data2[0]
for i in range(1, len(data2)):
    x_t = data1[i]
    x_f = RungeKutta4(Lorenz96, x_a, F, dt)
    y = data2[i]
    x_a = (x_f*5 + y)/6
    Fig5.append(np.linalg.norm(x_t- x_a))
plt.xlim(0, 1460)
plt.ylim(0, 40)
plt.plot(Fig1)
plt.show()
plt.xlim(0, 1460)
plt.ylim(0, 40)
plt.plot(Fig2)
plt.show()
plt.xlim(0, 1460)
plt.ylim(0, 40)
plt.plot(Fig3)
plt.show()
plt.xlim(0, 1460)
plt.ylim(0, 40)
plt.plot(Fig4)
plt.show()
plt.xlim(0, 1460)
plt.ylim(0, 40)
plt.plot(Fig5)
plt.show()

Fig1 = np.array(Fig1)
Fig2 = np.array(Fig2)
Fig3 = np.array(Fig3)
Fig4 = np.array(Fig4)
Fig5 = np.array(Fig5)
print(Fig1.mean(), Fig2.mean(), Fig3.mean(), Fig4.mean(), Fig5.mean())
