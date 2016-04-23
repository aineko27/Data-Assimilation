# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:56:35 2016

@author: seis
"""

import matplotlib.pyplot as plt
import numpy as np
#np.seterr(all="ignore")
#np.seterr(all="raise")

#初期条件を設定する関数
def initArray(x, F):
    global T
    F = F
    x[:] = F
    x[len(x)//2] *= 1.001
    T = 0
    
#グラフを描写する関数
def draw(x, ylim=False, title="", label0="", label1=""):
    x = np.array(x)
    plt.title(title)
    plt.xlabel(label0)
    plt.ylabel(label1)
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    else:
        plt.ylim(np.min(x), np.max(x))
    
    if x.ndim == 1:
        plt.xlim(0, x.shape[0])
        plt.plot(np.append(x, x[0]))
    else:
        plt.xlim(0, x.shape[1])
        plt.plot(np.append(x, x[:,0:1], axis=1).T)
    plt.show()

#ローレンツ96の式を定義する関数
def Lorenz96(x, F):
    y = np.zeros_like(x)
    y = np.append(x[1:], x[:1])* np.append(x[-1:], x[:-1])- np.append(x[-2:], x[:-2])* np.append(x[-1:], x[:-1]) - x[:] + F
    return y

#関数fに対して4次のルンゲクッタを計算する関数
def RungeKutta4(f, x, F, dt):
    k1 = f(x, F)* dt
    k2 = f(x+ k1/2, F)* dt
    k3 = f(x+ k2/2, F)* dt
    k4 = f(x+ k3, F)* dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/ 6
    
#リアプノフ指数の計算
def calcLyapunov1(f, x, F, dt):
    x_copy1 = x.copy()
    x_copy2 = x.copy()
    x_copy2[0] += 0.000001
    error_init = np.linalg.norm(x_copy2- x_copy1)
    error = error_init
    error_exponent = [0]
    T = 0
    while error < error_init*1000000 and error > error_init/1000000:
        x_copy1 = RungeKutta4(f, x_copy1, F, dt)
        x_copy2 = RungeKutta4(f, x_copy2, F, dt)
        error = np.linalg.norm(x_copy2- x_copy1, F)
        T += dt
        plt.ylim(-1,15)
        error_exponent.append(np.log(error/ error_init))
    print(len(error_exponent))
    error_exponent = np.array(error_exponent)
    t = np.arange(0, dt*len(error_exponent), dt)
    L1 = error_exponent.dot(error_exponent)/ error_exponent.dot(t)
    L2 = np.log((error/ error_init))/ T
    test = np.arange(0, len(error_exponent), 1)
    plt.plot(test, test*L1*dt)
    plt.plot(test, test*L2*dt)
    plt.plot(error_exponent)
    print(T, L1, L2)
    return L2, error_exponent

def calcLyapunov2(f, x, F, dt):
    x_new = x.copy()
    epsilon = 0.00001
    x_new[0] += epsilon
    sum = 0
    n = 100000
    for i in range(n):
        x = RungeKutta4(f, x, F, dt)
        x_new = RungeKutta4(f, x_new, F, dt)
        error = np.linalg.norm(x- x_new)
        x_new = x + epsilon/ error* (x_new- x)
        if i > n/10:
            sum += np.log(error/epsilon)
    sum /= n*0.9* dt
    print(sum, np.log(2)/sum)

















