# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:17:32 2016

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
isObserved = np.in1d(np.arange(J), np.round(np.arange(0, J, J/(J-n))))
isObserved = np.in1d(np.arange(J), np.arange(n, J))
H = np.eye(J)[isObserved]
R = np.eye(J-n)*1
RMSE = []
#P_f_true = np.zeros([14600, J, J])

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")
norm = np.zeros([40, 40])
norm_t = np.zeros(40)
std = np.zeros(40)
std_t = np.zeros(40)

X_a = (np.loadtxt("X_init.txt", delimiter=",")[:40]).T
X_a_true = (np.loadtxt("X_init.txt", delimiter=",")).T
m = X_a.shape[1]

gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
a = np.arange(J)
gauss_matrix = np.zeros([J, J])
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]

sigma = 4.
inf = 1.18
delta_mean = 1
#下のコメントアウトを解除することで適当な値(localizationがない場合)から推定を始めることが出きる。コメントアウトをしないと推定した値が上書きされてしまうので、推定したlocを使いたい場合はコメントアウトすること
#loc = gauss_matrix**(1/(2*sigma*sigma))*1
#loc_temp = loc.copy()
#loc = loc* loc
#loc=1
for i in range(1, 1460*10):
    x_t = data1[i]
    y = data2[i][isObserved]
    X_f = RungeKutta4(Lorenz96, X_a, F, dt)
#    X_f_true = RungeKutta4(Lorenz96, X_a_true, F, dt)
    X_a, delta, P_f, dX = EnKF5(X_f, y, m, R, H, loc, delta_mean)
#    dX_true = ((X_f_true- X_f_true.mean(axis=1, keepdims=True))* delta_mean)
#    P_f_true[i] = dX_true@ dX_true.T/ (X_f_true.shape[1]-1)
#    imshow(P_f- P_f_ture)
    if i>100000:
        print(i)
        imshow(P_f/(m-1))
        imshow(P_f_true[i])
        imshow(P_f/(m-1)- P_f_true[i])
    #ここで対角方向のnormをけいさんする。個々の計算はもしかしたらnp.stdでも行けるかもしれないがいまのところnormを使う方が精度が良い
    if i>500:norm += np.abs(P_f/ (m-1))**2
#    for j in range(40):
##        norm[j] += np.linalg.norm(P_f[a, a-j]/ (m-1))
##        norm_t[j] += np.linalg.norm((P_f_true[i])[a, a-j])
#        norm[j] += np.abs(P_f/ (m-1))
#        norm_t[j] += np.abs(P_f/ (m-1))
#        std[j] += np.std(P_f[a, a-j]/ (m-1))
#        std_t[j] += np.std((P_f_true[i])[a,a-j])
#    for j in range(40):
        
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
norm /= np.sqrt(J)
norm_t /= np.sqrt(J)
#print(norm[20])
##plt.xlabel("row")
##plt.ylabel("column")
##imshow(loc)
##imshow(loc_temp)
#print("norm_t")
#print(norm_t[0:21])
#print("norm")
#print(norm[0:21])
#print("std_t")
#print(std_t[0:21])lo
#print("std")
#print(std[0:21])

for i in range(40):
    loc[i, a] = (norm[i, a]- norm[i, i-20])/ norm[i, a]
    loc[i,a] = (norm[i,a]**2- norm[i, i-20]**2)/ (norm[i,a]**2)
loc = (norm**2- (np.min(norm))**2)/ norm**2
print(loc[0])
#%%
test = norm_t/ norm
test = np.zeros(J)
test[a] = (norm[a]**2- norm[20]**2)/norm[a]**2
for i in range(40):
    loc[a, a-i] = test[i]**1
print("norm**2- norm_t**2")
print(np.sqrt(norm**2- norm_t**2)[0:21])
print("std**2- std_t**2")
print(np.sqrt(std**2- std_t**2)[0:21])
print("loc")
print(loc[0][0:21])
#norm = np.sqrt(norm**2- norm_t**2)
#test[a] = (norm[a]**2- norm[20]**2)/norm[a]**2
#test[a] = norm_t[a]**1/ norm[a]**1
#for i in range(40):
#    loc[a, a-i] = test[i]**1
