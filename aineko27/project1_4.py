# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 21:08:06 2016

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
n = 5
isObserved = np.in1d(np.arange(J), np.round(np.arange(0, J, J/(J-n))))
isObserved = np.in1d(np.arange(J), np.arange(n, J))
#isObserved = isObserved[a-20]
H = np.eye(J)[isObserved]
R = np.eye(J-n)*1
RMSE = []
#P_f_true = np.zeros([14600, J, J])
Norm = np.zeros([14600, 40])
Norm_t = np.zeros([14600, 40])

data1 = np.loadtxt("data01.txt", delimiter=", ")
data2 = np.loadtxt("data02.txt", delimiter=", ")
norm = np.zeros([40,40])
norm_t = np.zeros([40,40])
norm_sub = np.zeros(40)
norm_n = 0
std = np.zeros(40)
std_t = np.zeros(40)
std_sub = np.zeros(40)
std_n = 0
dX_sum = np.zeros(40)
#noise = np.zeros([14600, 40,40])
sum1 = np.zeros([40, 40])
sum2 = 0

X_a = (np.loadtxt("X_init.txt", delimiter=",")[:40]).T
m = X_a.shape[1]

gauss = np.exp(-np.min([np.arange(J), J- np.arange(J)], axis=0)**2)
a = np.arange(J)
gauss_matrix = np.zeros([J, J])
for i in range(J):
    gauss_matrix[a, a-i] = gauss[i]

sigma = 10.3
inf = 1.18
delta_mean = 1
#下のコメントアウトを解除することで適当な値(localizationがない場合)から推定を始めることが出きる。コメントアウトをしないと推定した値が上書きされてしまうので、推定したlocを使いたい場合はコメントアウトすること
#loc = gauss_matrix**(1/(2*sigma*sigma))*1
#loc_temp = loc.copy()
#loc = loc* diag_matrix
#loc=1
for i in range(1, 1460*10):
    x_t = data1[i]
    y = data2[i][isObserved]
    X_f = RungeKutta4(Lorenz96, X_a, F, dt)
    X_a, delta, P_f, dX = EnKF5(X_f, y, m, R, H, loc, delta_mean, P_f_true[i]*(m-1))
#    P_f_true[i] = P_f.copy()/ (m-1)
    #ここで対角方向のnormをけいさんする。個々の計算はもしかしたらnp.stdでも行けるかもしれないがいまのところnormを使う方が精度が良い
    if i>100000:
        print(i)
        imshow(P_f/ (m-1))
        imshow(P_f_true[i])
        imshow(P_f/ (m-1)- P_f_true[i])
#    noise[i] = P_f[a, a-20][a]/ (m-1)
#    noise[i] = np.random.normal(0, 0.3, 40)
    noise[i] = np.random.normal(0, .1, [40,40])
    if i>100: dX_sum += np.std(dX, axis=1)**2
    sum2 = np.linalg.norm(dX, axis=1)
#    for j in range(40):
##        if j==1 or j==39: print(j, P_f[a,(a-np.abs(20-j))], "======================================================", noise[i][a])
##        norm_t[j] += np.linalg.norm(P_f_true[i][a, a-j])
#        norm_t[j] += np.linalg.norm(P_f_true[i][a, a-j])
#        norm[j] += np.linalg.norm(P_f[a, a-j]/ (m-1)/(dX_sum*dX_sum.reshape(-1,1)))
##        norm_t[j] += np.linalg.norm(P_f_true[i][0, j])
##        norm[j] += np.linalg.norm(P_f[0, j]/ (m-1))
#        norm_sub[j] += (P_f[a, a-j]).mean()
##        std[j] += np.std(P_f[a, a-j]/ (m-1))
##        std_t[j] += np.std(P_f_true[i][a,a-j])
##        std_sub[j] += np.var(P_f[a, a-j]/ (m-1)- (P_f*loc)[a, a-j]/ (m-1))
#        std_t[j] += np.std((P_f[a, a-j]/ (m-1) + noise[i][a-j]))
#        std[j] += np.std(P_f[a, a-j]/ (m-1))
#        Norm_t[i, j] = ((P_f[a, a-j]/ (m-1)).mean())**1
#        Norm[i, j] = np.linalg.norm(P_f[a, a-j]/ (m-1))
    if i>100: norm += np.abs(P_f/(sum2*sum2.reshape(-1,1)))**2
    if i>100: norm_t += np.abs(P_f/(sum2*sum2.reshape(-1,1)) + noise[i])**2
    delta_mean = 0.97*delta_mean + 0.03* delta
    RMSE.append(np.linalg.norm(x_t- X_a.mean(axis=1))/np.sqrt(J))

#%%
#ここから下は平均の計算とかグラフの表示とか
RMSE = np.array(RMSE)
print(RMSE.mean())
plt.xlabel("Time Steps")
plt.ylabel("RMSE")
plt.plot(RMSE)
plt.plot(AAA)
plt.xlim(0, 1460)
plt.ylim(0, 1)
plt.show()
#norm /= np.sqrt(J)
#norm_t /= np.sqrt(J)
#norm_sub /= np.sqrt(J)
#norm_n /= np.sqrt(J)
#
#print(norm[20])
##plt.xlabel("row")
##plt.ylabel("column")
##imshow(loc)
##imshow(loc_temp)
#print("norm_t")
#print(norm_t[0:21])
#print("norm")
#print(norm[0:21])
#print("norm_sub")
#print(norm_sub[0:21])
##print("std_t")
##print(std_t[0:21])
#print("std")
#print(std[0:21])
##print("std_sub")
##print(std_sub[0:21])
#loc_temp = loc.copy()
#%%
test = np.zeros(J)
test[a] = np.abs(norm[a]**2- norm[20]**2*(norm[a]/norm[20])**0.25)/norm[a]**2
test = ((Norm**2).sum(axis=0)- Norm[:,20]@ Norm[:,20])/ ((Norm**2).sum(axis=0))
test[a] = ((norm[a])**2- norm[20]**2)/ (norm[a]**2)
test = (norm**2- norm[20]**2).mean(axis=0)/ (norm**2).mean(axis=0)
aaa = (Norm[0:]**2).mean(axis=0)
aaa = Norm[200:].mean(axis=0)
test = (norm**2- norm[20]**2)/ norm**2
test = (aaa- aaa[20])/aaa
test = (aaa**2- aaa[20]**2)/ aaa**2
#test[a] = ((np.sqrt(norm**2- std**2)- np.sqrt(norm[20]**2- std[20]**2))**2 + std**2- std[20]**2)/ (norm[a]**2+norm[20]**2)
#test[0] = np.abs(norm[0]**2- norm[20]**2)/norm[0]**2
#test[2] = np.abs(norm[2]**2- norm[20]**2)/norm[2]**2
#test[38] = np.abs(norm[38]**2- norm[20]**2)/norm[38]**2
for i in range(40):
    loc[a, a-i] = test[i]**2
ccc = norm/(dX_sum*dX_sum.reshape(-1,1))**1
print(ccc[a,a-20])
loc = (ccc**1- (ccc[a,a-20].mean())**1)/ ccc**1
loc = (norm**1- (norm[a,a-20].mean())**1)/ norm**1
imshow(loc)
print("loc")
#loc=loc/loc[0,0]
print(loc[0][0:21])
imshow(norm_t/14600- norm/14600)
loc[loc<0] = 0
gauss_matrix = loc.copy()
#
##print(dX_sum/16000)
#print(np.sqrt(norm_t**2- norm**2)[:21])
#print(norm[:21]/norm[20])
#print((((np.sqrt(norm_t**2- std_t**2)- np.sqrt(norm**2- std**2))**2 + std_t**2- std**2)/ (norm_t**2))[:21])
#%%
#aaa = np.sqrt(norm_t**2- norm**2)
#test[a] = np.abs((norm[a])**2- aaa[a]**2)/ (norm[a])**2
##print(loc[0]**2)
##print(test)
##for i in range(40):
##    loc[a, a-i] = test[i]**1
#plt.plot(dX_sum)
#plt.show()
#print(dX_sum.mean(), np.std(dX_sum))
