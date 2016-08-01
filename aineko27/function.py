# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:56:35 2016

@author: seis
"""

import matplotlib.pyplot as plt
import numpy as np
import time
#np.seterr(all="ignore")
#np.seterr(all="raise")

#初期条件を設定する関数
def initArray(x, F):
    global T
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
    
def imshow(M, v_max=0, v_min=0):
    if v_max==v_min:
        plt.colorbar(plt.imshow(M, interpolation="nearest"))
    else:
        plt.colorbar(plt.imshow(M, interpolation="nearest", vmax=v_max, vmin=v_min))
    plt.show()

#ローレンツ96の式を定義する関数
def Lorenz96(x, F):
    y = np.zeros_like(x)
    y = np.append(x[1:], x[:1], axis=0)* np.append(x[-1:], x[:-1], axis=0)- np.append(x[-2:], x[:-2], axis=0)* np.append(x[-1:], x[:-1], axis=0) - x[:] + F
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
    x_copy2[0] += 0.0001
    error_init = np.linalg.norm(x_copy2- x_copy1)
    error = error_init
    error_exponent = [0]
    T = 0
    while error < error_init*100000 and error > error_init/100000:
        x_copy1 = RungeKutta4(f, x_copy1, F, dt)
        x_copy2 = RungeKutta4(f, x_copy2, F, dt)
        error = np.linalg.norm(x_copy2- x_copy1)
        T += dt
        plt.ylim(-1,15)
        error_exponent.append(np.log(error/ error_init))
    error_exponent = np.array(error_exponent)
    t = np.arange(0, dt*error_exponent.shape[0]-1e-4, dt)
    L1 = np.log((error/ error_init))/ T
    L2 = error_exponent.dot(error_exponent)/ error_exponent.dot(t)
    test = np.arange(0, len(error_exponent), 1)
    #plt.title("Fig1")
    plt.ylim(-0.5, 12.5)
    plt.xlabel("TimeStep")
    plt.ylabel("ln($\epsilon_t$/$\epsilon_0$)")
    plt.plot(t, error_exponent, label="plot1")
    plt.plot(t, test*L1*dt, label="plot2")
    plt.plot(t, test*L2*dt, label="plot3")
    plt.legend(loc="upper left")
    #plt.legend(("plot1", "plot2", "plot3"), "upper left")
    plt.savefig("Fig1.png",format = 'png', dpi=300)
    plt.show()
    print(T, L1, L2)
    return L1, L2

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

#カルマンフィルターの計算
def KF(x_a, x_f, y, dt, P_a, H, R, mul=0.08):
    #ヤコビアンの求め方その２
    J = len(x_a)
    M2 = np.zeros([J, J])
    delta = 1e-5
    M2 = (RungeKutta4(Lorenz96, (np.tile(x_a.reshape(J, 1), [1, J]) + np.eye(J)* delta), 8.0, dt) - x_f.reshape(J, 1))/ delta
    
    #dotは行列の掛け算、a.Tは転置、np.linalg.invは逆行列を意味する
    P_f = M2.dot(P_a).dot(M2.T)
    K = P_f.dot(H.T).dot(np.linalg.inv(R + H.dot(P_f).dot(H.T)))
    P_a = (np.eye(J)- K.dot(H)).dot(P_f)
    P_a = (1+mul)* P_a
    x = x_f + K.dot(y- H.dot(x_f))
    return x, P_a

#三次元変分法の計算
def calc3DVAR(x_f, y, H, B, R):
    x = x_f + np.linalg.inv(np.linalg.inv(B)+ H.T.dot(np.linalg.inv(R).dot(H))).dot(H.T).dot(np.linalg.inv(R)).dot(y- H.dot(x_f))
    #x = x_f + (y- H.dot(x_f)).dot(np.linalg.inv(np.linalg.inv(B)+ H.T.dot(np.linalg.inv(R).dot(H))).dot(H.T).dot(np.linalg.inv(R)))
    return x

#アンサンブルカルマンフィルターの計算。
def EnKF(X_f, y, m, R, H, rho=1, inf=1):
    y = y.reshape(len(y), 1)
    dX = (X_f- X_f.mean(axis=1, keepdims=True))*inf#+ np.random.normal(0, 1, X_f.shape)
#    print(np.diag(dX@dX.T))
    dY = H.dot(dX)
    K = rho*(dX@ dX.T)@H.T@ (np.linalg.inv(H@ (rho* (dX@dX.T))@ H.T+ (m-1)*R))
    #K = rho*(dX.dot(dY.T)).dot(np.linalg.inv(rho*(dY.dot(dY.T))+ (m-1)*R))
    e = np.random.normal(0, 1, dY.shape)
    X_a = X_f + K@(y + e- H@X_f)
    R_temp = (y- X_a)@ (y- X_f).T/ 40
    HBH = (X_a- X_f)@ (y- X_f).T/ 40
    HBH_R = (y- X_f)@ (y- X_f).T/ 40
    return X_a, R_temp
    
#アンサンブルカルマンフィルターの計算。観測時刻が同一でなくとも計算できるようにした。計算式は授業のものを参考にした
def EnKF2(X_f_temp, X_f, y, m, R, H, rho=1, inf=1):
    y = y.reshape(len(y), 1)
    dX = (X_f- X_f.mean(axis=1, keepdims=True))*inf
    dX_temp = (X_f_temp- X_f_temp.mean(axis=1, keepdims=True))*inf
    dY = H.dot(dX)
    K = rho*(dX@ dX_temp.T)@H.T@ (np.linalg.inv(H@ (rho* (dX_temp@dX_temp.T))@ H.T+ (m-1)*R))
    #K = rho*(dX.dot(dY.T)).dot(np.linalg.inv(rho*(dY.dot(dY.T))+ (m-1)*R))
    e = np.random.normal(0, 1, dY.shape)
    return X_f + K.dot(y + e - H.dot(X_f_temp))
    
#アダプティブ法で計算する
def EnKF3(X_f, X_a, y, m, R, H, rho, delta_mean):
    y = y.reshape(len(y), 1)
    dX = X_f- X_f.mean(axis=1, keepdims=True)
    dY = H.dot(dX)
    d_ob = y- H@ X_f.mean(axis=1, keepdims=True)
    d_ab = H@(X_a.mean(axis=1, keepdims=True)- X_f.mean(axis=1, keepdims=True))
    Pb = dX@ dX.T/(m-1)
    delta = (d_ab.T@ d_ob)/ np.trace((H@ Pb@ H.T))
    delta_mean = 0.9* delta_mean + 0.1* np.abs(delta)
    Pb = Pb *(delta_mean)
    K = rho*(Pb)@H.T@ (np.linalg.inv(H@ (rho* (Pb))@ H.T+ R))
    e = np.random.normal(0, 1, dY.shape)
    return X_f + K.dot(y + e - H.dot(X_f)), delta_mean
    
#アンサンブルカルマンフィルターの計算。
def EnKF4(X_f, y, m, R, H, rho=1, inf=1):
    y = y.reshape(len(y), 1)
    dX = (X_f- X_f.mean(axis=1, keepdims=True))*inf
    dY = H.dot(dX)
    K = rho*(dX@ dX.T)@H.T@ (np.linalg.inv(H@ (rho* (dX@dX.T))@ H.T+ (m-1)*R))
    e = np.random.normal(0, 1, dY.shape)
    X_a = X_f + K@(y + e- H@X_f)
    R_temp = (y- X_a)@ (y- X_f).T/ 39
    HBH = (X_a- X_f)@ (y- X_f).T/ 39
    HBH_R = (y- X_f)@ (y- X_f).T/ 39
    delta = np.trace(HBH)/ (np.trace(dX@dX.T)/ 39)
    return X_a, R_temp, HBH, delta
    
#アンサンブルカルマンフィルターの計算。
def EnKF5(X_f, y, m, R, H, rho=1, inf=1, P_f=1):
    y = y.reshape(len(y), 1)
    dX = (X_f- X_f.mean(axis=1, keepdims=True))*inf
    dY = H.dot(dX)
    P_f = dX@ dX.T
    K = rho*(P_f)@H.T@ (np.linalg.inv(H@ (rho* (P_f))@ H.T+ (m-1)*R))
    e = np.random.normal(0, np.sqrt(R[0,0]), dY.shape)
    X_a = X_f + K@(y + e- H@X_f)
    R_temp = (y- H@X_a)@ (y- H@X_f).T/ (m-1)
    HBH = (H@X_a- H@X_f)@ (y- H@X_f).T/ (m-1)
    HBH_R = (y- H@X_f)@ (y- H@X_f).T/ (m-1)
    delta = np.trace(HBH)/ (np.trace(dX@dX.T)/ (m-1))
#    delta = delta/ np.trace(R_temp)*40
    return X_a, delta, R_temp, HBH, HBH_R, dX@dX.T, K

#アンサンブルカルマンフィルターの計算その２
def LETKF(X_f, y, m, R, H, inf):
    #yを一応縦の行列にしとく。dX,dYを計算してTT^Tをもとめる。求めたTT^Tを固有値分解してTの値を求める。
    y = y.reshape(len(y), 1)
    dX = X_f- X_f.mean(axis=1, keepdims=True)
    dY = H.dot(dX)
    D, U = np.linalg.eig((m-1)*np.eye(m)/inf + dY.T.dot(np.linalg.inv(R)).dot(dY))
    D = 1/D
    #求めたTの値からX_aの値を求める。ここまでpdfのDerivation of LETKFを参考にした
    X_a = X_f.mean(axis=1, keepdims=True) + dX.dot(U.dot(np.diag(D)).dot(U.T).dot(dY.T).dot(np.linalg.inv(R)).dot(y-H.dot(X_f.mean(axis=1, keepdims=True)))+np.sqrt(m-1)*U.dot(np.diag(np.sqrt(D))).dot(U.T))
    return X_a
    
#ローカリゼーションしたバージョン
def LETKF2(X_f, y, m, R, H, inf):
    J = len(y)
    y = y.reshape(len(y), 1)
    x_f = X_f.mean(axis=1, keepdims=True)
    dX = X_f- x_f
    dY = H.dot(dX)
    X_a = np.zeros(X_f.shape)
    for i in range(J):
        D, U = np.linalg.eigh((m-1)*np.eye(m)/inf + dY.T.dot(np.linalg.inv(R[i])).dot(dY))
        D = 1/D
        X_a[i,:] = (x_f + dX.dot(U.dot(np.diag(D)).dot(U.T).dot(dY.T).dot(np.linalg.inv(R[i])).dot(y-H.dot(x_f))+np.sqrt(m-1)*U.dot(np.diag(np.sqrt(D))).dot(U.T)))[i,:]
    return X_a
    
#計算を高速化した場合。上手くいかなかったので無視していい
def LETKF3(X_f, y, m, R, H, inf, rho):
    J = len(y)
    a = np.arange(J)
    y = y.reshape(len(y), 1)
    x_f = X_f.mean(axis=1, keepdims=True)
    dX = X_f - x_f
    dY = H.dot(dX)
    R_loc = R/ rho
    D, U = np.linalg.eigh((m-1)*np.eye(m)/inf + dY.T@(np.linalg.inv(R_loc))@(dY))
    D = 1/ D
    M1 = np.tile(np.eye(J).reshape(J, 1, J), [1, m, 1])
    M2 = np.eye(m)
    X_a = x_f + dX.reshape(1, 40, 39)@(U@(M1@D*M2)@(np.transpose(U, axes=[0,2,1]))@(dY.T)@(R_loc)@(y-H@(x_f))+np.sqrt(m-1)*U@(M1@np.sqrt(D)*M2)@(np.transpose(U, axes=[0,2,1])))
    return X_a[np.arange(J), np.arange(J), :]
    
    
    
    
    

