from scipy.special import erf
import numpy as np


# 線形カーネル（sigma_eがある程度大きくないとエラー吐きます）
def linear(x_1, x_2):
    return x_1*x_2

# 多項式カーネル（sigma_eがある程度大きくないとエラー吐きます）
def cubic(x_1, x_2, d=1):
    return np.power(1+x_1*x_2, d)

# RBFカーネル
def rbf(x_1, x_2, theta_1=1, theta_2=1):
    return theta_1*np.exp(-np.abs(x_1-x_2)**2/theta_2)

# 指数カーネル
def exponential(x_1, x_2, theta=1):
    return np.exp(-np.abs(x_1-x_2)/theta)

# 周期カーネル
def periodic(x_1, x_2, theta_1=1, theta_2=1):
    return np.exp(theta_1*np.cos(np.abs(x_1-x_2)/theta_2))

# step関数族の定義に使われる関数
def J(theta, m):
    if m==0:
        return np.pi - theta
    elif m==1:
        return np.sin(theta) + (np.pi - theta)*np.cos(theta)
    elif m==2:
        return 3*np.sin(theta)*np.cos(theta) + (np.pi - theta)*(1+2*np.cos(theta)**2)

# step関数族の活性化関数に対応するカーネル関数
def step(x_1, x_2, m=0, sigma_w=1, sigma_v=1, sigma_b=0):
    Sigma_11 = sigma_w**2*x_1*x_1+sigma_b**2
    Sigma_12 = sigma_w**2*x_1*x_2+sigma_b**2
    Sigma_22 = sigma_w**2*x_2*x_2+sigma_b**2

    cos = Sigma_12/np.sqrt(Sigma_11*Sigma_22)
    theta = np.arccos(np.clip(cos, -1, 1))
    return np.power(Sigma_11*Sigma_22, m/2)*J(theta, m)/2/np.pi + sigma_b**2

# 誤差関数の活性化関数に対応するカーネル関数
def erf(x_1, x_2, sigma_w=1, sigma_v=1, sigma_b=0):
    Sigma_11 = sigma_w**2*x_1*x_1+sigma_b**2
    Sigma_12 = sigma_w**2*x_1*x_2+sigma_b**2
    Sigma_22 = sigma_w**2*x_2*x_2+sigma_b**2

    sin = 2*Sigma_12/np.sqrt((1+2*Sigma_11)*(1+2*Sigma_22))
    return 2*sigma_v**2*np.arcsin(sin)/np.pi + sigma_b**2
