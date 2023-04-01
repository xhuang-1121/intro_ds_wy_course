# -*- coding: UTF-8 -*-
"""
此脚本用于展示使用惩罚项解决模型幻觉的问题
"""


import os

import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd


def read_data(path):
    """
    使用pandas读取数据
    """
    return pd.read_csv(path)


def generate_random_var():
    """
    生成不相关的特征
    """
    np.random.seed(4873)
    return np.random.randint(2, size=20)


def train_model(x, y, alpha):
    """
    训练模型
    """
    # 数据里面已经包含常变量，所以fit_intercept=False
    model = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    model.fit(x, y)
    return model


def visualize_model(X, Y, alphas, coefs):
    """
    模型可视化
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, coefs[:, 1], "r:", label='a')
    ax.plot(alphas, coefs[:, 2], "g", label='b')
    ax.plot(alphas, coefs[:, 0], "b-.", label='c')
    legend = plt.legend(loc=4, shadow=True)
    legend.get_frame().set_facecolor("#6F93AE")
    ax.set_yticks(np.arange(-1, 1.3, 0.3))
    ax.set_xscale("log")
    ax.set_xlabel("$alpha$")
    plt.show()


def run_model(data):
    """
    运行模型
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    X = data[features]
    # 加入新的随机变量，这个变量的系数应为0
    X["z"] = generate_random_var()
    # 加入常变量const
    X = sm.add_constant(X)
    alphas = np.logspace(-4, -0.5, 100)
    coefs = []
    for alpha in alphas:
        model = train_model(X, Y, alpha)
        coefs.append(model.coef_)
    coefs = np.array(coefs)
    # 可视化惩罚项效果
    visualize_model(X, Y, alphas, coefs)


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\simple_example.csv" % home_path
    else:
        data_path = f"{home_path}/simple_example.csv"
    data = read_data(data_path)
    run_model(data)