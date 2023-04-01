# -*- coding: UTF-8 -*-
"""
此脚本用于展示模型参数估计值的分布
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model


def generate_data():
    """
    随机生成数据
    """
    # Python2和Python3的range并不兼容，所以使用list(range(xx, xx))
    x = np.array(list(range(100)))
    error = np.round(np.random.randn(100), 2)
    y = x + error
    return pd.DataFrame({"x": x, "y": y})


def train_model(x, y):
    """
    利用训练数据，估计模型参数
    """
    # 创建一个线性回归模型
    model = linear_model.LinearRegression()
    # 训练模型，估计模型参数
    model.fit(x, y)
    return model


def visualize_params(params):
    """
    可视化模型参数估计值的分布
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    n, bins, _ = ax.hist(params, bins=50, density=True, color="b",
            edgecolor='black', linewidth=1.2,  alpha=0.5)
    # 用多项式拟合得到的直方图
    z1 = np.polyfit(bins[:-1], n, 10)
    p1 = np.poly1d(z1)
    ax.plot(bins[:-1], p1(bins[:-1]), "r-.")
    plt.show()


def run():
    """
    产生“结构”相似的随机数据，并它训练线性回归模型
    以此展示模型参数估计值服从正态分布
    """
    features = ["x"]
    label = ["y"]
    coefs = []
    intercepts = []
    # 循环运行1000次
    for _ in range(1000):
        data = generate_data()
        model = train_model(data[features], data[label])
        # 记录每一次参数a的估计值
        coefs.append(model.coef_[0][0])
        # 记录每一次参数b的估计值
        intercepts.append(model.intercept_[0])
    visualize_params(coefs)
    visualize_params(intercepts)

    
if __name__ == "__main__":
    run()