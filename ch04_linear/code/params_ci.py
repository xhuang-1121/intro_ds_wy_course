# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何正确理解参数的置信区间
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd


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
    model = sm.OLS(y, x)
    return model.fit()


def visualize_ci(ci):
    """
    可视化参数a估计值置信区间的分布
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 将每一个95%置信区间用竖线表示
    for i in range(len(ci) - 1):
        ci_low = ci[i][0]
        ci_up = ci[i][1]
        # 如果置信区间不包含1，则用红色表示，否则用蓝色表示
        include_one = (ci_low < 1) & (ci_up > 1)
        colors = "b" if include_one else "r"
        ax.vlines(x=i + 1, ymin=ci_low, ymax=ci_up, colors=colors)
    # 用黑线将真实值1表示出来
    ax.hlines(1, xmin=0, xmax=len(ci))
    plt.show()


def run():
    """
    产生“结构”相似的随机数据，并它训练线性回归模型，得到模型参数的置信区间
    以此展示模型参数估计值置信区间的真实含义
    """
    features = ["x"]
    label = ["y"]
    ci = []
    # 循环运行100次
    for _ in range(100):
        data = generate_data()
        X = sm.add_constant(data[features])
        re = train_model(X, data[label])
        # 记录每一次参数a的95置信区间
        ci.append(re.conf_int(alpha=0.05).loc["x"].values)
    visualize_ci(ci)


if __name__ == "__main__":
    run()
