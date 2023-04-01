# -*- coding: UTF-8 -*-
"""
此脚本用于比较LAD线性回归和OLS线性回归
"""


import statsmodels.api as sm
from sklearn import linear_model
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_data():
    """
    随机生成数据
    """
    np.random.seed(4889)
    # Python2和Python3的range并不兼容，所以使用list(range(10, 29))
    x = np.array([10] + list(range(10, 29)))
    error = np.round(np.random.randn(20), 2)
    y = x + error
    # 增加异常点
    x = np.append(x, 29)
    y = np.append(y, 29 * 10)
    return pd.DataFrame({"x": x, "y": y})


def train_OLS(x, y):
    """
    训练OLS线性回归模型，并返回模型预测值
    """
    model = linear_model.LinearRegression()
    model.fit(x, y)
    return model.predict(x)


def train_LAD(x, y):
    """
    训练LAD线性回归模型，并返回模型预测值
    """
    X = sm.add_constant(x)
    model = QuantReg(y, X)
    model = model.fit(q=0.5)
    return model.predict(X)
    
    
def visualize_model(x, y, ols, lad):
    """
    模型结果可视化
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 设置坐标轴
    ax.set_xlabel("$x$")
    ax.set_xticks(range(10, 31, 5))
    ax.set_ylabel("$y$")
    # 画点图，点的颜色为蓝色，半透明
    ax.scatter(x, y, color="b", alpha=0.4)
    # 将模型结果可视化出来
    # 用红色虚线表示OLS线性回归模型的结果
    ax.plot(x, ols, 'r--', label="OLS")
    # 用黑色实线表示LAD线性回归模型的结果
    ax.plot(x, lad, 'k', label="LAD")
    plt.legend(shadow=True)
    # 展示上面所画的图片。图片将阻断程序的运行，直至所有的图片被关闭
    # 在Python shell里面，可以设置参数"block=False"，使阻断失效
    plt.show()


def OLS_vs_LAD(data):
    """
    比较OLS模型和LAD模型的差异
    """
    features = ["x"]
    label = ["y"]
    ols = train_OLS(data[features], data[label])
    lad = train_LAD(data[features], data[label])
    visualize_model(data[features], data[label], ols, lad)

    
if __name__ == "__main__":
    data = generate_data()
    OLS_vs_LAD(data)