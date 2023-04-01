# -*- coding: UTF-8 -*-
"""
此脚本用于展示过度拟合的问题
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


def read_data(path):
    """
    使用pandas读取数据
    """
    return pd.read_csv(path)


def evaluate_model(model, test_data, features, labels, featurizer):
    """
    计算线性模型的均方差和决定系数
    """
    # 均方差(The mean squared error)，均方差越小越好
    error = np.mean(
        (model.predict(featurizer.fit_transform(test_data[features])) - test_data[labels]) ** 2)
    # 决定系数(Coefficient of determination)，决定系数越接近1越好
    score = model.score(featurizer.fit_transform(test_data[features]), test_data[labels])
    return error, score


def train_model(train_data, features, labels, featurizer):
    """
    利用训练数据，估计模型参数
    """
    # 创建一个线性回归模型
    model = linear_model.LinearRegression(fit_intercept=False)
    # 训练模型，估计模型参数
    model.fit(featurizer.fit_transform(train_data[features]), train_data[labels])
    return model


def visualize_model(model, featurizer, data, features, labels, evaluation):
    """
    模型可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 创建一个图形框
    fig = plt.figure(figsize=(10, 10), dpi=80)
    # 在图形框里只画一幅图
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        _visualization(ax, data, model[i], featurizer[i], evaluation[i], features, labels)
    plt.show()


def _visualization(ax, data, model, featurizer, evaluation, features, labels):
    """
    实现可视化
    """
    # 画点图，用蓝色圆点表示原始数据
    ax.scatter(data[features], data[labels], color='b')
    # 画线图，用红色线条表示模型结果
    ax.plot(data[features], model.predict(featurizer.fit_transform(data[features])),
            color="r")
    # 显示均方差和决定系数
    ax.text(0.01, 0.99,
            u'%s%.3f\n%s%.3f' % ("mse: ", evaluation[0], "R2: ", evaluation[1]),
            style="italic", verticalalignment="top", horizontalalignment="left",
            transform=ax.transAxes, color="m", fontsize=13)


def run_model(data):
    """
    运行模型
    """
    features = ["x"]
    labels = ["y"]
    # 划分训练集和测试集
    train_data = data[:15]
    test_data = data[15:]
    featurizer = []
    # overfitting_model和overfitting_evaluation记录区分训练集和测试集的模型效果
    overfitting_model = []
    overfitting_evaluation = []
    # model和evaluation记录过度拟合的模型效果
    model = []
    evaluation = []
    for i in range(1, 11, 3):
        featurizer.append(PolynomialFeatures(degree=i))
        # 产生并训练模型
        overfitting_model.append(train_model(train_data, features, labels, featurizer[-1]))
        # 评价模型效果
        overfitting_evaluation.append(
            evaluate_model(overfitting_model[-1], test_data, features, labels, featurizer[-1]))
        # 过度拟合
        model.append(train_model(data, features, labels, featurizer[-1]))
        evaluation.append(evaluate_model(model[-1], data, features, labels, featurizer[-1]))
    # 图形化模型结果
    visualize_model(model, featurizer, data, features, labels, evaluation)
    visualize_model(overfitting_model, featurizer,
                    data, features, labels, overfitting_evaluation)


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\simple_example.csv" % home_path
    else:
        data_path = f"{home_path}/simple_example.csv"
    data = read_data(data_path)
    run_model(data)