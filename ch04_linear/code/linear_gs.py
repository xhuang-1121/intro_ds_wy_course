# -*- coding: UTF-8 -*-
"""
此脚本用于展示使用网格搜寻的方法找到最佳的超参数
"""


import os

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
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


def visualize_model(alphas, scores):
    """
    将结果可视化
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(1, 1, 1)
    for i, (alpha, score) in enumerate(zip(alphas, scores)):
        ax.bar(i, score)
    # 调整x轴刻度的说明
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels(alphas)
    ax.set_xlabel("$alpha$")
    ax.set_ylabel("$R^2$")
    plt.show()


def run_model(data):
    """
    运行模型
    """
    # 定义X, Y
    features = ["x"]
    label = ["y"]
    X = data[features]
    X["z"] = generate_random_var()
    Y = data[label]
    # 定义备选的超参数集
    params = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 2]}
    # 定义使用模型
    model = linear_model.Lasso()
    # 定义网格搜寻
    gs = GridSearchCV(model, params, cv=2)
    # 训练模型
    gs.fit(X, Y)
    # mean_test_score里面保存的是模型'score'里面保存的得分
    # 在这里score表示的决定系数，越靠近1，模型的效果越好
    visualize_model(params["alpha"], gs.cv_results_['mean_test_score'])


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\simple_example.csv" % home_path
    else:
        data_path = f"{home_path}/simple_example.csv"
    data = read_data(data_path)
    run_model(data)