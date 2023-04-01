# -*- coding: UTF-8 -*-
"""
此脚本用于实现线性回归模型的统计分析
"""

# 保证脚本与Python2兼容
from __future__ import print_function

import os

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import pandas as pd


def read_data(path):
    """
    使用pandas读取数据
    
    参数
    ----
    path: String，数据的路径
    
    返回
    ----
    data: DataFrame，建模数据
    """
    return pd.read_csv(path)


def train_model(x, y):
    """
    利用训练数据，估计模型参数
    
    参数
    ----
    x: DataFrame，特征
    
    y: DataFrame，标签
    
    返回
    ----
    res : RegressionResults, 训练好的线性模型
    """
    # 创建一个线性回归模型
    model = sm.OLS(y, x)
    return model.fit()


def model_summary(res):
    """
    分析线性回归模型的统计性质
    """
    # 整体统计分析结果
    print(res.summary())
    # 用f test检测x对应的系数a是否显著
    print("检验假设x的系数等于0：")
    print(res.f_test("x=0"))
    # 用f test检测常量b是否显著
    print("检测假设const的系数等于0：")
    print(res.f_test("const=0"))
    # 用f test检测a=1, b=0同时成立的显著性
    print("检测假设x的系数等于1和const的系数等于0同时成立：")
    print(res.f_test(["x=1", "const=0"]))

    
def get_prediction(res, x):
    """
    得到模型的预测结果以及结果的上下限
    """
    prstd, ci_low, ci_up = wls_prediction_std(res, alpha=0.05)
    pred = res.predict(x)
    return pd.DataFrame({"ci_low": ci_low, "pred": pred, "ci_up": ci_up})


def visualize_model(pred, x, y):
    """
    模型可视化
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    # 画点图，用蓝色圆点表示原始数据
    ax.scatter(x, y, color='b')
    # 将模型预测结果画在图上
    ax.plot(x, pred["pred"], "r", label="prediction")
    # 将预测结果置信区间的上下限画在图上
    ax.plot(x, pred["ci_low"], "r--", label="95% confidence interval")
    ax.plot(x, pred["ci_up"], "r--", label="")
    plt.legend(shadow=True)
    # 展示上面所画的图片。图片将阻断程序的运行，直至所有的图片被关闭
    # 在Python shell里面，可以设置参数"block=False"，使阻断失效。
    plt.show()


def run_model(data):
    """
    线性回归模型统计建模步骤展示
    """
    features = ["x"]
    labels = ["y"]
    # 加入常量变量
    X = sm.add_constant(data[features])
    # 构建模型
    res = train_model(X, data[labels])
    # 分析模型效果
    model_summary(res)
    # 得到模型的预测结果
    pred = get_prediction(res, X)
    # 将模型结果可视化
    visualize_model(pred, data[features], data[labels])


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\simple_example.csv" % home_path
    else:
        data_path = f"{home_path}/simple_example.csv"
    data = read_data(data_path)
    run_model(data)