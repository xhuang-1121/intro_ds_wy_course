# -*- coding: UTF-8 -*-
"""
此脚本用于展示使用假设检验解决模型幻觉的问题
"""


# 保证脚本与Python2兼容
from __future__ import print_function

import os

import numpy as np
import statsmodels.api as sm
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


def train_model(X, Y):
    """
    训练模型
    """
    model = sm.OLS(Y, X)
    return model.fit()


def evaluate_model(res):
    """
    分析线性回归模型的统计性质
    """
    # 整体统计分析结果
    print(res.summary())
    # 用f test检测x对应的系数a是否显著
    print("检验假设z的系数等于0：")
    print(res.f_test("z=0"))
    # 用f test检测常量b是否显著
    print("检测假设const的系数等于0：")
    print(res.f_test("const=0"))
    # 用f test检测a=1, b=0同时成立的显著性
    print("检测假设z和const的系数同时等于0：")
    print(res.f_test(["z=0", "const=0"]))


def run_model(data):
    """
    运行模型
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    _X = data[features]
    # 加入新的随机变量，次变量的系数应为0
    _X["z"] = generate_random_var()
    # 加入常量变量
    X = sm.add_constant(_X)
    res = train_model(X, Y)
    evaluate_model(res)
    # 使用如下的代码去掉变量z，然后重新训练模型
    # X = X.drop(["z"], axis=1)
    # train_model(X, Y)


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\simple_example.csv" % home_path
    else:
        data_path = f"{home_path}/simple_example.csv"
    data = read_data(data_path)
    run_model(data)