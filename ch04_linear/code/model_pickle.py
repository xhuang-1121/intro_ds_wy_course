# -*- coding: UTF-8 -*-
"""
此脚本用于展示使用pickle来保存和读取模型
"""


# 保证脚本与Python2兼容
from __future__ import print_function

import os

import pandas as pd
import pickle
from sklearn import linear_model


def read_data(path):
    """
    使用pandas读取数据
    """
    return pd.read_csv(path)


def train_and_save_model(data, model_path):
    """
    使用pickle保存训练好的模型
    """
    model = linear_model.LinearRegression()
    model.fit(data[["x"]], data[["y"]])
    pickle.dump(model, open(model_path, "wb"))
    return model


def load_model(model_path):
    """
    读取模型
    """
    return pickle.load(open(model_path, "rb"))


def run_model(data, model_path):
    """
    运行模型
    """
    # 保存模型
    original_model = train_and_save_model(data, model_path)
    print(f"保存的模型对1的预测值：{original_model.predict([[1]])}")
    # 读取模型
    model = load_model(model_path)
    print(f"读取的模型对1的预测值：{model.predict([[1]])}")
    return model


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\simple_example.csv" % home_path
    else:
        data_path = f"{home_path}/simple_example.csv"
    data = read_data(data_path)
    model_path = "linear_model_pickle"
    run_model(data, model_path)