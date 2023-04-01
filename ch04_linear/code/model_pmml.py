# -*- coding: UTF-8 -*-
"""
此脚本用于展示使用PMML来保存模型
"""


import os

import pandas as pd
from sklearn import linear_model
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml


def read_data(path):
    """
    使用pandas读取数据
    """
    return pd.read_csv(path)


def train_and_save_model(data, model_path):
    """
    利用sklearn2pmml将模型存储为PMML
    """
    model = PMMLPipeline([("regressor", linear_model.LinearRegression())])
    model.fit(data[["x"]], data["y"])
    sklearn2pmml(model, model_path)
    

def run_model(data, model_path):
    """
    运行模型
    """
    train_and_save_model(data, model_path)


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\simple_example.csv" % home_path
    else:
        data_path = f"{home_path}/simple_example.csv"
    data = read_data(data_path)
    model_path = "linear_model_pmml"
    run_model(data, model_path)