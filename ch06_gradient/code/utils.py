# -*- coding: UTF-8 -*-
"""
此脚本用于随机生成线性模型数据、定义模型以及其他工具
"""


import numpy as np
import tensorflow as tf
import os
import math


def generate_linear_data(dimension, num):
    """
    随机产生线性模型数据
    参数
    ----
    dimension ：int，自变量个数
    num ：int，数据个数
    返回
    ----
    x ：np.array，自变量
    y ：np.array，因变量
    """
    np.random.seed(1024)
    beta = np.array(range(dimension)) + 1
    x = np.random.random((num, dimension))
    epsilon = np.random.random((num, 1))
    # 将被预测值写成矩阵形式，会极大加快速度
    y = x.dot(beta).reshape((-1, 1)) + epsilon
    return x, y


def create_linear_model(dimension):
    """
    搭建模型，包括数据中的自变量，应变量和损失函数
    参数
    ----
    dimension : int，自变量的个数
    返回
    ----
    model ：dict，里面包含模型的参数，损失函数，自变量，应变量
    """
    np.random.seed(1024)
    # 定义自变量和应变量
    x = tf.placeholder(tf.float64, shape=[None, dimension], name='x')
    # 将被预测值写成矩阵形式，会极大加快速度
    y = tf.placeholder(tf.float64, shape=[None, 1], name="y")
    # 定义参数估计值和预测值
    beta_pred = tf.Variable(np.random.random([dimension, 1]))
    y_pred = tf.matmul(x, beta_pred, name="y_pred")
    # 定义损失函数
    loss = tf.reduce_mean(tf.square(y_pred - y))
    return {
        "loss_function": loss,
        "independent_variable": x,
        "dependent_variable": y,
        "prediction": y_pred,
        "model_params": beta_pred,
    }


def create_summary_writer(log_path):
    """
    检查所给路径是否已存在，如果存在删除原有日志。并创建日志写入对象
    参数
    ----
    logPath ：string，日志存储路径
    返回
    ----
    summaryWriter ：FileWriter，日志写入器
    """
    if tf.gfile.Exists(log_path):
        tf.gfile.DeleteRecursively(log_path)
    return tf.summary.FileWriter(log_path, graph=tf.get_default_graph())


def gradient_descent(X, Y, model, learning_rate=0.01, max_iter=10000, tol=1.e-6):
    """
    利用梯度下降法训练模型。
    参数
    ----
    X : np.array, 自变量数据
    Y : np.array, 因变量数据
    model : dict, 里面包含模型的参数，损失函数，自变量，应变量。
    """
    # 确定最优化算法
    method = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = method.minimize(model["loss_function"])
    # 增加日志
    tf.summary.scalar("loss_function", model["loss_function"])
    tf.summary.histogram("params", model["model_params"])
    tf.summary.scalar("first_param", tf.reduce_mean(model["model_params"][0]))
    tf.summary.scalar("last_param", tf.reduce_mean(model["model_params"][-1]))
    summary = tf.summary.merge_all()
    # 在程序运行结束之后，运行如下命令，查看日志
    # tensorboard --logdir logs/
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        summary_writer = create_summary_writer("logs\\gradient_descent")
    else:
        summary_writer = create_summary_writer("logs/gradient_descent")
    # tensorflow开始运行
    sess = tf.Session()
    # 产生初始参数
    init = tf.global_variables_initializer()
    # 用之前产生的初始参数初始化模型
    sess.run(init)
    # 迭代梯度下降法
    step = 0
    prev_loss = np.inf
    diff = np.inf
    # 当损失函数的变动小于阈值或达到最大循环次数，则停止迭代
    while (step < max_iter) & (diff > tol):
        _, summary_str, loss = sess.run(
            [optimizer, summary, model["loss_function"]],
            feed_dict={model["independent_variable"]: X,
                       model["dependent_variable"]: Y})
        # 将运行细节写入目录
        summary_writer.add_summary(summary_str, step)
        # 计算损失函数的变动
        diff = abs(prev_loss - loss)
        prev_loss = loss
        step += 1
    summary_writer.close()


def stochastic_gradient_descent(X, Y, model, method,
                                mini_batch_fraction=0.01, epoch=10000, tol=1.e-6):
    """
    利用随机梯度下降法训练模型。
    参数
    ----
    X : np.array, 自变量数据
    Y : np.array, 因变量数据
    model : dict, 里面包含模型的参数，损失函数，自变量，应变量
    """
    # 确定最优化算法
    optimizer = method.minimize(model["loss_function"])
    # 增加日志
    tf.summary.scalar("loss_function", model["loss_function"])
    tf.summary.histogram("params", model["model_params"])
    tf.summary.scalar("first_param", tf.reduce_mean(model["model_params"][0]))
    tf.summary.scalar("last_param", tf.reduce_mean(model["model_params"][-1]))
    summary = tf.summary.merge_all()
    # 在程序运行结束之后，运行如下命令，查看日志
    # tensorboard --logdir logs/
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        summary_writer = create_summary_writer("logs\\stochastic_gradient_descent")
    else:
        summary_writer = create_summary_writer("logs/stochastic_gradient_descent")
    # tensorflow开始运行
    sess = tf.Session()
    # 产生初始参数
    init = tf.global_variables_initializer()
    # 用之前产生的初始参数初始化模型
    sess.run(init)
    # 迭代梯度下降法
    step = 0
    batch_size = int(X.shape[0] * mini_batch_fraction)
    batch_num = int(math.ceil(1 / mini_batch_fraction))
    prev_loss = np.inf
    diff = np.inf
    # 当损失函数的变动小于阈值或达到最大训练轮次，则停止迭代
    while (step < epoch) & (diff > tol):
        for i in range(batch_num):
            # 选取小批次训练数据
            batch_x = X[i * batch_size: (i + 1) * batch_size]
            batch_y = Y[i * batch_size: (i + 1) * batch_size]
            # 迭代模型参数
            sess.run([optimizer],
                     feed_dict={model["independent_variable"]: batch_x,
                                model["dependent_variable"]: batch_y})
            # 计算损失函数并写入日志
            summary_str, loss = sess.run(
                [summary, model["loss_function"]],
                feed_dict={model["independent_variable"]: X,
                           model["dependent_variable"]: Y})
            # 将运行细节写入目录
            summary_writer.add_summary(summary_str, step * batch_num + i)
            # 计算损失函数的变动
            diff = abs(prev_loss - loss)
            prev_loss = loss
            if diff <= tol:
                break
        step += 1
    summary_writer.close()
