#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:54:51 2018

@author: tgbaggio
"""



# 元组的操作方法
t = (1, 2, 3, "a", "b")

# 取第一个元素
t[0]

# 取最后一个元素
t[-1]

# 取1-3个元素
t[:3]

# 列表的操作方法
y = [1, 2, 3, 4]

# 取第一个元素
y[0]

# 取最后一个元素
y[-1]

# 在列表的末尾追加一个元素
y.append("aa")

# 将两个列表合并成为一个列表
y += ["bb", "cc"]

# 在index=3处，插入new_item这样一个元素
y.insert(3, "new_item")

# 去除列表里面排在最前面的元素3
y.remove(3)

# 字典的操作方法
d = {"a": 1, "b": 2}

# 取出key=“a”的value
d["a"]

# 取出key=“c”的value
d["c"]

# 取出key=“b”的value，如果“b”不是字典的key，则返回not exist
d.get("b", "not exist")

# 新定义键值对“c”, 3
d["c"] = 3

# 遍历元祖
for i in t:
    print(i)

# 遍历字典
for i in d:
    print(f"the key is {i}", f"the value is {d[i]}")
