#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:18:18 2018

@author: tgbaggio
"""


def word_count(data):
    """
    输入一个字符串列表，统计列表中字符串出现的次数

    参数
    ----
    data: list[str]，需要统计的字符串列表

    返回
    ----
    re: dict，结果hash表，key为字符串，value为对应的出现次数
    """
    re = {}
    for i in data:
        re[i] = re.get(i, 0) + 1
    return re


if __name__ == "__main__":
    data = ["a", "b", "a", "c"]
    print(f"The result is {word_count(data)}")
