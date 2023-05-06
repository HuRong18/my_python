import torch
import numpy as np
import math


worker_number=6
threshold_number=5
dimension = 5
alpha_list = [1, 2, 3, 4, 5, 6]
dimension_mm = dimension * dimension
dimension_s = 1
input_k = 3
input_k_BACC=3



def alpha_list_AICC(k):
    alpha_list = []
    for index in range(1, k + 1):
        alpha_list.append(index / (k + 1))
    return alpha_list


def alpha_list_BACC(k):
    alpha_list = []
    for index in range(0, k):
        alpha_list.append(np.cos(((2 * index + 1) * math.pi) / (2 * k)))
    return alpha_list


def z_list_BACC(k):
    alpha_list = []
    for index in range(0, k+1):
        alpha_list.append(np.cos((index * math.pi) / k))
    return alpha_list

if __name__ == '__main__':
    # a = np.random.rand(4, 1, 64, 1)
    # b = main_rebuild.interpolate(a)
    # vander = np.vander([i + 1 for i in range(4)], increasing=True)
    # c = []
    # for i in range(4):
    #     tmp = np.zeros((1, 64, 1))
    #     for j in range(4):
    #         tmp += vander[i][j] * b[j]
    #     c.append(tmp)
    # c = np.concatenate(c, 0)
    # loss = (c - a).sum()
    # print(loss)
    tmp = alpha_list_BACC(5)
    print(z_list_BACC(5))
