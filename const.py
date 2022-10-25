import torch
import main_rebuild

dimension = 8
alpha_list = [1, 2, 3, 4]
dimension_mm = 64

import numpy as np

if __name__ == '__main__':
    a=np.random.rand(4, 1, 64, 1)
    b=main_rebuild.interpolate(a)
    vander = np.vander([i + 1 for i in range(4)], increasing=True)
    c=[]
    for i in range(4):
        tmp=np.zeros((1, 64, 1))
        for j in range(4):
            tmp+=vander[i][j]*b[j]
        c.append(tmp)
    c=np.concatenate(c, 0)
    loss=(c-a).sum()
    print(loss)

