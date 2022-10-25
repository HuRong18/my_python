import math
import numpy as np
import torch
from scipy.linalg import expm

# 定义插值函数
import DNN_re
import util
import const

device = util.get_device()


def interpolate(matrix_c):
    len0 = len(matrix_c)
    len1 = len(matrix_c[0])
    len2 = len(matrix_c[0][0])
    len3 = len(matrix_c[0][0][0])
    vander = np.vander([i + 1 for i in range(len0)], increasing=True)
    # 结果
    res = np.zeros((len2, len3, len0, len1))

    matrix_c = matrix_c.swapaxes(0, 2).swapaxes(1, 3)
    for i in range(len2):
        for j in range(len3):
            res[i][j] = np.linalg.solve(vander, matrix_c[i][j])
    res = res.swapaxes(0, 2).swapaxes(1, 3)
    return res


# 改变输入矩阵维度
def transfer(*args):
    res = []
    for matrix in args:
        res.append(np.expand_dims(matrix, axis=0))
    return np.array(res)


def dataset(batch_size):
    data_set = []
    feature_set = []
    for i in range(batch_size):
        data = []
        feature = []
        for j in range(3):
            # A = 2*np.random.rand(const.dimension, const.dimension)-1
            # a = A.T
            # x = (A + a)/2
            # f = np.linalg.eigvals(x)
            A = np.random.rand(const.dimension, const.dimension)
            a = np.linalg.norm(A)
            x = A / a
            f = np.exp(x)
            # x = A
            # f = np.sqrt(x)
            data.append(x)
            feature.append(f)
        data_set.append(data)
        feature_set.append(feature)

    return np.asarray(data_set), np.asarray(feature_set)


def train():
    dnn = DNN_re.DNN([1 / 4, 2 / 4, 3 / 4], device)
    loss = 1
    i = 1
    for i in range(3000):
        # torch.set_default_dtype(torch.float64)
        x, feature = dataset(32)
        input = torch.from_numpy(x).to(torch.float32).to(device)
        expect: torch.Tensor = torch.from_numpy(feature).to(torch.float32).reshape(32, 3, const.dimension_mm, 1).to(
            device)
        loss = dnn.train_by_data(input, expect)
        NRMSE = math.sqrt(loss.item() / torch.pow(expect, 2).sum().item())
        print('第{}次,loss:{},NRMSE:{}'.format(i, loss, NRMSE))
        i += 1
    torch.save(dnn.state_dict(), 'dnn_params_test.pth')
    # dnn.load_state_dict(torch.load('dnn_params.pth'))
    print(dnn)
    # test = dnn([1, 2, 3], device)


def predict(filename):
    alpha_list = [1 / 4, 1 / 2, 3 / 4]
    test_net = DNN_re.DNN(const.alpha_list, device)
    test_net.load_state_dict(torch.load(filename))
    test_net.eval()
    with torch.no_grad():
        for route in range(1000):
            x, feature = dataset(1)
            input = torch.from_numpy(x).to(torch.float32).to(device)
            expect: torch.Tensor = torch.from_numpy(feature).to(torch.float32).reshape(3, const.dimension_mm, 1) \
                .to(device)
            appro = test_net.forward(input)  # 发送给workers的计算任务
            coff = interpolate(np.expand_dims(appro.squeeze(0).numpy(), axis=1))  # 得到的多项式系数
            res = []
            for alpha in alpha_list:
                tmp = np.zeros((1, const.dimension_mm, 1))
                for index, u in enumerate(coff):
                    tmp += u * alpha ** index
                res.append(tmp)
            res = torch.from_numpy(np.concatenate(res, 0))
            for i in range(len(alpha_list)):
                NRMSE = math.sqrt(torch.pow(expect[i] - res[i], 2).sum().item()
                                  / torch.pow(expect[i], 2).sum().item())
                print('第{}轮测试的第{}个NRMSE:{}'.format(route + 1, i + 1, NRMSE))
                i += 1


if __name__ == '__main__':
    # train()
    predict('./dnn_params_test.pth')
