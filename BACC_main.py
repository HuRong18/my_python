import math
import numpy as np
import torch
from numpy import pi
from scipy.linalg import expm

# 定义插值函数
import BACC_DNN
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
        for j in range(const.input_k_BACC):
            # A = 2 * np.random.rand(const.dimension, const.dimension) - 1
            k = []
            for d in range(const.dimension_mm):
                tmp = np.random.uniform(-1, 1)
                k.append(tmp)
            A = np.asarray(k).reshape(const.dimension, const.dimension)
            a = A.T
            x = (A + a) / 2
            f = np.linalg.eigvalsh(x)
            f = np.sort(f, 0)  # 计算矩阵特征值

            # A = np.random.rand(const.dimension, const.dimension)
            # a = A.T
            # x = (A + a) / 2
            # w, v = np.linalg.eigh(x)
            # # lamda=np.linalg.eig(x)
            # index = np.argmax(w)
            # f = v[:, index]  # 计算矩阵最大特征值对应的特征向量
            # if f[0] < 0:
            #     f = f * (-1)

            # I=np.identity(const.dimension)
            # k = []
            # for d in range(const.dimension_mm):
            #     tmp = np.random.uniform(-2/const.dimension, 2/const.dimension)
            #     k.append(tmp)
            # E = np.asarray(k).reshape(const.dimension, const.dimension)
            # for i in range(const.dimension):
            #     for j in range(const.dimension):
            #         if i==j:
            #           E[i][j]=0
            # x=I-E
            # f=np.linalg.det(x)#计算实值方阵的行列式
            # # E=np.diag([0]*const.dimensionm)

            # A = np.random.rand(const.dimension, const.dimension)
            # a = np.linalg.norm(A)
            # x = A / a
            # f = np.exp(x)#计算指数函数

            # A=np.random.rand(const.dimension,const.dimension_mm)
            # x = A
            # f = np.sqrt(x)#计算开方根函数
            # k = []
            # for d in range(const.dimension_mm):
            #     tmp = np.random.uniform(0, pi*2)
            #     k.append(tmp)
            # x = np.asarray(k).reshape(const.dimension, const.dimension)
            # f = np.sin(x)  # 计算正余弦函数

            data.append(x)
            feature.append(f)
        data_set.append(data)
        feature_set.append(feature)

    return np.asarray(data_set), np.asarray(feature_set)


def train():
    dnn = BACC_DNN.DNN(const.alpha_list_BACC(const.input_k_BACC), device)
    loss = 1
    i = 1
    tmp = 0
    # dnn.load_state_dict(torch.load('dnn_params_BACC_10.pth'))
    while i <= 50000:
        # torch.set_default_dtype(torch.float64)
        x, feature = dataset(32)
        input = torch.from_numpy(x).to(torch.float32).to(device)
        loss = dnn.train_by_data(input, input)
        NRMSE = math.sqrt(loss.item() / torch.pow(input, 2).sum().item())
        print('第{}次,loss:{},NRMSE:{}'.format(i, loss, NRMSE))
        if i in range(49900, 50000):
            tmp += NRMSE
        i += 1
    NRMSE_mean = tmp / 100
    print(NRMSE_mean)
    torch.save(dnn.state_dict(), 'dnn_params_BACC_5.pth')
    # dnn.load_state_dict(torch.load('dnn_params.pth'))
    # print(dnn)
    # test = dnn([1, 2, 3], device)


def train_compute():
    dnn = BACC_DNN.DNN(const.alpha_list_BACC(3), device)
    i = 1
    tmp = 0
    dnn.load_state_dict(torch.load('dnn_params_BACC_10.pth'))
    while i <= 50000:
        with torch.no_grad():
            # torch.set_default_dtype(torch.float64)
            x, feature = dataset(32)
            input = torch.from_numpy(x).to(torch.float32).to(device)
            output = dnn.forward(input)
        expect: torch.Tensor = torch.from_numpy(feature).to(torch.float32).reshape(-1, const.input_k, const.dimension,
                                                                                   1).to(device)
        loss = dnn.train_by_data_compute(input, output, expect)
        NRMSE = math.sqrt(loss.item() / torch.pow(input, 2).sum().item())
        print('第{}次,loss:{},NRMSE:{}'.format(i, loss, NRMSE))
        if i in range(49900, 50000):
            tmp += NRMSE
        i += 1
    NRMSE_mean = tmp / 100
    print(NRMSE_mean)
    torch.save(dnn.state_dict(), 'dnn_params_BACC_10.pth')


def predict(filename):
    alpha_list = const.alpha_list_BACC(3)
    test_net = BACC_DNN.DNN(const.z_list_BACC(10), device)
    test_net.load_state_dict(torch.load(filename))
    test_net.eval()
    with torch.no_grad():
        for route in range(1000):
            x, feature = dataset(1)
            input = torch.from_numpy(x).to(torch.float32).to(device)
            expect: torch.Tensor = torch.from_numpy(feature).to(torch.float32). \
                reshape(const.input_k, const.dimension, 1).to(device)
            code_input = test_net.forward(input)  # 发送给workers的计算任务
            y = []
            res_Berrut = []
            for i in range(len(const.z_list_BACC(10))):
                # tmp = code_input[:, i:i + 1, :, :]
                f = np.linalg.eigvalsh(code_input[:, i:i + 1, :, :].numpy())
                # f_input = np.linalg.eigvalsh(input[:, i:i + 1, :, :].numpy())
                # f_input = np.sort(f_input, 0)
                f = np.sort(f, 0)  # 计算矩阵特征值
                y.append(f)
            for i in range(len(alpha_list)):
                res_Berrut.append(util.interpolate(alpha_list[i], 8, y, const.z_list_BACC(10)))
            res_Berrut = torch.from_numpy(np.concatenate(res_Berrut, 1)).reshape(const.input_k, const.dimension, 1)
            for i in range(len(alpha_list)):
                NRMSE = math.sqrt(torch.pow(expect[i] - res_Berrut[i], 2).sum().item()
                                  / torch.pow(expect[i], 2).sum().item())
                print('第{}轮测试的第{}个NRMSE:{}'.format(route + 1, i + 1, NRMSE))
            # coff = interpolate(np.expand_dims(code_input.squeeze(0).numpy(), axis=1))  # 得到的多项式系数
            # res = []
            # for alpha in alpha_list:
            #     tmp = np.zeros((1, const.dimension, const.dimension))
            #     for index, u in enumerate(coff):
            #         tmp += u * alpha ** index
            #     res.append(tmp)
            # res = torch.from_numpy(np.concatenate(res, 1))
            # for i in range(len(alpha_list)):
            # NRMSE = math.sqrt(torch.pow(expect - res_Berrut, 2).sum().item()
            #                   / torch.pow(expect, 2).sum().item())
            # print('第{}轮测试的NRMSE:{}'.format(route + 1, NRMSE))


if __name__ == '__main__':
    # train_compute()
    # train()
    predict('./dnn_params_BACC_10.pth')
    # print(torch.cuda.is_available())
