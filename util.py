import torch
from gpu_project_hr.python_project.pythonProject_values import const
from gpu_project_hr.python_project.pythonProject_values import DNN_re
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


device = get_device()


def interpolate(interpolate_points, receive_points_number, function_value, z_list):
    numerator = 0
    res_Berrut = 0
    index = [0 for i in range(receive_points_number)]
    flag = 0
    i = 0
    z_receive = []
    fuz_receive = []
    while i < receive_points_number:
        k = np.random.randint(0, len(z_list))
        for j in range(i):
            if index[j] == k:
                flag = 1
                break
            else:
                flag = 0
        if flag == 0:
            index[i] = k
            i = i + 1
    index.sort()
    for i in range(receive_points_number):
        z_receive.append(z_list[index[i]])
        fuz_receive.append(function_value[index[i]])
    for i in range(receive_points_number):
        numerator += ((-1) ** i) / (interpolate_points - z_receive[i])
    for i in range(receive_points_number):
        # tmp = function_value[i]
        res_Berrut += ((-1) ** i) * fuz_receive[i] / ((interpolate_points - z_receive[i]) * numerator)
    return res_Berrut


def predict(filename):
    alpha_list = [1 / 4, 1 / 2, 3 / 4]
    test_net = DNN_re.DNN(const.alpha_list, device)
    test_net.load_state_dict(torch.load(filename))
    test_net.eval()
    with torch.no_grad():
        for route in range(1000):
            input = torch.from_numpy(x).to(torch.float32).to(device)
            expect: torch.Tensor = torch.from_numpy(feature).to(torch.float32).reshape(3, const.dimension, 1) \
                .to(device)
            appro = test_net.forward(input)  # 发送给workers的计算任务
            coff = interpolate(np.expand_dims(appro.squeeze(0).numpy(), axis=1))  # 得到的多项式系数
            res = []
            for alpha in alpha_list:
                tmp = np.zeros((1, const.dimension, 1))
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
    z_j_list = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    y_list = [1 / 36, 1 / 9, 1 / 4, 4 / 9, 25 / 36]
    alpha_i = [1, 2, 3]
    res = interpolate(alpha_i[0], 5, y_list, z_j_list)
    print(res)
