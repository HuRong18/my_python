import os
import pickle
import threading
import time

import torch

import worker as w
import countdown_latch
from python_encode import function as F
from python_encode import const_write_file
from python_encode import find_difference
from gpu_project_hr.python_project.pythonProject_values import main_rebuild, const


def start_thread(worker: w.Worker, lock: threading.RLock, countdown_latch: countdown_latch.CountDownLatch,
                 res_list: list, sequence: list):
    res, alpha = worker.compute()
    lock.acquire()
    res_list.append(res)
    sequence.append(alpha)
    lock.release()
    countdown_latch.countDown()


if __name__ == '__main__':
    # F.data_set()
    # F.monitor()
    # current_milli_time = lambda: int(round(time.time() * 1000))
    # print(current_milli_time())
    # x, feature = main_rebuild.dataset(1)
    # pathname = 'D:\gnuradio\\ldpc_gmsk_rx.txt'
    pathname_list = ['D:\\gnuradio\\worker_receive_file\\rx_worker_1.txt',
                     'D:\\gnuradio\\worker_receive_file\\rx_worker_2.txt',
                     'D:\\gnuradio\\worker_receive_file\\rx_worker_3.txt',
                     'D:\\gnuradio\\worker_receive_file\\rx_worker_4.txt',
                     'D:\\gnuradio\\worker_receive_file\\rx_worker_5.txt',
                     'D:\\gnuradio\\worker_receive_file\\rx_worker_6.txt']
    dnn_filename = 'D:\\Pythonproject\\gpu_project_hr\\python_project\\pythonProject_values\\dnn_params_eigenvalue_5.pth'
    worker_list = []
    for i in range(const.worker_number):
        worker = w.Worker(pathname_list[i], dnn_filename, const.alpha_list[i])
        worker_list.append(worker)
    res = []
    sequence = []
    lock = threading.RLock()
    countdown_latch = countdown_latch.CountDownLatch(const.threshold_number)
    for i in range(len(worker_list)):
        thread = threading.Thread(target=start_thread, args=[worker_list[i], lock, countdown_latch, res, sequence])
        thread.start()
    countdown_latch.wait()
    res = res[:const.threshold_number]
    sequence = sequence[:const.threshold_number]

    res = torch.cat(res, 1)
    print('compute_complete')
    feature = F.data_read_file('D:\\Pythonproject\\python_encode\\file\\output_matrix_100_matrixs.txt')
    for i in range(100):
        F.final_interpolate_result(res[i:i + 1, :, :, :], feature[i:i + 1, :, :], sequence, i)
    current_milli_time = lambda: int(round(time.time() * 1000))
    print(current_milli_time())
