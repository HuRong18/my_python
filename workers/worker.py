import os
import pickle
import time

from python_encode import function as F
from python_encode import const_write_file
from python_encode import find_difference
from gpu_project_hr.python_project.pythonProject_values import main_rebuild


class Worker:
    def __init__(self, pathname, dnn_filename, alpha):
        self.pathname = pathname
        self.alpha = alpha
        self.dnn_filename=dnn_filename

    def compute(self):
        # F.data_set()
        # F.monitor()
        # current_milli_time = lambda: int(round(time.time() * 1000))
        # print(current_milli_time())
        # x, feature = main_rebuild.dataset(1)
        # pathname = 'D:\gnuradio\worker_receive_file\\rx_worker_1.txt'
        # pathname_tx = 'D:\Pythonproject\python_encode\\file\ldpc_gmsk_tx.txt'
        bytes_matrix = F.monitor(self.pathname)
        tmp = pickle.loads(bytes_matrix)
        res = F.predict(self.dnn_filename, tmp, [self.alpha])
        return res, self.alpha
