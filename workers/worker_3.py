import os
import pickle
import time
import worker
from python_encode import function as F
from python_encode import const_write_file
from python_encode import find_difference
from gpu_project_hr.python_project.pythonProject_values import main_rebuild


if __name__=='__main__':
    # F.data_set()
    # F.monitor()
    # current_milli_time = lambda: int(round(time.time() * 1000))
    # print(current_milli_time())
    # x, feature = main_rebuild.dataset(1)
    pathname = 'D:\gnuradio\worker_receive_file\\rx_worker_5.txt'
    # pathname_tx = 'D:\Pythonproject\python_encode\\file\ldpc_gmsk_tx.txt'
    bytes_matrix = F.monitor(pathname)
    print(bytes_matrix)
    tmp = pickle.loads(bytes_matrix)
    # res_3 = F.predict('D:\Pythonproject\gpu_project_hr\python_project\pythonProject_values\dnn_params_eigvector_5.pth',
    #                   tmp, [3])
    print(tmp)
