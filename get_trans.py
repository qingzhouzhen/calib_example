from ctypes import cdll
import ctypes
import numpy as np

path = "calib/build/libpose_calib.so"
mydll = cdll.LoadLibrary(path)
file_name_in = 'data/1.pcd'.encode()
file_name_trans = 'data/2.pcd'.encode()
mydll.calib.restype = ctypes.POINTER(ctypes.c_double)
c_array = mydll.calib(file_name_in, file_name_trans)
ret_array = np.array([c_array[i] for i in range(18)])
trans_array = ret_array[0:16].reshape((4,4))
is_conv = ret_array[-2]
conf = ret_array[-1]
print('trans_array: ', trans_array)
print('is_conv: ', is_conv)
print('score: ', conf)
