
import os
import scipy.io as sio
import numpy as np
class read_files:
    def __init__(self):
        pass
    def read_files_merge_reshape(size,name):
        names=[]
        total=[]
        for i in range(len(name)):
            for file in os.listdir(os.getcwd()+"\\AP1\\"+str(name[i])+"\\T3\\"):
                if file.endswith(".mat"):
                    names.append(file)
        print(names)
        for i in range(len(name)):
            for i in range(len(names)):
                total.append((sio.loadmat(os.getcwd()+"\\AP1\\"+str(name[i])+"\\T3\\"+names[i])['a'][0:size,0:size]))
        total_merged_arr= np.array(total)
        total_merged_all= total_merged_arr.reshape(size,size,-1)
        print('Done process 1')
        return total_merged_all