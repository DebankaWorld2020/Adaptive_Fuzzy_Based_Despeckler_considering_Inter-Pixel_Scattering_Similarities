
import numpy as np
class count_zeros:
    def __init__(self):
        pass
    def zero_count(count_all):
        c_zeros=[]
        for i in range(count_all.shape[0]):
            if count_all[i]==1:
                c_zeros.append(i)
        c_zeros= np.array(c_zeros)
        print('Done process 6')
        return c_zeros