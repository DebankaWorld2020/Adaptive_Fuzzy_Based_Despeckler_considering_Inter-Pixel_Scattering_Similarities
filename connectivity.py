
import numpy as np
class conn_count:
    def __init__(self):
        pass
    def connected(dist_all, size):
        count_all= np.zeros((size**2))
        for i in range(dist_all.shape[0]):
            count=1
            for j in range(dist_all.shape[1]):
                for k in range(dist_all.shape[2]):
                    if dist_all[i][j][k]==1 or dist_all[i][j][k]==np.sqrt(2):#for 8 connectivity check
                        count+=1
            count_all[i]= count
        print('Done process 5')
        return count_all