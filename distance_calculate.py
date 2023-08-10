
import numpy as np
class calc_dist:
    def __init__(self):
        pass
    def calc_distance_cp(p_all,size,wsize):
        centre_x= centre_y= wsize//2
        dist_all= np.zeros((size**2, wsize, wsize))
        for i in range(p_all.shape[0]):
            cp= p_all[i][centre_x][centre_y]
            # count=0
            for j in range(p_all.shape[1]):
                for k in range(p_all.shape[2]):
                    if cp==p_all[i][j][k]:
                        dist_all[i][j][k]= np.sqrt(((centre_x-j)**2)+(centre_y-k)**2)
        print('Done process 4')
        return dist_all