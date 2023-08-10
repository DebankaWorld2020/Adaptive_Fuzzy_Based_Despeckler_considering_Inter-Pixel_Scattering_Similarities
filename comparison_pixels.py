
import numpy as np
class comparison:
    def __init(self):
        pass
    def comparison_pixels(p_total,size,wsize):
        comp_all= np.zeros((size**2,wsize,wsize))
        x= p_total.shape[3]
        if x==3:
            for i in range(p_total.shape[0]):
                for j in range(p_total.shape[1]):
                    for k in range(p_total.shape[2]):
                        if p_total[i][j][k][0] > p_total[i][j][k][1] and p_total[i][j][k][0] > p_total[i][j][k][2]:
                            comp_all[i][j][k]= x-(x-1)
                        elif p_total[i][j][k][1] > p_total[i][j][k][0] and p_total[i][j][k][1] > p_total[i][j][k][2]:
                            comp_all[i][j][k]= x-(x-2)
                        elif p_total[i][j][k][2] > p_total[i][j][k][0] and p_total[i][j][k][2] > p_total[i][j][k][1]:
                            comp_all[i][j][k]= x
        elif x==4:
            for i in range(p_total.shape[0]):
                for j in range(p_total.shape[1]):
                    for k in range(p_total.shape[2]):
                        if p_total[i][j][k][0] > p_total[i][j][k][1] and p_total[i][j][k][0] > p_total[i][j][k][2] and p_total[i][j][k][0] > p_total[i][j][k][3]:
                            comp_all[i][j][k]= x-(x-1)
                        elif p_total[i][j][k][1] > p_total[i][j][k][0] and p_total[i][j][k][1] > p_total[i][j][k][2] and p_total[i][j][k][1] > p_total[i][j][k][3]:
                            comp_all[i][j][k]= x-(x-2)
                        elif p_total[i][j][k][2] > p_total[i][j][k][0] and p_total[i][j][k][2] > p_total[i][j][k][1] and p_total[i][j][k][2] > p_total[i][j][k][3]:
                            comp_all[i][j][k]= x-(x-3)
                        elif p_total[i][j][k][3] > p_total[i][j][k][0] and p_total[i][j][k][3] > p_total[i][j][k][1] and p_total[i][j][k][3] > p_total[i][j][k][2]:
                            comp_all[i][j][k]= x
        elif x==5:
            for i in range(p_total.shape[0]):
                for j in range(p_total.shape[1]):
                    for k in range(p_total.shape[2]):
                        if p_total[i][j][k][0] > p_total[i][j][k][1] and p_total[i][j][k][0] > p_total[i][j][k][2] and p_total[i][j][k][0] > p_total[i][j][k][3] and p_total[i][j][k][0] > p_total[i][j][k][4]:
                            comp_all[i][j][k]= x-(x-1)
                        elif p_total[i][j][k][1] > p_total[i][j][k][0] and p_total[i][j][k][1] > p_total[i][j][k][2] and p_total[i][j][k][1] > p_total[i][j][k][3] and p_total[i][j][k][1] > p_total[i][j][k][4]:
                            comp_all[i][j][k]= x-(x-2)
                        elif p_total[i][j][k][2] > p_total[i][j][k][0] and p_total[i][j][k][2] > p_total[i][j][k][1] and p_total[i][j][k][2] > p_total[i][j][k][3] and p_total[i][j][k][2] > p_total[i][j][k][4]:
                            comp_all[i][j][k]= x-(x-3)
                        elif p_total[i][j][k][3] > p_total[i][j][k][0] and p_total[i][j][k][3] > p_total[i][j][k][1] and p_total[i][j][k][3] > p_total[i][j][k][2] and p_total[i][j][k][3] > p_total[i][j][k][4]:
                            comp_all[i][j][k]= x-(x-4)
                        elif p_total[i][j][k][4] > p_total[i][j][k][0] and p_total[i][j][k][4] > p_total[i][j][k][1] and p_total[i][j][k][4] > p_total[i][j][k][2] and p_total[i][j][k][4] > p_total[i][j][k][3]:
                            comp_all[i][j][k]= x
        elif x==6:
            for i in range(p_total.shape[0]):
                for j in range(p_total.shape[1]):
                    for k in range(p_total.shape[2]):
                        if p_total[i][j][k][0] > p_total[i][j][k][1] and p_total[i][j][k][0] > p_total[i][j][k][2] and p_total[i][j][k][0] > p_total[i][j][k][3] and p_total[i][j][k][0] > p_total[i][j][k][4] and p_total[i][j][k][0] > p_total[i][j][k][5]:
                            comp_all[i][j][k]= x-(x-1)
                        elif p_total[i][j][k][1] > p_total[i][j][k][0] and p_total[i][j][k][1] > p_total[i][j][k][2] and p_total[i][j][k][1] > p_total[i][j][k][3] and p_total[i][j][k][1] > p_total[i][j][k][4] and p_total[i][j][k][1] > p_total[i][j][k][5]:
                            comp_all[i][j][k]= x-(x-2)
                        elif p_total[i][j][k][2] > p_total[i][j][k][0] and p_total[i][j][k][2] > p_total[i][j][k][1] and p_total[i][j][k][2] > p_total[i][j][k][3] and p_total[i][j][k][2] > p_total[i][j][k][4] and p_total[i][j][k][2] > p_total[i][j][k][5]:
                            comp_all[i][j][k]= x-(x-3)
                        elif p_total[i][j][k][3] > p_total[i][j][k][0] and p_total[i][j][k][3] > p_total[i][j][k][1] and p_total[i][j][k][3] > p_total[i][j][k][2] and p_total[i][j][k][3] > p_total[i][j][k][4] and p_total[i][j][k][3] > p_total[i][j][k][5]:
                            comp_all[i][j][k]= x-(x-4)
                        elif p_total[i][j][k][4] > p_total[i][j][k][0] and p_total[i][j][k][4] > p_total[i][j][k][1] and p_total[i][j][k][4] > p_total[i][j][k][2] and p_total[i][j][k][4] > p_total[i][j][k][3] and p_total[i][j][k][4] > p_total[i][j][k][5]:
                            comp_all[i][j][k]= x-(x-5)
                        elif p_total[i][j][k][5] > p_total[i][j][k][0] and p_total[i][j][k][5] > p_total[i][j][k][1] and p_total[i][j][k][5] > p_total[i][j][k][2] and p_total[i][j][k][5] > p_total[i][j][k][3] and p_total[i][j][k][5] > p_total[i][j][k][4]:
                            comp_all[i][j][k]= x
        print('Done process 3')
        return comp_all