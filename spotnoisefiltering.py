
import numpy as np
class snoisefiltering:
    def __init__(self):
        pass
    def spot_noise_filtering(pos_all, count_final1, b, wsize):
        count= 0
        centre_x = centre_y = wsize//2
        for i in count_final1:
            patch_number= i[0]
            nop= i[1]
            pos= pos_all[count:count+nop]
            for j in range(b.shape[3]):
                avg= 0
                cp= b[int(patch_number)][centre_x][centre_y][j]
                for k in pos:
                    avg+= b[int(patch_number)][k[1]][k[2]][j]
                avg/= nop
                sigmas= np.sqrt((cp-avg)**2/wsize**2)
                if sigmas== 0:
                    sigmas= 1e-6
                ENLS = (avg/sigmas)**2
                if ENLS== 0:
                    ENLS= 1e-6
                sx2s= ((ENLS*(sigmas)**2)-(avg**2))/(ENLS+1)
                if sx2s== 0:
                    sx2s= 1e-6
                ifiltered= avg+(sx2s*(cp-avg)/(sx2s+(avg**2/ENLS)))
                    # print(ifiltered)
                b[int(patch_number)][centre_x][centre_y][j]= ifiltered
            count+= nop
        print('Done process 8')
        return b