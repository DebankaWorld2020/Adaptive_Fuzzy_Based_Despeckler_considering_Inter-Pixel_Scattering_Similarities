
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
import spotnoisefiltering as spf
import fuzzyfiltering as fzf
class readpauli:
    def __init__(self):
        pass
    def readpauli_img(size, wsize, pos_all, count_final1):
        b= plt.imread(os.getcwd()+'\\AP1\\T3\\PauliRGB.bmp')[0:size,0:size,:]
        centre_x= wsize//2
        centre_y= wsize//2
        # b= plt.imread(os.getcwd()+'\\PauliRGB_AP2.bmp')[0:size,0:size,:]
        plt.subplot(311)
        plt.axis('off')
        plt.imshow(b)
        rows,cols,channels= b.shape
        b= np.dstack((np.pad(b[:,:,0],((size-1)//2),'edge'),np.pad(b[:,:,1],((size-1)//2),'edge'),
                      np.pad(b[:,:,2],((size-1)//2),'edge')))
        b= image.extract_patches_2d(b, (wsize,wsize))
        defuzz_values_nspf, b_nspnf= fzf.fuzzyfilter.fuzzy(size, b, wsize)
        b_sp= spf.snoisefiltering.spot_noise_filtering(pos_all, count_final1, b, wsize)
        defuzz_values_spf, b_spnf= fzf.fuzzyfilter.fuzzy(size, b_sp, wsize)
        
        filtered_image_sp= np.zeros((rows,cols,channels))
        filtered_image_nspnf= np.zeros((rows,cols,channels))
        c=0
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    filtered_image_sp[i,j,k]= b_spnf[c,centre_x,centre_y,k]
                    filtered_image_nspnf[i,j,k] = b_nspnf[c,centre_x,centre_y,k]
                c+=1
        plt.subplot(312)
        plt.axis('off')
        plt.imshow(filtered_image_sp.astype(np.uint8))
        # plt.imsave(os.getcwd()+'\\paulirgb_noisy_fuzzy_sp.jpg',filtered_image_sp.astype(np.uint8))
        plt.imsave(os.getcwd()+'\\AP1\\paulirgb_fuzzy_sp.jpg',filtered_image_sp.astype(np.uint8))
        plt.subplot(313)
        plt.axis('off')
        plt.imshow(filtered_image_sp.astype(np.uint8))
        # plt.imsave(os.getcwd()+'\\paulirgb_noisy_fuzzy_nspnf.jpg',filtered_image_nspnf.astype(np.uint8))
        plt.imsave(os.getcwd()+'\\AP1\\paulirgb_fuzzy_nspnf.jpg',filtered_image_nspnf.astype(np.uint8))