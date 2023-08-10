
import os
import numpy as np
from sklearn.feature_extraction import image
from matplotlib import pyplot as plt
import gui
import comparison_pixels as cp
import dstack as d
import readfiles as rd
import distance_calculate as dc
import connectivity as ct
import zero_count as zc
import spotnoisedetection as spd
import spotnoisefiltering as spf
import fuzzyfiltering as fzf

z= gui.ui.maingui()
print(z)
centre_x=centre_y= int(z[4])//2
comp_all= np.zeros((int(z[3])**2, int(z[4]), int(z[4])))
defuzz_values= np.zeros((int(z[3])**2))
names=[z[0],z[1],z[2]]
merged_list_arr_reshape= rd.read_files.read_files_merge_reshape(int(z[3]),names)
patches= d.patches.remove_nan_overlapping_patches(merged_list_arr_reshape,int(z[4]))
comp_all = cp.comparison.comparison_pixels(patches, int(z[3]), int(z[4]))
dist_all = dc.calc_dist.calc_distance_cp(comp_all, int(z[3]), int(z[4]))
count_all = ct.conn_count.connected(dist_all, int(z[3]))
c_zeros = zc.count_zeros.zero_count(count_all)
pos_all, count_final1= spd.sn_detection.positions(c_zeros,comp_all, int(z[4]))


b= plt.imread(os.getcwd()+'\\AP1\\T3\\PauliRGB.bmp')[0:int(z[3]),0:int(z[3]),:]
# b= plt.imread(os.getcwd()+'\\PauliRGB_AP2.bmp')[0:size,0:size,:]
plt.subplot(311)
plt.axis('off')
plt.imshow(b)
rows, cols, channels= b.shape
b= np.dstack((np.pad(b[:,:,0],((int(z[4])-1)//2),'edge'),np.pad(b[:,:,1],((int(z[4])-1)//2),'edge'),np.pad(b[:,:,2],((int(z[4])-1)//2),'edge')))
b= image.extract_patches_2d(b, (int(z[4]),int(z[4])))
n_pa= b.shape[0]
count=0
defuzz_values_nspf, b_nspnf= fzf.fuzzyfilter.fuzzy(int(z[3]), b, int(z[4]))

b_sp= spf.snoisefiltering.spot_noise_filtering(pos_all, count_final1, b, int(z[4]))
defuzz_values_spf, b_spnf= fzf.fuzzyfilter.fuzzy(int(z[3]), b_sp, int(z[4]))

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
plt.imshow(filtered_image_nspnf.astype(np.uint8))
# plt.imsave(os.getcwd()+'\\paulirgb_noisy_fuzzy_nspnf.jpg',filtered_image_nspnf.astype(np.uint8))
plt.imsave(os.getcwd()+'\\AP1\\paulirgb_fuzzy_nspnf.jpg',filtered_image_nspnf.astype(np.uint8))