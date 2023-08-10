

import scipy.io as sio
import os
import numpy as np
from sklearn.feature_extraction import image
from matplotlib import pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as plt
size= 500    
wsize= 5
centre_x=centre_y= wsize//2
pos_all= []
count_final1= []
c_zeros= []
total= []
dist_all= np.zeros((size**2,wsize,wsize))
comp_all= np.zeros((size**2,wsize,wsize))
count_all= np.zeros((dist_all.shape[0]))
defuzz_values= np.zeros((size**2))

def read_files_merge_reshape(size):
    names= ['An_Yang3_Dbl.mat','An_Yang3_Odd.mat','An_Yang3_Vol.mat']
    for i in range(len(names)):
        total.append((sio.loadmat(os.getcwd()+'\\An & Yang 3\\T3\\'+names[i])['A'][0:size,0:size]))
    total_merged_arr=np.array(total)
    total_merged_all=total_merged_arr.reshape(size,size,-1)
    print('Done process 1')
    return total_merged_all

def remove_nan_overlapping_patches(X,wsize):
    # patches=patches_pic=[]
    where_are_NaNs = np.isnan(X)
    where_are_inf = np.isinf(X)
    X[where_are_NaNs] = 1e-6
    X[where_are_inf] = 1e-6
    where_are_zeros= X==0
    X[where_are_zeros]= 1e-3
    if X.ndim ==3:
        X= np.dstack((np.pad(X[:,:,0],((wsize-1)//2),'edge'),np.pad(X[:,:,1],((wsize-1)//2),'edge'),np.pad(X[:,:,2],((wsize-1)//2),'edge')))
    elif X.ndim==2:
        X= np.pad(X[:,:],((wsize-1)//2),'edge')
    X= image.extract_patches_2d(X, (wsize,wsize))
    print('Done process 2')
    return X

def comparison_pixels(p_total):
    for i in range(p_total.shape[0]):
        for j in range(p_total.shape[1]):
            for k in range(p_total.shape[2]):
                if p_total[i][j][k][0] > p_total[i][j][k][1] and p_total[i][j][k][0] > p_total[i][j][k][2]:
                    comp_all[i][j][k]=1
                elif p_total[i][j][k][1] > p_total[i][j][k][0] and p_total[i][j][k][1] > p_total[i][j][k][2]:
                    comp_all[i][j][k]=2
                elif p_total[i][j][k][2] > p_total[i][j][k][0] and p_total[i][j][k][2] > p_total[i][j][k][1]:
                    comp_all[i][j][k]=3
    print('Done process 3')
    return comp_all

def calc_distance_cp(p_all):
    for i in range(p_all.shape[0]):
        cp= p_all[i][centre_x][centre_y]
        # count=0
        for j in range(p_all.shape[1]):
            for k in range(p_all.shape[2]):
                if cp==p_all[i][j][k]:
                    dist_all[i][j][k]= np.sqrt(((centre_x-j)**2)+(centre_y-k)**2)
    print('Done process 4')
    return dist_all

def connected(dist_all):
    for i in range(dist_all.shape[0]):
        count=1
        for j in range(dist_all.shape[1]):
            for k in range(dist_all.shape[2]):
                if dist_all[i][j][k]==1 or dist_all[i][j][k]==np.sqrt(2):#for 8 connectivity check
                    count+=1
        count_all[i]= count
    print('Done process 5')
    return count_all

def zero_count(count_all):
    c_zeros=[]
    for i in range(count_all.shape[0]):
        if count_all[i]==1:
            c_zeros.append(i)
    c_zeros= np.array(c_zeros)
    print('Done process 6')
    return c_zeros

def positions(c_zeros,p_all):
    count=0
    tmp={}
    t1=t2=a=b=0
    for i in range(len(c_zeros)):
        y= p_all[c_zeros[i],:,:] #navigating to that patch
        a=b=1
        centre= y[centre_x][centre_y]
        if centre==1:
            t1=2
            t2=3
            for l in range(y.shape[0]):
                for m in range(y.shape[1]):
                    if y[l][m]==t1:
                        a+=1
                    elif y[l][m]==t2:
                        b+=1
        elif centre==2:
            t1=1
            t2=3
            for l in range(y.shape[0]):
                for m in range(y.shape[1]):
                    if y[l][m]==t1:
                        a+=1
                    elif y[l][m]==t2:
                        b+=1
        elif centre==3:
            t1=1
            t2=2
            for l in range(y.shape[0]):
                for m in range(y.shape[1]):
                    if y[l][m]==t1:
                        a+=1
                    elif y[l][m]==t2:
                        b+=1
        tmp={t1:a,t2:b}
        sorted(tmp.values())
        pos_all.append([c_zeros[i],wsize//2,wsize//2])
        for l in range(y.shape[0]):
            for m in range(y.shape[1]):
                if y[l][m]==list(tmp.keys())[0]:
                    pos_all.append([c_zeros[i],l,m])
                    count+=1
        count_final1.append([c_zeros[i],list(tmp.values())[0]])
    print('Done process 7')
    return pos_all,count_final1

def spot_noise_filtering(pos_all, count_final1, b):
    count= 0
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


def fuzzy(size, b_sp, wsize):
    names_ea= ['DP\\T3\\degree_purity.mat','SD\\T3\\scatt_diversity.mat']
    defuzz_values= np.zeros((b_sp.shape[0]))
    total= [sio.loadmat(os.getcwd()+'\\'+names_ea[i])['A'][0:size,0:size]
            for i in range(len(names_ea))]
    
    dp= remove_nan_overlapping_patches(total[0],wsize)
    sd= remove_nan_overlapping_patches(total[1],wsize)
    
    dp= np.asarray([np.abs(dp[i][wsize//2][wsize//2]-dp[i,:,:]) for i in range(dp.shape[0])])
    sd= np.asarray([np.abs(sd[i][wsize//2][wsize//2]-sd[i,:,:]) for i in range(sd.shape[0])])
    
    dp_min= np.amin(dp)
    dp_max= np.amax(dp)
    dp_diff= dp_max-dp_min
    
    sd_min= np.amin(sd)
    sd_max= np.amax(sd)
    sd_diff= sd_max-sd_min
    
    sd1= 0.5*dp_diff/(np.sqrt(-2*np.log(0.5))*4)
    sd2= 0.5*sd_diff/(np.sqrt(-2*np.log(0.5))*4)
    sd5= 0.5*1/(np.sqrt(-2*np.log(0.5))*4)
    
    deg_pur= ctrl.Antecedent(np.linspace(dp_min,dp_max,num=100,endpoint=True), 'deg_pur')
    s_div= ctrl.Antecedent(np.linspace(sd_min,sd_max,num=100,endpoint=True), 's_div')
    wt = ctrl.Consequent(np.linspace(0,1,num=100,endpoint=True), 'weight')
    
    deg_pur['Very_Small'] = fuzz.gaussmf(s_div.universe, 0,sd1)
    deg_pur['Small'] = fuzz.gaussmf(s_div.universe, dp_diff/4,sd1)
    deg_pur['Average'] = fuzz.gaussmf(s_div.universe, dp_diff/2,sd1)
    deg_pur['Large'] = fuzz.gaussmf(s_div.universe, 3*dp_diff/4,sd1)
    deg_pur['Very_Large'] = fuzz.gaussmf(s_div.universe, dp_diff,sd1)
    
    s_div['Very_Low'] = fuzz.gaussmf(s_div.universe, 0,sd2)
    s_div['Low'] = fuzz.gaussmf(s_div.universe, sd_diff/4,sd2)
    s_div['Medium'] = fuzz.gaussmf(s_div.universe, sd_diff/2,sd2)
    s_div['High'] = fuzz.gaussmf(s_div.universe, 3*sd_diff/4,sd2)
    s_div['Very_High'] = fuzz.gaussmf(s_div.universe, sd_diff,sd2)
    
    wt['Very_Small_Weight'] = fuzz.gaussmf(wt.universe, 0,sd5)
    wt['Small_Weight'] = fuzz.gaussmf(wt.universe, 0.25,sd5)
    wt['Average_Weight'] = fuzz.gaussmf(wt.universe, 0.50,sd5)
    wt['Large_Weight'] = fuzz.gaussmf(wt.universe, 0.75,sd5)
    wt['Very_Large_Weight'] = fuzz.gaussmf(wt.universe, 1.0,sd5)
    
    with open(os.getcwd()+"\\Rules4_25set.txt","r") as f:
        lines= f.readlines()
    dp_r= [lines[i].rstrip('\n').split('\t')[1] for i in range(1,len(lines))]
    sd_r= [lines[i].rstrip('\n').split('\t')[0] for i in range(1,len(lines))]
    w_r= [lines[i].rstrip('\n').split('\t')[2] for i in range(1,len(lines))]
    
    rule= [ctrl.Rule(s_div[sd_r[i]] & deg_pur[dp_r[i]], wt[w_r[i]])
            for i in range(len(dp_r))]
    
    r_cs = ctrl.ControlSystem(rule)
    r_all = ctrl.ControlSystemSimulation(r_cs)
    for i in range(b_sp.shape[0]):
        r_all.input['deg_pur'] = dp[i][centre_x//2][centre_y//2]#pixel of degree purity
        r_all.input['s_div'] = sd[i][centre_x//2][centre_y//2]#pixel of scattering diversity
        r_all.compute()
    # Crunch the numbers
        defuzz_values[i]= r_all.output['weight']
    #         # print (r_all.output['randomness'])
 
    # randomness.view(sim=r_all)
    for i  in range(b_sp.shape[0]):
        b_sp[i,centre_x,centre_y,0]= b_sp[i,centre_x,centre_y,0]*defuzz_values[i]
        b_sp[i,centre_x,centre_y,1]= b_sp[i,centre_x,centre_y,1]*defuzz_values[i]
        b_sp[i,centre_x,centre_y,2]= b_sp[i,centre_x,centre_y,2]*defuzz_values[i]
    print('Done process 9')
    return defuzz_values,b_sp

merged_list_arr_reshape= read_files_merge_reshape(size)
patches= remove_nan_overlapping_patches(merged_list_arr_reshape,wsize)
comp_all = comparison_pixels(patches)
dist_all = calc_distance_cp(comp_all)
count_all = connected(dist_all)
c_zeros = zero_count(count_all)
final_1,count_final1= positions(c_zeros,comp_all)


b= plt.imread(os.getcwd()+'\\T3\\PauliRGB.bmp')[0:size,0:size,:]
# b= plt.imread(os.getcwd()+'\\PauliRGB_AP2.bmp')[0:size,0:size,:]
plt.subplot(311)
plt.axis('off')
plt.imshow(b)
rows,cols,channels= b.shape
b= np.dstack((np.pad(b[:,:,0],((wsize-1)//2),'edge'),np.pad(b[:,:,1],((wsize-1)//2),'edge'),np.pad(b[:,:,2],((wsize-1)//2),'edge')))
b= image.extract_patches_2d(b, (wsize,wsize))
n_pa= b.shape[0]
count=0
defuzz_values_nspf, b_nspnf= fuzzy(size, b, wsize)
b_sp= spot_noise_filtering(pos_all, count_final1, b)
defuzz_values_spf, b_spnf= fuzzy(size, b_sp, wsize)

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
plt.imsave(os.getcwd()+'\\paulirgb_fuzzy_sp.jpg',filtered_image_sp.astype(np.uint8))
plt.subplot(313)
plt.axis('off')
plt.imshow(filtered_image_sp.astype(np.uint8))
# plt.imsave(os.getcwd()+'\\paulirgb_noisy_fuzzy_nspnf.jpg',filtered_image_nspnf.astype(np.uint8))
plt.imsave(os.getcwd()+'\\paulirgb_fuzzy_nspnf.jpg',filtered_image_nspnf.astype(np.uint8))