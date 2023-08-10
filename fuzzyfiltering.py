
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import scipy.io as sio
import os
import dstack as d
class fuzzyfilter:
    def __init__(self):
        pass
    def fuzzy(size, b_sp, wsize):
        names_ea= ['DI\\T3\\depolarisation_index.mat','SD\\T3\\scatt_diversity.mat','SP\\T3\\scatt_predominance.mat']
        centre_x = centre_y = wsize//2
        defuzz_values= np.zeros((b_sp.shape[0]))
        total= [sio.loadmat(os.getcwd()+'\\AP1\\'+names_ea[i])['a'][0:size,0:size]
                for i in range(len(names_ea))]
        
        dp= d.patches.remove_nan_overlapping_patches(total[0],wsize)
        sd= d.patches.remove_nan_overlapping_patches(total[1],wsize)
        sp= d.patches.remove_nan_overlapping_patches(total[2],wsize)
        
        dp= np.asarray([np.abs(dp[i][wsize//2][wsize//2]-dp[i,:,:]) for i in range(dp.shape[0])])
        sd= np.asarray([np.abs(sd[i][wsize//2][wsize//2]-sd[i,:,:]) for i in range(sd.shape[0])])
        sp= np.asarray([np.abs(sp[i][wsize//2][wsize//2]-sp[i,:,:]) for i in range(sp.shape[0])])
        
        dp_min= np.amin(dp)
        dp_max= np.amax(dp)
        dp_diff= dp_max-dp_min
        
        sd_min= np.amin(sd)
        sd_max= np.amax(sd)
        sd_diff= sd_max-sd_min
        
        sp_min= np.amin(sp)
        sp_max= np.amax(sp)
        sp_diff= sp_max-sp_min
        
        sd1= 0.5*dp_diff/(np.sqrt(-2*np.log(0.5))*4)
        sd2= 0.5*sd_diff/(np.sqrt(-2*np.log(0.5))*4)
        sd3= 0.5*sp_diff/(np.sqrt(-2*np.log(0.5))*4)
        sd5= 0.5*1/(np.sqrt(-2*np.log(0.5))*4)
        
        deg_pur= ctrl.Antecedent(np.linspace(dp_min,dp_max,num=100,endpoint=True), 'deg_pur')
        s_div= ctrl.Antecedent(np.linspace(sd_min,sd_max,num=100,endpoint=True), 's_div')
        scatt_p= ctrl.Antecedent(np.linspace(sp_min,sp_max,num=100,endpoint=True), 'scatt_p')
        wt = ctrl.Consequent(np.linspace(0,1,num=100,endpoint=True), 'weight')
        
        deg_pur['VLDPD'] = fuzz.gaussmf(s_div.universe, 0,sd1)
        deg_pur['LDPD'] = fuzz.gaussmf(s_div.universe, dp_diff/4,sd1)
        deg_pur['ADPD'] = fuzz.gaussmf(s_div.universe, dp_diff/2,sd1)
        deg_pur['HDPD'] = fuzz.gaussmf(s_div.universe, 3*dp_diff/4,sd1)
        deg_pur['VHDPD'] = fuzz.gaussmf(s_div.universe, dp_diff,sd1)
        
        s_div['VLSDD'] = fuzz.gaussmf(s_div.universe, 0,sd2)
        s_div['LSDD'] = fuzz.gaussmf(s_div.universe, sd_diff/4,sd2)
        s_div['ASDD'] = fuzz.gaussmf(s_div.universe, sd_diff/2,sd2)
        s_div['HSDD'] = fuzz.gaussmf(s_div.universe, 3*sd_diff/4,sd2)
        s_div['VHSDD'] = fuzz.gaussmf(s_div.universe, sd_diff,sd2)
        
        scatt_p['VLSPD'] = fuzz.gaussmf(scatt_p.universe, 0,sd3)
        scatt_p['LSPD'] = fuzz.gaussmf(scatt_p.universe, sp_diff/4,sd3)
        scatt_p['ASPD'] = fuzz.gaussmf(scatt_p.universe, sp_diff/2,sd3)
        scatt_p['HSPD'] = fuzz.gaussmf(scatt_p.universe, 3*sp_diff/4,sd3)
        scatt_p['VHSPD'] = fuzz.gaussmf(scatt_p.universe, sp_diff,sd3)
        
        wt['VLW'] = fuzz.gaussmf(wt.universe, 0,sd5)
        wt['LW'] = fuzz.gaussmf(wt.universe, 0.25,sd5)
        wt['AW'] = fuzz.gaussmf(wt.universe, 0.50,sd5)
        wt['HW'] = fuzz.gaussmf(wt.universe, 0.75,sd5)
        wt['VHW'] = fuzz.gaussmf(wt.universe, 1.0,sd5)
        
        # deg_pur['Very_Small'] = fuzz.gaussmf(s_div.universe, 0,sd1)
        # deg_pur['Small'] = fuzz.gaussmf(s_div.universe, dp_diff/4,sd1)
        # deg_pur['Average'] = fuzz.gaussmf(s_div.universe, dp_diff/2,sd1)
        # deg_pur['Large'] = fuzz.gaussmf(s_div.universe, 3*dp_diff/4,sd1)
        # deg_pur['Very_Large'] = fuzz.gaussmf(s_div.universe, dp_diff,sd1)
        
        # s_div['Very_Low'] = fuzz.gaussmf(s_div.universe, 0,sd2)
        # s_div['Low'] = fuzz.gaussmf(s_div.universe, sd_diff/4,sd2)
        # s_div['Medium'] = fuzz.gaussmf(s_div.universe, sd_diff/2,sd2)
        # s_div['High'] = fuzz.gaussmf(s_div.universe, 3*sd_diff/4,sd2)
        # s_div['Very_High'] = fuzz.gaussmf(s_div.universe, sd_diff,sd2)
        
        # s_div['Very_Low'] = fuzz.gaussmf(scatt_p.universe, 0,sd3)
        # s_div['Low'] = fuzz.gaussmf(scatt_p.universe, sd_diff/4,sd3)
        # s_div['Medium'] = fuzz.gaussmf(scatt_p.universe, sd_diff/2,sd3)
        # s_div['High'] = fuzz.gaussmf(scatt_p.universe, 3*sd_diff/4,sd3)
        # s_div['Very_High'] = fuzz.gaussmf(scatt_p.universe, sd_diff,sd3)
        
        # wt['Very_Small_Weight'] = fuzz.gaussmf(wt.universe, 0,sd5)
        # wt['Small_Weight'] = fuzz.gaussmf(wt.universe, 0.25,sd5)
        # wt['Average_Weight'] = fuzz.gaussmf(wt.universe, 0.50,sd5)
        # wt['Large_Weight'] = fuzz.gaussmf(wt.universe, 0.75,sd5)
        # wt['Very_Large_Weight'] = fuzz.gaussmf(wt.universe, 1.0,sd5)
        
        # with open(os.getcwd()+"\\AP1\\Rules4_25set.txt","r") as f:
        #     lines= f.readlines()
        # dp_r= [lines[i].rstrip('\n').split('\t')[1] for i in range(1,len(lines))]
        # sd_r= [lines[i].rstrip('\n').split('\t')[0] for i in range(1,len(lines))]
        # w_r= [lines[i].rstrip('\n').split('\t')[2] for i in range(1,len(lines))]
        with open(os.getcwd()+"\\AP1\\rules_125.txt","r") as f:
            lines= f.readlines()
        dp_r= [lines[i].rstrip('\n').split('\t')[1] for i in range(len(lines))]
        sd_r= [lines[i].rstrip('\n').split('\t')[2] for i in range(len(lines))]
        sp_r= [lines[i].rstrip('\n').split('\t')[3] for i in range(len(lines))]
        w_r= [lines[i].rstrip('\n').split('\t')[4] for i in range(len(lines))]
        
        rule= [ctrl.Rule(s_div[sd_r[i]] & deg_pur[dp_r[i]] & scatt_p[sp_r[i]] , wt[w_r[i]])
                for i in range(len(dp_r))]
        
        r_cs = ctrl.ControlSystem(rule)
        r_all = ctrl.ControlSystemSimulation(r_cs)
        for i in range(b_sp.shape[0]):
            r_all.input['deg_pur'] = dp[i][centre_x//2][centre_y//2]#pixel of degree purity
            r_all.input['s_div'] = sd[i][centre_x//2][centre_y//2]#pixel of scattering diversity
            r_all.input['scatt_p'] = sp[i][centre_x//2][centre_y//2]#pixel of scattering diversity
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
        return defuzz_values, b_sp