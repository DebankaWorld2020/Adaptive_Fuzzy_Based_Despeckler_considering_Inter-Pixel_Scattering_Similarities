
class sn_detection:
    def __init__(self):
        pass
    def positions(c_zeros, p_all, wsize):
        count=0
        tmp={}
        pos_all= []
        centre_x= centre_y= wsize//2
        count_final1= []
        
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
        return pos_all, count_final1
    