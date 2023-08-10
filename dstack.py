
import numpy as np
from sklearn.feature_extraction import image
class patches:
    def __init__(self):
        pass
    def remove_nan_overlapping_patches(X,wsize):
        # patches=patches_pic=[]
        where_are_NaNs = np.isnan(X)
        where_are_inf = np.isinf(X)
        X[where_are_NaNs] = 1e-6
        X[where_are_inf] = 1e-6
        where_are_zeros= X==0
        X[where_are_zeros]= 1e-3
        if X.ndim ==3:
            if X.shape[2]==2:
                X= np.dstack((np.pad(X[:,:,0],((wsize-1)//2),'edge'),np.pad(X[:,:,1],((wsize-1)//2),'edge')))
            elif X.shape[2]==3:
                X= np.dstack((np.pad(X[:,:,0],((wsize-1)//2),'edge'),np.pad(X[:,:,1],((wsize-1)//2),'edge'),
                              np.pad(X[:,:,2],((wsize-1)//2),'edge')))
            elif X.shape[2]==4:
                X= np.dstack((np.pad(X[:,:,0],((wsize-1)//2),'edge'),np.pad(X[:,:,1],((wsize-1)//2),'edge'),
                              np.pad(X[:,:,2],((wsize-1)//2),'edge'), np.pad(X[:,:,3],((wsize-1)//2),'edge')))
            elif X.shape[2]==5:
                X= np.dstack((np.pad(X[:,:,0],((wsize-1)//2),'edge'),np.pad(X[:,:,1],((wsize-1)//2),'edge'),
                              np.pad(X[:,:,2],((wsize-1)//2),'edge'), np.pad(X[:,:,3],((wsize-1)//2),'edge'),
                              np.pad(X[:,:,4],((wsize-1)//2),'edge')))
            elif X.shape[2]==6:
                X= np.dstack((np.pad(X[:,:,0],((wsize-1)//2),'edge'),np.pad(X[:,:,1],((wsize-1)//2),'edge'),
                              np.pad(X[:,:,2],((wsize-1)//2),'edge'), np.pad(X[:,:,3],((wsize-1)//2),'edge'),
                              np.pad(X[:,:,4],((wsize-1)//2),'edge'), np.pad(X[:,:,5],((wsize-1)//2),'edge')))
        elif X.ndim==2:
            X= np.pad(X[:,:],((wsize-1)//2),'edge')
        X= image.extract_patches_2d(X, (wsize,wsize))
        print('Done process 2')
        return X