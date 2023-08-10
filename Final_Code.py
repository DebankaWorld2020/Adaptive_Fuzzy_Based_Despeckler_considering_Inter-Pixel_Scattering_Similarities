

#Import the libraries
from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization, Dropout, Input, AveragePooling2D, Activation, GlobalAveragePooling2D, add, concatenate, MaxPooling3D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, CSVLogger, EarlyStopping
# from keras.utils import np_utils
from tensorflow.keras import utils
from tensorflow.keras.regularizers import l2

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, hamming_loss, jaccard_score, log_loss, roc_curve, auc

from operator import truediv

from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from spectral import imshow as spyShow
from spectral import spy_colors
from matplotlib import patches
import tensorflow as tf
from clr_callback import CyclicLR

import tkinter as tk 
import tkinter.filedialog
import glob


def merging(rows,cols,path):
    labels = sio.loadmat(path+'\\label final.mat')['a']
    cwd= path+'\\s1'
    lines= open(cwd+'\\s1Common.txt','r').read().split('\n')
    arr_merged= np.zeros(shape=(int(rows),int(cols),len(lines)))
    c=0
    for i in range(len(lines)-1):
        x= sio.loadmat(path+'\\features\\'+lines[i])['A']
        x= np.array(x)
        if x.ndim==2:
            arr_merged[:,:,c]=x[0:rows,0:cols]
            c+= 1
        if x.ndim==3:
            for y in range(x.shape[2]):
                arr_merged[:,:,y]=x[0:rows,0:cols,y]
                c+=1
    return arr_merged, labels

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=9, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState)
    print("X train {}\n X test {}\n Y train{}\n Y test {}\n".format(X_train[0],X_test[0], y_train[0], y_test[0]))
    return X_train, X_test, y_train, y_test

def remove_nan(X):
  where_are_NaNs = np.isnan(X)
  where_are_inf = np.isinf(X)
  X[where_are_NaNs] = 1e-6
  X[where_are_inf] = 1e-6
  # print(np.where(where_are_NaNs==True))
  # print(np.min(X[:,:,0]))
    # putting the 3 channels back together:
  t=np.zeros((np.shape(X)[0], np.shape(X)[1]), dtype=np.float64)
  for i in range(X.shape[2]):
    t=X[:,:,i]
    #t=(t - np.min(t)) / (np.max(t) - np.min(t))
    # print(t)
    #X[:,:,i]=t

  # for i in range(x.shape[2]):
  #   t=x[:,:,i]
  #   t = (t - np.mean(t)) / np.std(t)
  #   print(t)
  #   X[:,:,i]=t
  where_are_NaNs = np.isnan(X)
  where_are_inf = np.isinf(X)
  X[where_are_NaNs] = 1e-6
  X[where_are_inf] = 1e-6
  return X

# #test-train split
def training_testing_validation(X,y,val_ratio,test_ratio):
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(np.array((X)), np.array((y)), test_ratio)
    Xtrain, Xvalid, ytrain, yvalid = splitTrainTestSet(Xtrain, ytrain, val_ratio)
    print("Training data",Xtrain.shape)
    ytrain = utils.to_categorical(ytrain)
    print("Training GT",ytrain.shape)
    print("Validation data",Xvalid.shape)
    yvalid = utils.to_categorical(yvalid) 
    print("Validation GT",yvalid.shape)
    return Xtrain,ytrain,Xtest,ytest,Xvalid,yvalid

def model_arch(windowSize,Xvalid,num_classes,epochs,name,blr,mlr):
    S = windowSize
    L = Xvalid.shape[3]
    
    # convolutional layers
    # input layer
    input_layer = Input((S, S, L, 1))
    conv_layer1 = Conv3D(filters=16, padding="same", kernel_size=(5, 5, 5), 
                         kernel_initializer="VarianceScaling", kernel_regularizer=tf.keras.regularizers.l2(2e-4), 
                         activation='elu')(input_layer)#7
    conv_layer1=BatchNormalization(epsilon=1e-03, momentum=0.9, weights=None)(conv_layer1)
    conv_layer2 = Conv3D(filters=16, padding="same", kernel_size=(5, 5, 5), 
                         kernel_initializer="VarianceScaling", kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                         activation='elu')(conv_layer1)#5
    conv_layer2=BatchNormalization(epsilon=1e-03, momentum=0.9, weights=None)(conv_layer2)
    conv_layer3=add([conv_layer2,conv_layer1])
    conv_layer4 = Conv3D(filters=32, padding="same", kernel_size=(5, 5, 5),
                         kernel_initializer="VarianceScaling", kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                         activation='elu')(conv_layer3)#3
    #conv_layer5=BatchNormalization(epsilon=1e-03, momentum=0.9, weights=None)(conv_layer4)
    #conv_layer4=add([conv_layer3,conv_layer1])
    conv3d_shape = conv_layer4.shape
    conv_layer5 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer4)
    conv_layer6 = Conv2D(filters=32, padding="same",  kernel_size=(3,3), kernel_initializer="VarianceScaling",
                         kernel_regularizer=tf.keras.regularizers.l2(2e-4), activation='elu')(conv_layer5)
    
    flatten_layer = Flatten()(conv_layer6)
    
    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='elu')(flatten_layer)
    dense_layer1 = Dropout(0.3)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='elu')(dense_layer1)
    dense_layer2 = Dropout(0.3)(dense_layer2)
    output_layer = Dense(units=num_classes, activation='softmax')(dense_layer2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.summary()
    
    model.compile(optimizer=Adam(lr= 1e-4, beta_1=0.9, beta_2=0.95, epsilon=1e-02), loss='categorical_crossentropy', metrics=['accuracy'])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.001), patience=5, verbose=1, min_lr=0.001) #on plateaus
    
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10 ,verbose=1, mode='auto')
    
    checkpoint = ModelCheckpoint(name, verbose=1, save_best_only = True, monitor="val_loss")
    
    #clr_triangular = CyclicLR(mode='exp_range', gamma=0.5)
    clr_triangular = CyclicLR(base_lr=0.001, max_lr=0.006,step_size=2000,mode='exp_range', gamma=0.8)
    
    callbacks_list = [earlystop,reduce_lr,checkpoint,clr_triangular]
    return model, callbacks_list

def train(model,callbacks_list,Xtrain,ytrain,Xvalid,yvalid,epochs):
    history = model.fit(x=Xtrain, y=ytrain, validation_data = (Xvalid,yvalid[:,:,1]), batch_size=128, 
                        epochs=epochs, callbacks=callbacks_list)
    return history

def prediction(model,Xtest):
    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)
    return y_pred_test, Y_pred_test

def plots(history):
    plt.figure(figsize=(15,15)) 
    plt.grid() 
    plt.xlabel('Number of epochs', fontsize=20)
    plt.ylabel('MSE', fontsize=20)
    plt.plot(history.history['loss'], linewidth=5)
    plt.plot(history.history['val_loss'], linewidth=5)
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right', fontsize='x-large')
    
    plt.figure(figsize=(15,15)) 
    plt.ylim(0,1.1) 
    plt.grid()
    plt.xlabel('Number of epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.plot(history.history['accuracy'], linewidth=5)
    plt.plot(history.history['val_accuracy'], linewidth=5)
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper right', fontsize='x-large')

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports (X_test,y_test,name,model):
    file_name = name+'\\s1\\cropnet_report_s1.txt'
    
    Y_pred, y_pred= prediction(model,X_test)
    print("Y_prediction",y_pred.shape)
    print("y_test",np.argmax(y_test, axis=1).shape)
    target_names = [ 'Onion', 'Potato']
    # print(y_pred.shape)
    # print(y_test.shape)
    classification = classification_report(np.argmax(y_test, axis=1), Y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), Y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), Y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), Y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    # fpr, tpr, threshold = roc_curve(np.argmax(y_test, axis=1), y_pred, pos_label=1)
    # roc_auc= auc(fpr, tpr)
    ham_loss= hamming_loss(np.argmax(y_test,axis=1), Y_pred)
    jac_score= jaccard_score(np.argmax(y_test,axis=1), Y_pred, average=None)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    
    if os.path.exists(file_name): 
            try:
                os.remove(file_name)
            except OSError:
                    pass   
    with open(file_name, 'w') as x_file:
        x_file.write('Test loss {}(%)'.format(Test_Loss))
        x_file.write('\n \n')
        x_file.write('Test accuracy {}(%)'.format(Test_accuracy))
        x_file.write('\n \n')
        x_file.write('Kappa accuracy {}(%)'.format(kappa))
        x_file.write('\n \n')
        x_file.write('Overall accuracy {}(%)'.format(oa))
        x_file.write('\n \n')
        x_file.write('Average accuracy {}(%)'.format(aa))
        x_file.write('\n \n')
        x_file.write('Confusion Matrix\n')
        x_file.write('\n \n')
        x_file.write('{}'.format(classification))
        x_file.write('\n \n')
        x_file.write('{}'.format(confusion))
        x_file.write('\n \n')
        x_file.write('Each class accuracy \t{}'.format(each_acc))
        x_file.write('\n \n')
        x_file.write('Hamming loss \t{}(%)'.format(ham_loss))
        x_file.write('\n \n')
        x_file.write('Jaccard score \t{}(%)'.format(jac_score))
        x_file.write('\n \n')
    print(confusion)
    print("Test loss and accuracy is",Test_Loss, Test_accuracy, ham_loss)
    print(jac_score)
    print(oa, each_acc, aa, kappa)
    
    
def Patch(data,height_index,width_index,windowSize):
    PATCH_SIZE = windowSize
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch

def visualize(height,width,X,y,model,name,windowSize):
    outputs = np.zeros((height,width))
    count=0;
    for i in range(height):
        for j in range(width):
            target = int(y[i,j])
            if target == 0 :
                count+=1
                continue
            else :
                image_patch=Patch(X,i,j,windowSize)
                X_test_image = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], 
                                                   image_patch.shape[2], 1).astype('float64')
                # print(image_patch.shape[0],image_patch.shape[1],image_patch.shape[2])                                  
                prediction = (model.predict(X_test_image))
                prediction = np.argmax(prediction, axis=1)
                # print(prediction)
                outputs[i][j] = prediction+1
                count+=1
                print(count)
                print(outputs[i][j])

    mdic = {"outputs": outputs, "label": "predicted"}
    sio.savemat(name,mdic)
    
    imageView = spyShow(classes=y, fignum=1 ,figsize =(15,15), interpolation='none')
    # imageView.set_display_mode('overlay')
    labelDictionary={0:'Unknown', 1:'Onion',2:'Potato'}
    labelPatches = [ patches.Patch(color=spy_colors[x]/255.,
                     label=labelDictionary[x]) for x in np.unique(y) ]
    plt.legend(handles=labelPatches, ncol=5, fontsize='medium', 
               loc='upper left',bbox_to_anchor=(0.5, -0.05))
    
    predict_image = spyShow(classes = outputs.astype(int),figsize =(15,15))
    plt.legend(handles=labelPatches, ncol=5, fontsize='medium', 
               loc='upper left',bbox_to_anchor=(0.5, -0.05))
    

def main():
    #declaring the various parameters of the network
    root = tkinter.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.title("Python GUI")
    root.geometry(str(screen_width)+'x'+str(screen_height))
    
    rows=cols=p=t_ratio=w_size=num_classes=val_ratio=epochs=blr=mlr=""
    
    
    rows_label = tk.Label(root, text = 'Number of rows', font=('calibre', 14, 'bold'),
                          bg='orange') 
    rows_entry = tk.Entry(root, textvariable = rows,font=('calibre',14,'normal', 'italic'), 
                          bg='yellow', bd=8)
    tk.messagebox.showinfo("Kindly Read Carefully!!", "Number of rows: User defined\n Number of cols : User defined\n Number of classes : User defined\n Window size : 5(Recommended)\nRecommended Test to train ratio 0.20\n Recommended Train to validation ratio 0.333")
    
    
    cols_label = tk.Label(root, text = 'Number of columns', font = ('calibre',14,'bold'),
                          bg='orange') 
    cols_entry=tk.Entry(root, textvariable = cols, font = ('calibre',14,'normal','italic'),
                        bg='yellow', bd=8) 
    
    # p_label= tk.Label(root, text = 'Please enter the directory for saved model(.hdf5) and features 1-24(.mat)', 
    #                   font=('calibre', 10, 'bold'),bg='orange')
    # p_entry= tk.Entry(root, textvariable = p,
    #                   font=('calibre',10,'normal','italic'),bg='green', bd=8)
        
    t_ratio_label= tk.Label(root, text='Test to Train Ratio [0-1]', 
                            font=('calibre', 14, 'bold'), bg='orange')
    t_ratio_entry= tk.Entry(root, textvariable = t_ratio,
                            font=('calibre',14,'normal','italic'),bg='yellow', bd=8)
    
    epochs_label= tk.Label(root, text='Max Epochs to train the model', 
                            font=('calibre', 14, 'bold'), bg='orange')
    epochs_entry= tk.Entry(root, textvariable = epochs,
                            font=('calibre',14,'normal','italic'),bg='yellow', bd=8)
    
    blr_label= tk.Label(root, text='Base Learning Rate to start training the model', 
                            font=('calibre', 14, 'bold'), bg='orange')
    blr_entry= tk.Entry(root, textvariable = blr,
                            font=('calibre',14,'normal','italic'),bg='yellow', bd=8)
    
    mlr_label= tk.Label(root, text='Max Learning Rate at an Epoch', 
                            font=('calibre', 14, 'bold'), bg='orange')
    mlr_entry= tk.Entry(root, textvariable = mlr,
                            font=('calibre',14,'normal','italic'),bg='yellow', bd=8)
    
    w_size_label= tk.Label(root, text='Window Size', font=('calibre', 14, 'bold'),
                           bg='orange')
    w_size_entry= tk.Entry(root, textvariable = w_size,
                           font=('calibre',14,'normal','italic'),bg='yellow', bd=8)
    
    num_classes_label=  tk.Label(root, text='Number of Classes', 
                                 font=('calibre', 14, 'bold'),bg='orange')
    num_classes_entry= tk.Entry(root, textvariable = num_classes,
                                font=('calibre',14,'normal','italic'),bg='yellow', bd=8)
    
    val_ratio_label=  tk.Label(root, text='Train to Validation Ratio [0-1]',
                               font=('calibre', 14, 'bold'), bg='orange')
    val_ratio_entry= tk.Entry(root, textvariable = rows,
                              font=('calibre',14,'normal','italic'),bg='yellow',bd=8)
    def submit(): 
        rows=int(rows_entry.get())
        cols=int(cols_entry.get())
        # p= p_entry.get()
        # p_gt= p_gt_entry.get()
        # p_model= p_model_entry.get()
        # p_op= p_op_entry.get()
        t_ratio= float(t_ratio_entry.get())
        epochs=int(epochs_entry.get())
        blr=float(blr_entry.get())
        mlr=float(mlr_entry.get())
        w_size= int(w_size_entry.get())
        num_classes= int(num_classes_entry.get())
        val_ratio= float(val_ratio_entry.get())
        
        p = tk.filedialog.askdirectory(title='Select the Parent Directory')
        
        root.destroy()
        
        p=p.replace('/','\\')
        
        print(p)
        
        X,y= merging(rows,cols,p)
        
        #removing any nan or infy values
        X= remove_nan(X)
        
        print(X.shape)
        print(np.unique(y))
        num_classes= len(np.unique(y))-1 
        
        #creating small cubes from the 3d matrix
        X, y = createImageCubes(X, y, w_size)
        print(X.shape, y.shape)
        #test-train split
        Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, t_ratio)
        print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
        
        Xtrain = Xtrain.reshape(-1, w_size, w_size, Xtrain.shape[3], 1)
        ytrain = utils.to_categorical(ytrain)
        Xtrain, Xvalid, ytrain, yvalid = splitTrainTestSet(Xtrain, ytrain, val_ratio)
        print(Xtrain.shape, Xvalid.shape, ytrain.shape, yvalid.shape)
        
        Xvalid = Xvalid.reshape(-1, w_size, w_size, Xtrain.shape[3], 1)
        print(Xvalid.shape)
        
        yvalid = utils.to_categorical(yvalid)
        print(yvalid[0:20,:,1]) #first 20 rows,all cols, 2nd channel
        
        model,callbacks_list= model_arch(w_size,Xvalid,num_classes,epochs,p+'\\model1.hdf5',blr,mlr)
        history= train(model,callbacks_list,Xtrain,ytrain,Xvalid,yvalid,epochs)
        
        plots(history)
        
        model = load_model(p+'\\model1.hdf5', custom_objects={'tf': tf})
        
        Xtest = Xtest.reshape(-1, w_size, w_size, Xtrain.shape[3], 1)
        print(Xtest.shape)
        ytest = utils.to_categorical(ytest)
        print(ytest.shape)
        
        reports(Xtest,ytest,p,model)
        
        X, y = merging(rows,cols,p)
        X= remove_nan(X)
        height = y.shape[0]
        width = y.shape[1]
        #numComponents = 
        print(height,width)
        # X,pca = applyPCA(X, numComponents=numComponents)
        X = padWithZeros(X, w_size//2)
        visualize(height,width,X,y,model,p+'\\model1_predicted.mat',w_size)
        
    sub_btn=tk.Button(root,text = 'Next', command = submit, bg='white', fg='brown', font=('helvetica', 14, 'bold'), padx=25,pady=10) 
    
    rows_label.grid(row=0,column=0, padx=25,pady=10,ipady=3, sticky='W') 
    rows_entry.grid(row=0,column=1, padx=25,pady=10,ipady=3, sticky='E') 
    
    cols_label.grid(row=1,column=0, padx=25,pady=10,ipady=3, sticky='W') 
    cols_entry.grid(row=1,column=1, padx=25,pady=10,ipady=3, sticky='E')
    
    # p_label.grid(row=2, column=0, padx=5,pady=10,ipady=3)
    # p_entry.grid(row=2,column=1, padx=5,pady=10,ipady=3)
    
    epochs_label.grid(row=3, column=0, padx=25,pady=10,ipady=3, sticky='W')
    epochs_entry.grid(row=3, column=1, padx=25,pady=10,ipady=3, sticky='E')
    
    blr_label.grid(row=4, column=0, padx=25,pady=10,ipady=3, sticky='W')
    blr_entry.grid(row=4, column=1, padx=25,pady=10,ipady=3, sticky='E')
    
    mlr_label.grid(row=5, column=0, padx=25,pady=10,ipady=3, sticky='W')
    mlr_entry.grid(row=5, column=1, padx=25,pady=10,ipady=3, sticky='E')
    
    t_ratio_label.grid(row=6, column=0, padx=25,pady=10,ipady=3, sticky='W')
    t_ratio_entry.grid(row=6, column=1, padx=25,pady=10,ipady=3, sticky='E')
    
    w_size_label.grid(row=7, column=0, padx=25,pady=10,ipady=3, sticky='W')
    w_size_entry.grid(row=7, column=1, padx=25,pady=10,ipady=3, sticky='E')
    
    num_classes_label.grid(row=8, column=0, padx=25,pady=10,ipady=3, sticky='W')
    num_classes_entry.grid(row=8, column=1, padx=25,pady=10,ipady=3, sticky='E')
    
    val_ratio_label.grid(row=9, column=0, padx=25,pady=10,ipady=3, sticky='W')
    val_ratio_entry.grid(row=9, column=1, padx=25,pady=10,ipady=3, sticky='E')
    
    sub_btn.grid(row=10,column=1) 
       
    # performing an infinite loop  
    # for the window to display 
    root.mainloop()
if __name__ == '__main__':
    main()
    print("DONE!!")