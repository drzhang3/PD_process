# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:45:12 2019

@author: win10
"""



import csv
import os
from math import sqrt
import numpy as np
import scipy.signal as signal
import pandas as pd
import glob
import matplotlib.pyplot as plt
import pylab as pl

import pywt
#,'ZCN[1]','ZCN[2]'
column_name1=['range[0]','max[1]','mean[1]','mean_abs[1]','RMS[1]','STD[1]','skew[1]',
              'ZCN[1]','CoV[1]']
column_name2=['range[1]','max[2]','mean[2]','mean_abs[2]','RMS[2]','STD[2]','skew[2]',
              'ZCN[2]','CoV[2]']


def load_data(path):
    os.chdir(path)
    #files=os.listdir(path)
    data=[]
    for file in glob.glob("*.csv"): #遍历文件夹
        meta_data=pd.read_csv(file,skiprows=range(0, 2))
        data.append(meta_data) #每个文件的文本存到list中
    #data=[create_data(file) for file in glob.glob("*.csv")]
    return data

def band_filter(data):
        b,a=signal.butter(8,[0.005,0.6],'bandpass')
        buffer_data=signal.filtfilt(b,a,data)
        return buffer_data
    
def wavelet_dec(data):
        db4=pywt.Wavelet('db4')
        A5,D5,D4,D3,D2,D1= pywt.wavedec(data,db4,mode='symmetric',level=5)
        D5=np.zeros(D5.shape[0])
        D4=np.zeros(D4.shape[0])
        D3=np.zeros(D3.shape[0])
        D2=np.zeros(D2.shape[0])
        D1=np.zeros(D1.shape[0])
        data_rec=pywt.waverec([A5,D5,D4,D3,D2,D1],db4)
        return data_rec
    
def data_pre(data):
    b=[]
    #meta=['Acce_0','Gyro_0','Acce_1','Gyro_1','Acce_2','Gyro_2']
    meta=['Acce_x[0]','Acce_y[0]','Acce_z[0]','Gyro_x[0]','Gyro_y[0]','Gyro_z[0]',
          'Acce_x[1]','Acce_y[1]','Acce_z[1]','Gyro_x[1]','Gyro_y[1]','Gyro_z[1]',
          'Acce_x[2]','Acce_y[2]','Acce_z[2]','Gyro_x[2]','Gyro_y[2]','Gyro_z[2]']
    for i in data:
        a=pd.DataFrame(columns=meta)
        for j in meta:   
            a[j]=wavelet_dec(i[j])
        b.append(a)
    return b

def merge_data(i,item1,item2,item3):
    merge=i[item1]**2+ i[item2]**2+ i[item3]**2
    return np.sqrt(merge)

def data_merged(raw_data):
    data=[]   
    #for i in data_pre(raw_data):
    for i in raw_data:
        a=pd.DataFrame(columns=['Acce_0','Gyro_0','Acce_1','Gyro_1','Acce_2','Gyro_2'])
        Acce_0=merge_data(i,meta[0],meta[1],meta[2])
        Gyro_0=merge_data(i,meta[3],meta[4],meta[5])
        Acce_1=merge_data(i,meta[6],meta[7],meta[8])
        Gyro_1=merge_data(i,meta[9],meta[10],meta[11])
        Acce_2=merge_data(i,meta[12],meta[13],meta[14])
        Gyro_2=merge_data(i,meta[15],meta[16],meta[17])
        a['Acce_0']=Acce_0
        a['Acce_1']=Acce_1
        a['Acce_2']=Acce_2
        a['Gyro_0']=Gyro_0
        a['Gyro_1']=Gyro_1
        a['Gyro_2']=Gyro_2
        data.append(a)
    #data=data_pre(data)
    return data

def ext_fea(item):
    return [np.ptp(item),np.max(item),np.mean(item),np.mean(abs(item)),
            mse(item),np.std(item),item.skew(),calZeroCrossingNum(item),
            np.std(item)/np.mean(item)]

def segment(data,column):
    data=data[column]-data[column][0:100].mean()
    data_mid=signal.medfilt(abs(data),251)
    start=[]
    end=[]
    low=1
    high=0
    for i in range(len(data_mid)):
        if low==1:
            if data_mid[i]>0.6:
                start.append(i)
                low=0
                high=1
                continue
        if high==1:
            if data_mid[i]<0.4:
                end.append(i)
                high=0
                low=1
                continue

    drop_i=[]
    for i in range(len(start)):
        #print(i)
        if end[i]-start[i]<500:
            drop_i.append(i)
    
    for i in drop_i:
        start[i]=0
        end[i]=0
    #start.remove(0)
    #end.drop(0)
    for i in range(len(drop_i)):
        start.remove(0)
        end.remove(0)
    start=[i-120 for i in start]
    end=[i+100 for i in end]
    return start,end
#
#    feature=pd.DataFrame(columns=[column_name1])
#    for i in range(len(start)):
#        feature.loc[i]=(ext_fea(data[start[i]:end[i]]))
#    m.plot()
#              
##    return [feature['max[1]'].mean(),feature['mean[1]'].mean(),
##            feature['Varience[1]'].mean(),feature['STD[1]'].mean()]
#    return [feature[col].mean() for col in feature.columns]
#for i in range(len(start)):        
#    data[3]['Acce_0'][start[i]:end[i]].plot()    
#

def seg_fea(start,end,data,fea):
    data=data[fea]-data[fea][0:100].mean()
    return (ext_fea(data[start:end]))

#def ext_fea(item,feature):
#    return [np.max(item[feature]),np.mean(item[feature]),mse(item[feature]),
#            np.std(item[feature])**2,np.std(item[feature])]
#    
#def make_dataset(data):
#    feature1=pd.DataFrame(columns=column_name1)
#    feature2=pd.DataFrame(columns=column_name2)   
#    for i,item in enumerate(data):
#        start,end=segment(item,'Acce_1')
#        feature1.loc[i]=seg_fea(start,end,item,'Acce_1')   
#        #print(i)
#        feature2.loc[i]=seg_fea(start,end,item,'Gyro_1')
#    feature=pd.concat([feature1,feature2],axis=1)
#    return feature

def make_dataset(data):
    
    feature1=pd.DataFrame(columns=column_name1)
    feature2=pd.DataFrame(columns=column_name2)
    feature=pd.concat([feature1,feature2],axis=1)
    for i,item in enumerate(data):
        start,end=segment(item,'Acce_1')
        for j in range(len(start)):
            
            feature1.loc[j]=seg_fea(start[j],end[j],item,'Acce_1')   
        #print(i)
            feature2.loc[j]=seg_fea(start[j],end[j],item,'Gyro_1')
            feature3=pd.concat([feature1,feature2],axis=1)
        feature=pd.concat([feature,feature3],axis=0)
    return feature


def mse(data):
    SUM=0
    for i in data:
        SUM=SUM+i*i
    return sqrt(SUM/len(data))


def neg_label(num):
    a=np.ones(num)
    for i in range(num):
        a[i]=-1
    return a

def pos_label(num):
    a=np.ones(num)
    for i in range(num):
        a[i]=1
    return a
  
def data_made(path):
    data_meta=load_data(path)
    data_filt=data_merged(data_meta)
    data=data_pre(data_filt)
    return data_meta,data_filt,data,make_dataset(data)

def sgn(data):
    if data >= 0 :
        return 1
    else :
        return 0   

def calZeroCrossingNum(data) :
    data=data.reset_index(drop=True)
    SUM = 0
    for i in range(len(data)-1) :
        SUM = SUM + np.abs(sgn(data[i+1]) - sgn(data[i]))
    return SUM


if __name__=="__main__":
    #print(os.getcwd()) # 打印当前工作目录    
    #导入病人数据
    meta=['Acce_x[0]','Acce_y[0]','Acce_z[0]','Gyro_x[0]','Gyro_y[0]','Gyro_z[0]',
          'Acce_x[1]','Acce_y[1]','Acce_z[1]','Gyro_x[1]','Gyro_y[1]','Gyro_z[1]',
          'Acce_x[2]','Acce_y[2]','Acce_z[2]','Gyro_x[2]','Gyro_y[2]','Gyro_z[2]']

    path1=r'C:\Users\win10\Desktop\JOB\PD2019.1.8__8'
    path2=r'C:\Users\win10\Desktop\JOB\PD2019.1.15__10'
    path3=r'C:\Users\win10\Desktop\JOB\HP2019.1.9__10'
    data=load_data(path1)+load_data(path2)
    #data=data_merged(data_parkinson)
    data_wavedec=data_pre(data)
    data_parkinson=data_merged(data_wavedec)
    feature_pd=make_dataset(data_parkinson)
    
    
    data3=load_data(path3)
    data_wavedec3=data_pre(data3)
    data_healthy=data_merged(data_wavedec3)
    feature_hp=make_dataset(data_healthy)
    
    
    feature=pd.concat([feature_pd,feature_hp],axis=0,ignore_index=True)
    a=neg_label(feature_pd.shape[0])
    b=pos_label(feature_hp.shape[0])
    labels=np.concatenate((a,b))
    feature['label']=labels
#    cc=load_data(path3)
#    c=data_parkinson[10]['Acce_1'].tolist()
#    d=data[10]['Acce_x[1]'].tolist()
#    e=data_wavedec[10]['Acce_x[1]'].tolist()
    from sklearn.utils import shuffle
    feature=shuffle(feature)
    
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature.iloc[:,0:-1])
    

    
    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train, y_test =train_test_split(scaled,
                            feature.iloc[:,-1],test_size=0.3, random_state=0)
#    
#    X_train=feature.iloc[:,0:-1]
#    y_train=labels
#    #y_train=feature.iloc[:,-1]
#       
#    X_test=TEfeature
#    y_test=neg_label(len(TEdata))
#    
#   
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score


  
    from sklearn.svm import SVC
    clf=SVC(C=1, gamma=0.05)
    
    from sklearn.linear_model.logistic import LogisticRegression
    classifier=LogisticRegression(solver='liblinear')
    scores = cross_val_score(classifier,scaled, feature.iloc[:,-1], cv=5)
    print(scores.mean())


    
    

     
    