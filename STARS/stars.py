# -*- coding: utf-8 -*-
import sklearn
from sklearn import datasets as db
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import os
import time
import scipy as sp
from scipy.stats import bernoulli
from collections import Counter
import math
from sklearn.gaussian_process import kernels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import clone
import scipy
# partitioan the train/pool/test 
def partitionForAL(Y,leastTrain=1,leastPool=30,randomSeed=0,signi=None):
    trainIndex=[]
    testIndex=[]
    poolIndex=[]
    #if class distribution is balanced, can directly assign how many data instance per class for train/candidate
    if(leastTrain>=1 or leastPool>=1):
        for className in list(set(Y)):
            #get the index of each class
            classIndex=[i for i in range(len(Y)) if Y[i]==className]
            if signi is None:
                np.random.seed(randomSeed)                    
                np.random.shuffle(classIndex)
            else:#if pass a significance list of samples,(how many none zero emptys in the data.)    
                
                classIndex=[classIndex[k] for k in np.argsort(signi[classIndex])]
                #classIndex[np.argsort(signi[classIndex])]
                
            trainIndex=trainIndex+classIndex[0:leastTrain]
            poolIndex=poolIndex+classIndex[leastTrain:leastTrain+leastPool]
            testIndex=testIndex+classIndex[leastTrain+leastPool:len(classIndex)]
        return trainIndex,testIndex,poolIndex
    #for unbalanced class distribution, assign the percentage of instances for each class used for train/candidate    
    if(leastTrain<1 or leastPool<1):
        for className in list(set(Y)):
            #get the index of each class
            classIndex=[i for i in range(len(Y)) if Y[i]==className]    
            np.random.seed(randomSeed)                    
            np.random.shuffle(classIndex)
            trainIndex=trainIndex+classIndex[0:int(np.ceil(leastTrain*len(classIndex)))]
            poolIndex=poolIndex+classIndex[int(np.ceil(leastTrain*len(classIndex))):int(np.ceil((leastTrain+leastPool)*len(classIndex)))]                                             
            testIndex=testIndex+classIndex[int(np.ceil((leastTrain+leastPool)*len(classIndex))):len(classIndex)]                                
        return trainIndex,testIndex,poolIndex

# AL sampling function: best vs second best
def bvs(pred):
    res=np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        insPred=pred[i,:]
        sortedIndPred=np.sort(insPred)[::-1]
        res[i]=sortedIndPred[0]-sortedIndPred[1]
    return res

# resample step, use resampleDic0 to keep track of repeated times and apply \
# resampling cap
def resample(resampleScore,resampleCap,resampleDic):
    resampleDic0=resampleDic.copy()
    sampleIndex=np.argsort(resampleScore)
    for ind in sampleIndex:
        if(ind not in resampleDic0.keys()):
            resampleDic0[ind]=1
            return ind,resampleDic0
        elif (resampleDic0[ind]<resampleCap):
            resampleDic0[ind]=resampleDic0[ind]+1
            return ind,resampleDic0

# AL with resampling framework
def simpleALsvm(clf,X,y,y0,trainIndex1,testIndex1,poolIndex1,sample=20,\
                method='bvs',resampleFreq=5,noiseLev=0.3,resampleMethod='STARS',\
                    resampleCap=3,EMAcoef=0.2,EMAscope=10,tau_1 = 0.7,\
                        tau_0 = 0.2):
    res=[]
    #need new label space to store noise label
    yNoise=y.copy()
    labels=np.unique(yNoise)
    trainIndex=trainIndex1.copy()
    testIndex=testIndex1.copy()
    poolIndex=poolIndex1.copy()
    hit=[]#1: resampled data is noise. 0: resampled data is pure
    resampleIndex=[]#index of resampled data points (on entire data)
    resampleIter=[]#al iteration when resample happend
    resampleDic=dict()
    EMAScore = [0 for i in trainIndex]#save all training samples
    for i in range(sample):
        #train model
        clf.fit(X[trainIndex,:],yNoise[trainIndex])
        res.append(clf.score(X[testIndex,:],y0[testIndex]))
        #select sample
        pred=clf.predict_proba(X[poolIndex,:])
        EMAScore.append(0)
        # AL sampling step
        if method=='bvs':
            bvsScore=bvs(pred)
            sampleIndex0=np.argmin(bvsScore)
        trainIndex.append(poolIndex[sampleIndex0])
        #flip coin to see whether the new added data is with noise label.
        if( bernoulli.rvs(noiseLev, size=1)[0] ==1):#need add noise label
            #print('add a noisy data')
            candiWrong=[x for x in labels if x!=y[poolIndex[sampleIndex0]]]
            np.random.shuffle(candiWrong)
            yNoise[poolIndex[sampleIndex0]]=candiWrong[0]
        dec=np.mean(np.abs(clf.decision_function(X[trainIndex,:])),axis = 1)
        y_train = yNoise[trainIndex]
        RBF = rbf_kernel(X[trainIndex,:],gamma = clf.estimators_[0].gamma)
        loss = []
        l = 0
        LC = []
        # prepare the resampling scores
        for c in clf.estimators_:
            dec_c = c.decision_function(X[trainIndex,:])
            y_c = np.zeros(y_train.shape)
            y_c[y_train==l]=2
            y_c = y_c-1
            l = l+1
            loss_c = np.zeros(y_train.shape)
            lc_c = np.zeros(y_train.shape)
            for k in range(len(trainIndex)):
                lc_c[k] = np.linalg.norm(np.multiply(RBF[k,:],y_c)-dec_c)
            LC.append(np.reciprocal(lc_c))
            for k in range(len(trainIndex)):
                loss_c[k] = max(0,1-dec_c[k]*y_c[k])
            loss.append(loss_c)
        loss = np.array(loss).T
        lossm = np.mean(loss,axis = 1)
        LC = np.array(LC).T
        LCm = np.mean(LC,axis = 1)
        
        #if we need to resample
        if (resampleFreq>0 and i%resampleFreq==0):
            resampleIter.append(i)
            #find from training
            resampleScore=0
            # DEC
            if resampleMethod=='DEC':
                resampleScore=dec
                sampleIndex,resampleDic=resample(resampleScore,resampleCap,resampleDic)
            # LOSS
            elif resampleMethod=='LOSS':
                resampleScore = -lossm
                sampleIndex,resampleDic=resample(resampleScore,resampleCap,resampleDic)
            # LIC
            elif resampleMethod=='LIC':
                resampleScore = LCm
                sampleIndex,resampleDic=resample(resampleScore,resampleCap,resampleDic)    
            # STARS
            elif resampleMethod=='STARS':
                tau_c = tau_1-(tau_1-tau_0)*i/sample
                resampleScore = tau_c*dec-(1-tau_c)*LCm
                focusInd=np.argsort(resampleScore)[0:min(EMAscope,len(trainIndex)-1)]
                trivialInd=np.argsort(resampleScore)[min(EMAscope,len(trainIndex)-1):]
                for ind in focusInd:
                    EMAScore[ind]=resampleScore[ind]*(1-EMAcoef)+EMAScore[ind]*EMAcoef
                for ind in trivialInd:
                    EMAScore[ind]=resampleScore[ind]
                sampleIndex,resampleDic=resample(EMAScore,resampleCap,resampleDic)
# 			Random	
            elif resampleMethod=='Rand':
                sampleIndex=np.random.choice(np.arange(len(trainIndex)))
            resampleIndex.append(trainIndex[sampleIndex])
            #we first record whether the picked on is noisy label
            if y0[trainIndex[sampleIndex]]!=yNoise[trainIndex[sampleIndex]]:
                hit.append(1)
                #print('find a noisy label')
            else:
                hit.append(0)
            #next we relabel. It has higher chance(1-0.2) to be correct
            if( bernoulli.rvs(noiseLev, size=1)[0] ==0):
                yNoise[trainIndex[sampleIndex]]=y0[trainIndex[sampleIndex]]
        else:
            pass 
        del poolIndex[sampleIndex0]
    # return the model accuracy on test set along with the indices
    return [res,trainIndex,testIndex,poolIndex,resampleIter,resampleIndex]

