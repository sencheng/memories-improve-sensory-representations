# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:54:05 2016

@author: richaopf
"""

from core import input_params, semantic, sensory
from sklearn import svm
import numpy as np
import sys
from sklearn import linear_model

def info_loss(function, inparams, verbose=False, **extras) :
    ingen = sensory.SensorySystem(inparams)
    
    train_seq, train_cat, train_lat = ingen.generate()
    trainY = np.array(train_lat)   
    trainX = np.array(function(train_seq))
    
    if verbose: 
        print 'training input generated; ready to regress'
        sys.stdout.flush()
    
    regressors = [ linear_model.SGDRegressor(**extras) for i in range(trainY.shape[1]) ]
    for i, r in enumerate(regressors):
        r.fit(trainX, trainY[:,i])    
        if verbose: 
            print 'regressor %d trained'%i, r.score(trainX, trainY[:,i])
            sys.stdout.flush()

    test_seq, test_cat, test_lat = ingen.generate()
    testY = np.array(test_lat)
    testX = np.array(function(test_seq))
    if verbose: print 'testing input generated; ready to test'
    
    scores = []
    for i,r in enumerate(regressors):
        scores.append(r.score(testX, testY[:,i]))
        print scores[-1]
    
    return scores
        

def correls(sfa, inparams) :
    tools.feature_latent_correlation(SFA1_output[indices], forming_latent[indices])

    pass    

