#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:40:26 2018

@author: olya
"""

def retrieval(probes,targets,ret_cond,memory_noise=False,ret_noise=False):
    """Perform memory retrieval based on a cue
    
    Parameters
    ----------
    
    probes :   numpy.ndarray 
               test cues
    targets : numpy.ndarray
              target items
    ret_cond: string
              specifies what distance measure to use
    memory_noise :  float
        memory noise value (std of the distribution)
    ret_noise: float
        retrieval noise (std of the distribution)
    cond: string
          study condition
    Returns
    --------
    
    the distances between the retrieved items and the cues, the proportion of correct retrievals
     
    """
    import scipy.spatial.distance
    import numpy as np
    if ret_cond=='corr':
        from scipy.stats import pearsonr
    elif ret_cond=='euclidean':
        pdist=getattr(scipy.spatial.distance, 'euclidean')
    elif ret_cond=='cosine':
         pdist=getattr(scipy.spatial.distance, 'cosine')
        
    dist_f=list(range(len(probes))) # for storing the distances btw cue and retrieved item
    d=np.shape(targets)[-1] # pattern dimension
    if memory_noise!=False:
            mem_noise=np.asarray([np.random.normal(0,memory_noise+1e-20,d) for i in range(len(targets))])
            targets_noisy=np.add(mem_noise,targets) # memory/encoding noise is added to the target items  
    else:
        targets_noisy=targets
        
    correct=np.zeros((len(targets_noisy),1))
    
    for r in range(len(probes)):
        cue=probes[r] # test cue
       
        'add retrieval noise if desired'
        if ret_noise!=False:
            r_noise=np.random.normal(0,(ret_noise+1e-20),d)# retrieval noise with 0 mean and given variance
            n_cue=cue+r_noise # retrieval cue
        else: 
            n_cue=cue
       
        'compare the cue to all the stored items using the speicified metric'    
        if ret_cond=='corr':
            dist_f[r]=[pearsonr(n_cue,item)[0] for item in targets_noisy] 
        else:
            dist_f[r]=[pdist(n_cue,item) for item in targets_noisy]
        
        'find the index of the most similar item retrieved from the memory'
        if ret_cond=='corr':
            ret_ind=dist_f[r].index(max(dist_f[r]))#  stored item with maximal correlation
        else:
            ret_ind=dist_f[r].index(min(dist_f[r])) # stored item with minimal distance

#            if len(ret_ind)>1:
#                ret_ind=rd.choice(ret_ind)     
        'check if the correct item is retrieved'
        if r<len(targets):
            if ret_ind==r:
                correct[r]=1 
        if ret_cond=='corr':
            dist_f[r]=max(dist_f[r])
        else:
            dist_f[r]=min(dist_f[r]) # distance between the retrieval cue and the closest item
        
    return dist_f,np.sum(correct)/len(targets)

def rec_test(thr,targets,probes,dist_f,ret_cond):
    """
    Perform recognition test
    
    Parameters
    ----------
    thr: 
        decision threshold
    targets: 
        stored items
    probes: 
        test items
    dist_f: 
        distance between each cue to all stored items
        
    Returns
    --------
    Hit rate, false alarm rate, cumulative hit and false alarm instances: lists
    """
    import numpy as np
    hits,falarms=np.zeros((len(thr),1)),np.zeros((len(thr),1))
    for t in range(len(thr)): # different threshold values, i.e. criteria for ROC        
        hit,fa=0,0
        for r in range(len(probes)): # go through all test items
            dist=dist_f[r] # distance between that item and the retrieved item
            'condition when an item is recognized as old'
            if ret_cond=='corr':
               post_cond=dist>=thr[t]
            else:
                post_cond=dist<=thr[t]
                
            if post_cond:
                if r<len(targets):
                    hit+=1 # hit if in ol
                else:
                    fa+=1 #false recognition
    
        "calculation of hit and false alarm rates"
        hits[t]=hit  # /len(targets)
        falarms[t]=fa  # /len(targets)#  N of targets and lures is the same in current sim
    return hits,falarms


def plot_hdist(dist,N, title=""):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,1)
    ax.hist(dist[:N], label='targets',alpha=.5,color='g')
    ax.hist(dist[N:], label='lures',alpha=.5,color='r')
    ax.set_xlabel('Distance cue-retrieved item')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

def plot_roc(hits,falarm, title=""):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,1)
    ax.plot(falarm,hits,'-o')
    ax.set_xlabel('False alarm')
    ax.set_ylabel('Hits')
    ax.set_title(title)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.set_xlim([0, xl[1]])
    ax.set_ylim([0, yl[1]])