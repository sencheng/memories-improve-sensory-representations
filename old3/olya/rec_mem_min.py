#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:58:09 2018

@author: olya
"""

import numpy as np
import utils
from matplotlib import pyplot as plt
d=4 # dimensionality of the patterns
N=50 # number of study items
noise=.5 # memory noise
old=[np.random.normal(0,3,4) for i in range(N)] # targets or old items
new=[np.random.normal(0,3,4) for i in range(N)]  # lures or new items
patterns=np.concatenate((old,new))

thr=np.linspace(0,5,8) # ROC threshold values
ret_cond='euclidean' # distance metric used the other options are 'cosine' and 'corr'

dist,correct=utils.retrieval(patterns, old, ret_cond,memory_noise=noise)
hits,falarms=utils.rec_test(thr,old,patterns,dist,ret_cond)

utils.plot_hdist(dist,N)
utils.plot_roc(hits,falarms)
plt.show()