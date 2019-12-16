#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:58:09 2018

@author: olya
"""

import numpy as np
from olya import utils
from matplotlib import pyplot as plt
from core import semantic, sensory, system_params, input_params, streamlined
import pickle

PATH = "lro02_o1850t"
PATH_PRE = "/local/results/"
# SFA1name = "sfadef1train0.sfa"
SFA1name = "sfa1.p"

d=3 # dimensionality of the patterns
N=150 # number of study items
noiseS=0.20 # memory noise Hp lesion
noiseE=0.1 # memory noise controls

PARAMETERS = system_params.SysParamSet()
param_overload = {
    'movement_type': 'uniform',
    'movement_params': dict(),
    'object_code': input_params.make_object_code('L'),
    'sequence': [0],
    'number_of_snippets': 1,
    'snippet_length': N,
    'interleaved': True,
    'blank_frame': False,
    'glue': 'random',
    'input_noise': 0.1,
    'sfa2_noise': 0}

# old=[np.random.normal(0,3,4) for i in range(N)] # targets or old items
# new=[np.random.normal(0,3,4) for i in range(N)]  # lures or new items
# patterns=np.concatenate((old,new))

# thr=np.linspace(0,5,8) # ROC threshold values
rocsteps = 10
ret_cond='cosine' # distance metric used, options are 'euclidean', 'cosine' and 'corr'

sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
seq1, _, _ = sensys.generate(**param_overload)
seq2, _, _ = sensys.generate(**param_overload)

sfa1 = semantic.load_SFA(PATH_PRE + PATH + "/" + SFA1name)
# with open(PATH_PRE + PATH + "/res0.p", 'rb') as f:
#     res0 = pickle.load(f)
# sfa2S = res0.sfa2S
# sfa2E = res0.sfa2E
sfa2S = semantic.load_SFA(PATH_PRE + PATH + "/inc1_eps1_0.sfa")
sfa2E = semantic.load_SFA(PATH_PRE + PATH + "/inc1_eps1_39.sfa")

seq1_y = semantic.exec_SFA(sfa1, seq1)
seq1_yw = streamlined.normalizer(seq1_y, PARAMETERS.normalization)(seq1_y)
seq1_zS = semantic.exec_SFA(sfa2S, seq1_yw)
seq1_zwS = streamlined.normalizer(seq1_zS, PARAMETERS.normalization)(seq1_zS)
seq1_zE = semantic.exec_SFA(sfa2E, seq1_yw)
seq1_zwE = streamlined.normalizer(seq1_zE, PARAMETERS.normalization)(seq1_zE)
seq2_y = semantic.exec_SFA(sfa1, seq2)
seq2_yw = streamlined.normalizer(seq2_y, PARAMETERS.normalization)(seq2_y)
seq2_zS = semantic.exec_SFA(sfa2S, seq2_yw)
seq2_zwS = streamlined.normalizer(seq2_zS, PARAMETERS.normalization)(seq2_zS)
seq2_zE = semantic.exec_SFA(sfa2E, seq2_yw)
seq2_zwE = streamlined.normalizer(seq2_zE, PARAMETERS.normalization)(seq2_zE)

oldS = seq1_zwS[:,:d]
oldE = seq1_zwE[:,:d]
newS = seq2_zwS[:,:d]
newE = seq2_zwE[:,:d]
patternsS = np.concatenate((oldS, newS))
patternsE = np.concatenate((oldE, newE))

distS,correctS=utils.retrieval(patternsS, oldS, ret_cond,memory_noise=noiseS)
distE,correctE=utils.retrieval(patternsE, oldE, ret_cond,memory_noise=noiseE)

all_dists = np.concatenate((distS, distE))
mindist = min(min(distS[:len(oldS)]), min(distE[:len(oldE)]))
maxdist = max(all_dists)
thr=np.linspace(mindist*5,maxdist/2,rocsteps)

hitsRawS,falarmsRawS=utils.rec_test(thr,oldS,patternsS,distS,ret_cond)
hitsRawE,falarmsRawE=utils.rec_test(thr,oldE,patternsE,distE,ret_cond)
hitsS = hitsRawS/len(oldS)
hitsE = hitsRawE/len(oldE)
falarmsS = falarmsRawS/len(oldS)
falarmsE = falarmsRawE/len(oldE)

targfS = np.array(hitsRawS.transpose())
lurefS = np.array(falarmsRawS.transpose())
targfE = np.array(hitsRawE.transpose())
lurefE = np.array(falarmsRawE.transpose())
for i in range(len(hitsS)-1):
    targfS[0, i+1:] -= targfS[0,i]
    lurefS[0, i + 1:] -= lurefS[0, i]
    targfE[0, i + 1:] -= targfE[0, i]
    lurefE[0, i + 1:] -= lurefE[0, i]
np.savetxt("/local/results/targfS.csv", targfS, delimiter=',')
np.savetxt("/local/results/lurefS.csv", lurefS, delimiter=',')
np.savetxt("/local/results/targfE.csv", targfE, delimiter=',')
np.savetxt("/local/results/lurefE.csv", lurefE, delimiter=',')

utils.plot_hdist(distS,N,"HP lesion")
utils.plot_roc(hitsS,falarmsS, "HP lesion")
utils.plot_hdist(distE,N, "Control")
utils.plot_roc(hitsE,falarmsE, "Control")
plt.show()
