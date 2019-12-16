# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:05:54 2016

@author: richaopf
"""

from matplotlib import pyplot
import numpy as np

from core import system_params, semantic_params, input_params, sensory, episodic, semantic, tools, result, streamlined
PARAMS = system_params.SysParamSet()
PARAMS.input_params_default = input_params.rails
PARAMS.setsem2(semantic_params.make_layer_series(16,16,20,20,20,16))

PARAMS.input_params_default['object_code'] = input_params.make_object_code('ST', 15)
#PARAMS.preview = True
PARAMS.st2['memory']['depress_params'] = dict(cost=1, recovery_time_constant=10, activation_function='lambda X : X')
PARAMS.st2['memory']['retrieval_length'] = 200
PARAMS.st2['memory']['retrieval_noise'] = 0.05

sfa1st = semantic.load_SFA('./sfaved/ST.sfa')
res = streamlined.program(PARAMS,sfa1st)

from analysis import sfa_analysis
def loss():
    sfa_analysis.info_loss(sfa1, dict(PARAMS.input_params_default, **PARAMS.st2), eta0=0.0001)
