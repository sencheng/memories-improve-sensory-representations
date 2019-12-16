# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:01:09 2016

@author: richaopf
"""

from core import system_params, semantic_params, input_params, sensory, episodic, semantic, tools, result, streamlined
PARAMS = system_params.SysParamSet()
PARAMS.setsem2(semantic_params.make_layer_series(48,36,40,40,36,10))

sfa1st = semantic.load_SFA('./sfaved/XO12-stroll.sfa')
