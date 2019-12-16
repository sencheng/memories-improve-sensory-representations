# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:40:26 2016

@author: richaopf
"""

import sys
sys.path.insert(0,'./')

import matplotlib
matplotlib.use('Agg')

from core import sensory, tools, input_params, trajgen, system_params

from matplotlib import pyplot
import matplotlib.animation as animation

PARAMS = system_params.SysParamSet()

ss = sensory.SensorySystem(PARAMS.input_params_default)
names = ['catsndogs', 'stroll', 'rails', 'lissa']

imr = animation.writers['imagemagick'](fps=30, metadata=dict(artist='Me'))

for n in names:
    seq, cat, lat = ss.generate(**getattr(input_params,n))
    anim = tools.preview_input(seq)
    anim.save('./%s.gif'%n, writer=imr) 