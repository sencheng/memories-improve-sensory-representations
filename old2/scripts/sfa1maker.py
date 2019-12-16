# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:04:48 2016

@author: richaopf
"""

import warnings, sys

if len(sys.argv) > 1 :
    DESCRIPTION = sys.argv[1]
warnings.simplefilter('ignore')

import mdp
from core import system_params, streamlined, input_params, semantic_params, result
from core import tools, sensory, semantic

import numpy as np
from matplotlib import pyplot

PARAMS = system_params.SysParamSet()
PARAMS.input_params_default = input_params.stroll
PARAMS.input_params_default['object_code'] = input_params.make_object_code('ST',15)

PARAMS.input_params_default['sequence'] = [0,1]
PARAMS.input_params_default['interleaved'] = False
PARAMS.input_params_default['number_of_snippets'] = 450

PARAMS.st1 = {}
PARAMS.st1['snippet_length'] = 100
#PARAMS.st1['number_of_snippets'] = 1
#PARAMS.st1['sequence'] = [0,1]

PARAMS.st2 = {}
PARAMS.st2['movement_params'] = input_params.still['movement_params']
PARAMS.st2['movement_type'] = input_params.still['movement_type']
PARAMS.st2['snippet_length'] = 2
PARAMS.st2['number_of_snippets'] = 5000

print '================= final parameters: ==================='
print '------ st1 ------'
print PARAMS.st1
print '------ st2 ------'
print PARAMS.st2
print '-----default-----'
print PARAMS.input_params_default
print '======================================================='


sensory_system = sensory.SensorySystem(PARAMS.input_params_default, save_input=False)
print 'ready to generate input'
training_sequence, training_categories,training_latent = sensory_system.generate(**PARAMS.st1)
print 'training input generated'
forming_sequence, forming_categories, forming_latent = sensory_system.generate(**PARAMS.st2)
print 'input generated'

print 'shape=', np.shape(training_sequence)

#tools.preview_input(training_sequence)
sfa1 =  semantic.make_network(semantic_params.make_jingnet(16))
sfa1.train(training_sequence)
sfa1.save('./sfaved/%s.sfa' % DESCRIPTION)

print 'sfa saved'
SFA1_output = sfa1.execute(forming_sequence)

print 'sfa executed'

forming_corr = tools.feature_latent_correlation(SFA1_output, forming_latent, forming_categories)

options = list(set(forming_categories))

SFA1_output = np.array(SFA1_output)
forming_latent = np.array(forming_latent)    

buff = tools.LocalBuffer('~/__temp%s__.py'%DESCRIPTION)

buff.code("""
from matplotlib import pyplot
from numpy import *
import sys, os
srcdir = os.path.expanduser('~/episodic-driven-semantic-learning/source')
sys.path.insert(0, srcdir)

from core import tools
""")

buff.declare('options', options)
buff.declare('forming_corr', forming_corr)

colors = ['Greys', 'Blues', 'Reds', 'Greens']
#buff.code('fig2, histaxes = pyplot.subplots(1+len(options),1)')

names = []
names.append('forming_corr')

for i,o in enumerate(options):
    indices = forming_categories == o
    buff.declare('cor%d'%i, tools.feature_latent_correlation(SFA1_output[indices], forming_latent[indices]))
    names.append('cor%d'%i)
#    buff.declare('hist%d'%i, result.histogram_2d(SFA1_output[indices], forming_latent[indices, 0:2]))
#    buff.code("histaxes[{0}+1].matshow(hist{0}, cmap='{1}')".format(i, colors[i%len(colors)]))

buff.code("""
tools.display_corr(["""+', '.join(names)+"""])
pyplot.title('{}')
pyplot.show()
""".format(DESCRIPTION))
buff.execute()

print '================= done ================='