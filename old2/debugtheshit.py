import numpy as np
import random

from core import system_params
from old2 import episodic3 as episodic

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st2['memory']['category_weight'] = 5
PARAMETERS.st2['memory']['completeness_weight'] = 0
PARAMETERS.st2['memory']['retrieval_length'] = 10
PARAMETERS.st2['memory']['retrieval_noise'] = 10
PARAMETERS.st2['memory']['weight_vector'] = 1
#PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')
PARAMETERS.st2['memory']['depress_params'] = None

seq = np.array([[1,0],[0.8,0],[0,0.6],[0,0.8],[0.6,0],[1,0]])
cats = np.array([0,0,0,1,1,1])

memory = episodic.EpisodicMemory(2, **PARAMETERS.st2['memory'])
memory.store_sequence(seq, categories=cats)

cue_index = random.randint(0,5)
print(cue_index)
res = memory.retrieve_sequence(seq[cue_index], cats[cue_index], return_indices=True)