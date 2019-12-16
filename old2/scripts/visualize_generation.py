#import sys
#sys.path.insert(0,'../')

from core import sensory, semantic, episodic
from core import tools, result, system_params, streamlined

import numpy as np
import random
import itertools

PARAMETERS = system_params.SysParamSet()

PARAMETERS.input_params_default['movement_params']['step'] = 2
PARAMETERS.st1['number_of_snippets'] = 500

PARAMETERS.st2['memory']['retrieval_length'] = 200
PARAMETERS.st2['memory']['retrieval_noise'] = 0.02
PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['weight_vector'] = 1

PARAMETERS.st2['number_of_retrieval_trials'] = 30


###########################################################

semantic_system = semantic.SemanticSystem()
sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

sfa1 = streamlined.construct_SFA1(PARAMETERS, sensory_system, semantic_system)
memory   = episodic.EpisodicMemory( sfa1[-1].output_dim, categories=True, **PARAMETERS.st2['memory'])

print "Generating input.. "
selection_sequence, selection_categories, selection_latent, selection_ranges = sensory_system.generate(fetch_indices=True,**PARAMETERS.st2)
SFA1_output = sfa1.execute(selection_sequence)

print "storing+scenarios... "
for ran in selection_ranges:
    liran = list(ran)
    memory.store_sequence(SFA1_output[liran], categories=selection_categories[liran])


stage2_retrievals = []

retrieved_sequence = []
perfect_sequence = []

BLACK = np.ones((30,30))*255

ret_length = PARAMETERS.st2['memory']['retrieval_length']

for j in range(PARAMETERS.st2['number_of_retrieval_trials']):
    cue_index = random.randint(0,len(SFA1_output)-ret_length)
#    print 'cue:', cue_index

    retrieved,retrieved_indices = memory.retrieve_sequence(
                SFA1_output[cue_index], selection_categories[cue_index], return_indices=True)
                
#    print 'retrieved_indices: ', len(retrieved_indices), retrieved_indices
                
    stage2_retrievals.append(retrieved)
    
    retrieved_sequence.extend(selection_sequence[retrieved_indices])
    perfect_sequence.extend(selection_sequence[range(cue_index, cue_index+ret_length)])
    
    retrieved_sequence.extend([BLACK]*5)
    perfect_sequence.extend([BLACK]*5)

ANIM = tools.compare_inputs([retrieved_sequence, perfect_sequence], rate=20)
#ANIM.save('./noise={}.gif'.format(PARAMETERS.st2['memory']['retrieval_noise']), writer='imagemagick', fps=30)