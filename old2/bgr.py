from core import sensory, system_params, input_params

from core import tools

import numpy as np

import profile
PARAMETERS = system_params.SysParamSet()

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

# PARAMETERS.st1 = input_params.rand
# PARAMETERS.st1["snippet_length"] = 50
PARAMETERS.st1['input_noise'] = 0.05
PARAMETERS.st1['number_of_snippets'] = 5
PARAMETERS.st1['scale_clip'] = True
PARAMETERS.st1['movement_type'] = 'stillframe'
PARAMETERS.st1['movement_params'] = dict()
PARAMETERS.st1['snippet_length'] = 2

seq1, _, _ = sensory_system.generate(**PARAMETERS.st1)

PARAMETERS.st1['background_params'] = dict(seed_prob = 0.1, spread_prob = 0.3, spread_scaling = 0.1, constant = False)

seq2, _, _ = sensory_system.generate(**PARAMETERS.st1)

# ======================================

print("seq1 min max", np.min(seq1), np.max(seq1))
print("seq2 min max", np.min(seq2), np.max(seq2))

tools.compare_inputs((seq1, seq2))