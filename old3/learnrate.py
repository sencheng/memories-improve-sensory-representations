from core import semantic, sensory, system_params, input_params, tools

import numpy as np
import sys
import os
import time
import pickle

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st1['number_of_snippets'] = 50
PARAMETERS.st1['input_noise'] = 0.2
PARAMETERS.st1['frame_shape'] = (12,12)
PARAMETERS.st1['object_code'] = input_params.make_object_code(['-'], sizes=22)
PARAMETERS.st1['sequence'] = [0]

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=True)
print("Generating input")
training_sequence, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
pickle.dump(sensory_system, open("../results/learnrate1_2/sensory.p", 'wb'))

time.sleep(2)

eps_default = 0.001
eps_list = [0.0001, 0.001, 0.01, 0.1]
incsfa1_list = []
incsfa2_list = []
for dim_id in [1,2]:
    for ei, eps in enumerate(eps_list):
        os.system("srun python learnrate_ex.py {} {} {} &".format(dim_id, ei, eps))
