import numpy as np
import random
import sys

from core import semantic, system_params, sensory, semantic_params
from core import streamlined2 as streamlined
from old2 import episodic3 as episodic

cat_weight = 0
use_depression = False
try:
    cat_weight = int(sys.argv[1])
except:
    pass
try:
    use_depression = bool(sys.argv[2])
except:
    pass

PARAMETERS = system_params.SysParamSet()

PARAMETERS.filepath_sfa1 = "../results/sfanew.sfa"

PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = False, False

PARAMETERS.program_extent = 4
PARAMETERS.which = 'SE'

PARAMETERS.same_input_for_all = False

PARAMETERS.sem_params1 = semantic_params.make_jingnet(8)

PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = semantic_params.make_layer_series(8,8,16,16,16,16)

PARAMETERS.st2['movement_type'] = 'random_rails'
PARAMETERS.st2['movement_params'] = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50
PARAMETERS.st2['number_of_retrieval_trials'] = 200

PARAMETERS.st2['memory']['category_weight'] = cat_weight
PARAMETERS.st2['memory']['retrieval_noise'] = 0.5
PARAMETERS.st2['memory']['weight_vector'] = 1
if use_depression:
    PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')
else:
    PARAMETERS.st2['memory']['depress_params'] = None

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=100, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50

print("training sfa1")
sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
sfa1 = streamlined.construct_SFA1(PARAMETERS, sensys)

print("generating forming data")
forming_sequenceX, forming_categories, forming_latent, forming_ranges = sensys.generate(fetch_indices=True, **PARAMETERS.st2)

print("executing sfa1")
forming_sequenceY = semantic.exec_SFA(sfa1, forming_sequenceX)
forming_sequenceY = streamlined.normalizer(forming_sequenceY, PARAMETERS.normalization)(forming_sequenceY)

memory = episodic.EpisodicMemory(sfa1[-1].output_dim, **PARAMETERS.st2['memory'])

lat = np.array(forming_latent)
print("storing sequences")
for ran in forming_ranges:
    liran = list(ran)
    memory.store_sequence(forming_sequenceY[liran], categories=forming_categories[liran])

def useGenerator(repeats=100, sv = True):
    reslist = []
    for iii in range(repeats):
        cue_index = random.randint(0, len(forming_sequenceY) - 2 * PARAMETERS.st2['memory']['retrieval_length'])
        res = memory.retrieve_sequence(forming_sequenceY[cue_index], forming_categories[cue_index], return_indices=True)
        if len(res) == 2:
            reslist.append(res)
    return reslist

reslist = useGenerator(200)
arr = np.array(reslist)
np.save("../results/test8_{}{}.npy".format(cat_weight, int(use_depression)), arr)

print("Finished. Result array length: {}".format(len(reslist)))