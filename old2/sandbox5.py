import numpy as np
import random

from core import semantic, system_params, streamlined
from old2 import episodic3 as episodic

PARAMETERS = system_params.SysParamSet()

PARAMETERS.filepath_sfa1 = "../results/sfaonelayer/onelayer.sfa"

PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['weight_vector'] = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]*16
PARAMETERS.st2['memory']['retrieval_noise'] = 0.5
#PARAMETERS.st2['memory']['depress_params'] = dict(cost=5OS X, recovery_time_constant=10, activation_function='lambda X : X')
PARAMETERS.st2['memory']['depress_params'] = None

forming_data = np.load("../results/sfaonelayer/sfaonelayer/forming.npz")
forming_sequenceX, forming_categories, forming_latent, forming_ranges = forming_data["forming_sequenceX"], forming_data["forming_categories"], forming_data["forming_latent"], forming_data["forming_ranges"]

print("loading SFA")
sfa1 = semantic.load_SFA(PARAMETERS.filepath_sfa1)

print("executing sfa")
forming_sequenceY = semantic.exec_SFA(sfa1, forming_sequenceX)
forming_sequenceY = streamlined.normalizer(forming_sequenceY, PARAMETERS.normalization)(forming_sequenceY)

memory = episodic.EpisodicMemory(sfa1[-1].output_dim, **PARAMETERS.st2['memory'])

lat = np.array(forming_latent)
print("storing sequences")
for ran in forming_ranges:
    liran = list(ran)
    memory.store_sequence(forming_sequenceY[liran], categories=forming_categories[liran])

def doRetrieval():
    cue_index = random.randint(0, len(forming_sequenceY) - 2 * PARAMETERS.st2['memory']['retrieval_length'])
    ret_Y, ret_Y2, ret_i, ret_i2 = memory.retrieve_sequence(
        forming_sequenceY[cue_index], forming_categories[cue_index], return_indices=True)
    print("retrieval done")
    return ret_Y, ret_Y2, ret_i, ret_i2

def doLoop(repeats=100):
    for iii in range(repeats):
        _, _, ret_i, ret_i2 = doRetrieval()
        if not np.array_equal(ret_i, ret_i2):
            print("\nERROR")
            print(ret_i)
            print(ret_i2)
            print(ret_i==ret_i2)

def useGenerator(repeats=100, sv = True):
    reslist = []
    for iii in range(repeats):
        cue_index = random.randint(0, len(forming_sequenceY) - 2 * PARAMETERS.st2['memory']['retrieval_length'])
        reslist.append(memory.retrieve_sequence(forming_sequenceY[cue_index], forming_categories[cue_index], return_indices=True))
    if sv:
        np.save("usegen.npy", reslist)
    return reslist

res = useGenerator(1, False)

print("DONE")