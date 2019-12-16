from core import semantic, sensory, system_params, input_params, tools

import numpy as np
import sys
import pickle

dim_id = int(sys.argv[1])
ei = int(sys.argv[2])
eps = float(sys.argv[3])

bsfa_parms1 = [
        ('single.linear', {
            'dim_in': 144,
            'dim_out': 16
        })
    ]

incsfa_parms1 = [
    ('inc.linear', {
        'dim_in': 144,
        'dim_out': 16
    })
]

bsfa_parms2 = [
        ('single.square', {
            'dim_in': 144,
            'dim_mid': 32,
            'dim_out': 16
        })
    ]

incsfa_parms2 = [
        ('inc.onesquare', {
            'dim_in': 144,
            'dim_mid': 32,
            'dim_out': 16
        })
    ]

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st1['number_of_snippets'] = 50
PARAMETERS.st1['input_noise'] = 0.2
PARAMETERS.st1['frame_shape'] = (12,12)
PARAMETERS.st1['object_code'] = input_params.make_object_code(['-'], sizes=22)
PARAMETERS.st1['sequence'] = [0]

if ei == 0:
    bsfa = semantic.build_module([bsfa_parms1, bsfa_parms2][dim_id-1])
incsfa = semantic.build_module([incsfa_parms1, incsfa_parms2][dim_id-1], eps=eps)

sensory_system = pickle.load(open("../results/learnrate1_2/sensory.p",'rb'))
ran = np.arange(PARAMETERS.st1['number_of_snippets'])
for i in range(40):
    training_sequence, training_categories, training_latent = sensory_system.recall(numbers=ran, fetch_indices=False, **PARAMETERS.st1)
    if ei == 0 and i == 0:
        semantic.train_SFA(bsfa, training_sequence)
        bsfa.save("../results/learnrate1_2/b{}.sfa".format(dim_id))
    semantic.train_SFA(incsfa, training_sequence)
    incsfa.save("../results/learnrate1_2/inc{}_eps{}_{}.sfa".format(dim_id,ei,i))
    ran = np.random.permutation(PARAMETERS.st1['number_of_snippets'])
