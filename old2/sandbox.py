from core import semantic, sensory, system_params

import numpy as np
import sys
import pickle

arg = int(sys.argv[1])

if arg == 1:
    bsfa_parms1 = [
            ('single.linear', {
                'dim_in': 900,
                'dim_out': 16
            })
        ]

    incsfa_parms1 = [
        ('inc.linear', {
            'dim_in': 900,
            'dim_out': 16
        })
    ]

if arg == 2:
    bsfa_parms2 = [
            ('single.square', {
                'dim_in': 900,
                'dim_mid': 128,
                'dim_out': 16
            })
        ]

    incsfa_parms2 = [
            ('inc.square', {
                'dim_in': 900,
                'dim_mid': 128,
                'dim_out': 16
            })
        ]

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st1['number_of_snippets'] = 50
PARAMETERS.st1['input_noise'] = 0.2

if arg == 0:
    sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=True)
    print("Generating input")
    training_sequence, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    pickle.dump(sensory_system, open("../results/traininc/sensory.p", 'wb'))

if arg > 0:
    print("Building SFAs")
if arg == 1:
    bsfa1 = semantic.build_module(bsfa_parms1)
    incsfa1 = semantic.build_module(incsfa_parms1)
if arg == 2:
    bsfa2 = semantic.build_module(bsfa_parms2)
    incsfa2 = semantic.build_module(incsfa_parms2)

if arg > 0:
    sensory_system = pickle.load(open("../results/traininc/sensory.p",'rb'))
    ran = np.arange(PARAMETERS.st1['number_of_snippets'])
    for i in range(20):
        training_sequence, training_categories, training_latent = sensory_system.recall(numbers = ran, fetch_indices=False)
        if i == 0:
            if arg == 1:
                print("Training bSFA1")
                semantic.train_SFA(bsfa1, training_sequence)
                bsfa1.save("../results/traininc/b1.sfa")
            elif arg == 2:
                print("Training bSFA2")
                semantic.train_SFA(bsfa2, training_sequence)
                bsfa2.save("../results/traininc/b2.sfa")
        if arg == 1:
            print("Training incSFA1")
            semantic.train_SFA(incsfa1, training_sequence)
            incsfa1.save("../results/traininc/inc1_{}.sfa".format(i))
        elif arg == 2:
            print("Training incSFA2")
            semantic.train_SFA(incsfa2, training_sequence)
            incsfa2.save("../results/traininc/inc2_{}.sfa".format(i))
        np.save("../results/traininc/ran{}.npy".format(i), ran)
        ran = np.random.permutation(PARAMETERS.st1['number_of_snippets'])


# print("Executing incSFA1")
# inc_seq1 = semantic.exec_SFA(incsfa1, training_sequence)
# print("Executing incSFA2")
# inc_seq2 = semantic.exec_SFA(incsfa2, training_sequence)

# print("Executing bSFA1")
# b_seq1 = semantic.exec_SFA(bsfa1, training_sequence)
# print("Executing bSFA2")
# b_seq2 = semantic.exec_SFA(bsfa2, training_sequence)