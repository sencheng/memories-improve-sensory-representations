"""Part of the *learnrate2* simulation for replay in episodic memory.
This generates the forming data and trains SFA2.

Forming data has ``number_of_snippets = 50`` and ``snippet_length = 50``.

SFA parameter lists are fixed as follows::

    bsfa_parms1 = [
            ('single.linear', {
                'dim_in': 288,
                'dim_out': 16
            })
        ]

    incsfa_parms1 = [
        ('inc.linear', {
            'dim_in': 288,
            'dim_out': 16
        })
    ]

Results are stored in the path specified by :py:data:`PATH`.
Also, the required files are loaded from this path, so it has to be the same as in :py:mod:`learnrate_pre_ex2`.

The results are the following pickle files

- **b1.sfa**: The batch sfa for comparison, that was trained with ``number_of_snippets = 600``.
- **inc1_eps[r]_[i].sfa**: The incremental sfa after the [i]-th repetition using the [r]-th learnrate from the :py:data:`learnrate2.eps_list` list.

"""

from core import semantic, system_params, input_params, streamlined, sensory

import numpy as np
import sys
import pickle
import time
import os

PATH = "../results/replay_o18/"
"""
Where to save the results in (Batch and incremental SFA2 instances).
Has to be identical to :py:data:`learnrate_pre_ex2.PATH`.
"""

if __name__ == "__main__":

    dim_id = int(sys.argv[1])
    ei = int(sys.argv[2])
    eps = float(sys.argv[3])
    typ = sys.argv[4]
    fram = int(sys.argv[5])
    rot = sys.argv[6]

    # PATH = "../results/lroW_{}{}{}/".format(typ, fram, rot)

    print("this is learnrate_ex in folder " + PATH)

    bsfa_parms1 = [
            ('single.linear', {
                'dim_in': 288,
                'dim_out': 16
            })
        ]

    incsfa_parms1 = [
        ('inc.linear', {
            'dim_in': 288,
            'dim_out': 16
        })
    ]

    bsfa_parms2 = None

    incsfa_parms2 = None

    PARAMETERS = system_params.SysParamSet()
    PARAMETERS.st1.update(dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                    object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=0.1))

    # poll for completion of pre script
    while True:
        time.sleep(10)
        if os.path.isfile([PATH + "sfa1.p", PATH + "sfa2.p"][dim_id-1]):
            break
    time.sleep(2)

    sfa = semantic.load_SFA([PATH + "sfa1.p", PATH + "sfa2.p"][dim_id-1])

    with open(PATH + "whitener.p", 'rb') as f:
        whitener = pickle.load(f)

    if ei == 0:
        bsfa = semantic.build_module([bsfa_parms1, bsfa_parms2][dim_id-1])
    incsfa = semantic.build_module([incsfa_parms1, incsfa_parms2][dim_id-1], eps=eps)

    sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=True)
    sensysb = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
    bparms = dict(PARAMETERS.st1)
    bparms['number_of_snippets'] = 600
    training_sequence, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    bseq, bcat, blat = sensysb.generate(fetch_indices=False, **bparms)
    ran = np.arange(PARAMETERS.st1['number_of_snippets'])
    for i in range(40):
        training_sequence, training_categories, training_latent = sensory_system.recall(numbers=ran, fetch_indices=False, **PARAMETERS.st1)
        seq = semantic.exec_SFA(sfa, training_sequence)
        # seq_w = streamlined.normalizer(seq, PARAMETERS.normalization)(seq)
        seq_w = whitener(seq)
        if ei == 0 and i == 0:
            yb = semantic.exec_SFA(sfa, bseq)
            # yb_w = streamlined.normalizer(yb, PARAMETERS.normalization)(yb)
            yb_w = whitener(yb)
            semantic.train_SFA(bsfa, yb_w)
            bsfa.save(PATH + "b{}.sfa".format(dim_id))
        semantic.train_SFA(incsfa, seq_w)
        incsfa.save(PATH + "inc{}_eps{}_{}.sfa".format(dim_id,ei,i))
        ran = np.random.permutation(PARAMETERS.st1['number_of_snippets'])
