"""
Script that generates secondary results for the **repeated replay of the episodes** part.
As opposed to the primary result, where the input is generated once, and training on this
same input is executed 40 times, here 40 different inputs are generated.
This is basically a condensed version of the *learnrate2* simulation
with a different input generation method.
"""

from core import semantic, sensory, system_params, input_params, streamlined

import pickle, os

import numpy as np

PATH = "../results/replay_o18_rep_gen/"
"""
Where to save the results in (SensorySystem, SFAs, st1 parameters, whitener).
"""

if not os.path.isdir(PATH):
    os.system("mkdir " + PATH)

if __name__ == "__main__":

    sfa1_parms = [
        ('layer.linear', {
            'bo': 30,
            'rec_field_ch': 18,
            'spacing': 6,
            'in_channel_dim': 1,
            'out_sfa_dim': 32
        })]

    sfa2parms = None

    PARAMETERS = system_params.SysParamSet()

    PARAMETERS.st1.update(dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                               object_code=input_params.make_object_code('TL'), sequence=[0, 1], input_noise=0.1))
    bparms = dict(PARAMETERS.st1)
    bparms['number_of_snippets'] = 600

    sensory_system1 = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=True)
    sensory_system2 = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
    print("Generating input")
    training_sequence, training_categories, training_latent = sensory_system2.generate(fetch_indices=False, **PARAMETERS.st1)
    # whitening_sequence, _, _ = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    whitening_sequence = training_sequence

    sfa1 = semantic.build_module(sfa1_parms)
    # sfa2 = semantic.build_module(sfa2_parms)

    semantic.train_SFA(sfa1, training_sequence)
    # semantic.train_SFA(sfa2, training_sequence)

    whitening_sequenceY = semantic.exec_SFA(sfa1, whitening_sequence)
    whitener = streamlined.normalizer(whitening_sequenceY, PARAMETERS.normalization)

    with open(PATH + "whitener.p", 'wb') as f:
        pickle.dump(whitener, f)

    sfa1.save(PATH + "sfa1.p")

    with open(PATH + "st1.p", 'wb') as f:
        pickle.dump(PARAMETERS.st1, f)

    incsfa_parms1 = [
        ('inc.linear', {
            'dim_in': 288,
            'dim_out': 16
        })
    ]

    bsfa_parms1 = [
            ('single.linear', {
                'dim_in': 288,
                'dim_out': 16
            })
        ]

    incsfa_rep = semantic.build_module(incsfa_parms1, eps=0.0005)
    incsfa_gen = semantic.build_module(incsfa_parms1, eps=0.0005)
    bsfa = semantic.build_module(bsfa_parms1)

    bseq, _, _ = sensory_system2.generate(fetch_indices=False, **bparms)
    yb = semantic.exec_SFA(sfa1, bseq)
    yb_w = whitener(yb)
    semantic.train_SFA(bsfa, yb_w)
    bsfa.save(PATH + "b.sfa")

    old_sequence, _, _ = sensory_system1.generate(fetch_indices=False, **PARAMETERS.st1)
    pickle.dump(sensory_system1, open(PATH + "sensory.p", 'wb'))
    ran = np.arange(PARAMETERS.st1['number_of_snippets'])

    for i in range(40):
        # repeat
        old_sequence, _, _ = sensory_system1.recall(numbers=ran, fetch_indices=False, **PARAMETERS.st1)
        seq_old = semantic.exec_SFA(sfa1, old_sequence)
        seq_old_w = whitener(seq_old)
        semantic.train_SFA(incsfa_rep, seq_old_w)
        incsfa_rep.save(PATH + "inc_rep{}.sfa".format(i))
        ran = np.random.permutation(PARAMETERS.st1['number_of_snippets'])

        # generate new
        if i == 0:   #if first iteration, take the same sequence to make sure start is equal
            new_sequence = old_sequence
        else:
            new_sequence, _, _ = sensory_system2.generate(fetch_indices=False, **PARAMETERS.st1)
        seq_new = semantic.exec_SFA(sfa1, new_sequence)
        seq_new_w = whitener(seq_new)
        semantic.train_SFA(incsfa_gen, seq_new_w)
        incsfa_gen.save(PATH + "inc_gen{}.sfa".format(i))
