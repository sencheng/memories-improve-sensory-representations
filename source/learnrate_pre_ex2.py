"""
Part of the *learnrate2* simulation for replay in episodic memory.
This generates the training data and trains SFA1.
Results are stored in the path specified by :py:data:`PATH`.

These results are the following pickle files

- **sensory.p**: The :py:class:`core.sensory.SensorySystem` that was used to generate the training input.
                 The generated input is saved in the object and can be recalled using the :py:func:`core.sensory.SensorySystem.recall` method.
- **sfa1.p**: The trained SFA1 instance.
- **st1.p**: The parameter dictionary that was used to generate input.
- **whitener.p**: The :py:class:`core.streamlined.normalizer` object that was initialized on the training data.

"""

from core import semantic, sensory, system_params, input_params, streamlined

import pickle, os, sys

PATH = "../results/replay_o18/"
"""
Where to save the results in (SensorySystem, SFA1, st1 parameters, whitener).
This path has to be dynamically set in case that :py:data:`learnrate2.typelist`,
:py:data:`learnrate2.framelist`, :py:data:`learnrate2.rotlist` have more than one element.
The names of the files generated by this script and loaded by :py:mod:`learnrate_ex2`
do not depend on these parameters so there would be no way of differentiating between
different settings. Example::

   PATH = "../results/lr2_{}{}{}/".format(typ, fram, rot)
"""

if __name__ == "__main__":

    typ = sys.argv[1]
    fram = sys.argv[2]
    rot = sys.argv[3]

    # PATH = "../results/lroW_{}{}{}/".format(typ, fram, rot)

    if not os.path.isdir(PATH):
        os.system("mkdir " + PATH)

    print("this is learnrate_pre in folder " + PATH)

    if typ == 'o14':
        print("set SFA parms to o14")
        sfa1_parms = [
                ('layer.linear', {
                    'bo':               30,
                    'rec_field_ch':     14,
                    'spacing':          8,
                    'in_channel_dim':   1,
                    'out_sfa_dim':     32
                })
            ]

    elif typ == 'o18':
        print("set SFA parms to o18")
        sfa1_parms = [
            ('layer.linear', {
                'bo': 30,
                'rec_field_ch': 18,
                'spacing': 6,
                'in_channel_dim': 1,
                'out_sfa_dim': 32
            })
        ]
    else:
        print("set SFA parms to single")
        sfa1_parms = [
            ('single.linear', {
                'dim_in': 900,
                'dim_out': 288
            })
        ]

    sfa2parms = None

    if fram == 50:
        nsnip = 50
        snlen = 50
    else:
        nsnip = 600
        snlen = 100

    if rot == 't':
        dehteh = 0.05
    else:
        dehteh = 0.0

    # if typ == 'j':
    #     noi = 0.2
    # else:
    #     noi = 0
    noi = 0.1

    PARAMETERS = system_params.SysParamSet()

    PARAMETERS.st1.update(dict(number_of_snippets=nsnip, snippet_length=snlen, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=dehteh, step=5),
                    object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=noi))
    # PARAMETERS.st1['number_of_snippets'] = 50
    # PARAMETERS.st1['input_noise'] = 0.2

    sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=True)
    print("Generating input")
    training_sequence, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    pickle.dump(sensory_system, open(PATH + "sensory.p", 'wb'))
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
    # sfa2.save(PATH + "sfa2.p")


    with open(PATH + "st1.p", 'wb') as f:
        pickle.dump(PARAMETERS.st1, f)