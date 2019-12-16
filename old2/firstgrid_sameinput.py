from core import semantic, system_params, input_params, streamlined, result, semantic_params, tools, sensory

import random
from matplotlib import pyplot
import numpy as np
import os

PARAMETERS = system_params.SysParamSet()

PARAMETERS.program_extent = 4
PARAMETERS.which = 'SE'

PARAMETERS.same_input_for_all = False

PARAMETERS.st2['movement_type'] = 'random_rails'
PARAMETERS.st2['movement_params'] = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50

PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['completeness_weight'] = 0
PARAMETERS.st2['memory']['retrieval_noise'] = 0.2
PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=400, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50

p1_1_edit = [
                            ('layer.square', {
                                'bo':               30,
                                'rec_field_ch':     10,
                                'spacing':          4,
                                'in_channel_dim':   1,
                                'out_sfa_dim1':     48,
                                'out_sfa_dim2':     32
                            }),
                            ('layer.square', {
                                'bo' :              6,
                                'rec_field_ch':     3,
                                'spacing':          1,
                                'in_channel_dim':   32,
                                'out_sfa_dim1':     16,
                                'out_sfa_dim2':     32
                            }),
                            ('layer.square', {
                                'bo' :              4,
                                'rec_field_ch':     3,
                                'spacing':          1,
                                'in_channel_dim':   32,
                                'out_sfa_dim1':     16,
                                'out_sfa_dim2':     32
                            }),
                            ('single', {
                                'dim_in':      128,
                                'dim_mid':     48,
                                'dim_out':     32
                            })
                        ]

p1_2_edit = semantic_params.make_layer_series(32,32,32,32,16,16)

# sem_par1_1 = semantic_params.make_jingnet(16)
# sem_par2_1 = semantic_params.make_layer_series(16,16,20,20,16,16)

inp_obj = [input_params.make_object_code('TL', 15)]
seq = [[0,1]]

sem_par = [(p1_1_edit, p1_2_edit)]

norm = ['whiten.ZCA', 'none']

cat_wei = [0,3]
ret_noi = [0.05, 0.1, 0.2]
com_wei = [0,3]

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
semantic_system = semantic.SemanticSystem()

file_idx = 0

for inp, s in zip(inp_obj, seq):
    PARAMETERS.input_params_default['object_code'] = inp
    PARAMETERS.input_params_default['sequence'] = s

    print("Generating input for file " + str(file_idx) + " and following...")
    training_sequence, _training_categories, training_latent, training_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st1)
    forming_sequenceX, forming_categories, forming_latent, forming_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st2)
    testing_sequenceX, testing_categories, testing_latent, testing_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st4)
    PARAMETERS.st1['movement_type'] = 'copy_traj'
    PARAMETERS.st1['movement_params'] = dict(latent=training_latent, ranges=iter(training_ranges))
    PARAMETERS.st2['movement_type'] = 'copy_traj'
    PARAMETERS.st2['movement_params'] = dict(latent=forming_latent, ranges=iter(forming_ranges))
    PARAMETERS.st4['movement_type'] = 'copy_traj'
    PARAMETERS.st4['movement_params'] = dict(latent=testing_latent, ranges=iter(testing_ranges))
    for sem in sem_par:
        PARAMETERS.sem_params1 = sem[0]
        PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = sem[1]

        sfa1 = semantic_system.build_module(PARAMETERS.sem_params1)
        print("Training SFA for file " + str(file_idx) + " and following...")
        semantic.train_SFA(sfa1, training_sequence)
        for n in norm:
            PARAMETERS.normalization = n
            for cat in cat_wei:
                PARAMETERS.st2['memory']['category_weight'] = cat
                for noi in ret_noi:
                    PARAMETERS.st2['memory']['retrieval_noise'] = noi
                    for com in com_wei:

                        print("......................" + str(file_idx) + "......................")

                        PARAMETERS.st2['memory']['completeness_weight'] = com

                        res = streamlined.program(PARAMETERS, sfa1, [[forming_sequenceX, forming_categories, forming_latent, forming_ranges],
                                                             [testing_sequenceX, testing_categories, testing_latent]])

                        res.save_to_file("../results/firstgrid" + str(file_idx) + ".p")

                        file_idx += 1

    if not os.path.exists("../results/firstgrid" + str(file_idx)):
        os.makedirs("../results/firstgrid" + str(file_idx))

    for i in range(10):

        r = random.randrange(0, 2000)
        form = np.reshape(res.forming_sequenceX[r], (30, 30))
        test = np.reshape(res.testing_sequenceX[r], (30, 30))
        pyplot.imshow(form, cmap="Greys", vmin=0, vmax=1, interpolation='none')
        pyplot.savefig("../results/firstgrid" + str(file_idx) + "/form" + str(i) + ".png")
        pyplot.imshow(test, cmap="Greys", vmin=0, vmax=1, interpolation='none')
        pyplot.savefig("../results/firstgrid" + str(file_idx) + "/test" + str(i) + ".png")