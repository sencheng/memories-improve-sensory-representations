from core import semantic, system_params, input_params, streamlined, result, semantic_params

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

sem_par1_2 = [
        ('layer.square', {
            'bo':               30,
            'rec_field_ch':     15,
            'spacing':          3,
            'in_channel_dim':   1,
            'out_sfa_dim1':     48,
            'out_sfa_dim2':     48
        }),
        ('single', {
            'dim_mid':     48,
            'dim_out':     32
        })
    ]

sem_par2_2 = semantic_params.make_layer_series(32,32,32,32,16,16)

sem_par1_1 = semantic_params.make_jingnet(16)
sem_par2_1 = semantic_params.make_layer_series(16,16,20,20,16,16)

inp_obj = [input_params.make_object_code('T', 15), input_params.make_object_code('TL', 15), input_params.make_object_code(('TX','LB'), 15)]
seq = [[0], [0,1], [0,1]]

sem_par = [(sem_par1_1, sem_par2_1),(sem_par1_2, sem_par2_2)]

norm = ['whiten.ZCA', 'none']

cat_wei = [0,3]
ret_noi = [0.2,0.4]
com_wei = [0,3]

file_idx = 0

for inp, s in zip(inp_obj, seq):
    PARAMETERS.input_params_default['object_code'] = inp
    PARAMETERS.input_params_default['sequence'] = s
    for sem in sem_par:
        PARAMETERS.sem_params1 = sem[0]
        PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = sem[1]

        sfa1 = streamlined.construct_SFA1(PARAMETERS)
        for n in norm:
            PARAMETERS.normalization = n
            for cat in cat_wei:
                PARAMETERS.st2['memory']['category_weight'] = cat
                # for noi in ret_noi:
                #     PARAMETERS.st2['memory']['retrieval_noise'] = noi
                #     for com in com_wei:
                #         PARAMETERS.st2['memory']['completeness_weight'] = com

                res = streamlined.program(PARAMETERS, sfa1)

                res.save_to_file("../results/firstgrid" + str(file_idx) + ".p")

                file_idx += 1

    if not os.path.exists("../results/firstgrid" + str(file_idx)):
        os.makedirs("../results/firstgrid" + str(file_idx))
    for i in range(10):
        print("......................" + str(file_idx) + "......................")

        r = random.randrange(0, 2000)
        form = np.reshape(res.forming_sequenceX[r], (30, 30))
        test = np.reshape(res.testing_sequenceX[r], (30, 30))
        pyplot.imshow(form, cmap="Greys", vmin=0, vmax=1, interpolation='none')
        pyplot.savefig("../results/firstgrid" + str(file_idx) + "/form" + str(i) + ".png")
        pyplot.imshow(test, cmap="Greys", vmin=0, vmax=1, interpolation='none')
        pyplot.savefig("../results/firstgrid" + str(file_idx) + "/test" + str(i) + ".png")