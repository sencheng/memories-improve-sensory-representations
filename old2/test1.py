from core import semantic, system_params, input_params, result, semantic_params
from core import streamlined2
from core import streamlined

import sys

import profile

c = int(sys.argv[1])
d = int(sys.argv[2])

PARAMETERS = system_params.SysParamSet()

PARAMETERS.filepath_sfa1 = "../results/sfanew.sfa"

PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = False, False

PARAMETERS.program_extent = 4
PARAMETERS.which = 'SE'

PARAMETERS.same_input_for_all = False

PARAMETERS.sem_params1 = [
                            ('layer.square', {
                                'bo':               30,
                                'rec_field_ch':     9,
                                'spacing':          3,
                                'in_channel_dim':   1,
                                'out_sfa_dim1':     48,
                                'out_sfa_dim2':     32
                            }),
                            ('layer.square', {
                                'bo' :              8,
                                'rec_field_ch':     6,
                                'spacing':          2,
                                'in_channel_dim':   32,
                                'out_sfa_dim1':     16,
                                'out_sfa_dim2':     32
                            }),
                            ('single', {
                                'dim_in':      128,
                                'dim_mid':     48,
                                'dim_out':     16
                            })
                        ]

PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = semantic_params.make_layer_series(16,16,20,20,16,16)

PARAMETERS.st2['movement_type'] = 'random_rails'
PARAMETERS.st2['movement_params'] = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50
PARAMETERS.st2['number_of_retrieval_trials'] = 200

PARAMETERS.st2['memory']['retrieval_noise'] = 0.2
PARAMETERS.st2['memory']['weight_vector'] = 1

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=100, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50

if not d and not c:
    PARAMETERS.st2['memory']['category_weight'] = 0
    PARAMETERS.st2['memory']['depress_params'] = None
    profile.run("res1 = streamlined.program(PARAMETERS)", "../results/test1.stats")
    profile.run("res2 = streamlined2.program(PARAMETERS)", "../results/test2.stats")

elif d and not c:
    PARAMETERS.st2['memory']['category_weight'] = 0
    PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')
    profile.run("res1 = streamlined.program(PARAMETERS)", "../results/test1d.stats")
    profile.run("res2 = streamlined2.program(PARAMETERS)", "../results/test2d.stats")

elif d and c:
    PARAMETERS.st2['memory']['category_weight'] = 5
    PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')
    profile.run("res1 = streamlined.program(PARAMETERS)", "../results/test1dc.stats")
    profile.run("res2 = streamlined2.program(PARAMETERS)", "../results/test2dc.stats")

elif not d and c:
    PARAMETERS.st2['memory']['category_weight'] = 5
    PARAMETERS.st2['memory']['depress_params'] = None
    profile.run("res1 = streamlined.program(PARAMETERS)", "../results/test1c.stats")
    profile.run("res2 = streamlined2.program(PARAMETERS)", "../results/test2c.stats")
#res1.save_to_file("../results/test1d.p")
print("Saved. category weight was {} and depress params was {}".format(PARAMETERS.st2['memory']['category_weight'],
                                                                       PARAMETERS.st2['memory']['depress_params']))
