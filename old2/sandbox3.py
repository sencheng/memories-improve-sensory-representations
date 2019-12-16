from core import system_params, streamlined
import numpy as np

PARAMETERS = system_params.SysParamSet()

PARAMETERS.filepath_sfa1 = "../results/sfanew.sfa"

PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = False, False

PARAMETERS.program_extent = 4
PARAMETERS.which = 'SE'

PARAMETERS.same_input_for_all = False

PARAMETERS.sem_params1 = [
                            ('layer.linear', {
                                'bo':               30,
                                'rec_field_ch':     14,
                                'spacing':          8,
                                'in_channel_dim':   1,
                                'out_sfa_dim':     16
                            })
                            ]
'''
PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = [
                            ('single.linear', {
                                'dim_in':      144,
                                'dim_out':     16
                            })
                        ]
                        '''
PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = [
                            ('inc.linear', {
                                'dim_in': 144,
                                'dim_wh': 144,
                                'dim_out': 16
                            })
                        ]

PARAMETERS.st1['movement_type'] = 'gaussian_walk'
PARAMETERS.st1['movement_params'] = dict(dx = 0.05, dt = 0.05, step=5)
PARAMETERS.st1['snippet_length'] = 25
PARAMETERS.st1['background_params'] = None

PARAMETERS.st2['movement_type'] = 'gaussian_walk'
PARAMETERS.st2['movement_params'] = dict(dx = 0.05, dt = 0.05, step=5)
PARAMETERS.st2["number_of_snippets"] = 1
PARAMETERS.st2['snippet_length'] = 600
PARAMETERS.st2['number_of_retrieval_trials'] = 60

PARAMETERS.st2['memory']['retrieval_length'] = 40
PARAMETERS.st2['memory']['category_weight'] = 40
PARAMETERS.st2['memory']['retrieval_noise'] = 0
PARAMETERS.st2['memory']['depress_params'] = dict(cost=400, recovery_time_constant=400, activation_function='lambda X : X')
PARAMETERS.st2['memory']['smoothing_percentile'] = 100
PARAMETERS.st2['memory']['use_latents'] = True

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=150, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 25
PARAMETERS.st4["sequence"] = [0]

PARAMETERS.st4b = dict(PARAMETERS.st4)
PARAMETERS.st4b["sequence"] = [1]

res = streamlined.program(PARAMETERS)

# res.save_to_file("../results/sb3.p")