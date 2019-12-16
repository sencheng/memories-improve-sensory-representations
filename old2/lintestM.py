from core import semantic, sensory, system_params, streamlined, tools
import numpy as np
from matplotlib import pyplot as plt

# sfaparm1 = [
#         ('layer.linear', {
#             'bo':               30,
#             'rec_field_ch':     21,
#             'spacing':          9,
#             'in_channel_dim':   1,
#             'out_sfa_dim':     32
#         })]
#
# sfaparm2=[('single.linear', {
#             'dim_in': 128,
#             'dim_out': 48
#         })
#     ]
PARAMETERS = system_params.SysParamSet()

PARAMETERS.sem_params1 = [
        ('layer.linear', {
            'bo': 30,
            'rec_field_ch': 18,
            'spacing': 6,
            'in_channel_dim': 1,
            'out_sfa_dim': 32
        })
    ]

PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = [
        ('single.linear', {
            'dim_in': 288,
            'dim_out': 16
        })
    ]

# PARAMETERS.st1['movement_type'] = 'timed_border_stroll'
# PARAMETERS.st1['movement_params'] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=50, border_extent=2.3)
# PARAMETERS.st1['snippet_length'] = None
PARAMETERS.st1['movement_type'] = 'gaussian_walk'
PARAMETERS.st1['movement_params'] = dict(dx=0.05, dt=0.05, step=5)
PARAMETERS.st1['snippet_length'] = 100
PARAMETERS.st1['number_of_snippets'] = 100
PARAMETERS.st1['input_noise'] = 0
PARAMETERS.st2['movement_type'] = 'gaussian_walk'
PARAMETERS.st2['movement_params'] = dict(dx=0.05, dt=0.05, step=5)
PARAMETERS.st2['snippet_length'] = 100
PARAMETERS.st2['number_of_snippets'] = 100
PARAMETERS.st2['input_noise'] = 0
PARAMETERS.st2['number_of_retrieval_trials'] = 100
PARAMETERS.st2['memory']['retrieval_length'] = 100
PARAMETERS.st2['memory']['optimization'] = False

res = streamlined.program(PARAMETERS)
