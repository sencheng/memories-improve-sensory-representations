from core import semantic, sensory, system_params, input_params, streamlined

import numpy as np
import sys
import pickle
from matplotlib import pyplot as plt
from core import tools

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "apfelstrudel"

PARAMETERS = system_params.SysParamSet()

PARAMETERS.sem_params1 = [
        ('layer.square', {
            'bo':               30,
            'rec_field_ch':     15,
            'spacing':          3,
            'in_channel_dim':   1,
            'out_sfa_dim1':     48,
            'out_sfa_dim2':     32
        }),
        ('layer.square', {
            # <bo>=  (bo-rec_field_ch)/spacing+1 := 6
            'rec_field_ch':     4,
            'spacing':          2,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     32
        })
    ]

PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = [
    ('inc.linear', {
        'dim_in': 128,
        'dim_out': 16
    })
]

# PARAMETERS.sem_params1 = [
#                 ('layer.square', {
#                     'bo':               30,
#                     'rec_field_ch':     14,
#                     'spacing':          8,
#                     'in_channel_dim':   1,
#                     'out_sfa_dim1':     32,
#                     'out_sfa_dim2':     16
#                 })
#                 ]
#
# PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = [
#                 ('inc.linear', {
#                     'dim_in':      144,
#                     'dim_out':     16
#                 })
#             ]

PARAMETERS.st1['movement_type'] = 'gaussian_walk'
PARAMETERS.st1['movement_params'] = dict(dx=0.05, dt=0.05, step=5)
PARAMETERS.st1['snippet_length'] = 50

PARAMETERS.st3['cue_equally'] = True

PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4['movement_params'] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=150, border_extent=2.3)
PARAMETERS.st4['number_of_snippets'] = 25
PARAMETERS.st4['snippet_length'] = None
PARAMETERS.st4['sequence'] = [0]

PARAMETERS.st2['movement_type'] = 'gaussian_walk'
PARAMETERS.st2['movement_params'] = dict(dx=0.05, dt=0.05, step=5)
PARAMETERS.st2['snippet_length'] = 10
PARAMETERS.st2["number_of_snippets"] = 20
PARAMETERS.st2["sequence"] = [0]

PARAMETERS.st2['number_of_retrieval_trials'] = 120
PARAMETERS.st2['sfa2_noise'] = 0

PARAMETERS.st2['memory'] = dict([])
PARAMETERS.st2['memory']['retrieval_length'] = 20
PARAMETERS.st2['memory']['category_weight'] = 40
PARAMETERS.st2['memory']['retrieval_noise'] = 0.2
PARAMETERS.st2['memory']['depress_params'] = dict(cost=400, recovery_time_constant=400, activation_function='lambda X : X')
PARAMETERS.st2['memory']['smoothing_percentile'] = 100

PARAMETERS.st2b = dict(PARAMETERS.st2)
PARAMETERS.st2b['sequence'] = [1]
PARAMETERS.st2b['snippet_length'] = 10
PARAMETERS.st2b['number_of_snippets'] = 220

PARAMETERS.st4b = dict(PARAMETERS.st4)
PARAMETERS.st4b['sequence'] = [1]

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

forming_sequenceXa, forming_categoriesa, forming_latenta, forming_rangesa = sensory_system.generate(fetch_indices=True, **PARAMETERS.st2)
forming_sequenceXb, forming_categoriesb, forming_latentb, forming_rangesb = sensory_system.generate(fetch_indices=True, **PARAMETERS.st2b)
testing_sequenceX, testing_categories, testing_latent = sensory_system.generate(**PARAMETERS.st4)
testing_sequenceXb, testing_categoriesb, testing_latentb = sensory_system.generate(**PARAMETERS.st4b)

forming_sequenceX = np.concatenate((forming_sequenceXa, forming_sequenceXb))
forming_categories = np.concatenate((forming_categoriesa, forming_categoriesb))
forming_latent = forming_latenta
forming_latent.extend(forming_latentb)
forming_ranges = forming_rangesa
forming_ranges.extend(forming_rangesb)

res = streamlined.program(PARAMETERS, input=[[forming_sequenceX, forming_categories, forming_latent, forming_ranges],
                                             [testing_sequenceX, testing_categories, testing_latent],[testing_sequenceXb, testing_categoriesb, testing_latentb]])

with open("../results/{}/res5.p".format(path), 'wb') as f:
    pickle.dump(res, f)
