from core import semantic, system_params, input_params, result, semantic_params, sensory

from matplotlib import pyplot
import numpy as np
import copy

import profile
import pstats

import sys

episodic_number = sys.argv[1]
statsfile = sys.argv[2]

if int(episodic_number) == 2:
    from core import streamlined2 as streamlined
else:
    from core import streamlined

PARAMETERS = system_params.SysParamSet()

PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = [
                            ('layer.square', {
                                'bo':               4,
                                'rec_field_ch':     3,
                                'spacing':          1,
                                'in_channel_dim':   16,
                                'out_sfa_dim1':     16,
                                'out_sfa_dim2':     32
                            }),
                            ('single', {
                                'dim_in':      128,
                                'dim_mid':     48,
                                'dim_out':     32
                            }),
                            ('single', {
                                'dim_in':      32,
                                'dim_mid':     32,
                                'dim_out':     32
                            }),
                            ('single', {
                                'dim_in':      32,
                                'dim_mid':     16,
                                'dim_out':     16
                            }),
                        ]

PARAMETERS.filepath_sfa1 = "../results/onelayer.sfa"
PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = True, False
PARAMETERS.program_extent = 3

PARAMETERS.which = 'SE'

PARAMETERS.same_input_for_all = False

PARAMETERS.st2['movement_type'] = 'random_rails'
PARAMETERS.st2['movement_params'] = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50
PARAMETERS.st2['number_of_retrieval_trials'] = 200

PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['weight_vector'] = 1
PARAMETERS.st2['memory']['completeness_weight'] = 0
PARAMETERS.st2['memory']['retrieval_noise'] = 0.1
PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=400, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

forming_data = np.load("../results/sfaonelayer/forming.npz")
testing_data = np.load("../results/sfaonelayer/testing.npz")

forming_sequenceX, forming_categories, forming_latent, forming_ranges = forming_data["forming_sequenceX"], forming_data["forming_categories"], forming_data["forming_latent"], forming_data["forming_ranges"]
testing_sequenceX, testing_categories, testing_latent, testing_ranges = testing_data["testing_sequenceX"], testing_data["testing_categories"], testing_data["testing_latent"], testing_data["testing_ranges"]

sfa1 = semantic.load_SFA(PARAMETERS.filepath_sfa1)
profile.run('res = streamlined.program(PARAMETERS, sfa1, [[forming_sequenceX, forming_categories, forming_latent, forming_ranges], [testing_sequenceX, testing_categories, testing_latent]])', statsfile)
#res = streamlined.program(PARAMETERS, sfa1, [[forming_sequenceX, forming_categories, forming_latent, forming_ranges], [testing_sequenceX, testing_categories, testing_latent]])