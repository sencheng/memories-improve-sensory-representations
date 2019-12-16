from core import semantic, system_params, input_params, streamlined, result, semantic_params, sensory

from matplotlib import pyplot
import numpy as np
import copy

PARAMETERS = system_params.SysParamSet()

s1_1 = [
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
                                'out_sfa_dim2':     16
                            })
                            ]

s2_1 = [
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

# s1_2 = copy.deepcopy(s1_1)
# s1_2[0][1]['out_sfa_dim2'] = 16
# s2_2 = copy.deepcopy(s2_1)
# s2_2[0][1]['in_channel_dim'] = 16

PARAMETERS.filepath_sfa1 = "../results/onelayer.sfa"
PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = False, False
PARAMETERS.program_extent = 4

PARAMETERS.which = 'SE'

PARAMETERS.same_input_for_all = False

PARAMETERS.st2['movement_type'] = 'random_rails'
PARAMETERS.st2['movement_params'] = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50

PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['weight_vector'] = 1
PARAMETERS.st2['memory']['completeness_weight'] = 0
PARAMETERS.st2['memory']['retrieval_noise'] = 0.1
PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=400, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50

# sem_par = [(s1_1, s2_1), (s1_2, s2_2)]
sem_par = [(s1_1, s2_1)]
cat_wei = [0,10]
sixteen = [1]*16
sixteen.extend([0]*16)
eight = [1]*8
eight.extend([0]*24)
wei_vec = [1, sixteen*6, eight*6]
ret_noi = [0.05, 0.1, 0.2]

PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = True, False
PARAMETERS.program_extent = 4

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

print("Generating and saving input...")

training_sequence, training_categories, training_latent, training_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st1)
forming_sequenceX, forming_categories, forming_latent, forming_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st2)
testing_sequenceX, testing_categories, testing_latent, testing_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st4)

np.savez("../results/sfaonelayer/training.npz", training_sequence=training_sequence, training_categories=training_categories, training_latent=training_latent, training_ranges=training_ranges)
np.savez("../results/sfaonelayer/forming.npz", forming_sequenceX=forming_sequenceX, forming_categories=forming_categories, forming_latent=forming_latent, forming_ranges=forming_ranges)
np.savez("../results/sfaonelayer/testing.npz", testing_sequenceX=testing_sequenceX, testing_categories=testing_categories, testing_latent=testing_latent, testing_ranges=testing_ranges)

print("Training and saving SFA...")

for sp in sem_par:
    PARAMETERS.sem_params1 = sp[0]
    PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = sp[1]

    sfa1 = semantic.build_module(PARAMETERS.sem_params1)
    semantic.train_SFA(sfa1, training_sequence)
    sfa1.save(PARAMETERS.filepath_sfa1)

print("FINISHED")