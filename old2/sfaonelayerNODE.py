from core import semantic, system_params, input_params, streamlined, result, semantic_params, sensory

from matplotlib import pyplot
import numpy as np
import copy
import sys
import traceback

import pickle

node_number = sys.argv[1]  #first argument is script name, so first arg is really [1]
gridfile = "sfaonelayer" + node_number
sys.path.insert(0,'gridfiles')
exec("from "+gridfile + " import gridparms")

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
cat_wei = gridparms.cat_wei
wei_vec = gridparms.wei_vec
ret_noi = gridparms.ret_noi

PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = True, False
PARAMETERS.program_extent = 4

forming_data = np.load("../results/sfaonelayer/forming.npz")
testing_data = np.load("../results/sfaonelayer/testing.npz")

forming_sequenceX, forming_categories, forming_latent, forming_ranges = forming_data["forming_sequenceX"], forming_data["forming_categories"], forming_data["forming_latent"], forming_data["forming_ranges"]
testing_sequenceX, testing_categories, testing_latent, testing_ranges = testing_data["testing_sequenceX"], testing_data["testing_categories"], testing_data["testing_latent"], testing_data["testing_ranges"]

file_number = gridparms.file_number

for sp in sem_par:
    PARAMETERS.sem_params1 = sp[0]
    PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = sp[1]
    print("Loading SFA for file " + str(file_number) + " and following...")
    sfa1 = semantic.load_SFA(PARAMETERS.filepath_sfa1)
    for cw in cat_wei:
        for wv in wei_vec:
            for rn in ret_noi:
                PARAMETERS.st2['memory']['category_weight'] = cw
                PARAMETERS.st2['memory']['weight_vector'] = wv
                PARAMETERS.st2['memory']['retrieval_noise'] = rn

                print("......................" + str(file_number) + "......................")
                try:
                    res = streamlined.program(PARAMETERS, sfa1, [[forming_sequenceX, forming_categories, forming_latent, forming_ranges],
                                                                 [testing_sequenceX, testing_categories, testing_latent]])
                except Exception as e:
                    exc_info = sys.exc_info()
                    str_tb = traceback.format_exception(*exc_info)
                    print(''.join(str_tb))
                    pickle.dump(str_tb, open("../results/sfaonelayer/res" + str(file_number)+".p",'wb'))
                    continue

                res.save_to_file("../results/sfaonelayer/res" + str(file_number)+".p")

                file_number += 1