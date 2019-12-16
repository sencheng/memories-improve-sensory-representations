from core import sensory, system_params, semantic, tools, result, input_params

import mdp
import numpy as np
import warnings
import pickle

warnings.simplefilter("module")

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
                                'bo' :              6,
                                'rec_field_ch':     4,
                                'spacing':          2,
                                'in_channel_dim':   32,
                                'out_sfa_dim1':     8,
                                'out_sfa_dim2':     32
                            }),
                            ('single', {
                                'dim_in':      128,
                                'dim_mid':     48,
                                'dim_out':     16
                            })
                        ]

PARAMETERS.sem_params2 = [
                            ('single', {
                                'dim_in': 16,
                                'dim_mid': 16,
                                'dim_out': 20
                            }),
                            ('single', {
                                'dim_in': 20,
                                'dim_mid': 16,
                                'dim_out': 16
                            })
                        ]

PARAMETERS.st1['object_code'] = input_params.make_object_code('T', 15)
PARAMETERS.st1['sequence'] = [0]

PARAMETERS.filepath_sfa1 = "sfaved/minimal.sfa"
PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = 0,0

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
input_sequence1, input_categories1, input_latent1, input_ranges1 = sensory_system.generate(fetch_indices=True,**PARAMETERS.st1)
input_sequence2, input_categories2, input_latent2, input_ranges2 = sensory_system.generate(fetch_indices=True,**PARAMETERS.st1)
input_arr = [input_sequence1, input_sequence2]
input_arr2 = np.load("inputs.npy")
input_arr = [input_arr2[1], input_arr2[1]]

sfa1_arr = []

for _ in range(0):
    switchboard1, sfa_layer1 = semantic._switchboard_based_layer( **PARAMETERS.sem_params1[0][1])
    switchboard2, sfa_layer2 = semantic._switchboard_based_layer( **PARAMETERS.sem_params1[1][1])
    top_node = semantic._sfa_flow_node(**PARAMETERS.sem_params1[2][1])
    sfa1_arr.append(mdp.Flow([switchboard1, sfa_layer1, switchboard2, sfa_layer2, top_node]))
    sfa1_arr[-1].train(input_arr[0])

#sfa1_arr[0].save(PARAMETERS.filepath_sfa1)

sfa1_load = pickle.load(open(PARAMETERS.filepath_sfa1, "rb"))
#sfa1_gen = sfa1_arr[0]

sfa1_arr = [sfa1_load, sfa1_load]

#np.save("inputs.npz", input_arr)
feature_arr = []
for i in range(4):
    feature_arr.append(sfa1_arr[int(i/2)].execute(input_arr[i%2]))

sfa2_arr = []
for _ in range(4):
    node1 = semantic._sfa_flow_node(**PARAMETERS.sem_params2[0][1])
    node2 = semantic._sfa_flow_node(**PARAMETERS.sem_params2[1][1])
    sfa2_arr.append(mdp.Flow([node1, node2]))

final_arr = []
for j in range(4):
    print(j)
    sfa2_arr[j].train(feature_arr[j])
    final_arr.append(sfa2_arr[j].execute(feature_arr[j]))

captions = ["gen1", "gen2", "load1", "load2"]
latent_arr = [input_latent1, input_latent2]
categories_arr = [input_categories1, input_categories2]
final_corr_arr = []
feature_corr_arr = []
for k in range(4):
    print("delta feature " + captions[k] + ": " + str(np.mean(tools.delta_diff(feature_arr[k]))))
    print("delta final " + captions[k] + ": " + str(np.mean(tools.delta_diff(final_arr[k]))))
    print("mean, std feature " + captions[k] + ": " + str(np.mean(feature_arr[k])) + ", " + str(np.std(feature_arr[k])))
    print("mean, std final " + captions[k] + ": " + str(np.mean(final_arr[k])) + ", " + str(np.std(final_arr[k])))

for l in range(2):
    final_corr_arr.append(tools.feature_latent_correlation(final_arr[l], latent_arr[l], categories_arr[l]))
    feature_corr_arr.append(tools.feature_latent_correlation(feature_arr[l], latent_arr[l], categories_arr[l]))