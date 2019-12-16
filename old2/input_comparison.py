from core import sensory, system_params, input_params, semantic, tools
import mdp, pickle
import numpy as np
import warnings

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

PARAMETERS.st1['object_code'] = input_params.make_object_code('L', 15)
PARAMETERS.st1['sequence'] = [0]

PARAMETERS.st1['movement_type'] = 'timed_border_stroll'
PARAMETERS.st1["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=50, border_extent=2.3)
PARAMETERS.st1["number_of_snippets"] = 200
PARAMETERS.st1["snippet_length"] = None
PARAMETERS.st1["interleaved"] = True
PARAMETERS.st1["blank_frame"] = False
PARAMETERS.st1["glue"] = "random"

# PARAMETERS.preview = True

print("Generating input")
sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
input_sequence, input_categories, input_latent, input_ranges = sensory_system.generate(fetch_indices=True,**PARAMETERS.st1)

print("Loading input")
input_sequence_loaded = np.load("input_clust.npy")

tools.compare_inputs((input_sequence,input_sequence_loaded), rate = 60)

# print("Saving input")
# np.save("input_clust", input_sequence)

# print("Generating and training SFA")
# switchboard1, sfa_layer1 = semantic._switchboard_based_layer( **PARAMETERS.sem_params1[0][1])
# switchboard2, sfa_layer2 = semantic._switchboard_based_layer( **PARAMETERS.sem_params1[1][1])
# top_node = semantic._sfa_flow_node(**PARAMETERS.sem_params1[2][1])
# sfa1 = mdp.Flow([switchboard1, sfa_layer1, switchboard2, sfa_layer2, top_node])
# sfa1.train(input_sequence[0])

# print("Saving and loading SFA")
# sfa1.save("inputtest.sfa")

print("Loading SFA")
sfa1_loaded = pickle.load(open("inputtest.sfa", "rb"))

print("Executing SFAs")
# feature_sequence = sfa1.execute(input_sequence)
feature_sequence_loaded = sfa1_loaded.execute(input_sequence)
feature_sequence_ll = sfa1_loaded.execute(input_sequence_loaded)

print("")
# print("delta feature: " + str(np.mean(tools.delta_diff(feature_sequence))))
print("delta feature loaded: " + str(np.mean(tools.delta_diff(feature_sequence_loaded))))
print("delta feature ll: " + str(np.mean(tools.delta_diff(feature_sequence_ll))))
# print("mean, std feature: " + str(np.mean(feature_sequence)) + ", " + str(np.std(feature_sequence)))
print("mean, std feature loaded: " + str(np.mean(feature_sequence_loaded)) + ", " + str(np.std(feature_sequence_loaded)))
print("mean, std feature ll: " + str(np.mean(feature_sequence_ll)) + ", " + str(np.std(feature_sequence_ll)))