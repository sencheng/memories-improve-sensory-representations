from core import sensory, system_params, semantic, tools, semantic_params, input_params

import copy
import mdp
import numpy as np
import warnings
import pickle

warnings.simplefilter("module")

def makeSFA(parm1, parm2):
    l = []
    parm = parm1[:]
    parm.extend(parm2)
    for p in parm:
        if 'square' in p[0]:
            switchboard, sfa_layer = semantic._switchboard_based_layer(**p[1])
            l.append(switchboard)
            l.append(sfa_layer)
        elif 'single' in p[0]:
            top_node = semantic._sfa_flow_node(**p[1])
            l.append(top_node)
    sfa = mdp.Flow(l)
    return sfa


p1_1 = [
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
p1_2 = [
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



p2_1 = [
                            ('layer.square', {
                                'bo':               30,
                                'rec_field_ch':     12,
                                'spacing':          6,
                                'in_channel_dim':   1,
                                'out_sfa_dim1':     48,
                                'out_sfa_dim2':     32
                            })
                            ]
p2_2 = [
                            ('layer.square', {
                                'bo':               4,
                                'rec_field_ch':     3,
                                'spacing':          1,
                                'in_channel_dim':   32,
                                'out_sfa_dim1':     32,
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



p3_1 = [
                            ('layer.square', {
                                'bo':               30,
                                'rec_field_ch':     12,
                                'spacing':          6,
                                'in_channel_dim':   1,
                                'out_sfa_dim1':     48,
                                'out_sfa_dim2':     16
                            })
                            ]
p3_2 = [
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


#============================= SFA COLLECTION =====================================
p = [[p1_1,p1_2],[p2_1, p2_2],[p3_1,p3_2]]

# ============================ INPUTS =============================================
#inp_obj = [input_params.make_object_code('T', 15), input_params.make_object_code('TL', 15), input_params.make_object_code(('TX','LB'), 15)]
#seq = [[0], [0,1], [0,1]]
#numb = [400,200,200]
inp_obj = [input_params.make_object_code('TL', 15)]
seq = [[0,1]]
numb = [200]

PARAMETERS = system_params.SysParamSet()
sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

training_seq, training_cat, training_lat, training_ran = list(np.zeros(len(inp_obj))), list(np.zeros(len(inp_obj))), list(np.zeros(len(inp_obj))), list(np.zeros(len(inp_obj)))

print("Generating input...")

for i, (inp, s, num) in enumerate(zip(inp_obj, seq, numb)):
    PARAMETERS.input_params_default['object_code'] = inp
    PARAMETERS.input_params_default['sequence'] = s
    PARAMETERS.st1['number_of_snippets'] = num
    print(i)
    training_seq[i], training_cat[i], training_lat[i], training_ran[i] = sensory_system.generate(fetch_indices=True, **PARAMETERS.st1)
    PARAMETERS.st1['movement_type'] = 'copy_traj'
    PARAMETERS.st1['movement_params'] = dict(latent=training_lat[i], ranges=iter(training_ran[i]))
sfalist = []
structlist = []     # holding structure of sfa: for each sfa a list of two numbers: sfa1_len, sfa2_len

print("Creating SFAs...")
for j, plist in enumerate(p):
    print(j)
    templist = []
    templist_struct = []
    for jj in range(len(inp_obj)):
        templist.append(makeSFA(plist[0],plist[1]))
        templist_struct.append([len(plist[0]), len(plist[1])])
    sfalist.append(templist)
    structlist.append(templist_struct)


d = []
print("Training SFAs...")
for s, sfa in enumerate(sfalist):
    d.append([])
    structs = structlist[s]
    print("SFA " + str(s))
    for ip, (seq, cat, lat, ran) in enumerate(zip(training_seq, training_cat, training_lat, training_ran)):
        sfa1_len = structs[ip][0]
        print("     Input " + str(ip))
        sfa[ip].train(seq)
        d_vals = semantic.get_d_values(sfa[ip],get_all=True)
        sfa1list = []
        for dd in range(sfa1_len):
            sfa1list.append(d_vals[dd])
        sfa2list = []
        for dd2 in range(sfa1_len, len(d_vals)):
            sfa2list.append(d_vals[dd2])
        d[-1].append(np.array([np.array(sfa1list),np.array(sfa2list)]))
        sfa[ip].save("../results/sandbox4/sfa" + str(s) + "_ip" + str(ip) + ".sfa")
d_arr = np.array(d)
np.save("../results/sandbox4/d.npy",d_arr)



# ========================== BIG 1 SQUARE + 2xSINGLE ============================
# p2_1 = [
#         ('layer.square', {
#             'bo':               30,
#             'rec_field_ch':     15,
#             'spacing':          3,
#             'in_channel_dim':   1,
#             'out_sfa_dim1':     48,
#             'out_sfa_dim2':     48
#         }),
#         ('single', {
#             'dim_in':      48*36,
#             'dim_mid':     48*4,
#             'dim_out':     32
#         })
#     ]
#
# p2_2 = semantic_params.make_layer_series(32,32,32,32,16,16)
#
# # =========================== SMALL 1 SQUARE + 2xSINGLE ==========================
# p3_1 = [
#         ('layer.square', {
#             'bo':               30,
#             'rec_field_ch':     12,
#             'spacing':          6,
#             'in_channel_dim':   1,
#             'out_sfa_dim1':     48,
#             'out_sfa_dim2':     48
#         }),
#         ('single', {
#             'dim_in':      48*16,
#             'dim_mid':     48*2,
#             'dim_out':     32
#         })
#     ]
#
# p3_2 = semantic_params.make_layer_series(32,32,32,32,16,16)