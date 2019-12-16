from core import sensory, system_params, semantic, streamlined, tools
import numpy as np
import sklearn.linear_model
import scipy.stats
import time

PARAMETERS = system_params.SysParamSet()

# generate input
print("generate input")
sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

PARAMETERS.st1["movement_type"] = "gaussian_walk"
PARAMETERS.st1["movement_params"] = dict(dx=0.05, dt=0.05, step=5)
PARAMETERS.st1["number_of_snippets"] = 50
PARAMETERS.st1["snippet_length"] = 50

sequence, categories, latents = sensys.generate(**PARAMETERS.st1)
sequence2, categories2, latents2 = sensys.generate(**PARAMETERS.st1)
cat_arr = np.array(categories)
lat_arr = np.array(latents)
cat_arr2 = np.array(categories2)
lat_arr2 = np.array(latents2)

# make sfa
sfa1a_parms = [
        ('layer.linear', {
            'bo': 30,
            'rec_field_ch': 18,
            'spacing': 6,
            'in_channel_dim': 1,
            'out_sfa_dim': 32
        })
    ]
sfa1b_parms = [
    ('single.linear', {
        'dim_in': 900,
        'dim_out': 288
    })
]
# sfa1b_parms = [
#     ('inc.linear', {
#         'dim_in': 900,
#         'dim_out': 288
#     })
# ]

sfa2_parms = [
    ('inc.linear', {
        'dim_in': 288,
        'dim_out': 16
    })
]

sfa1a = semantic.build_module(sfa1a_parms)
sfa1b = semantic.build_module(sfa1b_parms)

sfa2a = semantic.build_module(sfa2_parms, eps=0.0005)
sfa2b = semantic.build_module(sfa2_parms, eps=0.0005)

# train sfa
print("train SFAlo")
# t = time.time()
semantic.train_SFA(sfa1a, sequence)
semantic.train_SFA(sfa1b, sequence)
# print("Time elapsed: {}".format(time.time()-t))

# execute sfa
print("execute SFAlo")
ya = semantic.exec_SFA(sfa1a, sequence)
yb = semantic.exec_SFA(sfa1b, sequence)
ya2 = semantic.exec_SFA(sfa1a, sequence2)
yb2 = semantic.exec_SFA(sfa1b, sequence2)

print("train SFAhi...")

d_a = []
d_b = []
r_a = []
r_b = []
for i in range(40):
    print("")
    print("=={:02d}==".format(i))

    semantic.train_SFA(sfa2a, ya)
    semantic.train_SFA(sfa2b, yb)

    sfa2a.save("../results/sandbox190128/sfa2a{:02d}.sfa".format(i))
    sfa2b.save("../results/sandbox190128/sfa2b{:02d}.sfa".format(i))

    za = semantic.exec_SFA(sfa2a, ya)
    zb = semantic.exec_SFA(sfa2b, yb)
    za2 = semantic.exec_SFA(sfa2a, ya2)
    zb2 = semantic.exec_SFA(sfa2b, yb2)

    # whitening
    za2_w = streamlined.normalizer(za2, PARAMETERS.normalization)(za2)
    zb2_w = streamlined.normalizer(zb2, PARAMETERS.normalization)(zb2)

    # training regressor
    # print("training regressor")

    training_matrix_a = za
    training_matrix_b = yb
    target_matrix = np.append(lat_arr, cat_arr[:, None], axis=1)
    learner_a = sklearn.linear_model.LinearRegression()
    learner_b = sklearn.linear_model.LinearRegression()
    learner_a.fit(training_matrix_a, target_matrix)
    learner_b.fit(training_matrix_b, target_matrix)

    # using regressor prediction to measure feature quality
    prediction_a = learner_a.predict(za2)
    prediction_b = learner_a.predict(zb2)
    _, _, r_valueX_a, _, _ = scipy.stats.linregress(lat_arr2[:, 0], prediction_a[:, 0])
    _, _, r_valueY_a, _, _ = scipy.stats.linregress(lat_arr2[:, 1], prediction_a[:, 1])
    _, _, r_valueCOS_a, _, _ = scipy.stats.linregress(lat_arr2[:, 2], prediction_a[:, 2])
    _, _, r_valueSIN_a, _, _ = scipy.stats.linregress(lat_arr2[:, 3], prediction_a[:, 3])
    _, _, r_valueCAT_a, _, _ = scipy.stats.linregress(cat_arr2, prediction_a[:, 4])
    _, _, r_valueX_b, _, _ = scipy.stats.linregress(lat_arr2[:, 0], prediction_b[:, 0])
    _, _, r_valueY_b, _, _ = scipy.stats.linregress(lat_arr2[:, 1], prediction_b[:, 1])
    _, _, r_valueCOS_b, _, _ = scipy.stats.linregress(lat_arr2[:, 2], prediction_b[:, 2])
    _, _, r_valueSIN_b, _, _ = scipy.stats.linregress(lat_arr2[:, 3], prediction_b[:, 3])
    _, _, r_valueCAT_b, _, _ = scipy.stats.linregress(cat_arr2, prediction_b[:, 4])

    r_a.append([r_valueX_a, r_valueY_a, r_valueSIN_a, r_valueCOS_a, r_valueCAT_a])
    r_b.append([r_valueX_b, r_valueY_b, r_valueSIN_b, r_valueCOS_b, r_valueCAT_b])

    delta_za = tools.delta_diff(za2_w)
    delta_zb = tools.delta_diff(zb2_w)

    d_a.append(delta_za)
    d_b.append(delta_zb)

    print("delta_a", d_a[-1])
    print("delta_b", d_b[-1])
    print("corr_a", r_a[-1])
    print("corr_b", r_b[-1])

np.save("../results/sandbox190128/r_a.npy", r_a)
np.save("../results/sandbox190128/r_b.npy", r_b)
np.save("../results/sandbox190128/d_a.npy", d_a)
np.save("../results/sandbox190128/d_b.npy", d_b)
