from core import sensory, system_params, semantic, streamlined, tools
import numpy as np
import sklearn.linear_model
import scipy.stats

PARAMETERS = system_params.SysParamSet()

# generate input
sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

PARAMETERS.st1["movement_type"] = "gaussian_walk"
PARAMETERS.st1["movement_params"] = dict(dx=0.05, dt=0.05, step=5)
PARAMETERS.st1["number_of_snippets"] = 200
PARAMETERS.st1["snippet_length"] = 50

sequence, categories, latents = sensys.generate(**PARAMETERS.st1)
cat_arr = np.array(categories)
lat_arr = np.array(latents)

# make sfa
sfa_parms = [
        ('layer.linear', {
            'bo': 30,
            'rec_field_ch': 18,
            'spacing': 6,
            'in_channel_dim': 1,
            'out_sfa_dim': 32
        })
    ]

sfa = semantic.build_module(sfa_parms)

# train sfa
semantic.train_SFA(sfa, sequence)

# execute sfa
y = semantic.exec_SFA(sfa, sequence)

# whitening
whitener = streamlined.normalizer(y, PARAMETERS.normalization)
y_w = whitener(y)

# delta values
deltas = tools.delta_diff(y_w)

# correlations of individual features with latents
tools.feature_latent_correlation(y, lat_arr, cat_arr)

# training regressor
training_matrix = y
target_matrix = np.append(latents, cat_arr[:, None], axis=1)
learner = sklearn.linear_model.LinearRegression()
learner.fit(training_matrix, target_matrix)

# using regressor prediction to measure feature quality
prediction = learner.predict(y)
_, _, r_valueX, _, _ = scipy.stats.linregress(lat_arr[:, 0], prediction[:, 0])
_, _, r_valueY, _, _ = scipy.stats.linregress(lat_arr[:, 1], prediction[:, 1])
_, _, r_valueCOS, _, _ = scipy.stats.linregress(lat_arr[:, 2], prediction[:, 2])
_, _, r_valueSIN, _, _ = scipy.stats.linregress(lat_arr[:, 3], prediction[:, 3])
_, _, r_valueCAT, _, _ = scipy.stats.linregress(cat_arr, prediction[:, 4])
