from core import semantic, system_params, input_params, tools, streamlined, sensory, result

import numpy as np
import scipy.stats
import os
import pickle
from matplotlib import pyplot as plt

import sklearn.linear_model

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

NFEAT = 3

PATH = "/local/results/lro02_o1850t/"

RES_FILE = PATH + "res28.p"
SFA1_FILE = PATH + "sfa1.p"
SFA2_FILE = PATH + "inc1_eps1_39.sfa"

TRAIN_DATA = PATH + "forming28.npz"
TEST_DATA = PATH + "testing0.npz"

TYPE = "S"           # S or E

# with open(RES_FILE, 'rb') as f:
#     res = pickle.load(f)
do_train = True
# if "learner"+TYPE in res.__dict__:
#     learner = eval("res.learner" + TYPE)
#     do_train = False
#     print("Pre-trained learner found in res-file")
# sfa2 = eval("res.sfa2" + TYPE)
sfa1 = semantic.load_SFA(SFA1_FILE)
sfa2 = semantic.load_SFA(SFA2_FILE)

PARAMETERS = system_params.SysParamSet()

with open(PATH + "sensory.p", 'rb') as f:
    sensory_system = pickle.load(f)
ran = np.arange(PARAMETERS.st1['number_of_snippets'])
with open(PATH + "st1.p", 'rb') as f:
    PARAMETERS.st1 = pickle.load(f)
seq1, cat1, lat1 = sensory_system.recall(numbers = ran, fetch_indices=False, **PARAMETERS.st1)

# if TRAIN_DATA == "" or TEST_DATA == "":
#     PARAMETERS.st1.update(dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
#                     object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=0))
#
#     sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
#
#     if do_train:
#         seq1, cat1, lat1 = sensys.generate(fetch_indices=False, **PARAMETERS.st1)
#     seq2, cat2, lat2 = sensys.generate(fetch_indices=False, **PARAMETERS.st1)
# else:
#     if do_train:
#         forming_data = np.load(TRAIN_DATA)
#         seq1, cat1, lat1, _ = forming_data['forming_sequenceX'], forming_data['forming_categories'], forming_data['forming_latent'], forming_data['forming_ranges']
#     testing_data = np.load(TEST_DATA)
#     seq2, cat2, lat2, _ = testing_data['testing_sequenceX'], testing_data['testing_categories'], testing_data['testing_latent'], testing_data['testing_ranges']

PARAMETERS.st1['number_of_snippets'] = 50
PARAMETERS.st1['snippet_length'] = 50
sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
seq2, cat2, lat2 = sensys.generate(fetch_indices=False, **PARAMETERS.st1)

cat2 = np.array(cat2)
lat2 = np.array(lat2)

if do_train:
    cat1 = np.array(cat1)
    lat1 = np.array(lat1)

    yy = semantic.exec_SFA(sfa1, seq1)
    yy_w = streamlined.normalizer(yy, PARAMETERS.normalization)(yy)
    zz = semantic.exec_SFA(sfa2, yy_w)
    # zz_w = streamlined.normalizer(zz, PARAMETERS.normalization)(zz)

    target_matrix = np.append(lat1, cat1[:,None], axis=1)
    training_matrix = zz
    # training_matrix = zz_w

    learner = sklearn.linear_model.LinearRegression()

    learner.fit(training_matrix[:,:NFEAT], target_matrix)

yy2 = semantic.exec_SFA(sfa1, seq2)
yy2_w = streamlined.normalizer(yy2, PARAMETERS.normalization)(yy2)
zz2 = semantic.exec_SFA(sfa2, yy2_w)
# zz2_w = streamlined.normalizer(zz2, PARAMETERS.normalization)(zz2)
prediction = learner.predict(zz2[:,:NFEAT])
# prediction = learner.predict(zz2_w)

f, ax = plt.subplots(1,2, sharex=True, squeeze=False, sharey=True)
# f, ax = plt.subplots(2,2, sharex=True)
ax[0,0].get_shared_y_axes().join(ax[0,0], ax[0,1])
ax[0,0].scatter(lat2[:,0], prediction[:,0], s=2, c='k')
slope00, intercept00, r_value00, _, _ = scipy.stats.linregress(lat2[:,0], prediction[:,0])
xx = np.arange(-1,1.005,0.1)
yy = slope00*xx+intercept00
ax[0,0].plot(xx,yy)
# rmse00 = rmse(prediction[:,0], lat2[:,0])
ax[0,0].set_title("r²={:.3f}".format(r_value00))
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel("coordinate prediction")

ax[0,1].scatter(lat2[:,1], prediction[:,1], s=2, c='k')
slope01, intercept01, r_value01, _, _ = scipy.stats.linregress(lat2[:,1], prediction[:,1])
xx = np.arange(-1,1.005,0.1)
yy = slope01*xx+intercept01
ax[0,1].plot(xx,yy)
# rmse01 = rmse(prediction[:,1], lat2[:,1])
ax[0,1].set_title("r²={:.3f}".format(r_value01))
ax[0,1].set_xlabel("y")

# ax[1,0].get_shared_y_axes().join(ax[1,0], ax[1,1])
# ax[1,0].scatter(lat2[:,2], prediction[:,2], s=2, c='k')
# rmse10 = rmse(prediction[:,2], lat2[:,2])
# ax[1,0].set_title("cos(phi) | RMSE={:.3f}".format(rmse10))
#
# ax[1,1].scatter(lat2[:,3], prediction[:,3], s=2, c='k')
# rmse11 = rmse(prediction[:,3], lat2[:,3])
# ax[1,1].set_title("sin(phi) | RMSE={:.3f}".format(rmse11))
plt.show()
