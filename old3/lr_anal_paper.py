from core import semantic, system_params, input_params, tools, streamlined, sensory, result

import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
import matplotlib
import sklearn.linear_model
import scipy.stats

PLOT_XY = True
PLOT_PHI = True

TEST_DATA = 1  # 0: Training only, 1: New only, 2: Both
LEARNRATE2 = True
EXP_MODE = 1  # 0: both, 1: linear, 2: square

PATH = "/local/results/lro02_o1850t/"

# colors = [['b', 'c'], ['r', 'm'], ['g', 'y'], ['k', '0.5']]
colors = ['b', 'r', 'g', 'k']
linest = ['-', '--', ':']

DCNT = 3
SINGLED = False

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 12
matplotlib.rcParams['lines.markeredgewidth'] = 2
font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

NRUNS = 40
eps_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
NSEQS = TEST_DATA if TEST_DATA > 0 else 1
NEPS = len(eps_list)

PARAMETERS = system_params.SysParamSet()
if os.path.isfile(PATH + "st1.p"):
    with open(PATH + "st1.p", 'rb') as f:
        PARAMETERS.st1 = pickle.load(f)
else:
    PARAMETERS.st1.update(dict(number_of_snippets=100, snippet_length=100, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=0.1))

PARAMETERS.st1['number_of_snippets'] = 50
# PARAMETERS.st1['input_noise'] = 0.1
# PARAMETERS.st1['object_code'] = input_params.make_object_code('L')
# PARAMETERS.st1['sequence'] = [0]
# if not LEARNRATE2:
#     PARAMETERS.st1['input_noise'] = 0.2
#     PARAMETERS.st1['frame_shape'] = (12,12)
#     PARAMETERS.st1['object_code'] = input_params.make_object_code(['-'], sizes=22)
#     PARAMETERS.st1['sequence'] = [0]

sample_parms = dict(PARAMETERS.st1)
sample_parms['number_of_snippets'] = 1
sample_parms['movement_type'] = 'sample'
sample_parms['movement_params'] = dict()

sensory_system = pickle.load(open(PATH + "sensory.p", 'rb'))
ran = np.arange(PARAMETERS.st1['number_of_snippets'])

sensys2 = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
parmdict = dict(PARAMETERS.st1)
parmdict['movement_type'] = 'sample'
parmdict['movement_params'] = dict(x_range=None, y_range=None, t_range=[0], x_step=0.05, y_step=0.05, t_step=22.5)
parmdict['number_of_snippets'] = 1
parmdict['snippet_length'] = None


# training_sequence = training_latent = training_categories = new_sequence1 = new_sequence2 = new_latent = new_categories = 0
# b1_d_n = b1_d_t = b2_d_n = b2_d_t = b1_xycorr_n = b1_xycorr_t = b2_xycorr_n = b2_xycorr_t = b1_phicorr_n = b1_phicorr_t = b2_phicorr_n = b2_phicorr_t = 0
# b1_catcorr_n = b1_catcorr_t = b2_catcorr_n = b2_catcorr_t = 0
# xy_d_n = xy_d_t = phi_d_n = phi_d_t = 0

# training_sequence, training_categories, training_latent = sensory_system.recall(numbers = ran, fetch_indices=False, **PARAMETERS.st1)
training_sequence, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
tcat = np.array(training_categories)
tlat = np.array(training_latent)
new_sequence1, new_categories, new_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
ncat = np.array(new_categories)
nlat = np.array(new_latent)

# tools.compare_inputs((training_sequence, new_sequence1))

# new_sequence1, new_categories, new_latent = sensys2.generate(fetch_indices=False, **parmdict)
input_delta = np.mean(tools.delta_diff(new_sequence1))
sfa1 = semantic.load_SFA(PATH + "sfa1.p")

training_sequence = semantic.exec_SFA(sfa1,training_sequence)
training_sequence = streamlined.normalizer(training_sequence, PARAMETERS.normalization)(training_sequence)
target_matrix = np.append(tlat, tcat[:, None], axis=1)

new_sequence1 = semantic.exec_SFA(sfa1,new_sequence1)
new_sequence1 = streamlined.normalizer(new_sequence1, PARAMETERS.normalization)(new_sequence1)
intermediate_delta1 = np.mean(np.sort(tools.delta_diff(new_sequence1))[:DCNT])

b1_sfa = semantic.load_SFA(PATH + "b1.sfa")

b1_y_n = semantic.exec_SFA(b1_sfa, new_sequence1)
b1_w_n = streamlined.normalizer(b1_y_n, PARAMETERS.normalization)(b1_y_n)
b1_ds_n = np.sort(tools.delta_diff(b1_w_n))[:DCNT]
if SINGLED:
    b1_d_n = []
    for di in range(DCNT):
        b1_d_n.append(b1_ds_n[di])
else:
    b1_d_n = np.mean(b1_ds_n)

x_d_n = tools.delta_diff(new_latent)[0]
y_d_n = tools.delta_diff(new_latent)[1]
xy_d_n = np.mean([x_d_n, y_d_n])

b1_corr_n = np.abs(tools.feature_latent_correlation(b1_w_n, new_latent, new_categories))
b1_xycorr_n = np.mean([np.max(b1_corr_n[0,:]),np.max(b1_corr_n[1,:])])
training_matrix = semantic.exec_SFA(b1_sfa, training_sequence)
# training_matrix = streamlined.normalizer(training_matrix, PARAMETERS.normalization)(training_matrix)
learner = sklearn.linear_model.LinearRegression()
learner.fit(training_matrix, target_matrix)
prediction = learner.predict(b1_y_n)
# prediction = learner.predict(b1_w_n)
_, _, r_valueX, _, _ = scipy.stats.linregress(nlat[:, 0], prediction[:, 0])
_, _, r_valueY, _, _ = scipy.stats.linregress(nlat[:, 1], prediction[:, 1])
b1_xyr = np.mean((r_valueX, r_valueY))
b1_catcorr_n = np.max(b1_corr_n[4, :])

x = list(range(NRUNS))

fd, ad = plt.subplots(1, NEPS, sharex=True, sharey=True)
fd.text(0.5, 0.04, 'number of training repetitions', ha='center', va='center')
# if SINGLED:
#     fd.text(0.06, 0.5, 'delta-value (individual features)'.format(DCNT), ha='center', va='center', rotation='vertical')
# else:
#     fd.text(0.06, 0.5, 'delta-value (average of {} slowest features)'.format(DCNT), ha='center', va='center', rotation='vertical')

fd.text(0.06, 0.5, 'delta-value of features', ha='center', va='center', rotation='vertical')

fxy, axy = plt.subplots(1, NEPS, sharex=True, sharey=True)
fxy.text(0.5, 0.04, 'number of training repetitions', ha='center', va='center')
fxy.text(0.06, 0.5, 'correlation of features with latents', ha='center', va='center', rotation='vertical')

corlst1 = []
corlst2 = []
corlst1_2 = []
corlst2_2 = []
for ei, eps in enumerate(eps_list):
    # =========================================================================================================================================================================
    # if ei == 0:
    #     continue
    # =========================================================================================================================================================================
    inc1_d = []
    inc2_d = []
    inc1_xycorr, inc1_catcorr, inc1_xyr = [], [], []
    for i in range(NRUNS):
        inc1_sfa = inc2_sfa = semantic.load_SFA(PATH + "inc1_eps{}_{}.sfa".format(ei, i))

        inc1_y = semantic.exec_SFA(inc1_sfa, new_sequence1)
        inc1_w = streamlined.normalizer(inc1_y, PARAMETERS.normalization)(inc1_y)
        inc1_ds = np.sort(tools.delta_diff(inc1_w))[:DCNT]
        if SINGLED:
            for di in range(DCNT):
                if i == 0:
                    inc1_d.append([])
                    inc2_d.append([])
                inc1_d[di].append(inc1_ds[di])
        else:
            inc1_d.append(np.mean(inc1_ds))
        inc1_corr_raw = tools.feature_latent_correlation(inc1_w, new_latent, new_categories)
        inc1_corr = np.abs(inc1_corr_raw)
        if ei == 1:
            corlst1.append(inc1_corr_raw)
        if ei == 2:
            corlst1_2.append(inc1_corr_raw)

        inc1_xycorr.append(np.mean([np.max(inc1_corr[0, :]), np.max(inc1_corr[1, :])]))

        training_matrix = semantic.exec_SFA(inc1_sfa, training_sequence)
        # training_matrix = streamlined.normalizer(training_matrix, PARAMETERS.normalization)(training_matrix)
        learner = sklearn.linear_model.LinearRegression()
        learner.fit(training_matrix, target_matrix)
        prediction = learner.predict(inc1_y)
        # prediction = learner.predict(inc1_w)
        _, _, r_valueX, _, _ = scipy.stats.linregress(nlat[:, 0], prediction[:, 0])
        _, _, r_valueY, _, _ = scipy.stats.linregress(nlat[:, 1], prediction[:, 1])
        _, _, r_valueC, _, _ = scipy.stats.linregress(ncat, prediction[:, 4])
        inc1_xyr.append(np.mean((r_valueX, r_valueY)))
        inc1_catcorr.append(r_valueC)
        # inc1_catcorr.append(np.max(inc1_corr[4, :]))
    axd = ad[ei]
    fd.suptitle("avg d-values")
    axd.set_title("eps = {}".format(eps))
    if SINGLED:
        for di in range(DCNT):
            line_bd, = axd.plot(x, [b1_d_n[di]] * NRUNS, label="batch", c=colors[0], ls=linest[0])
            line_incd, = axd.plot(x, inc1_d[di], label="inc", c=colors[1], ls=linest[0])
            if eps == 0.0005:
                axd.plot(x[0], inc1_d[di][0], marker='^', c='k', mfc='none')
                axd.plot(x[-1], inc1_d[di][-1], marker='v', c='k', mfc='none')
    else:
        line_bd, = axd.plot(x, [b1_d_n] * NRUNS, label="batch", c=colors[0], ls=linest[0])
        line_incd, = axd.plot(x, inc1_d, label="inc", c=colors[1], ls=linest[0])
        if eps == 0.0005:
            axd.plot(x[0], inc1_d[0], marker='^', c='k', mfc='none')
            axd.plot(x[-1], inc1_d[-1], marker='v', c='k', mfc='none')
    line_pred, = axd.plot(x, [intermediate_delta1]*NRUNS, label="pre", c=colors[2], ls=linest[0])
    line_xyd, = axd.plot(x, [xy_d_n]*NRUNS, label="xy", c=colors[-1], ls=linest[0])

    axxy = axy[ei]
    fxy.suptitle("feature correlation with x/y-coordinate and with object category")
    axxy.set_title("eps = {}".format(eps))
    # line_bxy, = axxy.plot(x, [b1_xycorr_n] * NRUNS, label="batch", c=colors[0], ls=linest[2])
    line_bcat, = axxy.plot(x, [b1_catcorr_n] * NRUNS, label="batch cat", c=colors[0], ls=linest[1])
    line_br, = axxy.plot(x, [b1_xyr] * NRUNS, label="batch r²", c=colors[0], ls=linest[0])
    # line_incxy, = axxy.plot(x, inc1_xycorr, label="inc", c=colors[1], ls=linest[2])
    line_incr, = axxy.plot(x, inc1_xyr, label="inc r²", c=colors[1], ls=linest[0])
    line_inccat, = axxy.plot(x, inc1_catcorr, label="inc cat", c=colors[1], ls=linest[1])
    if eps == 0.0005:
        axxy.plot(x[0], inc1_xyr[0], marker='^', c='k', mfc='none')
        axxy.plot(x[-1], inc1_xyr[-1], marker='v', c='k', mfc='none')
        axxy.plot(x[0], inc1_catcorr[0], marker='^', c='k', mfc='none')
        axxy.plot(x[-1], inc1_catcorr[-1], marker='v', c='k', mfc='none')
# plt.figlegend((line_bxy, line_incxy), ("batch", "incremental"), 1)
# plt.figlegend((line_bxy, line_bcat, line_br), ("x/y - coordinate", "object category", "x/y regression r²"), 2)
plt.figlegend((line_br, line_incr), ("batch", "incremental"), 1)
plt.figlegend((line_bcat, line_br), ("category r", "x/y r"), 2)
plt.figure(fd.number)
plt.figlegend((line_bd, line_incd, line_pred, line_xyd), ("batch", "incremental", "SFAlo", "x/y-coordinate"), 1)
fxy.subplots_adjust(bottom=0.15, top=0.85)
fd.subplots_adjust(bottom=0.15, top=0.85)

# f1, ax1 = plt.subplots(2, 2, squeeze=True)
# for lr in [1,2]:
#     for ki, key in enumerate([0,NRUNS-1]):
#         ax1[ki][lr-1].matshow(np.abs([corlst1[key], corlst1_2[key]][lr-1]), cmap=plt.cm.Blues, vmin=0, vmax=1)
#         ax1[ki][lr-1].set_title(key)
#         for (ii, jj), z in np.ndenumerate([corlst1[key], corlst1_2[key]][lr-1]):
#             ax1[ki][lr-1].text(jj, ii, '{:.0f}'.format(z*100), ha='center', va='center', color="white")
# f1.suptitle("inc1: latent correlation of individual features")
plt.show()
