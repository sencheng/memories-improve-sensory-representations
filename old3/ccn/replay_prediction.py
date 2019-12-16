"""
(Figure 7)

Processes and plots data that was generated using :py:mod:`learnrate2`. Delta values of SFA features
and the prediction quality of the regressors are plotted against number of training repetitions of SFA2.
Also, scatter plots latent variable vs. regressor prediction are shown.

"""

from core import semantic, system_params, input_params, tools, streamlined, sensory, result

import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
import matplotlib
import sklearn.linear_model
import scipy.stats

if __name__ == "__main__":
    PLOT_XY = True
    PLOT_PHI = True

    TEST_DATA = 1  # 0: Training only, 1: New only, 2: Both
    LEARNRATE2 = True
    EXP_MODE = 1  # 0: both, 1: linear, 2: square

    MIX = False
    PATH_PRE = "/local/results/"  # Prefix for where to load results from
    # PATH_PRE = "/media/goerlrwh/Extreme Festplatte/results/"
    PATH = PATH_PRE + "replay_o18/"    # Directory to load results from

    # colors = [['b', 'c'], ['r', 'm'], ['g', 'y'], ['k', '0.5']]
    colors = ['b', 'r', 'g', 'k']
    linest = ['-', '--', ':']

    dcolors = ['k', 'k', 'k', 'k']
    dstyles = ['--', '-', '-.', ':']
    ccolors = ['g', 'g', 'b', 'b']
    cstyles = ['-', '--', '-', '--']

    rcolor = '0.6'
    rstyle = '--'

    DCNT = 3
    SINGLED = False

    NFEAT = 3

    # Whitening settings
    # of the data
    WHITENER = True
    NORM = False
    # for analysis
    CORRNORM = False
    DELTANORM = True

    # matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['lines.markersize'] = 22
    matplotlib.rcParams['lines.markeredgewidth'] = 2
    matplotlib.rcParams['lines.linewidth'] = 2
    font = {'family' : 'Sans',
            'size'   : 22}

    matplotlib.rc('font', **font)

    fig=plt.figure(figsize=(10,5))

    NRUNS = 40
    eps_list = [0.0005]
    NSEQS = TEST_DATA if TEST_DATA > 0 else 1
    NEPS = len(eps_list)

    PARAMETERS = system_params.SysParamSet()
    if os.path.isfile(PATH + "st1.p"):
        with open(PATH + "st1.p", 'rb') as f:
            PARAMETERS.st1 = pickle.load(f)
    else:
        PARAMETERS.st1.update(dict(number_of_snippets=100, snippet_length=100, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                    object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=0.1))

    if WHITENER:
        with open(PATH + "whitener.p", 'rb') as f:
            whitener = pickle.load(f)

    PARAMETERS.st1['number_of_snippets'] = 50

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


    training_sequence, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    new_sequence1, new_categories, new_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    tcat = np.array(training_categories)
    tlat = np.array(training_latent)
    ncat = np.array(new_categories)
    nlat = np.array(new_latent)

    input_delta = np.mean(tools.delta_diff(new_sequence1))
    sfa1 = semantic.load_SFA(PATH + "sfa1.p")

    training_sequence = semantic.exec_SFA(sfa1,training_sequence)
    if NORM:
        training_sequence = streamlined.normalizer(training_sequence, PARAMETERS.normalization)(training_sequence)
    elif WHITENER:
        training_sequence = whitener(training_sequence)
    target_matrix = np.append(tlat, tcat[:, None], axis=1)

    new_sequence1 = semantic.exec_SFA(sfa1,new_sequence1)
    if NORM:
        new_sequence1 = streamlined.normalizer(new_sequence1, PARAMETERS.normalization)(new_sequence1)
    elif WHITENER:
        new_sequence1 = whitener(new_sequence1)
    if DELTANORM:
        intermediate_delta1 = np.mean(np.sort(tools.delta_diff(streamlined.normalizer(new_sequence1, PARAMETERS.normalization)(new_sequence1)))[:DCNT])
    else:
        intermediate_delta1 = np.mean(np.sort(tools.delta_diff(new_sequence1))[:DCNT])

    learnerLo = sklearn.linear_model.LinearRegression()
    learnerLo.fit(training_sequence[:,:], target_matrix)
    predictionLo = learnerLo.predict(new_sequence1[:, :])
    _, _, r_XLo, _, _ = scipy.stats.linregress(nlat[:, 0], predictionLo[:, 0])
    _, _, r_YLo, _, _ = scipy.stats.linregress(nlat[:, 1], predictionLo[:, 1])
    _, _, r_CLo, _, _ = scipy.stats.linregress(ncat, predictionLo[:, 4])

    print(r_XLo, r_YLo, r_CLo)

    b1_sfa = semantic.load_SFA(PATH + "b1.sfa")

    b1_y_n = semantic.exec_SFA(b1_sfa, new_sequence1)
    b1_w_n = streamlined.normalizer(b1_y_n, PARAMETERS.normalization)(b1_y_n)
    if DELTANORM:
        b1_ds_n = np.sort(tools.delta_diff(b1_w_n))[:DCNT]
    else:
        b1_ds_n = np.sort(tools.delta_diff(b1_y_n))[:DCNT]
    b1_d_n = np.mean(b1_ds_n)

    all_d_n = tools.delta_diff(np.concatenate((new_latent, new_categories[:,None]), axis=1))
    x_d_n = all_d_n[0]
    y_d_n = all_d_n[1]
    cat_d_n = all_d_n[4]
    lat_d_n = np.mean([x_d_n, y_d_n, cat_d_n])


    b1_corr_n = np.abs(tools.feature_latent_correlation(b1_y_n, new_latent, new_categories))
    b1_xycorr_n = np.mean([np.max(b1_corr_n[0,:]),np.max(b1_corr_n[1,:])])
    training_matrix = semantic.exec_SFA(b1_sfa, training_sequence)
    if CORRNORM:
        training_matrix = streamlined.normalizer(training_matrix, PARAMETERS.normalization)(training_matrix)
    learner = sklearn.linear_model.LinearRegression()
    learner.fit(training_matrix[:,:NFEAT], target_matrix)
    if CORRNORM:
        prediction = learner.predict(b1_w_n[:, :NFEAT])
    else:
        prediction = learner.predict(b1_y_n[:,:NFEAT])
    # prediction = learner.predict(b1_w_n)
    _, _, r_valueX, _, _ = scipy.stats.linregress(nlat[:, 0], prediction[:, 0])
    _, _, r_valueY, _, _ = scipy.stats.linregress(nlat[:, 1], prediction[:, 1])
    _, _, r_valueC, _, _ = scipy.stats.linregress(ncat, prediction[:, 4])
    b1_xyr = np.mean((r_valueX, r_valueY))
    # b1_catcorr_n = np.max(b1_corr_n[4, :])
    b1_catcorr_n = r_valueC

    x = list(range(NRUNS))

    # fd, ad = plt.subplots(1, NEPS*2, sharex=True, sharey=False)
    # fd.text(0.5, 0.04, 'number of forming repetitions', ha='center', va='center')

    # axy[0].set_ylabel('r of variable prediction')
    # axy[0].set_title('correlation of\nfeatures with latents')

    corlst1 = []
    corlst2 = []
    corlst1_2 = []
    corlst2_2 = []
    for ei, eps in enumerate(eps_list):
        inc1_d = []
        inc2_d = []
        inc1_mean = []
        inc1_var = []
        inc1_xycorr, inc1_catcorr, inc1_xyr = [], [], []
        for i in range(NRUNS):
            inc1_sfa = inc2_sfa = semantic.load_SFA(PATH + "inc1_eps{}_{}.sfa".format(ei+1, i))

            inc1_y = semantic.exec_SFA(inc1_sfa, new_sequence1)
            inc1_w = streamlined.normalizer(inc1_y, PARAMETERS.normalization)(inc1_y)
            if DELTANORM:
                inc1_ds = np.sort(tools.delta_diff(inc1_w))[:DCNT]
            else:
                inc1_ds = np.sort(tools.delta_diff(inc1_y))[:DCNT]

            inc1_mean.append(np.mean(inc1_y))
            inc1_var.append(np.var(inc1_y))

            inc1_d.append(np.mean(inc1_ds))
            inc1_corr_raw = tools.feature_latent_correlation(inc1_y, new_latent, new_categories)
            inc1_corr = np.abs(inc1_corr_raw)
            if ei == 1:
                corlst1.append(inc1_corr_raw)
            if ei == 2:
                corlst1_2.append(inc1_corr_raw)

            inc1_xycorr.append(np.mean([np.max(inc1_corr[0, :]), np.max(inc1_corr[1, :])]))

            training_matrix = semantic.exec_SFA(inc1_sfa, training_sequence)
            if CORRNORM:
                training_matrix = streamlined.normalizer(training_matrix, PARAMETERS.normalization)(training_matrix)
            learner = sklearn.linear_model.LinearRegression()
            learner.fit(training_matrix[:,:NFEAT], target_matrix)
            if CORRNORM:
                prediction = learner.predict(inc1_w[:, :NFEAT])
            else:
                prediction = learner.predict(inc1_y[:,:NFEAT])
            # prediction = learner.predict(inc1_w)
            _, _, r_valueX, _, _ = scipy.stats.linregress(nlat[:, 0], prediction[:, 0])
            _, _, r_valueY, _, _ = scipy.stats.linregress(nlat[:, 1], prediction[:, 1])
            _, _, r_valueC, _, _ = scipy.stats.linregress(ncat, prediction[:, 4])
            inc1_xyr.append(np.mean((r_valueX, r_valueY)))
            inc1_catcorr.append(r_valueC)
            # inc1_catcorr.append(np.max(inc1_corr[4, :]))

        # axxy = axy[ei]
        # line_br, = fig.plot(x, [b1_xyr] * NRUNS, label="batch r", c=ccolors[0], ls=cstyles[0])
        # line_bcat, = fig.plot(x, [b1_catcorr_n] * NRUNS, label="batch cat", c=ccolors[1], ls=cstyles[1])
        line_incr, = plt.plot(x, inc1_xyr, label="inc r", c=ccolors[2], ls=cstyles[2])
        line_inccat, = plt.plot(x, inc1_catcorr, label="inc cat", c=ccolors[3], ls=cstyles[3])
        if eps == 0.0005:
            plt.plot(x[0], inc1_xyr[0], marker='^', c='k', mfc='none', clip_on=False)
            plt.plot(x[-1], inc1_xyr[-1], marker='v', c='k', mfc='none', clip_on=False)
        #     axxy.plot(x[0], inc1_catcorr[0], marker='^', c='k', mfc='none', clip_on=False)
        #     axxy.plot(x[-1], inc1_catcorr[-1], marker='v', c='k', mfc='none', clip_on=False)

    plt.legend((line_incr, line_inccat), ("x,y", "identity"), loc=4)
    plt.ylabel("encoding quality")
    plt.xlabel("training repetitions")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2)

    # ad[0].legend((line_bd, line_incd, line_pred, line_latd, line_incmean, line_incvar), ("batch", "incr.", "SFAlo", "latents", "incmean", "incvar/100"), loc=5)
    # ad[0].legend((line_bd, line_incd, line_pred, line_latd), ("batch", "incr.", "SFAlo", "latents"), loc=5)

    plt.show()
