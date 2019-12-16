"""
(Figure 8)

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
    PATH = PATH_PRE + "replay_o18_rep_gen/"    # Directory to load results from
    # GENPATH = PATH_PRE + "replay_o18_generation/"

    # colors = [['b', 'c'], ['r', 'm'], ['g', 'y'], ['k', '0.5']]
    colors = ['b', 'r', 'g', 'k']
    linest = ['-', '--', ':']

    dcolors = ['k', 'k', 'k', 'k']
    dstyles = ['--', '-', '-.', ':']
    ccolors = ['k', '0.6', 'k', '0.6']
    cstyles = ['--', '--', '-', '-']

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
    font = {'family' : 'Sans',
            'size'   : 22}

    matplotlib.rc('font', **font)

    fig=plt.figure()
    axtop , axbot1, axbot2 = [], [], []
    axtop.append(fig.add_subplot(2,2,1))
    axtop.append(fig.add_subplot(2,2,2))
    axbot1.append(fig.add_subplot(4,3,7))
    axbot2.append(fig.add_subplot(4,3,10, sharex=axbot1[0], sharey=axbot1[0]))
    axbot1.append(fig.add_subplot(4,3,8, sharex=axbot1[0], sharey=axbot1[0]))
    axbot2.append(fig.add_subplot(4,3,11, sharex=axbot1[0], sharey=axbot1[0]))
    axbot1.append(fig.add_subplot(4,3,9))
    axbot2.append(fig.add_subplot(4,3,12, sharex=axbot1[2], sharey=axbot1[2]))

    axbot1[0].set_ylim([-1.7, 1.7])
    axbot1[0].set_yticks([-1,0,1])
    # axbot1[0].set_xlim([-1, 1])
    axbot1[0].set_xticks([-1,0,1])
    corr_ylim = axtop[1].get_ylim()
    axtop[1].set_ylim([corr_ylim[0], 1])

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
        # with open(GENPATH + "whitener.p", 'rb') as f:
        #     whitener_gen = pickle.load(f)

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


    training_sequence1, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    new_sequence1, new_categories, new_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    # new_sequence_gen = new_sequence1
    seq2 = new_sequence1
    seq1 = training_sequence1
    tcat = np.array(training_categories)
    tlat = np.array(training_latent)
    ncat = np.array(new_categories)
    nlat = np.array(new_latent)

    input_delta = np.mean(tools.delta_diff(new_sequence1))
    sfa1 = semantic.load_SFA(PATH + "sfa1.p")

    # sfa1_gen = semantic.load_SFA(GENPATH + "sfa1.p")

    training_sequence = semantic.exec_SFA(sfa1,training_sequence1)
    if NORM:
        training_sequence = streamlined.normalizer(training_sequence, PARAMETERS.normalization)(training_sequence)
    elif WHITENER:
        training_sequence = whitener(training_sequence)
    target_matrix = np.append(tlat, tcat[:, None], axis=1)

    # training_sequence_gen = semantic.exec_SFA(sfa1_gen, training_sequence1)
    # if NORM:
    #     training_sequence_gen = streamlined.normalizer(training_sequence_gen, PARAMETERS.normalization)(training_sequence_gen)
    # elif WHITENER:
    #     training_sequence_gen = whitener_gen(training_sequence_gen)

    new_sequence1 = semantic.exec_SFA(sfa1,new_sequence1)
    if NORM:
        new_sequence1 = streamlined.normalizer(new_sequence1, PARAMETERS.normalization)(new_sequence1)
    elif WHITENER:
        new_sequence1 = whitener(new_sequence1)
    if DELTANORM:
        intermediate_delta1 = np.mean(np.sort(tools.delta_diff(streamlined.normalizer(new_sequence1, PARAMETERS.normalization)(new_sequence1)))[:DCNT])
    else:
        intermediate_delta1 = np.mean(np.sort(tools.delta_diff(new_sequence1))[:DCNT])

    # new_sequence_gen = semantic.exec_SFA(sfa1_gen, new_sequence_gen)
    # if NORM:
    #     new_sequence_gen = streamlined.normalizer(new_sequence_gen, PARAMETERS.normalization)(new_sequence_gen)
    # elif WHITENER:
    #     new_sequence_gen = whitener(new_sequence_gen)

    learnerLo = sklearn.linear_model.LinearRegression()
    learnerLo.fit(training_sequence[:,:], target_matrix)
    predictionLo = learnerLo.predict(new_sequence1[:, :])
    _, _, r_XLo, _, _ = scipy.stats.linregress(nlat[:, 0], predictionLo[:, 0])
    _, _, r_YLo, _, _ = scipy.stats.linregress(nlat[:, 1], predictionLo[:, 1])
    _, _, r_CLo, _, _ = scipy.stats.linregress(ncat, predictionLo[:, 4])

    print(r_XLo, r_YLo, r_CLo)

    b1_sfa = semantic.load_SFA(PATH + "b.sfa")

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

    ad = axtop
    ad[0].set_ylabel('delta-value')
    # ad[0].set_title('slowness of\n3 slowest features')

    axy =ad[1:]
    axy[0].set_ylabel('r of variable prediction')
    # axy[0].set_title('correlation of\nfeatures with latents')

    corlst1 = []
    corlst2 = []
    corlst1_2 = []
    corlst2_2 = []
    for ei, eps in enumerate(eps_list):
        inc1_d = []
        inc2_d = []
        incgen_d = []
        inc1_mean = []
        inc1_var = []
        inc1_xycorr, inc1_catcorr, inc1_xyr = [], [], []
        incgen_xycorr, incgen_catcorr, incgen_xyr = [], [], []
        for i in range(NRUNS):
            inc1_sfa = inc2_sfa = semantic.load_SFA(PATH + "inc_rep{}.sfa".format(i))

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

            # here replay _ generation results
            incgen_sfa = semantic.load_SFA(PATH + "inc_gen{}.sfa".format(i))

            incgen_y = semantic.exec_SFA(incgen_sfa, new_sequence1)
            incgen_w = streamlined.normalizer(incgen_y, PARAMETERS.normalization)(incgen_y)
            if DELTANORM:
                incgen_ds = np.sort(tools.delta_diff(incgen_w))[:DCNT]
            else:
                incgen_ds = np.sort(tools.delta_diff(incgen_y))[:DCNT]

            incgen_d.append(np.mean(incgen_ds))

            training_matrix_gen = semantic.exec_SFA(incgen_sfa, training_sequence)
            if CORRNORM:
                training_matrix = streamlined.normalizer(training_matrix_gen, PARAMETERS.normalization)(training_matrix_gen)
            learner_gen = sklearn.linear_model.LinearRegression()
            learner_gen.fit(training_matrix_gen[:, :NFEAT], target_matrix)
            if CORRNORM:
                prediction_gen = learner_gen.predict(incgen_w[:, :NFEAT])
            else:
                prediction_gen = learner_gen.predict(incgen_y[:, :NFEAT])
            # prediction = learner.predict(inc1_w)
            _, _, r_valueX_gen, _, _ = scipy.stats.linregress(nlat[:, 0], prediction_gen[:, 0])
            _, _, r_valueY_gen, _, _ = scipy.stats.linregress(nlat[:, 1], prediction_gen[:, 1])
            _, _, r_valueC_gen, _, _ = scipy.stats.linregress(ncat, prediction_gen[:, 4])
            incgen_xyr.append(np.mean((r_valueX_gen, r_valueY_gen)))
            incgen_catcorr.append(r_valueC_gen)

        axd = ad[ei]

        line_bd, = axd.plot(x, [b1_d_n] * NRUNS, label="batch", c=dcolors[0], ls=dstyles[0])
        line_incd, = axd.plot(x, inc1_d, label="inc", c=dcolors[1], ls=dstyles[1])
        line_gend, = axd.plot(x, incgen_d, label="incgen", c=dcolors[1], ls=dstyles[3])
        # line_incmean, = axd.plot(x, inc1_mean, label="inc_mean", c='r', ls=dstyles[1])
        # line_incvar, = axd.plot(x, np.array(inc1_var)/100, label="inc_var", c='g', ls=dstyles[1])
        if eps == 0.0005:
            axd.plot(x[0], inc1_d[0], marker='^', c='k', mfc='none', clip_on=False)
            axd.plot(x[-1], inc1_d[-1], marker='v', c='k', mfc='none', clip_on=False)
        line_pred, = axd.plot(x, [intermediate_delta1]*NRUNS, label="pre", c=dcolors[2], ls=dstyles[2])
        # line_latd, = axd.plot(x, [lat_d_n]*NRUNS, label="latents", c=dcolors[3], ls=dstyles[3])

        axxy = axy[ei]
        line_br, = axxy.plot(x, [b1_xyr] * NRUNS, label="batch r", c=ccolors[0], ls=cstyles[0])
        line_bcat, = axxy.plot(x, [b1_catcorr_n] * NRUNS, label="batch cat", c=ccolors[1], ls=cstyles[1])
        line_incr, = axxy.plot(x, inc1_xyr, label="inc r", c=ccolors[2], ls=cstyles[2])
        line_inccat, = axxy.plot(x, inc1_catcorr, label="inc cat", c=ccolors[3], ls=cstyles[3])
        line_incrgen = axxy.plot(x, incgen_xyr, label="incgen r", c=ccolors[2], ls=dstyles[3])
        line_inccatgen, = axxy.plot(x, incgen_catcorr, label="incgen cat", c=ccolors[3], ls=dstyles[3])
        # if eps == 0.0005:
        #     axxy.plot(x[0], inc1_xyr[0], marker='^', c='k', mfc='none', clip_on=False)
        #     axxy.plot(x[-1], inc1_xyr[-1], marker='v', c='k', mfc='none', clip_on=False)
        #     axxy.plot(x[0], inc1_catcorr[0], marker='^', c='k', mfc='none', clip_on=False)
        #     axxy.plot(x[-1], inc1_catcorr[-1], marker='v', c='k', mfc='none', clip_on=False)

    axy[0].legend((line_inccat, line_incr), ("identity", "x,y"), loc=5)

    # ad[0].legend((line_bd, line_incd, line_pred, line_latd, line_incmean, line_incvar), ("batch", "incr.", "SFAlo", "latents", "incmean", "incvar/100"), loc=5)
    # ad[0].legend((line_bd, line_incd, line_pred, line_latd), ("batch", "incr.", "SFAlo", "latents"), loc=5)
    ad[0].legend((line_incd, line_gend, line_bd, line_pred), ("replay", "more", "batch", "SFA$_{\\rm lo}$"), loc=5, bbox_to_anchor=(1,0.55))
    ylim_d = ad[0].get_ylim()
    ad[0].set_ylim([0, ylim_d[1]])

    # plt.show()


    # ====================================================================================================================================================================================================




    import sklearn.linear_model

    # PATH = PATH_PRE + "lroW_o1850t/"

    SFA1_FILE = PATH + "sfa1.p"
    SFA2_FILE = PATH + "inc_rep39.sfa"

    SFA2_FILE2 = PATH + "inc_rep0.sfa"

    sfa1 = semantic.load_SFA(SFA1_FILE)
    sfa2 = semantic.load_SFA(SFA2_FILE)
    sfa22 = semantic.load_SFA(SFA2_FILE2)

    # PARAMETERS = system_params.SysParamSet()
    #
    # with open(PATH + "sensory.p", 'rb') as f:
    #     sensory_system = pickle.load(f)
    # ran = np.arange(PARAMETERS.st1['number_of_snippets'])
    # with open(PATH + "st1.p", 'rb') as f:
    #     PARAMETERS.st1 = pickle.load(f)
    # seq1, cat1, lat1 = sensory_system.recall(numbers = ran, fetch_indices=False, **PARAMETERS.st1)
    #
    # PARAMETERS.st1['number_of_snippets'] = 50
    # PARAMETERS.st1['snippet_length'] = 50
    # sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
    # seq2, cat2, lat2 = sensys.generate(fetch_indices=False, **PARAMETERS.st1)

    # cat2 = np.array(cat2)
    # lat2 = np.array(lat2)
    #
    # cat1 = np.array(cat1)
    # lat1 = np.array(lat1)

    cat2 = ncat
    lat2 = nlat

    cat1 = tcat
    lat1 = tlat

    yy = semantic.exec_SFA(sfa1, seq1)
    if NORM:
        yy_w = streamlined.normalizer(yy, PARAMETERS.normalization)(yy)
    if WHITENER:
        yy_w = whitener(yy)

    # ==============================================================

    if NORM or WHITENER:
        zz = semantic.exec_SFA(sfa2, yy_w)
    else:
        zz = semantic.exec_SFA(sfa2, yy)

    if CORRNORM:
        zz_w = streamlined.normalizer(zz, PARAMETERS.normalization)(zz)

    target_matrix = np.append(lat1, cat1[:,None], axis=1)
    if CORRNORM:
        training_matrix = zz_w
    else:
        training_matrix = zz

    learner = sklearn.linear_model.LinearRegression()

    learner.fit(training_matrix[:,:NFEAT], target_matrix)

    yy2 = semantic.exec_SFA(sfa1, seq2)
    if NORM:
        yy2_w = streamlined.normalizer(yy2, PARAMETERS.normalization)(yy2)
    if WHITENER:
        yy2_w = whitener(yy2)
    if NORM or WHITENER:
        zz2 = semantic.exec_SFA(sfa2, yy2_w)
    else:
        zz2 = semantic.exec_SFA(sfa2, yy2)
    zz2_w = streamlined.normalizer(zz2, PARAMETERS.normalization)(zz2)
    if CORRNORM:
        prediction = learner.predict(zz2_w[:, :NFEAT])
    else:
        prediction = learner.predict(zz2[:,:NFEAT])
    # prediction = learner.predict(zz2_w)

    # f, ax = plt.subplots(1,3, sharex=True, squeeze=False, sharey=True)
    # f, ax = plt.subplots(2,2, sharex=True)
    ax = [axbot1, axbot2]
    ax[1][0].get_shared_y_axes().join(ax[1][0], ax[1][1])
    inds = np.random.randint(len(lat2[:,0]),size=1000)
    ax[1][0].scatter(lat2[:,0][inds], prediction[:,0][inds], s=2, c='k')
    slope00, intercept00, r_value00, _, _ = scipy.stats.linregress(lat2[:,0], prediction[:,0])
    xx = np.arange(-1,1.005,0.1)
    yyy = slope00*xx+intercept00
    ax[1][0].plot(xx,yyy, c=rcolor, ls=rstyle)
    ax[1][0].text(-1,1,"r={:.3f}".format(r_value00))

    # ax[1][0].set_ylabel("latent prediction")

    ax[1][1].scatter(lat2[:,1][inds], prediction[:,1][inds], s=2, c='k')
    slope01, intercept01, r_value01, _, _ = scipy.stats.linregress(lat2[:,1], prediction[:,1])
    xx = np.arange(-1,1.005,0.1)
    yyy = slope01*xx+intercept01
    ax[1][1].plot(xx,yyy, c=rcolor, ls=rstyle)
    ax[1][1].text(-1, 1, "r={:.3f}".format(r_value01))

    ax[1][2].scatter(cat2[inds], prediction[:,4][inds], s=2, c='k')
    slope02, intercept02, r_value02, _, _ = scipy.stats.linregress(cat2, prediction[:,4])
    xx = np.arange(0,1.005,0.1)
    yyy = slope02*xx+intercept02
    ax[1][2].plot(xx,yyy, c=rcolor, ls=rstyle)
    ax[1][2].text(0,1.07,"r={:.3f}".format(r_value02))
    ax[1][2].set_xticks([0,1])
    ax[1][2].set_yticks([0,1])
    # ax[1][2].set_ylim([-0.5, 1.5])

    #================================================================
    if NORM or WHITENER:
        zz = semantic.exec_SFA(sfa22, yy_w)
    else:
        zz = semantic.exec_SFA(sfa22, yy)
    if CORRNORM:
        zz_w = streamlined.normalizer(zz, PARAMETERS.normalization)(zz)

    target_matrix = np.append(lat1, cat1[:,None], axis=1)
    if CORRNORM:
        training_matrix = zz_w
    else:
        training_matrix = zz

    learner = sklearn.linear_model.LinearRegression()

    learner.fit(training_matrix[:,:NFEAT], target_matrix)

    if NORM or WHITENER:
        zz2 = semantic.exec_SFA(sfa22, yy2_w)
    else:
        zz2 = semantic.exec_SFA(sfa22, yy2)
    zz2_w = streamlined.normalizer(zz2, PARAMETERS.normalization)(zz2)
    if CORRNORM:
        prediction = learner.predict(zz2_w[:, :NFEAT])
    else:
        prediction = learner.predict(zz2[:,:NFEAT])
    # prediction = learner.predict(zz2_w)

    # f, ax = plt.subplots(1,3, sharex=True, squeeze=False, sharey=True)
    # f, ax = plt.subplots(2,2, sharex=True)
    ax = [axbot1, axbot2]
    ax[0][0].get_shared_y_axes().join(ax[0][0], ax[0][1])
    inds = np.random.randint(len(lat2[:,0]),size=1000)
    ax[0][0].scatter(lat2[:,0][inds], prediction[:,0][inds], s=2, c='k')
    slope00, intercept00, r_value00, _, _ = scipy.stats.linregress(lat2[:,0], prediction[:,0])
    xx = np.arange(-1,1.005,0.1)
    yyy = slope00*xx+intercept00
    ax[0][0].plot(xx,yyy, c=rcolor, ls=rstyle)
    ax[0][0].text(-1,1,"r={:.3f}".format(r_value00))
    ax[1][0].set_xlabel("x")
    # ax[0][0].set_ylabel("latent prediction")

    ax[0][1].scatter(lat2[:,1][inds], prediction[:,1][inds], s=2, c='k')
    slope01, intercept01, r_value01, _, _ = scipy.stats.linregress(lat2[:,1], prediction[:,1])
    xx = np.arange(-1,1.005,0.1)
    yyy = slope01*xx+intercept01
    ax[0][1].plot(xx,yyy, c=rcolor, ls=rstyle)
    ax[0][1].text(-1,1,"r={:.3f}".format(r_value01))
    ax[1][1].set_xlabel("y")

    ax[0][2].scatter(cat2[inds], prediction[:,4][inds], s=2, c='k')
    slope02, intercept02, r_value02, _, _ = scipy.stats.linregress(cat2, prediction[:,4])
    xx = np.arange(0,1.005,0.1)
    yyy = slope02*xx+intercept02
    ax[0][2].plot(xx,yyy, c=rcolor, ls=rstyle)
    ax[0][2].text(0,1.07,"r={:.3f}".format(r_value02))
    ax[1][2].set_xlabel("object identity")

    lims = ax[0][0].get_xlim()

    ax[1][0].plot(-2.05, 0, marker='v', c='k', mfc='none', ms=20, clip_on=False)
    ax[0][0].plot(-2.05, 0, marker='^', c='k', mfc='none', ms=20, clip_on=False)

    ax[0][0].set_xlim(lims)

    fig.text(0.07, 0.28, "variable prediction", rotation=90, ha='center', va='center')
    fig.text(0.51, 0.55, "number of training repetitions", ha='center', va='center')


    #tleft tright
    #|     |
    #v     v
    #A-----B-----  <-- top
    # ----- -----
    # ----- -----
    #C---D---E---  <-- mid
    # --- --- ---
    #F---G---H---  <-- bot
    # --- --- ---
    #^   ^   ^
    #|   |   |
    #bleft   bright
    #    bmid

    tleft = 0.1
    tright = 0.555
    bleft = tleft
    bmid = 0.38
    bright = 0.66
    top = 0.96
    mid = 0.485
    bot = 0.26

    col = '0.4'
    siz = 25

    fig.text(tleft, top, "A", color=col, fontsize=siz)
    fig.text(tright, top, "B", color=col, fontsize=siz)
    fig.text(bleft, mid, "C", color=col, fontsize=siz)
    fig.text(bmid, mid, "D", color=col, fontsize=siz)
    fig.text(bright, mid, "E", color=col, fontsize=siz)
    fig.text(bleft, bot, "F", color=col, fontsize=siz)
    fig.text(bmid, bot, "G", color=col, fontsize=siz)
    fig.text(bright, bot, "H", color=col, fontsize=siz)

    limstop = axtop[0].get_ylim()
    axtop[0].axhline(y=-0.48, xmin=0, xmax=2.3, clip_on=False, c='k')
    axtop[0].set_ylim(limstop)

    plt.subplots_adjust(top=0.95, wspace=0.3, hspace=0.5)

    corr_ylim = axtop[1].get_ylim()
    axtop[1].set_ylim([corr_ylim[0], 1])

    plt.show()
