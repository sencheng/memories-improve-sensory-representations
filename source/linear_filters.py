"""
(Figure 9)

Visualizes SFA responses as matrices.

"""

from core import semantic, streamlined, sensory, input_params, system_params, tools
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import gridspec as grd
import matplotlib
import scipy.stats
import sklearn.linear_model

if __name__ == "__main__":

    PATH_PRE = "/local/results/"
    # PATH_PRE = "/media/goerlrwh/Extreme Festplatte/results/"
    REORDER_PATH = "reorder_o18a/"     # not used atm
    REERROR_PATH_S = "reerrorN4a/"     # not used atm
    REERROR_PATH_E = "reerrorN0a/"     # not used atm
    REPLAY_PATH = "replay_o18/"

    NFEAT = 3
    DCNT = 3

    DATA_TYPE = 0       # 0: replay, 1: reorder, 2: reerror

    # matplotlib.rcParams['lines.markersize'] = 22
    # matplotlib.rcParams['lines.markeredgewidth'] = 2
    font = {'family' : 'Sans',
            'size'   : 18}

    matplotlib.rc('font', **font)

    with open(PATH_PRE+REORDER_PATH+"res0.p", 'rb') as f:
        reorder_res = pickle.load(f)
    with open(PATH_PRE+REERROR_PATH_E+"res0.p", 'rb') as f:
        reerror_res_E = pickle.load(f)
    with open(PATH_PRE+REERROR_PATH_S+"res0.p", 'rb') as f:
        reerror_res_S = pickle.load(f)

    if DATA_TYPE == 0:
        sfa1S = semantic.load_SFA(PATH_PRE+REPLAY_PATH+"sfa1.p")
        sfa1E = semantic.load_SFA(PATH_PRE + REPLAY_PATH + "sfa1.p")
        sfa2S = semantic.load_SFA(PATH_PRE+REPLAY_PATH+"inc1_eps1_0.sfa")
        sfa2E = semantic.load_SFA(PATH_PRE+REPLAY_PATH+"inc1_eps1_39.sfa")
    elif DATA_TYPE == 1:
        sfa1S = semantic.load_SFA(PATH_PRE + REORDER_PATH + "sfadef1train0.sfa")
        sfa1E = semantic.load_SFA(PATH_PRE + REORDER_PATH + "sfadef1train0.sfa")
        sfa2S = reorder_res.sfa2S
        sfa2E = reorder_res.sfa2E
    else:
        sfa1S = semantic.load_SFA(PATH_PRE+REERROR_PATH_S+"sfadef1train0.sfa")
        sfa1E = semantic.load_SFA(PATH_PRE+REERROR_PATH_E+"sfadef1train0.sfa")
        sfa2S = reerror_res_S.sfa2E
        sfa2E = reerror_res_E.sfa2E

    PARAMETERS = system_params.SysParamSet()
    with open(PATH_PRE+REPLAY_PATH+ "st1.p", 'rb') as f:
        PARAMETERS.st1 = pickle.load(f)
    with open(PATH_PRE+REPLAY_PATH+ "whitener.p", 'rb') as f:
        whitener = pickle.load(f)

    PARAMETERS.st1['number_of_snippets'] = 50

    sensory_system = pickle.load(open(PATH_PRE+REPLAY_PATH + "sensory.p", 'rb'))
    ran = np.arange(PARAMETERS.st1['number_of_snippets'])

    training_sequence, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    new_sequence1, new_categories, new_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    tcat = np.array(training_categories)
    tlat = np.array(training_latent)
    ncat = np.array(new_categories)
    nlat = np.array(new_latent)

    training_sequence = semantic.exec_SFA(sfa1S,training_sequence)
    training_sequence = whitener(training_sequence)
    target_matrix = np.append(tlat, tcat[:, None], axis=1)

    new_sequence1 = semantic.exec_SFA(sfa1S,new_sequence1)
    new_sequence1 = whitener(new_sequence1)

    inc1_y_n = semantic.exec_SFA(sfa2E, new_sequence1)
    inc1_w_n = streamlined.normalizer(inc1_y_n, PARAMETERS.normalization)(inc1_y_n)

    training_matrix = semantic.exec_SFA(sfa2E, training_sequence)
    learner = sklearn.linear_model.LinearRegression()
    learner.fit(training_matrix[:,:NFEAT], target_matrix)
    prediction = learner.predict(inc1_y_n[:,:NFEAT])
    # prediction = learner.predict(b1_w_n)
    _, _, r_valueX, _, _ = scipy.stats.linregress(nlat[:, 0], prediction[:, 0])
    _, _, r_valueY, _, _ = scipy.stats.linregress(nlat[:, 1], prediction[:, 1])
    _, _, r_valueC, _, _ = scipy.stats.linregress(ncat, prediction[:, 4])

    a = np.arange(-1 ,1+2/30, 2/30)
    d = np.diff(a)
    ran = a[:-1]+d/2    # 0 1 2 3
    y_lat = np.tile(ran, 30)   # 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
    x_lat = np.tile(ran, (30,1)).T.flatten()  # 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
    cos_lat = np.ones(900)-0.2
    sin_lat = np.zeros(900)+0.6
    lats = np.concatenate([x_lat[:,None], y_lat[:,None], cos_lat[:,None], sin_lat[:,None]], axis=1)
    parm = dict()
    parm["number_of_snippets"] = 1
    parm["snippet_length"] = None
    parm["movement_type"] = "copy_traj"
    parm["object_code"] = input_params.make_object_code('T')
    parm["sequence"] = [0]
    parm["movement_params"] = dict(latent=lats, ranges=iter([np.arange(900)]))
    filtseqT, _, _ = sensory_system.generate(**parm)
    # tools.preview_input(filtseq)

    parm["object_code"] = input_params.make_object_code('L')
    parm["movement_params"] = dict(latent=lats, ranges=iter([np.arange(900)]))
    filtseqL, _, _ = sensory_system.generate(**parm)

    pointseq = np.eye(900)
    nullseq = np.zeros((900,900))

    # input = []
    # for i in range(30):
    #     for j in range(30):
    #         zero = np.zeros((30, 30))
    #         zero[i,j] = 1
    #         input.append(zero)
    # input = np.array(input)
    # input0 = input[:,:18,:18]
    # input1 = input[:,6:24,:18]
    # input2 = input[:,12:,:18]
    # input3 = input[:,:18,6:24]
    # input4 = input[:,6:24,6:24]
    # input5 = input[:,12:,6:24]
    # input6 = input[:,:18,12:]
    # input7 = input[:,6:24,12:]
    # input8 = input[:,12:,12:]

    # def sfa1_filter(sfa1_matrix):
    #     seq0 = np.dot(input0.reshape((input0.shape[0], input0.shape[1] * input0.shape[2])), sfa1_matrix)
    #     seq1 = np.dot(input1.reshape((input1.shape[0], input1.shape[1] * input1.shape[2])), sfa1_matrix)
    #     seq2 = np.dot(input2.reshape((input2.shape[0], input2.shape[1] * input2.shape[2])), sfa1_matrix)
    #     seq3 = np.dot(input3.reshape((input3.shape[0], input3.shape[1] * input3.shape[2])), sfa1_matrix)
    #     seq4 = np.dot(input4.reshape((input4.shape[0], input4.shape[1] * input4.shape[2])), sfa1_matrix)
    #     seq5 = np.dot(input5.reshape((input5.shape[0], input5.shape[1] * input5.shape[2])), sfa1_matrix)
    #     seq6 = np.dot(input6.reshape((input6.shape[0], input6.shape[1] * input6.shape[2])), sfa1_matrix)
    #     seq7 = np.dot(input7.reshape((input7.shape[0], input7.shape[1] * input7.shape[2])), sfa1_matrix)
    #     seq8 = np.dot(input8.reshape((input8.shape[0], input8.shape[1] * input8.shape[2])), sfa1_matrix)
    #     ret = np.concatenate((seq0, seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8), axis=1)
    #     return ret

    # Areo = reorder_sfa1.flow[0]._flow.flow[0].sf
    # BSreo = reorder_sfa2S.layers[0]._node.v.T
    # BEreo = reorder_sfa2E.layers[0]._node.v.T
    # CSreo = np.dot(Areo,BSreo)
    # CEreo = np.dot(Areo,BEreo)

    # ree1_seqS = sfa1_filter(reerror_sfa1S.flow[1].node.flow.flow[0].sf)
    # ree1_seqE = sfa1_filter(reerror_sfa1E.flow[1].node.flow.flow[0].sf)
    # reerrorS = np.dot(ree1_seqS, reerror_sfa2S.layers[0]._node.v.T)
    # reerrorE = np.dot(ree1_seqE, reerror_sfa2E.layers[0]._node.v.T)


    # ==============================================
    # how to get matrixes from sfa and whitener

    # Arep = replay_sfa1.flow[0]._flow.flow[0].sf
    # Wrep = whitener.wn.v.dot(whitener.wn.get_eigenvectors().T)
    # BSrep = replay_sfa2S.layers[0]._node.v.T
    # BErep = replay_sfa2E.layers[0]._node.v.T
    # AWrep = np.dot(Arep, Wrep)
    # CSrep = np.dot(AWrep,BSrep)
    # CErep = np.dot(AWrep,BErep)
    #
    # CEpred = learner.predict(CErep[:,:NFEAT])
    #
    # regr_matrix = learner.coef_
    # regr_add = learner.intercept_
    # x_mat = np.zeros(900)
    # y_mat = np.zeros(900)
    # i_mat = np.zeros(900)
    # for ft in range(NFEAT):
    #     x_mat += CErep[:,ft]*regr_matrix[0,ft]
    #     y_mat += CErep[:, ft] * regr_matrix[1, ft]
    #     i_mat += CErep[:, ft] * regr_matrix[4, ft]
    # =============================================

    pseqY = semantic.exec_SFA(sfa1S, pointseq)
    pseqYw = whitener(pseqY)
    pseqZ = semantic.exec_SFA(sfa2E, pseqYw)
    pred_p = learner.predict(pseqZ[:,:NFEAT])

    nseqY = semantic.exec_SFA(sfa1S, nullseq)
    nseqYw = whitener(nseqY)
    nseqZ = semantic.exec_SFA(sfa2E, nseqYw)
    pred_n = learner.predict(nseqZ[:,:NFEAT])

    fseqTY = semantic.exec_SFA(sfa1S, filtseqT)
    fseqTYw = whitener(fseqTY)
    fseqTZ = semantic.exec_SFA(sfa2E, fseqTYw)
    ZT = fseqTZ

    F1T = np.zeros(900)
    F2T = np.zeros(900)
    F3T = np.zeros(900)
    P1T = np.zeros(900)
    P2T = np.zeros(900)
    P3T = np.zeros(900)
    pred = learner.predict(ZT[:,:NFEAT])
    for ei in range(len(filtseqT)):
        predel = pred[ei]
        P1T[ei] += predel[0]
        P2T[ei] += predel[1]
        P3T[ei] += predel[4]

    fseqLY = semantic.exec_SFA(sfa1S, filtseqL)
    fseqLYw = whitener(fseqLY)
    fseqLZ = semantic.exec_SFA(sfa2E, fseqLYw)
    ZL = fseqLZ

    F1L = np.zeros(900)
    F2L = np.zeros(900)
    F3L = np.zeros(900)
    P1L = np.zeros(900)
    P2L = np.zeros(900)
    P3L = np.zeros(900)
    pred = learner.predict(ZL[:,:NFEAT])
    for ei in range(len(filtseqL)):
        predel = pred[ei]
        P1L[ei] += predel[0]
        P2L[ei] += predel[1]
        P3L[ei] += predel[4]

    # ML1 = np.dot(filtseqL,x_mat) + regr_add[0]
    # ML2 = np.dot(filtseqL,y_mat) + regr_add[1]
    # ML3 = np.dot(filtseqL,i_mat) + regr_add[4]
    #
    # MT1 = np.dot(filtseqT,x_mat)+ regr_add[0]
    # MT2 = np.dot(filtseqT,y_mat)+ regr_add[1]
    # MT3 = np.dot(filtseqT,i_mat)+ regr_add[4]

    TEXTRIGHT = -5
    TEXTSIZE = 40

    LINEY = 0.63
    LINEY1 = 0.78
    LINEY2 = 0.36
    LINELEFT = 0.1
    LINERIGHT = 0.8

    # fig = plt.figure()
    # outer = grd.GridSpec(2, 1, wspace=0.2, hspace=0.2)
    # axes = []
    # for i in range(2):
    #     inner = grd.GridSpecFromSubplotSpec(3, NFEAT, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    #     for c in range(3):
    #         axes.append([])
    #         for r in range(3):
    #             axes[-1].append(plt.Subplot(fig, inner[c,r]))
    # axes = np.array(axes)

    fig, axes = plt.subplots(nrows=3, ncols=NFEAT)
    # plt.suptitle("features")
    maxv = np.amax((np.abs(np.amax(pseqZ[:,:NFEAT]-nseqZ[:,:NFEAT])), np.abs(np.amin(pseqZ[:,:NFEAT]-nseqZ[:,:NFEAT]))))
    for k in range(NFEAT):
        ax = axes[0,k]
        cset = ax.imshow(np.flip(np.reshape(pseqZ[:,k]- nseqZ[:,k], (30, 30)).T,0), interpolation='none', cmap='Greys', vmin = -maxv, vmax=maxv)
        # ax.set_title("feature {:d}\n[{:.2f}, {:.2f}]".format(k, np.amin(pseqZ[:, k]- nseqZ[:,k]), np.amax(pseqZ[:, k]- nseqZ[:,k])))
        ax.set_title("feature {:d}".format(k+1))
        ax.set_axis_off()
        if k == 0:
            ax.text(TEXTRIGHT, 15, ".", horizontalalignment='right', verticalalignment='center', fontsize=TEXTSIZE)
    # fig.colorbar(cset, ax=axes[0,NFEAT-1])
    maxbar = np.floor(100*maxv)/100
    cbar = fig.colorbar(cset, ax=axes[0,:].ravel().tolist(), ticks=[-maxbar,0,maxbar])

    axf = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
    axf.set_axis_off()
    axf.axhline(LINEY,LINELEFT,LINERIGHT, color='k', clip_on=False)

    maxvT = np.amax((np.abs(np.amax(ZT[:,:NFEAT]-nseqZ[:,:NFEAT])), np.abs(np.amin(ZT[:,:NFEAT]-nseqZ[:,:NFEAT]))))
    maxvL = np.amax((np.abs(np.amax(ZL[:,:NFEAT]-nseqZ[:,:NFEAT])), np.abs(np.amin(ZL[:,:NFEAT]-nseqZ[:,:NFEAT]))))
    maxv = max(maxvT, maxvL)
    for k in range(NFEAT):
        ax = axes[1, k]
        cset = ax.imshow(np.flip(np.reshape(ZT[:,k]- nseqZ[:,k], (30, 30)).T,0), interpolation='none', cmap='Greys', vmin = -maxv, vmax=maxv)
        # ax.set_title("[{:.2f}, {:.2f}]".format(np.amin(ZT[:,k]), np.amax(ZT[:,k])))
        ax.set_axis_off()
        if k == 0:
            ax.text(TEXTRIGHT, 15, "T", horizontalalignment='right', verticalalignment='center', fontsize=TEXTSIZE)
    # fig.colorbar(cset, ax=axes[1, NFEAT - 1])

    for k in range(NFEAT):
        ax = axes[2, k]
        cset = ax.imshow(np.flip(np.reshape(ZL[:,k]- nseqZ[:,k], (30, 30)).T,0), interpolation='none', cmap='Greys', vmin = -maxv, vmax=maxv)
        # ax.set_title("[{:.2f}, {:.2f}]".format(np.amin(ZL[:,k]), np.amax(ZL[:,k])))
        ax.set_axis_off()
        if k == 0:
            ax.text(TEXTRIGHT, 15, "L", horizontalalignment='right', verticalalignment='center', fontsize=TEXTSIZE)
    # fig.colorbar(cset, ax=axes[2,NFEAT-1])
    maxbar = np.floor(100*maxv)/100
    fig.colorbar(cset, ax=axes[1:3, :].ravel().tolist(), shrink=0.5, ticks=[-maxbar,0,maxbar])


    fig2, axes2 = plt.subplots(nrows=3, ncols=3)
    # plt.suptitle("regressor")
    val_list = [pred_p[:,0]-pred_n[:,0], pred_p[:,1]-pred_n[:,1], pred_p[:,4]-pred_n[:,4]]
    maxv = np.amax((np.abs(np.amax(val_list)), np.abs(np.amin(val_list))))
    for k in range(3):
        ax = axes2[0, k]
        cset = ax.imshow(np.flip(np.reshape(val_list[k], (30, 30)).T,0), interpolation='none', cmap='Greys', vmin=-maxv, vmax=maxv)
        # ax.set_title(["x", "y", "identity"][k] + " prediction\n[{:.2f}, {:.2f}]".format(np.amin(val_list[k]), np.amax(val_list[k])))
        ax.set_title(["x", "y", "identity\n"][k] + " prediction")
        ax.set_axis_off()
        if k == 0:
            ax.text(TEXTRIGHT, 15, ".", horizontalalignment='right', verticalalignment='center', fontsize=TEXTSIZE)
    # fig.colorbar(cset, ax=axes2[0,2])
    maxbar = np.floor(100*maxv)/100
    fig.colorbar(cset, ax=axes2[0,:].ravel().tolist(), ticks=[-maxbar,0,maxbar])

    axf2 = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
    axf2.set_axis_off()
    axf2.axhline(LINEY,LINELEFT,LINERIGHT, color='k', clip_on=False)

    val_listT = [P1T, P2T, P3T]
    val_listL = [P1L, P2L, P3L]
    maxvT = np.amax((np.abs(np.amax(val_listT)), np.abs(np.amin(val_listT))))
    maxvL = np.amax((np.abs(np.amax(val_listL)), np.abs(np.amin(val_listL))))
    maxv = max(maxvT, maxvL)
    for k in range(3):
        ax = axes2[1, k]
        cset = ax.imshow(np.flip(np.reshape(val_listT[k], (30, 30)).T, 0), interpolation='none', cmap='Greys', vmin = -maxv, vmax=maxv)
        # ax.set_title("[{:.2f}, {:.2f}]".format(np.amin(val_listT[k]), np.amax(val_listT[k])))
        ax.set_axis_off()
        if k == 0:
            ax.text(TEXTRIGHT, 15, "T", horizontalalignment='right', verticalalignment='center', fontsize=TEXTSIZE)
    # fig.colorbar(cset, ax=axes2[1,2])

    # maxv = np.amax((np.abs(np.amax(val_listL)), np.abs(np.amin(val_listL))))
    for k in range(3):
        ax = axes2[2, k]
        cset = ax.imshow(np.flip(np.reshape(val_listL[k], (30, 30)).T,0), interpolation='none', cmap='Greys', vmin=-maxv, vmax=maxv)
        # ax.set_title("[{:.2f}, {:.2f}]".format(np.amin(val_listL[k]), np.amax(val_listL[k])))
        ax.set_axis_off()
        if k == 0:
            ax.text(TEXTRIGHT, 15, "L", horizontalalignment='right', verticalalignment='center', fontsize=TEXTSIZE)
    # fig.colorbar(cset, ax=axes2[2,2])
    maxbar = np.floor(100*maxv)/100
    fig.colorbar(cset, ax=axes2[1:3, :].ravel().tolist(), shrink=0.5, ticks=[-maxbar,0,maxbar])

    plt.show()
