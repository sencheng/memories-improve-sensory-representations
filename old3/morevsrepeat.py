from core import semantic, system_params, streamlined, tools
import numpy as np
import sklearn.linear_model
import scipy.stats
import sys

if len(sys.argv) > 1:
    SAVEPATH = sys.argv[1]
else:
    SAVEPATH = "morevsrepeat/"

PATH_PRE = "../results/"

LOADPATH = "reerrorN0a/"

SNLENS_all = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
SNLENS = [2,5,10,32,80,200,600]
INDS = np.where(np.in1d(SNLENS_all, SNLENS))[0]

NFEAT = 16

sfa2parms = [
        ('inc.linear', {
            'dim_in': 288,
            'dim_out': 16
        })
    ]

PARAMETERS = system_params.SysParamSet()

sfa1 = semantic.load_SFA(PATH_PRE + LOADPATH + "sfadef1train0.sfa")
sfa2M = semantic.build_module(sfa2parms, eps=0.0005)
sfa2R = semantic.build_module(sfa2parms, eps=0.0005)

dMlist, dRlist, rMlist, rRlist = [], [], [], []

for ri in INDS:
    print(ri)
    fdata = np.load(PATH_PRE + LOADPATH + "forming{}.npz".format(ri))
    fseq = fdata["forming_sequenceX"]
    flat = np.array(fdata["forming_latent"])
    fcat = np.array(fdata["forming_categories"])
    fran = fdata["forming_ranges"]

    tdata = np.load(PATH_PRE + LOADPATH + "testing0.npz")
    tseq = tdata["testing_sequenceX"]
    tlat = np.array(tdata["testing_latent"])
    tcat = np.array(tdata["testing_categories"])

    y = semantic.exec_SFA(sfa1, fseq)
    y = streamlined.normalizer(y, PARAMETERS.normalization)(y)

    ran1 = np.random.permutation(fran[:len(fran)//2]).flatten()
    ran2 = np.random.permutation(fran[:len(fran)//2]).flatten()

    yR = np.concatenate((y[ran1], y[ran2]))
    latR = np.concatenate((flat[ran1], flat[ran2]))
    catR = np.concatenate((fcat[ran1], fcat[ran2]))

    semantic.train_SFA(sfa2M, y)
    semantic.train_SFA(sfa2R, yR)

    fzM = semantic.exec_SFA(sfa2M, y)
    fzM = streamlined.normalizer(fzM, PARAMETERS.normalization)(fzM)
    fzR = semantic.exec_SFA(sfa2R, y)
    fzR = streamlined.normalizer(fzR, PARAMETERS.normalization)(fzR)

    target_matrixM = np.append(flat, fcat[:, None], axis=1)
    learnerM = sklearn.linear_model.LinearRegression()
    learnerM.fit(fzM[:, :NFEAT], target_matrixM)

    target_matrixR = np.append(latR, catR[:, None], axis=1)
    learnerR = sklearn.linear_model.LinearRegression()
    learnerR.fit(fzR[:, :NFEAT], target_matrixR)

    ty = semantic.exec_SFA(sfa1, tseq)
    ty = streamlined.normalizer(ty, PARAMETERS.normalization)(ty)

    zM = semantic.exec_SFA(sfa2M, ty)
    zM = streamlined.normalizer(zM, PARAMETERS.normalization)(zM)

    zR = semantic.exec_SFA(sfa2R, ty)
    zR = streamlined.normalizer(zR, PARAMETERS.normalization)(zR)

    dM = tools.delta_diff(zM)
    dR = tools.delta_diff(zR)

    predictionM = learnerM.predict(zM[:, :NFEAT])
    predictionR = learnerR.predict(zR[:, :NFEAT])
    _, _, r_valueX_M, _, _ = scipy.stats.linregress(tlat[:, 0], predictionM[:, 0])
    _, _, r_valueY_M, _, _ = scipy.stats.linregress(tlat[:, 1], predictionM[:, 1])
    _, _, r_valueCAT_M, _, _ = scipy.stats.linregress(tcat, predictionM[:, 4])
    _, _, r_valueX_R, _, _ = scipy.stats.linregress(tlat[:, 0], predictionR[:, 0])
    _, _, r_valueY_R, _, _ = scipy.stats.linregress(tlat[:, 1], predictionR[:, 1])
    _, _, r_valueCAT_R, _, _ = scipy.stats.linregress(tcat, predictionR[:, 4])

    rM = [r_valueX_M, r_valueY_M, r_valueCAT_M]
    rR = [r_valueX_R, r_valueY_R, r_valueCAT_R]

    dMlist.append(dM)
    dRlist.append(dR)
    rMlist.append(rM)
    rRlist.append(rR)

np.save(PATH_PRE + SAVEPATH + "dMlist.npy", dMlist)
np.save(PATH_PRE + SAVEPATH + "dRlist.npy", dRlist)
np.save(PATH_PRE + SAVEPATH + "rMlist.npy", rMlist)
np.save(PATH_PRE + SAVEPATH + "rRlist.npy", rRlist)
np.save(PATH_PRE + SAVEPATH + "snlens.npy", SNLENS)
