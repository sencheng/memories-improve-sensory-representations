from core import semantic, system_params, input_params, streamlined, sensory
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats
import sklearn.linear_model
import pickle

PATH = "/local/results/lro02_o1850t/"
SFA1 = "sfa1.p"
SFA2S = "inc1_eps1_0.sfa"
SFA2E = "inc1_eps1_39.sfa"

DIFL = [0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
LETTER = "L"
NOI = 0.1

BIN_SIZE = 0.2

N=1000

PARAMETERS = system_params.SysParamSet()

normparm_alt = dict(number_of_snippets=100, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=NOI)
normparm = dict(number_of_snippets=1, snippet_length=None, movement_type='sample', movement_params=dict(x_range=None, y_range=None, t_range=None, x_step=0.05, y_step=0.05, t_step=22.5),
            object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=NOI)
parm = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
            object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=NOI)

sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

sfa1 = semantic.load_SFA(PATH+SFA1)
sfa2S = semantic.load_SFA(PATH+SFA2S)
sfa2E = semantic.load_SFA(PATH+SFA2E)

normseq, normcat, normlat = sensys.generate(**normparm)
normcat = np.array(normcat)
normlat = np.array(normlat)
norm_target = np.append(normlat, normcat[:,None], axis=1)
norm_y = semantic.exec_SFA(sfa1, normseq)
normalizer1 = streamlined.normalizer(norm_y, PARAMETERS.normalization)
norm_yw = normalizer1(norm_y)
norm_zS = semantic.exec_SFA(sfa2S, norm_yw)
norm_zE = semantic.exec_SFA(sfa2E, norm_yw)
normalizerS = streamlined.normalizer(norm_zS, PARAMETERS.normalization)
normalizerE = streamlined.normalizer(norm_zE, PARAMETERS.normalization)
norm_zSw = normalizerS(norm_zS)
norm_zEw = normalizerS(norm_zE)

regS = sklearn.linear_model.LinearRegression()
regE = sklearn.linear_model.LinearRegression()
regS.fit(norm_zSw, norm_target)
regE.fit(norm_zEw, norm_target)

cnt = len(DIFL)
latL = []
outS = []
outE = []
storS = []
storE = []
rS, rE, cS, cE = [], [], [], []
for idif, dif in enumerate(DIFL):
    latL.append([])
    outS.append([])
    outE.append([])
    storS.append([])
    storE.append([])
    for i in range(N):
        while True:
            loc1 = np.random.rand(2)*2-1
            if not dif == 0:
                offs = np.random.rand(2)*2-1
                offs /= np.linalg.norm(offs) / dif
                loc2 = loc1+offs
            else:
                loc2 = loc1
            if min(loc2) > -1 and max(loc2) < 1:
                break
        parm["movement_params"]["latent"] = [[loc1[0], loc1[1], 1, 0]]
        parm["movement_params"]["ranges"] = iter([[0]])
        x1, _, _ = sensys.generate(**parm)
        parm["movement_params"]["latent"] = [[loc2[0], loc2[1], 1, 0]]
        parm["movement_params"]["ranges"] = iter([[0]])
        x2, _, _ = sensys.generate(**parm)
        
        y1 = semantic.exec_SFA(sfa1, x1)[0]
        y2 = semantic.exec_SFA(sfa1, x2)[0]
        y1w = normalizer1(y1)
        y2w = normalizer1(y2)
        z1S = semantic.exec_SFA(sfa2S, y1w)
        z2S = semantic.exec_SFA(sfa2S, y2w)
        z1Sw = normalizerS(z1S)
        z2Sw = normalizerS(z2S)
        z1E = semantic.exec_SFA(sfa2E, y1w)
        z2E = semantic.exec_SFA(sfa2E, y2w)
        z1Ew = normalizerE(z1E)
        z2Ew = normalizerE(z2E)
        storS[idif].append(np.linalg.norm(z2Sw-z1Sw))
        storE[idif].append(np.linalg.norm(z2Ew - z1Ew))
        latL[idif].extend([loc1, loc2])
        outS[idif].extend([z1Sw[0], z2Sw[0]])
        outE[idif].extend([z1Ew[0], z2Ew[0]])

    lats = np.array(latL[idif])
    predictionS = regS.predict(outS[idif])
    predictionE = regE.predict(outE[idif])
    _, _, r_valueXS, _, _ = scipy.stats.linregress(lats[:, 0], predictionS[:, 0])
    _, _, r_valueYS, _, _ = scipy.stats.linregress(lats[:, 1], predictionS[:, 1])
    _, _, r_valueXE, _, _ = scipy.stats.linregress(lats[:, 0], predictionE[:, 0])
    _, _, r_valueYE, _, _ = scipy.stats.linregress(lats[:, 1], predictionE[:, 1])
    rS.append(np.mean((r_valueXS, r_valueYS)))
    rE.append(np.mean((r_valueXE, r_valueYE)))
    lat_trans = np.transpose(lats)
    feat_transS = np.transpose(outS[idif])
    feat_transE = np.transpose(outE[idif])
    corS = np.corrcoef(np.append(feat_transS,lat_trans, axis=0))[16:,:-2]
    corE = np.corrcoef(np.append(feat_transE, lat_trans, axis=0))[16:,:-2]
    cS.append(np.mean((np.max(np.abs(corS[0, :])), np.max(np.abs(corS[1, :])))))
    cE.append(np.mean((np.max(np.abs(corE[0, :])), np.max(np.abs(corE[1, :])))))

cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))
fh, axh = plt.subplots(rows, cols, sharex=True, squeeze=False)
fh.suptitle("input noise = {}".format(NOI))

modeS, modeE, widthS, widthE, meanS, meanE, stdS, stdE, medianS, medianE, skewS, skewE, kurtS, kurtE = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
for idif, dif in enumerate(DIFL):
    valS = storS[idif]
    valE = storE[idif]
    bnsS = int((max(valS) - min(valS))/BIN_SIZE)
    if bnsS == 0:
        bnsS = 1
    bnsE = int((max(valE) - min(valE))/BIN_SIZE)
    if bnsE == 0:
        bnsE = 1
    hiS, binsS, _ = axh[idif//cols, idif%cols].hist(valS, alpha=0.4, label="simple", bins=bnsS, color='b')
    hiE, binsE, _ = axh[idif//cols, idif%cols].hist(valE, alpha=0.4, label="episodic", bins=bnsE, color='r')
    bins = binsS if len(binsE) < len(binsS) else binsE
    xmax = max(max(np.where(hiS > 4)[0]), max(np.where(hiE > 4)[0]))
    axh[idif // cols, idif % cols].set_xlim((0, bins[xmax + 1]))
    axh[idif // cols, idif % cols].set_title("step={}".format(dif))

    temp = np.where(hiS > 4)[0]
    widthS.append(max(temp) - min(temp))
    temp = np.where(hiE > 4)[0]
    widthE.append(max(temp) - min(temp))
    modeS.append(max(0, BIN_SIZE+binsS[int(np.median(np.where(hiS == max(hiS))[0]))]))
    modeE.append(max(0, BIN_SIZE+binsE[int(np.median(np.where(hiE == max(hiE))[0]))]))
    meanS.append(np.mean(valS))
    meanE.append(np.mean(valE))
    stdS.append(np.std(valS))
    stdE.append(np.std(valE))
    medianS.append(np.median(valS))
    medianE.append(np.median(valE))
    skewS.append(scipy.stats.skew(valS))
    skewE.append(scipy.stats.skew(valE))
    kurtS.append(scipy.stats.kurtosis(valS))
    kurtE.append(scipy.stats.kurtosis(valE))

plt.legend()

ff, aax = plt.subplots(2,2)
ff.suptitle("input noise = {}".format(NOI))
lk1, = aax[0,0].plot(DIFL, modeS, 'k--')
lk, = aax[0,0].plot(DIFL, modeE, 'k-')
aax[0,0].plot(DIFL, meanS, 'b--')
lb, = aax[0,0].plot(DIFL, meanE, 'b-')
aax[0,0].plot(DIFL, medianS, 'c--')
lc, = aax[0,0].plot(DIFL, medianE, 'c-')
aax[1,0].plot(DIFL, widthS, 'g--')
lg, = aax[1,0].plot(DIFL, widthE, 'g-')
aax[0,1].plot(DIFL, skewS, 'm--')
lm, = aax[0,1].plot(DIFL, skewE, 'm-')
aax[0,1].plot(DIFL, kurtS, 'r--')
lr, = aax[0,1].plot(DIFL, kurtE, 'r-')
aax[1,1].plot(DIFL, rS, 'y--')
ly, = aax[1,1].plot(DIFL, rE, 'y-')
aax[1,1].plot(DIFL, cS, color='gold', linestyle='--')
lgold, = aax[1,1].plot(DIFL, cE, color='gold', linestyle='-')
if DIFL[0] == 0:
    tix = DIFL[1:]
else:
    tix = DIFL
for iax in range(2):
    for jax in range(2):
        aax[iax, jax].set_xscale('log')
        aax[iax, jax].set_xticks(tix)
        aax[iax, jax].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ff.legend((lk1, lk), ('simple', 'episodic'), 1)
ff.legend((lk, lb, lc, lg, lm, lr, ly, lgold), ('mode', 'mean', 'median', 'width', 'skewness', 'kurtosis', 'r² of x/y prediction', 'corr coef of x/y'), 2)

plt.show()