import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib
from core import streamlined, tools, sensory, system_params, semantic
import scipy.stats
import sklearn.linear_model

N=16
LOAD = True

PATHTYPE = "reorder4"
xlbls = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]

letters = "abcdefghijklmnopqrstuvqxyz"[:N]
PATHA = "/local/results/" + PATHTYPE + "{}/".format(letters[0])

if not LOAD:

    PARAMETERS = system_params.SysParamSet()

    sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

    nfram = 2500
    nsnip = list(2500//np.array(xlbls))

    dS = []
    dE = []

    for si in range(31):
        print("===============  {}  ===============".format(si))

        st4 = dict()
        st4['movement_type'] = 'gaussian_walk'
        st4['movement_params'] = dict(dx=0.05, dt=0.05, step=5)
        st4['number_of_snippets'] = nsnip[si]
        st4['snippet_length'] = xlbls[si]

        testing_sequence, _, _ = sensys.generate(fetch_indices=False, **st4)
        dS.append([])
        dE.append([])

        for li, let in enumerate(letters):
            PATH = "/local/results/" + PATHTYPE + "{}/".format(let)
            print(PATH + "...")
            dS[si].append([])
            dE[si].append([])

            for ri in range(31):
                print(ri)
                fname = "res{}.p".format(ri)
                with open(PATH + fname, 'rb') as f:
                    res = pickle.load(f)

                sfa1 = semantic.load_SFA(PATH + res.data_description + "train0.sfa")
                yy2 = semantic.exec_SFA(sfa1, testing_sequence)
                yy2_w = streamlined.normalizer(yy2, res.params.normalization)(yy2)
                zz2S = semantic.exec_SFA(res.sfa2S, yy2_w)
                zz2S_w = streamlined.normalizer(zz2S, res.params.normalization)(zz2S)
                zz2E = semantic.exec_SFA(res.sfa2E, yy2_w)
                zz2E_w = streamlined.normalizer(zz2E, res.params.normalization)(zz2E)

                dS[si][li].append(np.mean(np.sort(tools.delta_diff(zz2S_w))[:3]))
                dE[si][li].append(np.mean(np.sort(tools.delta_diff(zz2E_w))[:3]))

    np.save(PATHA+"dS.npy", dS)
    np.save(PATHA+"dE.npy", dE)

else:
    dS = np.load(PATHA + "dS.npy")
    dE = np.load(PATHA + "dE.npy")

f, ax = plt.subplots()
for pi, p in enumerate([1, 8, 20, 29]):  # snlen 3, 10, 50, 300
    dmean_s = np.mean(dS[p], axis=0)
    dmean_e = np.mean(dE[p], axis=0)
    ax.plot(xlbls, dmean_s, ['y:', 'y-', 'g:', 'g-'][pi], label="S {}".format([3, 10, 50, 300][pi]))
    ax.plot(xlbls, dmean_e, ['m:', 'm-', 'r:', 'r-'][pi], label="E {}".format([3, 10, 50, 300][pi]))
ax.set_xscale('log')
ax.set_xticks([2, 10, 100, 600])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend()
plt.show()