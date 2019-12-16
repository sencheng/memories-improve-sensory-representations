import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib

xlbls = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]

try:
    d_s = np.load("/local/results/posterrun_s.npy")
    d_e = np.load("/local/results/posterrun_e.npy")
except:
    d_s, d_e = [], []
    for i in range(69):
        d_s.append([])
        d_e.append([])
        dirname = "/local/results/posterrun{}/".format(i+1)
        for ri in range(31):
            fname = "res{}.p".format(ri)
            with open(dirname + fname, 'rb') as f:
                res = pickle.load(f)
            d_s[i].append(np.mean(np.sort(res.d_values['testingZ_S'])[:4]))
            d_e[i].append(np.mean(np.sort(res.d_values['testingZ_E'])[:4]))

    np.save("/local/results/posterrun_s.npy", d_s)
    np.save("/local/results/posterrun_e.npy", d_e)

mean_s = np.mean(d_s,axis=0)
mean_e = np.mean(d_e,axis=0)
std_s = np.std(d_s,axis=0)
std_e = np.std(d_e,axis=0)

f, ax = plt.subplots()
ax.plot(xlbls, mean_s, label="simple", c='b', ls='--')
ax.plot(xlbls, mean_e, label="episodic", c='b', ls='-')
ax.fill_between(xlbls, mean_s-std_s, mean_s+std_s, facecolor='b', alpha=0.2)
ax.fill_between(xlbls, mean_e-std_e, mean_e+std_e, facecolor='b', alpha=0.2)
ax.set_xscale('log')
ax.set_xticks([2, 10, 100, 600])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Length of training sequences")
ax.set_ylabel("Delta value of SFA output")
ax.legend()
f.subplots_adjust(bottom=0.15)
plt.show()
