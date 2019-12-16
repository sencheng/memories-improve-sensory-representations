import numpy as np
from matplotlib import pyplot as plt
import matplotlib

N = 4
letters = "abcdefghijklmnopqrstuvqxyz"[:N]

NFEAT = 16

PATH_PRE = "/local/results/"

PATH = "morevsrepeat"

# SNLENS = np.load(PATH_PRE + PATH + letters[0] + "/snlens.npy")
SNLENS = [2,5,10,80,200,600]

dM0, dR0, rXYM0, rCM0, rXYR0, rCR0 = [], [], [], [], [], []

for i, let in enumerate(letters):
    dM0.append(np.mean(np.load(PATH_PRE + PATH + let + "/dMlist.npy")[:, :NFEAT], axis=1))
    dR0.append(np.mean(np.load(PATH_PRE + PATH + let + "/dRlist.npy")[:, :NFEAT], axis=1))
    rMlist = np.load(PATH_PRE + PATH + let + "/rMlist.npy")
    rRlist = np.load(PATH_PRE + PATH + let + "/rRlist.npy")
    rXYM0.append(np.mean((np.abs(rMlist[:, 0]), np.abs(rMlist[:, 1])), axis=0))
    rXYR0.append(np.mean((np.abs(rRlist[:, 0]), np.abs(rRlist[:, 1])), axis=0))
    rCM0.append(np.abs(rMlist[:, 2]))
    rCR0.append(np.abs(rRlist[:, 2]))

dmeanM = np.mean(dM0, axis=0)
dmeanR = np.mean(dR0, axis=0)

xymeanM = np.mean(rXYM0, axis=0)
xymeanR = np.mean(rXYR0, axis=0)

cmeanM = np.mean(rCM0, axis=0)
cmeanR = np.mean(rCR0, axis=0)

f, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(SNLENS, dmeanM, label="more", c='g', ls='-')
ax[0].plot(SNLENS, dmeanR, label="repeat", c='r', ls='-')

ax[0].set_xscale('log')
ax[0].set_xticks([2, 10, 100, 600])
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax[0].set_ylabel("Delta value of SFA output")

lc1r, = ax[1].plot(SNLENS, cmeanM, c='g', ls='--')
lc2r, = ax[1].plot(SNLENS, cmeanR, c='r', ls='--')

l1, = ax[1].plot(SNLENS, xymeanM, label="more", c='g', ls='-')
l2, = ax[1].plot(SNLENS, xymeanR, label="repeat", c='r', ls='-')

ax[1].set_xscale('log')
ax[1].set_xticks([2, 10, 100, 600])
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_ylabel("r of variable prediction")

ax[1].legend((l1, lc1r), ("x/y", "category"), loc=4)
# f.subplots_adjust(bottom=0.15)

ax[1].set_xlabel("Length of training sequences")

plt.figlegend((l1,l2), ("more", "repeat"), 1)

plt.show()
