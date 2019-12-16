import numpy as np
from matplotlib import pyplot as plt
import matplotlib

PATH = "../results/flauschRand288xy/"

DRAW = False
# SNLEN_LIST = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
# SNLEN_LIST = [2,3,5,8,12,20,40,100,200,600]
SNLEN_LIST = [2,5,20,100]
indlist = [0, 3, 13, 25]
# RETNOI_LIST = [0,1,2,3,4]
RETNOI_LIST = [0,1,2,3,4]

COLORS = ['k', 'b', 'c', 'g', 'r', 'm']

delta_seq = np.load(PATH + "delta_seq.npy")

delta_retmem = np.load(PATH + "delta_retmem.npy")
error_distancesmem = np.load(PATH + "error_distancesmem.npy")
error_typesmem = np.load(PATH + "error_typesmem.npy")

f, ax = plt.subplots()
ax.plot(SNLEN_LIST, np.mean(delta_seq,axis=1), 'k-', label="seq")
for ni, nnn in enumerate(RETNOI_LIST):
    ax.plot(SNLEN_LIST, np.mean(delta_retmem[:, ni],axis=1), color=COLORS[ni], label="retmem {}".format(nnn))
ax.legend()
ax.set_xscale('log')
f.suptitle(PATH.split("/")[-2])

err_count, err_mean, err_sum, err_count1, err_frac, err_mean1, err_meanm1, err_sum1, err_summ1, err_made1 = [], [], [], [], [], [], [], [], [], []
for ni, noi in enumerate(RETNOI_LIST):
    err_count.append([])
    err_mean.append([])
    err_sum.append([])
    err_count1.append([])
    err_frac.append([])
    err_mean1.append([])
    err_meanm1.append([])
    err_sum1.append([])
    err_summ1.append([])
    err_made1.append([])
    for sni, snlen in enumerate(SNLEN_LIST):
        print(snlen)
        nnz = np.nonzero(error_distancesmem[sni,ni])[0]
        err_count[ni].append(len(nnz))
        err_sum[ni].append(np.sum(error_distancesmem[sni,ni]))
        err_mean[ni].append(np.mean(error_distancesmem[sni,ni][nnz]))
        where1 = np.where(error_typesmem[sni,ni] == 1)[0]
        wherem1 = np.where(error_typesmem[sni,ni] == -1)[0]
        where0 = np.where(error_typesmem[sni,ni] == 0)[0]
        err_count1[ni].append(len(where1))
        err_frac[ni].append(len(wherem1) / len(nnz))
        err_mean1[ni].append(np.mean(error_distancesmem[sni,ni][where1]))
        err_meanm1[ni].append(np.mean(error_distancesmem[sni,ni][wherem1]))
        err_sum1[ni].append(np.sum(error_distancesmem[sni,ni][where1]))
        err_summ1[ni].append(np.sum(error_distancesmem[sni,ni][wherem1]))
        len01 = len(where1) + len(where0)
        if 0 == len(where1):
            err_made1[ni].append(0)
        else:
            err_made1[ni].append(len(where1) / (len(where1) + len(where0)))

RES_VECTORS = [err_count, err_mean, err_sum, err_count1, err_frac, err_mean1, err_meanm1, err_sum1, err_summ1, err_made1]
RES_TITLES = ["error count", "mean error", "error sum", "error type 1 count", "fraction of error type -1", "error type 1 mean", "error type -1 mean", "error type 1 sum",
              "error type -1 sum", "error type 1 made of all possible type 1"]

cnt = len(RES_VECTORS)
cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))
f , ax = plt.subplots(cols, rows, sharex = True, squeeze=False)
for ri in range(len(RES_VECTORS)):
    rr = int(ri / cols)
    cc = ri % cols
    for ni, nn in enumerate(RETNOI_LIST):
        ax[cc,rr].plot(SNLEN_LIST, RES_VECTORS[ri][ni], color=COLORS[ni], label=nn)
    ax[cc,rr].set_xscale('log')
    ax[cc,rr].set_xticks([2, 10, 100, 600])
    ax[cc,rr].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[cc, rr].set_title(RES_TITLES[ri])
    ax[cc, rr].legend()
f.suptitle("retmem")

# =================================================================================================================================================================


delta_ret = np.load(PATH + "delta_ret.npy")
error_distances = np.load(PATH + "error_distances.npy")
error_types = np.load(PATH + "error_types.npy")

# f, ax = plt.subplots()
# ax.plot(SNLEN_LIST, np.mean(delta_seq,axis=1), 'k-', label="seq")
# for ni, nnn in enumerate(RETNOI_LIST):
#     ax.plot(SNLEN_LIST, np.mean(delta_ret[:, ni],axis=1), color=COLORS[ni], label="ret {}".format(nnn))
# ax.legend()
# ax.set_xscale('log')
# f.suptitle(PATH.split("/")[-2])

err_count, err_mean, err_sum, err_count1, err_frac, err_mean1, err_meanm1, err_sum1, err_summ1, err_made1 = [], [], [], [], [], [], [], [], [], []
for ni, noi in enumerate(RETNOI_LIST):
    err_count.append([])
    err_mean.append([])
    err_sum.append([])
    err_count1.append([])
    err_frac.append([])
    err_mean1.append([])
    err_meanm1.append([])
    err_sum1.append([])
    err_summ1.append([])
    err_made1.append([])
    for sni, snlen in enumerate(SNLEN_LIST):
        print(snlen)
        nnz = np.nonzero(error_distances[sni,ni])[0]
        err_count[ni].append(len(nnz))
        err_sum[ni].append(np.sum(error_distances[sni,ni]))
        err_mean[ni].append(np.mean(error_distances[sni,ni][nnz]))
        where1 = np.where(error_types[sni,ni] == 1)[0]
        wherem1 = np.where(error_types[sni,ni] == -1)[0]
        where0 = np.where(error_types[sni,ni] == 0)[0]
        err_count1[ni].append(len(where1))
        err_frac[ni].append(len(wherem1) / len(nnz))
        err_mean1[ni].append(np.mean(error_distances[sni,ni][where1]))
        err_meanm1[ni].append(np.mean(error_distances[sni,ni][wherem1]))
        err_sum1[ni].append(np.sum(error_distances[sni,ni][where1]))
        err_summ1[ni].append(np.sum(error_distances[sni,ni][wherem1]))
        len01 = len(where1) + len(where0)
        if 0 == len(where1):
            err_made1[ni].append(0)
        else:
            err_made1[ni].append(len(where1) / (len(where1) + len(where0)))

RES_VECTORS = [err_count, err_mean, err_sum, err_count1, err_frac, err_mean1, err_meanm1, err_sum1, err_summ1, err_made1]
RES_TITLES = ["error count", "mean error", "error sum", "error type 1 count", "fraction of error type -1", "error type 1 mean", "error type -1 mean", "error type 1 sum",
              "error type -1 sum", "error type 1 made of all possible type 1"]

cnt = len(RES_VECTORS)
cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))
f , ax = plt.subplots(cols, rows, sharex = True, squeeze=False)
for ri in range(len(RES_VECTORS)):
    rr = int(ri / cols)
    cc = ri % cols
    for ni, nn in enumerate(RETNOI_LIST):
        ax[cc,rr].plot(SNLEN_LIST, RES_VECTORS[ri][ni], color=COLORS[ni], label=nn)
    ax[cc,rr].set_xscale('log')
    ax[cc,rr].set_xticks([2, 10, 100, 600])
    ax[cc,rr].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[cc, rr].set_title(RES_TITLES[ri])
    ax[cc, rr].legend()
f.suptitle("ret")

plt.show()
