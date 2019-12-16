import numpy as np
from matplotlib import pyplot as plt
# from core import tools
# from core import episodic, semantic, streamlined, system_params
# import scipy.spatial.distance
# import pickle

PATH = "/local/results/bigreorder/"

DRAW = False
# SNLEN_LIST = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
# SNLEN_LIST = [2,3,5,8,12,20,40,100,200,600]
SNLEN_LIST = [2,5,20,100]
indlist = [0, 3, 13, 25]
# RETNOI_LIST = [0,1,2,3,4]
RETNOI_LIST = [0,1,2,3,4]
RET = True
RETMEM = True

delta_seq = np.load(PATH + "delta_seq.npy")
if RET:
    try:
        delta_ret = np.load(PATH + "delta_ret.npy")
    except:
        RET = False
if RETMEM:
    try:
        delta_retmem = np.load(PATH + "delta_retmem.npy")
    except:
        RETMEM = False
if RET:
    try:
        delta_retxyc = np.load(PATH + "delta_retxyc.npy")
    except:
        pass
if RETMEM:
    try:
        delta_retmemxyc = np.load(PATH + "delta_retmemxyc.npy")
    except:
        pass

if RET:
    plt.plot(SNLEN_LIST, np.mean(delta_seq, axis=1), 'k-', label="seq")
    for ni, nnn in enumerate(RETNOI_LIST):
        plt.plot(SNLEN_LIST, np.mean(delta_ret[:, ni:ni+1, :], axis=2), ['y-', 'g-', 'm-', 'r-', 'c-'][ni], label="ret {}".format(nnn))
        try:
            plt.plot(SNLEN_LIST, np.mean(delta_retxyc[:, ni:ni + 1, :], axis=2), ['y:', 'g:', 'm:', 'r:', 'c:'][ni], label="retxyc {}".format(nnn))
        except:
            pass
    plt.legend()
    plt.xscale('log')
    plt.suptitle(PATH.split("/")[-2])

if RETMEM:
    plt.figure()
    plt.plot(SNLEN_LIST, np.mean(delta_seq, axis=1), 'k-', label="seq")
    for ni, nnn in enumerate(RETNOI_LIST):
        plt.plot(SNLEN_LIST, np.mean(delta_retmem[:, ni:ni+1, :], axis=2), ['y-', 'g-', 'm-', 'r-', 'c-'][ni], label="retmem {}".format(nnn))
        try:
            plt.plot(SNLEN_LIST, np.mean(delta_retmemxyc[:, ni:ni + 1, :], axis=2), ['y:', 'g:', 'm:', 'r:', 'c:'][ni], label="retmemxyc {}".format(nnn))
        except:
            pass
    plt.legend()
    plt.xscale('log')
    plt.suptitle(PATH.split("/")[-2])

if not RET and not RETMEM:
    plt.plot(SNLEN_LIST, np.mean(delta_seq, axis=1), 'k-', label="seq")

plt.show()
