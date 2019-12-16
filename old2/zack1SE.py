import pickle
from matplotlib import pyplot as plt
import numpy as np

with open("/local/results/zackone2/zackoneSE.p", 'rb') as f:
    res = pickle.load(f)

print("xd", np.mean(np.sort(res['xd'])[:4]))
print("xTd", np.mean(np.sort(res['xTd'])[:4]))
print("xMd", np.mean(np.sort(res['xMd'])[:4]))
print("xKd", np.mean(np.sort(res['xKd'])[:4]))
print("xCd", np.mean(np.sort(res['xCd'])[:4]))
print("yd", np.mean(np.sort(res['yd'])[:4]))
print("yTd", np.mean(np.sort(res['yTd'])[:4]))
print("yMd", np.mean(np.sort(res['yMd'])[:4]))
print("yKd", np.mean(np.sort(res['yKd'])[:4]))
print("yCd", np.mean(np.sort(res['yCd'])[:4]))
print("zSd", np.mean(np.sort(res['zSd'])[:4]))
print("zSTd", np.mean(np.sort(res['zSTd'])[:4]))
print("zSMd", np.mean(np.sort(res['zSMd'])[:4]))
print("zSKd", np.mean(np.sort(res['zSKd'])[:4]))
print("zSCd", np.mean(np.sort(res['zSCd'])[:4]))
print("zEd", np.mean(np.sort(res['zEd'])[:4]))
print("zETd", np.mean(np.sort(res['zETd'])[:4]))
print("zEMd", np.mean(np.sort(res['zEMd'])[:4]))
print("zEKd", np.mean(np.sort(res['zEKd'])[:4]))
print("zECd", np.mean(np.sort(res['zECd'])[:4]))
for lis in [["corrS", "corrST", "corrSM", "corrSK", "corrSC"], ["corrE", "corrET", "corrEM", "corrEK", "corrEC"]]:
    f, ax = plt.subplots(len(lis), 1, squeeze=True)
    for ki, key in enumerate(lis):
        ax[ki].matshow(np.abs(res[key]), cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax[ki].set_title(key)
        for (ii, jj), z in np.ndenumerate(res[key]):
            ax[ki].text(jj, ii, '{:.0f}'.format(z*100), ha='center', va='center', color="white")
plt.show()