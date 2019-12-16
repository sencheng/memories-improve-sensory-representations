import pickle
from matplotlib import pyplot as plt
import numpy as np

with open("/local/results/zack/zackbSEu.p", 'rb') as f:
    res = pickle.load(f)

print("xd", np.mean(np.sort(res['xd'])[:4]))
print("xTd", np.mean(np.sort(res['xTd'])[:4]))
print("xMd", np.mean(np.sort(res['xMd'])[:4]))
print("xKd", np.mean(np.sort(res['xKd'])[:4]))
print("xCd", np.mean(np.sort(res['xCd'])[:4]))
print("ySd", np.mean(np.sort(res['ySd'])[:4]))
print("ySTd", np.mean(np.sort(res['ySTd'])[:4]))
print("ySMd", np.mean(np.sort(res['ySMd'])[:4]))
print("ySKd", np.mean(np.sort(res['ySKd'])[:4]))
print("ySCd", np.mean(np.sort(res['ySCd'])[:4]))
print("yEd", np.mean(np.sort(res['yEd'])[:4]))
print("yETd", np.mean(np.sort(res['yETd'])[:4]))
print("yEMd", np.mean(np.sort(res['yEMd'])[:4]))
print("yEKd", np.mean(np.sort(res['yEKd'])[:4]))
print("yECd", np.mean(np.sort(res['yECd'])[:4]))
for lis in [["corrS", "corrST", "corrSM", "corrSK", "corrSC"], ["corrE", "corrET", "corrEM", "corrEK", "corrEC"]]:
    f, ax = plt.subplots(len(lis), 1, squeeze=True)
    for ki, key in enumerate(lis):
        ax[ki].matshow(np.abs(res[key]), cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax[ki].set_title(key)
        for (ii, jj), z in np.ndenumerate(res[key]):
            ax[ki].text(jj, ii, '{:.0f}'.format(z*100), ha='center', va='center', color="white")
plt.show()