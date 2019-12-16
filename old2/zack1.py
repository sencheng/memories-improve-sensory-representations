import pickle
from matplotlib import pyplot as plt
import numpy as np

with open("/local/results/zack/zackzack2.p", 'rb') as f:
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
for key in ['corr', 'corrT', 'corrM', 'corrK', 'corrC']:
    plt.matshow(np.abs(res[key]), cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(key)
    for (ii, jj), z in np.ndenumerate(res[key]):
        plt.text(jj, ii, '{:.0f}'.format(z*100), ha='center', va='center', color="white")
plt.show()