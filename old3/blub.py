import numpy as np
from matplotlib import pyplot as plt

PATH = "/local/results/reerrorN0b/"
BINS = 50

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

data0 = np.load(PATH+"forming0.npz")
data3 = np.load(PATH+"forming3.npz")
data30 = np.load(PATH+"forming30.npz")

lat0 = data0["forming_latent"]
lat3 = data3["forming_latent"]
lat30 = data30["forming_latent"]

sumabs0 = np.abs(lat0[:,0]+ lat0[:,1])
sumabs3 = np.abs(lat3[:,0]+ lat3[:,1])
sumabs30 = np.abs(lat30[:,0]+ lat30[:,1])

end0 = sumabs0[1::2]
end3 = sumabs3[2::5]
end30 = sumabs30[599::600]

plt.figure()
plt.hist(sumabs0, bins=50, label="2")
plt.legend()
plt.figure()
plt.hist(sumabs3, bins=50, label="5")
plt.legend()
plt.figure()
plt.hist(end0, bins=10, label="2, ends")
plt.legend()
plt.figure()
plt.hist(end3, bins=10, label="5, ends")
plt.legend()
plt.show()

angle0 = np.arctan2(lat0[:,2], lat0[:,3])
hist0, bin0 = np.histogram(angle0, bins=BINS)
a0 = running_mean(bin0, 2)
angle3 = np.arctan2(lat3[:,2], lat3[:,3])
hist3, bin3 = np.histogram(angle3, bins=BINS)
a3 = running_mean(bin3, 2)

comb0 = np.arctan2(lat0[:,0], lat0[:,1])
chist0, cbin0 = np.histogram(comb0, bins=BINS)
c0 = running_mean(cbin0, 2)
comb3 = np.arctan2(lat3[:,0], lat3[:,1])
chist3, cbin3 = np.histogram(comb3, bins=BINS)
c3 = running_mean(cbin3, 2)

ax0 = plt.subplot(221, projection="polar")
axl0 = plt.subplot(222, projection="polar")
ax3 = plt.subplot(223, projection="polar")
axl3 = plt.subplot(224, projection="polar")

# ax0.scatter(lat0[:, 0], lat0[:, 1])
ax0.plot(c0, chist0)
axl0.plot(a0, hist0)
# ax30.scatter(lat30[:, 0], lat30[:, 1])
ax3.plot(c3, chist3)
axl3.plot(a3, hist3)

ax0.set_ylim(ax3.get_ylim())
axl0.set_ylim(axl3.get_ylim())

plt.show()