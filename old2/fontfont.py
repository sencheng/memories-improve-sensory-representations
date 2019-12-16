from core import trajgen

import numpy as np

from matplotlib import pyplot

tlist = []
llist = []
for s in range(5, 50, 5):
    tlist.append(trajgen.makeTextImageTrimmed("T", outsize=s))
    llist.append(trajgen.makeTextImageTrimmed("L", outsize=s))

tarr = np.array(tlist)
larr = np.array(llist)

# np.save("tarr.npy", tarr)
# np.save("larr.npy", larr)

tarr_clust = np.load("tarr.npy")
larr_clust = np.load("larr.npy")

for elt, ell, elt_c, ell_c in zip(tarr,larr,tarr_clust,larr_clust):
    f, axarr = pyplot.subplots(2,2)
    axarr[0][0].imshow(elt, interpolation="none")
    axarr[0][1].imshow(ell, interpolation="none")
    axarr[1][0].imshow(elt_c, interpolation="none")
    axarr[1][1].imshow(ell_c, interpolation="none")