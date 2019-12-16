"""
Calculates and plots retrieval offsets for all pattern-key pairs in memory.
For every value, patterns in memory are sorted by Euclidean distance to key.
x-axis is the index (closest pattern first). Values are average over all keys in memory.

What retrieval offset means, see *reerror_anal.py*.

"""

from core import streamlined, semantic, system_params
import numpy as np
import scipy.spatial.distance
from matplotlib import pyplot as plt
import matplotlib

if __name__ == "__main__":

    PATH = "/local/results/reorder_o18a/"
    LOAD = True

    font = {'family' : 'Sans',
            'size'   : 22}
    matplotlib.rc('font', **font)

    ZOOM = 500

    normcolor = '0.4'
    normcolor_line = '0.4'
    zoomcolor = 'k'
    zoomcolor_line = 'k'

    try:
        if not LOAD:
            raise Exception()
        difmean = np.load(PATH+"difmean.npy")
        difmeanout = np.load(PATH+"difmeanout.npy")
    except:

        params = system_params.SysParamSet()

        forming_data = np.load(PATH+"forming0.npz")
        formx = forming_data["forming_sequenceX"]

        sfa1 = semantic.load_SFA(PATH+"sfadef1train0.sfa")
        formy = semantic.exec_SFA(sfa1, formx)
        formyw = streamlined.normalizer(formy, params.normalization)(formy)

        keys = formyw[1::2]
        pats = formyw[::2]

        dmat = scipy.spatial.distance.cdist(keys, pats, "euclidean")
        refs = np.diagonal(dmat)[:, None]
        dsort = np.argsort(dmat)
        dmat = None
        difs = []
        for i in range(len(dsort)):
            difs.append(scipy.spatial.distance.cdist(pats[dsort[i]], pats[i:i+1], "euclidean") - refs[i])
        difs = np.array(difs)
        difmean = np.mean(difs,axis=0)
        difs[np.where(difs <= -min(refs))] = 0    # set outliers (where pattern-to-retrieve is the original pattern p1 to zero. If average is still negative, it will work also with depression turned on
        difmeanout = np.mean(difs,axis=0)

        np.save(PATH+"difmean.npy", difmean)
        np.save(PATH+"difmeanout.npy", difmeanout)

    difmean = difmeanout

    f, ax = plt.subplots(1,2, squeeze=False, sharex='col')
    # plt.subplots_adjust(top = 0.95, bottom=0.05, hspace = 0.3)
    ax[0,0].plot(np.arange(len(difmean)), difmean, c=normcolor_line)
    ax[0,1].plot(np.arange(ZOOM), difmean[:ZOOM], c=zoomcolor_line)
    # ax[1,0].plot(np.arange(len(difmeanout)), difmeanout)
    # ax[1,1].plot(np.arange(ZOOM), difmeanout[:ZOOM])

    ylims00 = ax[0,0].get_ylim()
    xlims00 = ax[0,0].get_xlim()
    xlims01 = ax[0,1].get_xlim()
    ylims01 = ax[0,1].get_ylim()
    # ylims10 = ax[1,0].get_ylim()
    # xlims10 = ax[1,0].get_xlim()
    # xlims11 = ax[1,1].get_xlim()
    # ylims11 = ax[1,1].get_ylim()

    rwidth = xlims01[1]-xlims01[0]
    rheight = ylims01[1]-ylims01[0]
    rect01 = matplotlib.patches.Rectangle((xlims01[0]-0.2*rwidth, ylims01[0]-0.2*rheight), rwidth*1.4, rheight*1.4, lw=1, ec=zoomcolor, fc='none', zorder=10)
    ax[0,0].plot([xlims01[1]+0.2*rwidth, xlims00[1]+0.2*(xlims00[1]-xlims00[0])], [ylims01[1]+0.2*rheight, ylims00[1]], lw=1, c=zoomcolor, clip_on=False)
    ax[0,0].plot([xlims01[1]+0.2*rwidth, xlims00[1]+0.2*(xlims00[1]-xlims00[0])], [ylims01[0]-0.2*rheight, ylims00[0]], lw=1, c=zoomcolor, clip_on=False)
    ax[0,0].set_ylim(ylims00)
    ax[0,0].set_xlim(xlims00)
    ax[0,0].add_patch(rect01)

    # ax[1,0].axhline(y=ylims10[1]*1.1, xmin=0, xmax=2.2, clip_on=False, c='k')
    #
    # rwidth = xlims11[1]-xlims11[0]
    # rheight = ylims11[1]-ylims11[0]
    # rect11 = matplotlib.patches.Rectangle((xlims11[0]-0.2*rwidth, ylims11[0]-0.2*rheight), rwidth*1.4, rheight*1.4, lw=1, ec='darkgreen', fc='none', zorder=10)
    # ax[1,0].plot([xlims11[1]+0.2*rwidth, xlims10[1]+0.2*(xlims10[1]-xlims10[0])], [ylims11[1]+0.2*rheight, ylims10[1]], lw=1, c='darkgreen', clip_on=False)
    # ax[1,0].plot([xlims11[1]+0.2*rwidth, xlims10[1]+0.2*(xlims10[1]-xlims10[0])], [ylims11[0]-0.2*rheight, ylims10[0]], lw=1, c='darkgreen', clip_on=False)
    # ax[1,0].set_ylim(ylims10)
    # ax[1,0].set_xlim(xlims10)
    # ax[1,0].add_patch(rect11)

    ax[0,0].spines['bottom'].set_color(normcolor)
    ax[0,0].spines['top'].set_color(normcolor)
    ax[0,0].spines['right'].set_color(normcolor)
    ax[0,0].spines['left'].set_color(normcolor)
    ax[0,0].tick_params(axis='x', colors=normcolor)
    ax[0,0].tick_params(axis='y', colors=normcolor)
    # ax[0,0].yaxis.label.set_color(normcolor)
    # ax[0,0].xaxis.label.set_color(normcolor)
    # ax[0,0].title.set_color(normcolor)

    ax[0,1].spines['bottom'].set_color(zoomcolor)
    ax[0,1].spines['top'].set_color(zoomcolor)
    ax[0,1].spines['right'].set_color(zoomcolor)
    ax[0,1].spines['left'].set_color(zoomcolor)
    ax[0,1].tick_params(axis='x', colors=zoomcolor)
    ax[0,1].tick_params(axis='y', colors=zoomcolor)
    # ax[0,1].yaxis.label.set_color(zoomcolor)
    # ax[0,1].xaxis.label.set_color(zoomcolor)
    # ax[0,1].title.set_color(zoomcolor)


    # ax[1,1].spines['bottom'].set_color("darkgreen")
    # ax[1,1].spines['top'].set_color("darkgreen")
    # ax[1,1].spines['right'].set_color("darkgreen")
    # ax[1,1].spines['left'].set_color("darkgreen")
    # ax[1,1].tick_params(axis='x', colors="darkgreen")
    # ax[1,1].tick_params(axis='y', colors="darkgreen")
    # ax[1,1].yaxis.label.set_color("darkgreen")
    # ax[1,1].xaxis.label.set_color("darkgreen")
    # ax[1,1].title.set_color("darkgreen")

    # ax[1,0].set_title("Excluding reference pattern")
    # ax[1,0].set_xlabel("Index of neighbor (closest first)")
    # ax[1,0].set_ylabel("Distance increase")
    # ax[0,0].set_ylabel("Distance increase")
    # ax[1,1].set_xlabel("Index of neighbor (closest first)")

    ax[0,0].set_xlabel("index $i$ of neighbor (closest first)")
    ax[0,0].set_xticks([0,5000,10000,15000])
    # tix = ax[0,1].get_yticks()
    # ax[0,1].set_yticks(tix[2:])
    ax[0,1].set_yticks([-0.2, -0.3, -0.4])
    ax[0,0].set_ylabel("retrieval offset $v_i$")
    ax[0,1].set_xlabel("$i$")

    # f.suptitle("Increase in distance when choosing pattern neighbors")

    plt.subplots_adjust(bottom=0.15, top = 0.95, left=0.1, right=0.95)
    plt.show()

    # for i in range(len(difs)):
    #     plt.semilogx(1+np.arange(len(difs[i])), difs[i])
    #     plt.show()