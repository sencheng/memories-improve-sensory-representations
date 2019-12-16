"""
(Figure7)

Loads the data generated with :py:mod:`figure_discrimination` and creates figures.

"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

if __name__ == "__main__":

        mcolor_s = 'k'
        mcolor_e = 'k'
        # mshape_s = '^'
        # mshape_e = 'v'
        mshape_s = 'x'
        mshape_e = '*'
        # mshape_s = 'o'
        # mshape_e = 's'


        font = {'family' : 'Sans',
                'size'   : 22}
        matplotlib.rc('font', **font)

        PATH = "discrimination_reerror_o18/"  # Path to load data from. Note that in two lines in the code below
                                              # more of the path has to be changed. Look for np.load(
        # PATH = "discrimination_lr02/"

        MIXING_LEVELS = [0.5, 0.49, 0.475, 0.45, 0.4, 0.3, 0.2, 0.1, 0]

        overlap_01_hitrateS = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH+"overlap_01_hitrateS.npy")
        overlap_01_hitrateE = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH+"overlap_01_hitrateE.npy")
        overlap_025_hitrateS = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH+"overlap_025_hitrateS.npy")
        overlap_025_hitrateE = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH+"overlap_025_hitrateE.npy")
        xy_hitrateS = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH+"xy_hitrateS.npy")
        xy_hitrateE = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH+"xy_hitrateE.npy")

        f, ax = plt.subplots(1,2, sharex=True, sharey=True, squeeze=False)

        axx = ax[0,0]
        ls, = axx.plot(MIXING_LEVELS, xy_hitrateS, 'k--')
        le, = axx.plot(MIXING_LEVELS, xy_hitrateE, 'k-')
        axx.set_ylabel('hitrate')
        axx.set_xlabel('mixing level')
        axx.plot(MIXING_LEVELS, [0.5]*len(MIXING_LEVELS), 'k:')
        axx.set_title("paradigm 1")
        # f.text(0.24, 0.93, "Paradigm 1", ha='center', va='center', fontdict=font)

        axx = ax[0,1]
        axx.plot(MIXING_LEVELS, overlap_01_hitrateS, 'k--')
        axx.plot(MIXING_LEVELS, overlap_01_hitrateE, 'k-')
        axx.set_xlabel('mixing level')
        lrand, = axx.plot(MIXING_LEVELS, [0.5]*len(MIXING_LEVELS), 'k:')
        # axx.set_title("$\sigma$ = 0.1")
        axx.set_title("paradigm 2")
        # f.text(0.65, 0.93, "Paradigm 2", ha='center', va='center', fontdict=font)

        # axx = ax[0,2]
        # axx.plot(MIXING_LEVELS, overlap_025_hitrateS, 'g-')
        # axx.plot(MIXING_LEVELS, overlap_025_hitrateE, 'r-')
        # axx.set_title("$\sigma$ = 0.25")
        # lrand, = axx.plot(MIXING_LEVELS, [0.5]*len(MIXING_LEVELS), 'k--')

        limix = axx.get_xlim()
        # ls, = axx.plot(-5, 0, c='k', ls='--', marker=mshape_s, mec=mcolor_s, mfc='none', markersize=22)
        # le, = axx.plot(-5, 0, c='k', ls='-',  marker=mshape_e, mec=mcolor_e, mfc='none', markersize=22)

        ax[0,0].axvline(x=0.58, ymin=-0.1, ymax=1.1, clip_on=False, c='k')

        axx.set_xlim(limix)
        axx.set_xticks([0,0.1,0.2,0.3,0.4,0.5])
        axx.set_ylim([0.4, 1.05])



        # axx.plot([0,0.5], [0.5, 0.5], 'k--')
        # f.text(0.5, 0.04, 'mixing level', ha='center', va='center', fontdict=font)
        # f.text(0.06, 0.5, 'hitrate', ha='center', va='center', rotation='vertical', fontdict=font)

        # ax[0,0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        f.subplots_adjust(bottom=0.15, right=0.82, left=0.08)
        f.legend((ls, le, lrand), ("$\sigma$=4  $\Delta$", "$\sigma$=0  $\\nabla$", "chance level"), 1)
        # f.legend((ls, le, lrand), ("simple    $\Delta$", "episodic  $\\nabla$", "chance level"), 1)
        # f.suptitle("Hitrate")


        top = 0.9
        left = 0.05
        right = 0.455
        col = '0.4'
        siz = 30
        f.text(left, top, "A", color=col, fontsize=siz)
        f.text(right, top, "B", color=col, fontsize=siz)

        plt.show()