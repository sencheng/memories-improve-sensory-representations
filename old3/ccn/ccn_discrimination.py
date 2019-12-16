"""
(Figures 9, 11, 14)

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

        PATH1 = "discrimination_replay_o18/"  # Path to load data from. Note that in two lines in the code below
                                              # more of the path has to be changed. Look for np.load(
        PATH2 = "discrimination_reorder_o18/"
        # PATH = "discrimination_lr02/"

        MIXING_LEVELS = [0.5, 0.49, 0.475, 0.45, 0.4, 0.3, 0.2, 0.1, 0]

        overlap_01_hitrateS = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH1+"overlap_01_hitrateS.npy")
        overlap_01_hitrateE = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH1+"overlap_01_hitrateE.npy")
        overlap_025_hitrateS = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH1+"overlap_025_hitrateS.npy")
        overlap_025_hitrateE = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH1+"overlap_025_hitrateE.npy")
        xy_hitrateS = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH1+"xy_hitrateS.npy")
        xy_hitrateE = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH1+"xy_hitrateE.npy")
        
        overlap_01_hitrateS_2 = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH2+"overlap_01_hitrateS.npy")
        overlap_01_hitrateE_2 = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH2+"overlap_01_hitrateE.npy")
        overlap_025_hitrateS_2 = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH2+"overlap_025_hitrateS.npy")
        overlap_025_hitrateE_2 = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH2+"overlap_025_hitrateE.npy")
        xy_hitrateS_2 = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH2+"xy_hitrateS.npy")
        xy_hitrateE_2 = np.load("/local/Sciebo/arbeit/Semantic-Episodic/figure data/"+PATH2+"xy_hitrateE.npy")

        f1 = plt.figure(figsize=(6,5))
        ls, = plt.plot(MIXING_LEVELS, overlap_01_hitrateS, 'g-')
        le, = plt.plot(MIXING_LEVELS, overlap_01_hitrateE, 'b-')
        plt.ylabel('hitrate')
        plt.xlabel('similarity level')
        lrand, = plt.plot(MIXING_LEVELS, [0.5]*len(MIXING_LEVELS), 'k:')

        plt.gca().yaxis.set_label_coords(-0.05, 0.5)
        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        plt.yticks([0.5,1.0])
        plt.ylim([0.45, 1.05])

        plt.subplots_adjust(left=0.15, right=0.9, bottom=0.2)
        plt.legend((ls, le, lrand), ("simple    $\Delta$", "episodic  $\\nabla$", "chance"), loc=3)

        f2 = plt.figure(figsize=(6, 5))
        plt.plot(MIXING_LEVELS, overlap_01_hitrateS_2, 'g-')
        plt.plot(MIXING_LEVELS, overlap_01_hitrateE_2, 'b-')
        plt.ylabel('hitrate')
        plt.xlabel('similarity level')
        plt.plot(MIXING_LEVELS, [0.5]*len(MIXING_LEVELS), 'k:')

        plt.gca().yaxis.set_label_coords(-0.05, 0.5)
        plt.xticks([0,0.1,0.2,0.3,0.4,0.5])
        plt.yticks([0.5, 1.0])
        plt.ylim([0.45, 1.05])

        plt.subplots_adjust(left=0.15, right=0.9, bottom=0.2)
        plt.legend((ls, le, lrand), ("simple    $\Delta$", "episodic  $\\nabla$", "chance"), loc=3)

        plt.show()