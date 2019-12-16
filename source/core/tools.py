#!/usr/bin/env python
"""
Contains some useful functions for data analysis.

The ones most commonly used are:

* :py:func:`delta_diff`: Computes delta values of a sequence
* :py:func:`compare_inputs`: Animate one or more image sequences
* :py:func:`feature_latent_correlation`: Compute a correlation matrix of SFA features and latent variables

"""

from matplotlib import pyplot
import numpy as np
import math
import itertools
import matplotlib.animation as animation
import os, subprocess
import scipy.stats

def _completeness(lats) :
    """ Computes a score in [0,1] for a numpy array of latent variables, where the first two
    code for position, and an "on screen position" is [-1,1]"""
    yc = np.clip(np.exp(1-np.abs(lats[:,0])), 0, 1)
    xc = np.clip(np.exp(1-np.abs(lats[:,1])), 0, 1)
    return yc * xc

def preview_input(inp, slow_features = None, retrieved_sequence=None, dimensions = None, rate=50):

    """Generate an animation to preview the input episodes.
    Corresponding sfa output (slow features) can be animated next to it at the same time.

    :param inp: input sequence as ndarray
    :param slow_features: ndarray containing slow features corresponding to selected input. If None, only input is shown.
    :param retrieved_sequence: ndarray containing sequence retrieved from episodic memory. Will be shown next to slow features if not None.
    :param dimensions: only evaluated if slow_features are given. Number of features to plot. If None, plots all features.
    :param rate: interval between two frames in ms. If small, drawing will be slower for computational reasons.
    """
    global anim
    fig = pyplot.figure()

    imrange = [np.min(inp), np.max(inp)]
        
    if slow_features is not None:
        if dimensions:
            slow = slow_features[:,:dimensions]
            if retrieved_sequence is not None:
                seq = retrieved_sequence[:,:dimensions]
        else:
            slow = slow_features
            seq = retrieved_sequence
        x_arr = np.arange(slow.shape[1])-0.4
        max_slow = np.amax(slow)
        min_slow = np.amin(slow)
        if min_slow > 0:
            min_slow = 0
    
    def frame_gen():
        """Local function. Generator for input frames"""
        frames_arr = frame_gen.source
        frame = frames_arr[0]
        width = int(math.sqrt(len(frame)))
        for frame in frames_arr:
            img = np.resize(frame,(width, width))
            yield img
            
    frame_gen.source = inp    #initialize local generator function
    
    frames = frame_gen()
    img = next(frames)

    if slow_features is not None:
        if retrieved_sequence is not None:
            pyplot.subplot(1,3,1)
        else:
            pyplot.subplot(1,2,1)

    im = pyplot.imshow(img, interpolation='none', cmap='Greys', vmin=imrange[0], vmax=imrange[1])
    
    if slow_features is not None:
        if retrieved_sequence is not None:
            ax2 = pyplot.subplot(1,3,2)
            ax3 = pyplot.subplot(1,3,3)
            ax3.set_xlim([-0.5, slow.shape[1]+0.5])
            ax3.set_ylim([math.floor(min_slow), math.ceil(max_slow)])
            rects3 = ax3.bar(x_arr, seq[0])    #bar plot
        else:
            ax2 = pyplot.subplot(1,2,2)
        ax2.set_xlim([-0.5, slow.shape[1]+0.5])
        ax2.set_ylim([math.floor(min_slow), math.ceil(max_slow)])
        
        rects2 = ax2.bar(x_arr, slow[0])   #bar plot
    
    # Ahem. So, this next line looks like an array (and is an array) but only needs to be there
    # so that the variable is mutable (thanks, non-typed languages). Anyway, python3 would have
    # been able to do this better, but since we run this with phython 2.7, we gotta do it this way. 
    # See http://stackoverflow.com/questions/9264763/unboundlocalerror-in-python
    # update: now we have python3, but we are lazy and keep it this way
    slow_idx = [1]
        
    def update(img):
        im.set_data(img)
        if slow_features is not None:
            if slow_idx[0] > len(slow)-1:
                slow_idx[0] = 0
            # Below is the offending line described above ^^
            for rect, feat in zip(rects2, slow[slow_idx[0]]):    #edit bar plot rects
                if feat < 0:
                    rect.set_y(feat)
                    rect.set_height(-feat)
                else:
                    rect.set_y(0)
                    rect.set_height(feat)
            if retrieved_sequence is not None and not slow_idx[0] > len(seq)-1:
                for rect, feat in zip(rects3, seq[slow_idx[0]]):  #edit bar plot rects
                    if feat < 0:
                        rect.set_y(feat)
                        rect.set_height(-feat)
                    else:
                        rect.set_y(0)
                        rect.set_height(feat)
            slow_idx[0] += 1

    anim = animation.FuncAnimation(fig, update, frame_gen, interval=rate)
    pyplot.show()
    return anim

def compare_inputs(INP, rate=10):
    """
    Given a tuple of input sequences, animate them next to each other

    :param INP: tuple of input sequence arrays
    :param rate: framerate in ms
    """
    global anim
    fig = pyplot.figure()
    cmapnames = ['Blues', 'Oranges', 'Purples', 'Greens', 'Reds' ]
    
    def frame_gen():
        """Local function. Generator for input frames"""
        width = int(math.sqrt(len(INP[0][0])))
        for (tup) in zip(*INP):
            yield [np.resize(q,(width, width)) for q in tup]
    
    frames = frame_gen()
    start = next(frames)
    imI = []
    imrange = [[np.min(a), np.max(a)] for a in INP]
    
    for i,startI in enumerate(start):
        imI.append(pyplot.subplot(1, len(INP), i + 1).matshow(startI, cmap=cmapnames[i % len(cmapnames)], vmin=imrange[i][0], vmax=imrange[i][1]))
        
    def update(tup):
        for data, image in zip(tup, imI):
            image.set_data(data)
    
    anim = animation.FuncAnimation(fig, update, frame_gen, interval=rate)
    pyplot.show()
    return anim

def listify(obj, desired_length):
    """
    makes a list containing desired_length copies of given obj.
    If obj is list, checks if its length matches desired length, otherwise raise Exception

    :param obj: obj to make list of (or to validate, if list)
    :returns: resulting list or original list in case of single object or list given, respectively
    """
    def flatten_stringlist(foo):
        for x in foo:
            if hasattr(x, '__iter__') and not isinstance(x, str):
                for y in flatten_stringlist(x):
                    yield y
            else:
                yield x
    
    l = [obj]
    merged = list(flatten_stringlist(l))
    if len(merged) == 1:
        res = merged*desired_length
        return res
    if not len(merged) == desired_length:
        raise Exception("Length of obj list does not match desired number.")
    return merged

def feature_latent_correlation(features, latent_vars, categories=None):
    """
    Computes a correlation matrix between SFA features and latent variables

    :param features: array containing SFA feature sequences
    :param latent_vars: array containing the corresponding latent variable sequences
    :param categories: array containing the corresponding object identity label sequences
    :return: correlation matrix of size (number of features, number of latents including object identity)
    """
    #latent_vars is of shape [...,[x,y,z],[x,y,z],None,[x,y,z],[x,y,z],...]
    none_indices = []   #at blank frames, None var is inserted
    latent = []
    for i, tup in enumerate(latent_vars):
        if tup is None:
            none_indices.append(i)    #find None indices
        else:
            latent.append(tup) #and only copy not None vars, effectively deleting None values
    lat = np.array(latent)
    lat_trans = np.transpose(lat)    #transposing, lat_trans is of shape [[x,x,x,x,x,....],[y,y,y,y,y,....],[z,z,z,z,z,....]]
    nLatent = len(lat_trans)   #number of latent variables
    
    feat = np.delete(features,none_indices, axis=0)   #delete feature corresponding to blank frames (None values)
    feat_trans = np.transpose(feat)       #transposing, feat_trans is of shape [[feat1,feat1,feat1,...],[feat2,feat2,feat2,...],...]
    nFeatures = np.shape(feat_trans)[0]
    
    if categories is not None:
        cat = np.delete(categories,none_indices, axis=0)
        app = np.append(lat_trans, [cat], axis=0)
        nLatent = nLatent+1
    else:
        app = lat_trans
    
    x = np.append(feat_trans, app, axis=0)   #make one observation vector, each row is one variable, each column one observation
    R = np.corrcoef(x)
    return R[nFeatures:,:-nLatent]    #only return correlation coefficients for latent_vars, excluding autocor of latent_vars or cor of latent_vars with each other

def delta_gradient(feature_array):
    """
    Calculate delta values of a sequence of vectors, one delta for every vector element.
    I.e. a sequence of vectors with 4 elements each (which could for instance be SFA features)
    would return an array with 4 delta values.

    This uses the numpy.gradient method to compute difference values. We have usually used
    the other delta method which is :py:func:`delta_diff`. See numpy documentation
    for more info.

    :param feature_array: sequence of which to calculate delta values
    :return: delta values
    """
    grad = np.transpose(np.gradient(feature_array)[0])
    delta = np.mean(grad**2,axis=1)
    return delta

def delta_diff(feature_array):
    """
    Calculate delta values of a sequence of vectors, one delta for every vector element.
    I.e. a sequence of vectors with 4 elements each (which could for instance be SFA features)
    would return an array with 4 delta values.

    This uses the numpy.diff method to compute difference values. This function was
    usually used to determine delta values, an alternative would be
    :py:func:`delta_gradient`. See numpy documentation
    for more info.

    :param feature_array: sequence of which to calculate delta values
    :return: delta values
    """
    diff = np.transpose(np.diff(feature_array,axis=0))
    delta = np.mean(diff**2,axis=1)
    return delta


def display_corr(matrices, titles=None):
    """
    Plot a tuple of correlation matrices (as subplots)

    :param matrices: tuple of correlation matrices
    :param titles: list of plot titles, same length as matrices tuple
    """
    if type(matrices) is not list:
        matrices = [ matrices ]

    fc, axarrc = pyplot.subplots(len(matrices),1)
    off = 0.05 # plotting offset

    if len(matrices) == 1:
        axarrc = [axarrc]

    for a, (ax, mat) in enumerate(zip(axarrc, matrices)):
        ax.set_xlabel("SFA Feature")
        ax.set_ylabel("Latent Feature")

        lastplot = ax.matshow(mat, cmap='seismic', vmin=-1, vmax=1)

        if titles is not None:
            ax.set_title(titles[a])

        for (ii, jj), z in np.ndenumerate(mat):
            #axarrc[i,p].text(jj, ii, '{:.2f}'.format(z), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            ax.text(jj+off, ii+off, '{:.2f}'.format(z), ha='center', va='center',color="black", fontsize=10)
            ax.text(jj, ii, '{:.2f}'.format(z), ha='center', va='center',color="white", fontsize=10)

#    pyplot.colorbar(lastplot)

    fc.show()

def gaussian(x, mu, sig):
    """
    gaussian function :math:`e^{\\frac{(x-mu)^2}{2*sig^2}}`

    :param x:
    :param mu:
    :param sig:
    :returns: function value
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def point_trailXY(preimage, DATA, meld=0.4, shape=(30,30), autoplay=1):
    """
    A function to inspect input sequences and corresponding x/y positions. Left plot
    shows the image, right plot shows all x/y positions in the data as small boxes and
    highlights the positions in the sequential vicinity of the currently shown image
    (so few elements before and after in the sequence). Sequence is automatically played
    but arrow/cursor keys can control progression. *Right* and *Left* key change the speed of
    autoplay, e.g pressing *Left* once stops the sequence, pressing *Left* again makes it play
    backwards. *Up* and *Down* keys step through the sequence, this probably only makes sense
    if autoplay is stopped (by pressing *Left* once at the beginning).

    :param preimage: array of images, shape (n,m)
    :param DATA: array of x/y coordinates, shape (n,2)
    :param meld: Changing this probably messes things up. Parameter controlling how lines scale when being updated.
    :param shape: How to reshape input images
    :param autoplay: Initialization of the parameter that can be controlled by *Right* and *Left* key. If 1,
                     autoplay will be forward with speed of 1, if 0 autoplay is off at the start etc.
    """
    index = [0]
    autoplay = [autoplay]
    speed = 5

    permute = np.argsort(-np.abs(2*speed- np.arange(4*speed)))
    invperm = np.argsort(permute)

    def get_positions():
        subset = np.array(range(max(index[0]-2*speed, 0), min(index[0]+2*speed, len(DATA)-1)))
        subset.resize(4*speed)

        return DATA[subset,:], subset

    def plotall():
        vals, indices = get_positions()

        g = gaussian(indices[permute], index[0], speed)
        pts = ax.scatter(vals[permute,0], vals[permute,1], marker='o', alpha=0.6, cmap='Greens', zorder=1,  c=g, s=30+g*100)
        lines, =  ax.plot(vals[:,0], vals[:,1], zorder=2)
        image = preax.matshow(preimage[index[0]].reshape(shape), cmap='Blues')
        return pts, lines, image

    I = np.arange(len(DATA))

    fig, (preax,ax) = pyplot.subplots(1,2)
    allpts = ax.plot(DATA[:,0], DATA[:,1], 's', color='0.95', mec='0.8', zorder=0)
    points,lines,image = plotall()

    def adjust(di):
        index[0] = max(min(index[0]+di,len(DATA)-1), 0)
        image.set_data(preimage[index[0]].reshape(shape))

    def automove(): adjust(autoplay[0])

    def press(event):
        if event.key == 'up':
            adjust(-1)
        elif event.key == 'down':
            adjust(1)
        elif event.key == 'left':
            autoplay[0] -= 1
        elif event.key == 'right':
            autoplay[0] += 1

    def update():
        valsto, ind = get_positions()
        valsnow = points.get_offsets()[invperm]

        newoff = (1-meld) * valsnow + meld*valsto
        points.set_offsets(newoff[permute,:])
        lines.set_data(newoff[:,0], newoff[:,1])

        fig.canvas.draw()


    fig.canvas.mpl_connect('key_press_event', press)
    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(update)
    timer.start()

    autotimer = fig.canvas.new_timer(interval=100)
    autotimer.add_callback(automove)
    autotimer.start()

    pyplot.show()

def _subcoordinates(ax, rect):
    fig = ax.get_figure()
    box = ax.get_position()
    x,y = fig.transFigure.inverted().transform(ax.transAxes.transform(rect[0:2]))
    return [x,y,rect[2]*box.width, rect[3]*box.height]

def _replace_with_joint_axes(ax):
    # definitions for the axes
    left, width = 0.0, 0.78
    bottom, height = 0.0, .78
    bottom_h = left_h = left + width + 0.02

    rect_scatter = _subcoordinates(ax, [left, bottom, width, height])
    rect_histx = _subcoordinates(ax, [left, bottom_h, width, 0.2])
    rect_histy = _subcoordinates(ax, [left_h, bottom, 0.2, height])

    axScatter = pyplot.axes(rect_scatter)
    axHistx = pyplot.axes(rect_histx)
    axHisty = pyplot.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_visible(False)
    axHistx.yaxis.set_visible(False)
    axHisty.xaxis.set_visible(False)
    axHisty.yaxis.set_visible(False)

    ax.get_figure().delaxes(ax)

    return axScatter, axHistx, axHisty

def _joint_scatter(axes, X, Y, barcolor='b', **scatterparams):
    scatartist = axes[0].scatter(X, Y, **scatterparams)

    extra = dict(bins=40, alpha=scatterparams['alpha'], color=barcolor, edgecolor="none")
    axes[1].hist(X, **extra)
    axes[2].hist(Y, orientation='horizontal', **extra)

    axes[1].set_xlim(axes[0].get_xlim())
    axes[2].set_ylim(axes[0].get_ylim())

    #    axes[0].set_axis_bgcolor('0.0')

    return scatartist

def _weighted_distance(A, B, weights=1):
    weights = np.array(listify(weights, A.shape[1]))
    return np.sqrt(np.sum(weights * (A - B) ** 2, axis=1))

def distpicker(dataX, dataY, lat, cat, M = 5000, weights=[1,1,1,1]):
    """
    Makes a interactive plot to analyze pairwise pattern distances. Takes input sequences,
    corresponding SFA features, x/y-coordinates and object identity labels and randomly selects two
    subsets of size *M*. A scatter plot of Euclidean distances in space (x-axis) and the distances
    in SFA feature values (y-axis) is made. Histograms in both directions are added. The pattern pairs
    are color coded: Blue when they share object identity and red when the object identity is different
    between the two patterns. The user can click on the dots in the plot to show the corresponding
    raw input images.

    :param dataX: N-element sequence of input images
    :param dataY: sequence of SFA features, shape (N,k) where k is number of SFA features
    :param lat: array with corresponding coordinates, shape (N,2)
    :param cat: array with object identity labels, shape (N,1)
    :param M: size of subsets to randomly select.
    :param weights: How to weight SFA features when calculating pairwise differences. Has to be either scalar
                    (then all features are equally weighted) or length k.
    """
    N = len(dataX)
    I1,I2 = np.random.randint(N,size=M), np.random.randint(N,size=M)

    latdists = _weighted_distance(lat[I1, :], lat[I2, :], weights=weights)
    ydists = _weighted_distance(dataY[I1, :], dataY[I2, :])

    complete = _completeness(lat[I1, :]) * _completeness(lat[I2, :])
    samecat = cat[I1] == cat[I2]
    diffcat = np.logical_not(samecat)

    fig = pyplot.figure()
    ax_overlay = pyplot.subplot2grid((2,2), (0,0), rowspan=2)
    axA = pyplot.subplot2grid((2,2), (0,1))
    axB = pyplot.subplot2grid((2,2), (1,1))
    axes = _replace_with_joint_axes(ax_overlay)

    def scat(axes, indx_subset, cmap, barcolor, picker=True):
        artist = _joint_scatter(axes, latdists[indx_subset], ydists[indx_subset], barcolor=barcolor, alpha=0.3, c=complete[indx_subset],
                                cmap=cmap, s=10+complete[indx_subset]*30, edgecolor='none', picker=picker, vmin=0, vmax=1)
        artist.indx_subset = indx_subset
        return artist

    def onpick(event):
        print(event.artist)
        index = event.ind[0] if len(np.shape(event.ind)) > 0 else event.ind
        indx_subset = event.artist.indx_subset
        i1, i2 = I1[indx_subset][index], I2[indx_subset][index]
        print(cat[i1], cat[i2])
        img1.set_data(dataX[i1].reshape((30,30)))
        img2.set_data(dataX[i2].reshape((30,30)))
        fig.canvas.draw()


    scat(axes, diffcat, 'Reds', 'r')
    scat(axes, samecat, 'Blues', 'b')
    img1 = axA.matshow(np.random.normal(0.5,0.5,(30,30)), cmap='Greys',vmin=0, vmax=1)
    img2 = axB.matshow(np.random.normal(0.5,0.5,(30,30)), cmap='Greys',vmin=0, vmax=1)

    IND = np.logical_and(samecat, complete==1)
    cor, p = scipy.stats.spearmanr(latdists[IND], ydists[IND])

    fun = lambda IND : scipy.stats.spearmanr(latdists[IND], ydists[IND])
    print(fun(np.logical_and(samecat, complete==1)))
    print(fun(samecat))
    print(fun(np.logical_and(diffcat, complete==1)))
    print(fun(diffcat))
    print(fun(range(M)))


    fig.canvas.mpl_connect('pick_event', onpick)
    pyplot.show()

# class LocalBuffer():
#     def __init__(self, filename):
#         self.filename = os.path.expanduser(filename)
#         self.writer = open(self.filename, 'w')
#
#     def declare(self, name, contents):
#         self.writer.write(name+' = '+repr(contents)+'\n')
#
#     def code(self, code) :
#         self.writer.write(code+'\n')
#
#     def execute(self):
#         self.writer.truncate()
#         self.writer.close()
#         subprocess.call([os.path.expanduser('~/execlocal.sh'), self.filename])
# #        self.writer = open(self.filename, 'w')

# #=====================================================
# #taken from
# #https://senselab.med.yale.edu/ModelDB/showModel.cshtml?model=150031&file=%5CGridCellModel%5Cgrid_cell_model%5Cgrid_cell_analysis.py#tabs-2
# #Copyright (C) 2012  Lukas Solanka <l.solanka@sms.ed.ac.uk>
# def SNAutoCorr(rateMap, arenaDiam, h):
#     precision = arenaDiam/h
#     xedges = np.linspace(-arenaDiam, arenaDiam, precision*2 + 1)
#     yedges = np.linspace(-arenaDiam, arenaDiam, precision*2 + 1)
#     X, Y = np.meshgrid(xedges, yedges)
#
#     corr = np.ma.masked_array(scipy.signal.correlate2d(rateMap, rateMap), mask = np.sqrt(X**2 + Y**2) > arenaDiam)   #@UndefinedVariable
#
#     return corr, xedges, yedges
#
# def cellGridnessScore(rateMap, arenaDiam, h, corr_cutRmin):
#     '''
#     Compute a cell gridness score by taking the auto correlation of the
#     firing rate map, rotating it, and subtracting maxima of the
#     correlation coefficients of the former and latter, at 30, 90 and 150 (max),
#     and 60 and 120 deg. (minima). This gives the gridness score.
#
#     The center of the auto correlation map (given by corr_cutRmin) is removed
#     from the map
#     '''
#     rateMap_mean = rateMap - np.mean(np.reshape(rateMap, (1, rateMap.size)))
#     autoCorr, autoC_xedges, autoC_yedges = SNAutoCorr(rateMap_mean, arenaDiam, h)
#
#     # Remove the center point and
#     X, Y = np.meshgrid(autoC_xedges, autoC_yedges)
#     autoCorr[np.sqrt(X**2 + Y**2) < corr_cutRmin] = 0
#
#     da = 3
#     angles = range(0, 180+da, da)
#     crossCorr = []
#     # Rotate and compute correlation coefficient
#     for angle in angles:
#         autoCorrRot = scipy.ndimage.interpolation.rotate(autoCorr, angle, reshape=False)   #@UndefinedVariable
#         C = np.corrcoef(np.reshape(autoCorr, (1, autoCorr.size)),
#             np.reshape(autoCorrRot, (1, autoCorrRot.size)))
#         crossCorr.append(C[0, 1])
#
#     max_angles_i = np.array([30, 90, 150]) / da
#     min_angles_i = np.array([60, 120]) / da
#
#     maxima = np.max(np.array(crossCorr)[max_angles_i])
#     minima = np.min(np.array(crossCorr)[min_angles_i])
#     G = minima - maxima
#
#     return G, np.array(crossCorr), angles
# #=========================================================0
