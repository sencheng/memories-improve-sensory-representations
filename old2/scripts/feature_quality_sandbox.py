# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:15:45 2016

@author: richaopf
"""

import numpy as np
from itertools import izip
from matplotlib import pyplot
from core.tools import _completeness

from core import sensory, input_params, tools

from scipy.stats import spearmanr

def subcoordinates(ax, rect):
    fig = ax.get_figure()
    box = ax.get_position()
    x,y = fig.transFigure.inverted().transform(ax.transAxes.transform(rect[0:2]))
    return [x,y,rect[2]*box.width, rect[3]*box.height]


def replace_with_joint_axes(ax):
    # definitions for the axes
    left, width = 0.0, 0.78
    bottom, height = 0.0, .78
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = subcoordinates(ax, [left, bottom, width, height])
    rect_histx = subcoordinates(ax, [left, bottom_h, width, 0.2])
    rect_histy = subcoordinates(ax, [left_h, bottom, 0.2, height])
        
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

def joint_scatter(axes, X, Y, barcolor='b', **scatterparams):
    scatartist = axes[0].scatter(X, Y, **scatterparams)
    
    extra = dict(bins=40,alpha=scatterparams['alpha'], color=barcolor, edgecolor = "none")
    axes[1].hist(X, **extra)
    axes[2].hist(Y, orientation='horizontal', **extra)
    
    axes[1].set_xlim(axes[0].get_xlim())
    axes[2].set_ylim(axes[0].get_ylim())
    
#    axes[0].set_axis_bgcolor('0.0')
    
    return scatartist


def weighted_distance(A,B,weights=1):
    weights = np.array(tools.listify(weights, A.shape[1]))
    return np.sqrt(np.sum(weights*(A-B)**2, axis=1))

def distpicker(dataX, dataY, lat, cat, M = 5000):
    N = len(dataX)
    I1,I2 = np.random.randint(N,size=M), np.random.randint(N,size=M)
    
    latdists = weighted_distance(lat[I1,:],lat[I2,:],weights=[1,1,1,1,0,0,0])
    ydists = weighted_distance(dataY[I1,:], dataY[I2,:])
    
    complete = _completeness(lat[I1, :]) * _completeness(lat[I2, :])
    samecat = cat[I1] == cat[I2]
    diffcat = np.logical_not(samecat)
    
#    samecat = np.logical_and(samecat, complete==1)
#    diffcat = np.logical_and(diffcat, complete==1)
    
    
    fig = pyplot.figure()
    ax_overlay = pyplot.subplot2grid((2,2), (0,0), rowspan=2)
    axA = pyplot.subplot2grid((2,2), (0,1))
    axB = pyplot.subplot2grid((2,2), (1,1))
    axes = replace_with_joint_axes(ax_overlay)
    
    def scat(axes, indx_subset, cmap, barcolor, picker=True):
        artist = joint_scatter(axes, latdists[indx_subset], ydists[indx_subset],barcolor=barcolor, alpha=0.3, c=complete[indx_subset], 
                           cmap=cmap, s=10+complete[indx_subset]*30, edgecolor='none',picker=picker,vmin=0, vmax=1)
        artist.indx_subset = indx_subset
        return artist
        
    def onpick(event):
        print event.artist
        index = event.ind[0] if len(np.shape(event.ind)) > 0 else event.ind
        indx_subset = event.artist.indx_subset
        i1, i2 = I1[indx_subset][index], I2[indx_subset][index]
        print cat[i1], cat[i2]
        img1.set_data(dataX[i1].reshape((30,30)))
        img2.set_data(dataX[i2].reshape((30,30)))
        fig.canvas.draw()
        

    scat(axes, diffcat, 'Reds', 'r')
    scat(axes, samecat, 'Blues', 'b')
    img1 = axA.matshow(np.random.normal(0.5,0.5,(30,30)), cmap='Greys',vmin=0, vmax=1)
    img2 = axB.matshow(np.random.normal(0.5,0.5,(30,30)), cmap='Greys',vmin=0, vmax=1)
    
    IND = np.logical_and(samecat, complete==1)
    cor, p = spearmanr(latdists[IND], ydists[IND])

    fun = lambda IND : spearmanr(latdists[IND], ydists[IND]) 
    print fun(np.logical_and(samecat, complete==1))
    print fun(samecat)
    print fun(np.logical_and(diffcat, complete==1))
    print fun(diffcat)
    print fun(range(M))
    
    
    fig.canvas.mpl_connect('pick_event', onpick)
    pyplot.show()

distpicker(dataX,dataY,lat,cat, M=10000)
#def score(function, input_parameters, samples=5000):
#    """  """
#    custom = dict(
#        movement_params=dict(),
#        movement_type='uniform',
#        snippet_length = samples,
#        number_of_snippets = 1,
#        blank_frame = False
#    )
#    
#    ss = sensory.SensorySystem(dict(input_parameters, **custom), save_input=False)
#    
#    seqX, cat, lat = ss.generate()
#    seqY = function(seqX)
#    
#    for (sx1, c1, l1, sy1), (sx2, c2, l2, sy2) in grouped(zip(seqX, cat, lat, seqY),2):
#        latdist = l1 - l2
    
        