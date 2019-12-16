# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:57:14 2016

@author: richaopf
"""

    
    
import numpy as np
from matplotlib import pyplot 
import math

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))




def point_trailXY(preimage, DATA, meld=1, shape=(30,30), autoplay=1, trail=10) :
    index = [0]
    autoplay = [autoplay]
    
    permute = np.argsort(-np.abs(2*trail- np.arange(4*trail)))
    invperm = np.argsort(permute)
    
    def get_positions():
        subset = np.array(range(max(index[0]-2*trail, 0), min(index[0]+2*trail, len(DATA)-1)))
        subset.resize(4*trail)
    
        return DATA[subset,:], subset
    
    def plotall():
        vals, indices = get_positions()
        
        g = gaussian(np.arange(4*trail)[permute], 2*trail, trail)
        pts = ax.scatter(vals[permute,0], vals[permute,1], marker='o', alpha=0.6, cmap='Greens', zorder=1,  c=g, s=10+g*100)
        lines, =  ax.plot(vals[:,0], vals[:,1],color='0.75', zorder=2)
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
            adjust(1)
        elif event.key == 'down':
            adjust(-1)
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
    
def point_trail(DATA, meld=0.4):
    index = [10]
    speed = 2
    
    permute = np.argsort(-np.abs(2*speed- np.arange(4*speed)))
    invperm = np.argsort(permute)
    
    def get_positions():
        subset = np.array(range(max(index[0]-2*speed, 0), min(index[0]+2*speed, len(DATA)-1)))
        subset.resize(4*speed)
    
        return DATA[subset,:], subset
    
    def scat():
        vals, indices = get_positions()
        
        g = gaussian(indices[permute], index[0], speed)
        pts = ax.scatter(vals[permute,0], vals[permute,1], marker='o', alpha=0.6, cmap='Greens', zorder=1,  c=g, s=30+g*100)
        lines, =  ax.plot(vals[:,0], vals[:,1], zorder=2)
        return pts, lines
        
    I = np.arange(len(DATA))
    
    fig, ax = pyplot.subplots()
    allpts = ax.plot(DATA[:,0], DATA[:,1], 's', color='0.95', mec='0.8', zorder=0)
    tempscatter,templines = scat()
    
    def press(event):
        if event.key == 'left':
            index[0] = max(index[0]-1, 0)
        if event.key == 'right':
            index[0] = min(index[0]+1,len(DATA)-1)
    
    def update():
        valsto, ind = get_positions()
        valsnow = tempscatter.get_offsets()[invperm]
        
        newoff = (1-meld) * valsnow + meld*valsto
        tempscatter.set_offsets(newoff[permute,:])
        templines.set_data(newoff[:,0], newoff[:,1])
        fig.canvas.draw()
    
    
    fig.canvas.mpl_connect('key_press_event', press)
    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(update)
    timer.start()
    pyplot.show()

