"""
(Figure 2)

Generates input episodes and shows frames and a plot of latent variables.

"""

from core import system_params, input_params, sensory
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

if __name__=="__main__":

    SNLEN = 4  # How long the two snippets (one for T, one for L) are
    NPLOTS = 8  # How many individual frames to plot.

    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['lines.markersize'] = 12
    matplotlib.rcParams['lines.markeredgewidth'] = 2

    font = {'family' : 'Sans',
            'size'   : 18}

    matplotlib.rc('font', **font)

    PARAMETERS = system_params.SysParamSet()

    # parms = dict(number_of_snippets=1, snippet_length=100, movement_type='lissajous', movement_params=dict(a=6, b=2*np.pi, deltaX=3, omega=7, step=0.02),
    #              object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=0)

    parms = dict(number_of_snippets=1, snippet_length=SNLEN, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                 object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=0.2)

    # parms = dict(number_of_snippets=1, snippet_length=None, movement_type='timed_border_stroll', movement_params=dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=80, border_extent=2.3),
    #              object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=0)

    # parms = dict(number_of_snippets=1, snippet_length=100, movement_type='random_stroll', movement_params=dict(d2x=0.005, d2t = 0.009, dx_max=0.05, dt_max=0.1, step=1),
    #              object_code=input_params.make_object_code('TL'), sequence=[0,1], input_noise=0)

    sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

    seq, cat, lat = sensys.generate(**parms)

    cat = np.array(cat)
    lat = np.array(lat)

    x = range(len(cat))
    step = (SNLEN*2)//NPLOTS
    for i in range(NPLOTS):
        plt.subplot(2, NPLOTS, i+1)
        ind = step*(i+1)-step//2 - step % 2
        # plt.imshow(np.transpose(np.reshape(seq[ind], (30, 30))), interpolation='none', cmap="Greys")
        plt.imshow(np.reshape(seq[ind], (30, 30)), interpolation='none', cmap="Greys")
        plt.title(ind)
        plt.axis('off')
    plt.subplot(2,1,2)
    lc, = plt.plot(x, cat, 'k-', label="object identity")
    lx, = plt.plot(x, lat[:,1], 'k--', label="x-coordinate")
    ly, = plt.plot(x, -lat[:,0], 'k-.', label="y-coordinate")
    # plt.ylabel('coordinate', color='k')
    plt.ylim((-1.2,1.2))
    plt.yticks([-1,0,1])
    # plt.plot(x, lat[:,2], label="cosine of rotation angle")
    # plt.plot(x, lat[:,3], label="sine of rotation angle")
    # plt.twinx()
    # la, = plt.plot(x, -np.arctan2(lat[:,2], lat[:,3])*180/np.pi, 'g', label="rotation angle")
    # plt.ylabel('angle', color='g')
    # plt.ylim((-180, 180))
    # plt.yticks([-180, -90, 0, 90, 180])
    # plt.tick_params('y', colors='g')
    # plt.figlegend([lc, lx, ly, la], ["object category", "x-coordinate", "y-coordinate", "rotation angle"], 0)
    plt.legend()

    plt.show()