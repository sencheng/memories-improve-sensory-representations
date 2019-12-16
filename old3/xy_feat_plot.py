from core import semantic, sensory, system_params, input_params, streamlined, tools

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

PATH = "/local/results/lro_o1850t/"
SFA1FILE = "sfa1.p"
SFA2FILE = "inc1_eps1_39.sfa"

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st2.update(dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('L'), sequence=[0], input_noise=0))

sfa1 = semantic.load_SFA(PATH+SFA1FILE)
sfa2 = semantic.load_SFA(PATH+SFA2FILE)
sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
seq, cat, lat = sensys.generate(**PARAMETERS.st2)

y = semantic.exec_SFA(sfa1, seq)
yw = streamlined.normalizer(y, PARAMETERS.normalization)(y)
z = semantic.exec_SFA(sfa2, yw)
zw = streamlined.normalizer(z, PARAMETERS.normalization)(z)

corr = tools.feature_latent_correlation(zw, lat, cat)
corrminus = -corr
xfeat = np.argmax(corr[0,:])
yfeat = np.argmax(corr[1,:])
xfeatminus = np.argmax(corrminus[0,:])
yfeatminus = np.argmax(corrminus[1,:])
xminus = yminus = False
if corr[0,xfeat] < corrminus[0,xfeatminus]:
    zw[xfeat] *= -1
if corr[1,yfeat] < corrminus[1,yfeatminus]:
    zw[yfeat] *= -1

fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))

# zw = lat
# xfeat = 0
# yfeat = 1

leng = len(zw[0])
rotmat = np.zeros((leng, leng))
rotmat[xfeat,yfeat] = 0
rotmat[yfeat,xfeat] = 0
rotmat[xfeat,xfeat] = 1
rotmat[yfeat,yfeat] = 1
zw = np.array([rotmat.dot(q) for q in zw])

def frame_gen():
    """Local function. Generator for input frames"""
    width = int(np.sqrt(len(seq[0])))
    for fr, fe in zip(seq,zw):
        yield [np.resize(fr, (width, width)), fe]

frames = frame_gen()
start = next(frames)
# Create initial objects
im = ax.imshow(start[0], cmap='Greys', vmin=0, vmax=1, extent=[-1,1,-1,1])
annotation = ax.scatter(start[1][xfeat], start[1][yfeat], s=500, c='red', marker='x')
annotation.set_animated(True)

# Create the update function that returns all the
# objects that have changed
def update(tup):
    newIm = tup[0]
    newFe = tup[1]
    im.set_data(newIm)
    # This is not working i 1.2.1
    # annotation.set_position((newData[0][0], newData[1][0]))
    annotation.set_offsets([newFe[xfeat], newFe[yfeat]])
    return im, annotation

anim = animation.FuncAnimation(fig, update, frame_gen, interval=100, blit=True)
plt.show()

# def compare_inputs(INP, rate=10):
#     global anim
#     fig = plt.figure()
#
#     def frame_gen():
#         """Local function. Generator for input frames"""
#         # width = int(np.sqrt(len(seq[0])))
#         # for fr, fe in zip(seq,zw):
#             # yield [np.resize(fr, (width, width)), fe]
#         for la, fe in zip(lat, zw):
#             yield [la, fe]
#
#     frames = frame_gen()
#     start = next(frames)
#     imI = []
#
#     for i, startI in enumerate(start):
#         imI.append(plt.subplot(1, len(INP), i + 1).matshow(startI)
#
#     def update(tup):
#         for data, image in zip(tup, imI):
#             image.set_data(data)
#
#     anim = animation.FuncAnimation(fig, update, frame_gen, interval=rate)
#     plt.show()
#     return anim
