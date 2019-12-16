from core import semantic, tools, streamlined
import numpy as np
from matplotlib import pyplot as plt

# This script randomly generates a sequence of two variables x,y. d = x-y varies slowly while s = x+y varies more quickly.
# s is chosen in a way to have a mean >> 0 and a variance >> 1
# Hence, whitening has a substantial influence on the x,y-sequence
# Ability of incremental sfa to extract d is tested on the raw and on the whitened sequence
# training is repeated 20 times and delta and correlation of the first feature is plotted.


seqlen = 1000
stepsize = 0.01

inc_params = [
        ('inc.linear', {
            'dim_in': 2,
            'dim_out': 2
        })
    ]

sfa1 = semantic.build_module(inc_params, eps=0.08)
sfa2 = semantic.build_module(inc_params, eps=0.08)

a = 1
b = 100

t = np.arange(seqlen)/stepsize
d = np.sin(a*t)
s = 1000*np.sin(b*t)+200
# var_diff = [np.random.rand()*10-5]
# var_sum = [np.random.rand()*100+10000]
#
# for i in range(seqlen):
#     var_diff.append(var_diff[-1]+np.random.rand()*0.2-0.1)
#     var_sum.append(var_sum[-1] + np.random.rand()*200-100)
#
# d = np.array(var_diff)
# s = np.array(var_sum)

y = ((s-d)/2)+np.random.randn(len(s))*100
x = s-y+np.random.randn(len(s))*100

seq = np.vstack((x,y)).T
seqW = streamlined.normalizer(seq, "whiten.ZCA")(seq)

def printStuff(inp, title=""):
    print("==================")
    print(title)

    print("mean", np.mean(inp))
    print("var", np.var(inp))

    print("meanX", np.mean(inp[:,0]))
    print("meanY", np.mean(inp[:,1]))
    print("varX", np.var(inp[:,0]))
    print("varY", np.var(inp[:,1]))

    print("meanD", np.mean(inp[:,0]-inp[:,1]))
    print("varD", np.var(inp[:,0]-inp[:,1]))
    print("meanS", np.mean(inp[:,0]+inp[:,1]))
    print("varS", np.var(inp[:,0]+inp[:,1]))

printStuff(seq, "seq")
printStuff(seqW, "Whitened")

d_out1 = []
d_out1W = []
d_out2 = []
d_out2W = []
r1 = []
r1W = []
r2 = []
r2W = []

runs = np.arange(20)

for t in runs:
    semantic.train_SFA(sfa1, seq)
    semantic.train_SFA(sfa2, seqW)

    out1 = semantic.exec_SFA(sfa1, seq)
    out2 = semantic.exec_SFA(sfa2, seqW)

    out1W = streamlined.normalizer(out1, "whiten.ZCA")(out1)
    out2W = streamlined.normalizer(out2, "whiten.ZCA")(out2)

    d_out1.append(tools.delta_diff(out1))
    d_out1W.append(tools.delta_diff(out1W))
    d_out2.append(tools.delta_diff(out2))
    d_out2W.append(tools.delta_diff(out2W))

    r1.append(np.corrcoef(np.vstack((out1[:,0], d)))[1,0])
    r1W.append(np.corrcoef(np.vstack((out1W[:,0], d)))[1,0])
    r2.append(np.corrcoef(np.vstack((out2[:,0], d)))[1,0])
    r2W.append(np.corrcoef(np.vstack((out2W[:,0], d)))[1,0])
    
d_out1 = np.array(d_out1)
d_out1W = np.array(d_out1W)
d_out2 = np.array(d_out2)
d_out2W = np.array(d_out2W)

r1 = np.abs(np.array(r1))
r1W = np.abs(np.array(r1W))
r2 = np.abs(np.array(r2))
r2W = np.abs(np.array(r2W))
#
# r1 = np.array(r1)
# r1W = np.array(r1W)
# r2 = np.array(r2)
# r2W = np.array(r2W)

plt.figure()
plt.subplot(1,2,1)
plt.plot(runs, d_out1[:,0],  marker = '+',label="out1")
plt.plot(runs, d_out1W[:,0],  marker = '+',label="out1W")
plt.plot(runs, d_out2[:,0], label="out2")
plt.plot(runs, d_out2W[:,0], label="out2W")
plt.ylabel("first delta")
plt.legend()

plt.subplot(1,2,2)
plt.plot(runs, r1,  marker = '+',label="r1")
plt.plot(runs, r1W,  marker = '+',label="r1W")
plt.plot(runs, r2, label="r2")
plt.plot(runs, r2W, label="r2W")
plt.ylabel("first corr")
plt.legend()

plt.show()
