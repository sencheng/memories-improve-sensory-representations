import tools

import numpy as np
from matplotlib import pyplot

input_sequence_full = np.load("../results/data1_input.npy")
input_sequence1 = input_sequence_full[:3000]
input_sequence2 = input_sequence_full[3000:]

SFA1_output = np.load("../results/data1_SFA1out.npy")
print(SFA1_output.shape)

arrB = np.ones(3000)
arrA = -arrB
category_array = np.concatenate((arrA,arrB))
print(category_array.shape)

corr_vector = []
for i in range(SFA1_output.shape[1]):
    corr_vector.append(np.correlate(SFA1_output[:,i],category_array))
print(corr_vector)

# x = np.arange(6000)
# for k in range(SFA1_output.shape[1]/4):
#     for j in range(4):
#         pyplot.subplot(2,2,j)
#         pyplot.plot(x, SFA1_output[:,j])
#     pyplot.show()



tools.preview_input(input_sequence2, rate=10)