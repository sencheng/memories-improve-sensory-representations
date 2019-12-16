"""
Here the parameters for the simulation runs started with :py:mod:`grid` are defined.
All parameters are in here, except for the sfa definitions. For those, sfadef files have
to be put in the *gridconfig* folder, see :py:mod:`gridconfig`.

This file defines the stage dictionaries (*st1*, *st2*, *st3*, *st4*) known from
:py:class:`core.system_params.SysParamSet` as well as two additional parameters that are explained below.

As opposed to a normal parameter set, the dictionary values (except for *st3*) have to be lists than can have
one or multiple elements. Generally, a simulation run is started for every combination of parameter values.
For instance, if all parameter lists have just one element, a single simulation run is started. If two of the lists have
two elements each, this would result in four simulation runs. You get the idea.
But there are exceptions, because input generation does not allow arbitrary parameter value combinations,
for example each *movement_type* requires a specific *movement_params* dict. See below.

"""
from core import input_params
import numpy as np

# WARNING: THIS ONLY WORKS WHEN PARAMETERS FOR ST1 AND ST4 ARE NOT MISSING
type_match = False
"""
If True, a simulation is only run for combinations where ``'movement_type'`` is the same for st1, st2 and st4
"""
param_match = False
"""
If True, a simulation is only run for combinations where ``'movement_params'`` is the same for st1, st2 and st4
"""

st1 = dict([])
"""
Parameters for training input generation. Not all combinations are possible but all parameters get zipped -
that means all parameter lists have to have the same number of elements and for each index
one parameter combination is generated. For instance::

   st1['movement_type'] = ['gaussian_walk', 'random_stroll']
   st1['movement_params'] = [dict(dx = 0.05, dt = 0.05, step=5), 
                            dict(d2x=0.005, d2t = 0.009, dx_max=0.05, dt_max=0.1, step=5)]

would lead to two different parameter combinations, one containing both first list elements and one
containing both second list elements.
"""

st2 = dict([])
"""
Can contain parameters for input generation, the two st2 parameters specific to *streamlined*
(``sfa2_noise'`` and ``'number_of_retrieval_trials'``) and the ``'memory'`` dictionary for
*EpisodicMemory* initialization.

Among the parameters for input generation, ``'movement_type'``, ``'movement_params'``, ``'snippet_length'``,
``'number_of_snippets'``, ``'background_params'`` and ``'sequence'`` are zipped, as in :py:data:`st1`.
That means the parameter lists have to have the same length. The first four are required, the latter
two can be left out. Additionally, the parameter ``'input_noise'`` can be supplied - it is used
independent of the aforementioned, such that all combinations are possible. For instance::

   st2['movement_type'] = ['gaussian_walk', 'random_stroll']
   st2['movement_params'] = [dict(dx = 0.05, dt = 0.05, step=5),
                            dict(d2x=0.005, d2t = 0.009, dx_max=0.05, dt_max=0.1, step=5)]
   st2['snippet_length'] = [50, 50]
   st2['number_of_snippets'] = [50, 50]
   
   st2['input_noise'] = [0.0, 0.1, 0.5]
   
would lead to six different parameter combinations, two from the upper parameter group that each get combined
with all three of ``'input_noise'``.

.. note:: Only the mentioned input parameters have an effect in st2. For instance, setting values for ``st2['object_code']``
          will have no effect. Has to be changed in :py:mod:`core.system_params`. However, in :py:data:`st1`
          and :py:data:`st4`, all input parameters can be set.

The *streamlined*-specific parameters and the elements of the ``'memory'`` dictionary are all evaluated in every possible
combination. See :py:class:`core.system_params.SysParamSet` for more information on the parameters.

"""

st3 = dict([])
"""
Single parameters are provided that just overwrite defaults from :py:class:`core.system_params.SysParamSet`.
Leaving *st3* in this file out and changing the values in :py:mod:`core.system_params` would have the exact same effect.
Having *st3* dict here is just for convenience.
"""

st4 = dict([])
"""
Same as :py:data:`st1`.
"""

st2b = None
"""
Second set of forming data, if not None.
In case a set of forming data is supposed to contained different trajectories etc.
This only needs to have input parameters, with lists of the same length as in st2 because st2b data is
appended to the corresponding st2 data. I do not know whether this function is useful
"""

st4b = None
"""
Second set of testing data, if not None. Parameter lists have to be the same length as in st4, because
st4b data is paired with st4 data.
"""

nsnip = 600*25   #number of frames per object. total number is actually this x2 (because we have two objects)

# movement_type, movement_params, snippet_length, number_of_snippets, background_params and sequence get zipped tuple-wise
# The first 4 are required, the latter two are optional
st2['snippet_length'] = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
# st2['snippet_length'] = [2,3,5,8,12,20,40,100,200,600]
st2['movement_type'] = ['gaussian_walk']*len(st2["snippet_length"])
st2['movement_params'] = [dict(dx = 0.05, dt = 0.05, step=5)]*len(st2["snippet_length"])
st2["number_of_snippets"] = list(nsnip//np.array(st2['snippet_length']))
# st2['snippet_length'] = [2,5,10,30,80,200,600]
# st2["number_of_snippets"] = list((np.array([300,120,60,20,7.5,3,1])*10).astype(int))

st2['input_noise'] = [0.1]
st2['number_of_retrieval_trials'] = [(2*nsnip)//80]
st2['sfa2_noise'] = [0]

st2['memory'] = dict([])
st2['memory']['retrieval_length'] = [80]
st2['memory']['category_weight'] = [0]
st2['memory']['retrieval_noise'] = [0.2]
st2['memory']['depress_params'] = [dict(cost=400, recovery_time_constant=400, activation_function='lambda X : X')]
st2['memory']['smoothing_percentile'] = [100]
st2['memory']['return_err_values'] = [False]

# all st1 params get zipped tuple-wise
st1['movement_type'] = ['gaussian_walk']
st1['movement_params'] = [dict(dx = 0.05, dt = 0.05, step=5)]
st1['snippet_length'] = [50]

# all st4 params get zipped tuple-wise
st4['movement_type'] = ['gaussian_walk']
st4['movement_params'] = [dict(dx=0.05, dt=0.05, step=5)]
st4['number_of_snippets'] = [50]
st4['snippet_length'] = [50]

st3['retr_repeats'] = 1
st3['inc_repeats_S'] = 1
st3['inc_repeats_E'] = 1
st3['learnrate'] = 0.0005
st3['cue_equally'] = False
st3['use_memory'] = True