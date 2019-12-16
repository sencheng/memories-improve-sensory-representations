# -*- coding: utf-8 -*-
"""
This module contains SFA parameter examples and information as well as few functions to generate
parameter sets.

The structure of a valid semantic parameter set is:
a list of tuples, each tuple representing a layer of the network,
where the first element is the name of the kind of layer (as string), and the second is a dictionary of parameters.
What the dictionary needs to contain is determined by the kind of layer chosen.

The kinds of layers currently available are:
 - layer.square
 - layer.square.parallel
 - layer.linear
 - single.square
 - single.square.parallel
 - single.linear
 - inc.linear
 - inc.square
 - inc.onesquare
 - inclayer.linear
 - inclayer.square

The term before the dot determines whether a receptive field structure is used and if SFA is batch or incremental:

================    ================    =============
       .            receptive field      single node
================    ================    =============
**batch**           layer               single
**incremental**     inclayer            inc
================    ================    =============

The term after the dot determines whether or not to use a quadratic expansion. When *square* is used,
a first SFA is followed by a quadratic expansion which outputs into a second SFA (see figure below, right side). A special case is
*inc.onesquare*, where the first SFA is omitted (because otherwise training is not stritly incremental,
as the first SFA is fully trained before the second can be trained).

However, it is easy to extend this to other kinds of layers and topologies.

-------------
Example
-------------
on how SFA networks can be constructed in this framework: Consider the SFA network from
Fang et al. 2018.

Fang, J., Rüther, N., Bellebaum, C., Wiskott, L., & Cheng, S. (2018).
The Interaction between Semantic Representation and Episodic Memory.
Neural Computation, 30(2), 293–332. https://doi.org/10.1162/neco_a_01044

The following parameter list is required to reproduce the SFA struture from that publication,
a parameter list that is also returned by the function :py:func:`make_jingnet`::

   [
        ('layer.square', {
            'bo':               30,
            'rec_field_ch':     15,
            'spacing':          3,
            'in_channel_dim':   1,
            'out_sfa_dim1':     48,
            'out_sfa_dim2':     32
        }),
        ('layer.square', {
            # <bo>=  (bo-rec_field_ch)/spacing+1 := 6
            'rec_field_ch':     4,
            'spacing':          2,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     32
        }),
        ('single.square', {
            'dim_in':      128,
            'dim_mid':     48,
            'dim_out':     4
        })
    ]

This generates the SFA network that is illustrated in the following figure.

.. image:: ./img/sfa_struct.svg
   :width: 1000px
   :alt: This is an alternative text
   :align: center

*layer.square* generates a layer of batch SFA nodes with receptive fields. This is used two times, so two consecutive
layers of these nodes (1. blue, 2. red) are generated. *single.square* generates a single SFA node
at the top (black dot). Because *square* was chosen, all SFA nodes in all three layers consist of a linear SFA,
a quadratic expansion and a second linear SFA (see right side of the figure). The parameters that determine input and output dimensionalities of
those SFAs are different between *layer.square* (red/blue text) and *single.square* (black text).

Mind that ``'in_channel_dim'`` and ``'bo'`` are only required in the first layer (then channels are usually pixels,
thus ``'in_channel_dim' = 1`` and ``'bo'`` is the width in pixels of the input images). On later levels
these parameters can be applied automatically.

------------------
More examples
------------------

The definitions used in the paper were the following::

    sfa1 = [
            ('layer.linear', {
                'bo': 30,
                'rec_field_ch': 18,
                'spacing': 6,
                'in_channel_dim': 1,
                'out_sfa_dim': 32
            })
        ]

    sfa2 = [
            ('inc.linear', {
                'dim_in': 288,
                'dim_out': 16
            })
        ]

In the following, more examples on how to set parameters are listed::

    params2_inc = [
        ('inc.linear', {
            'dim_in':   144,
            'dim_out':  16
        })
    ]

    params2_inc = [
        ('inc.square', {
            'dim_in':   144,
            'dim_mid':  48
            'dim_out':  16
        })
    ]

    params1_complete = [
        ('layer.square', {
            'bo':               30,
            'rec_field_ch':     15,
            'spacing':          5,
            'in_channel_dim':   1,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     8
        }),
        ('layer.square', {
            # <bo> = (bo-rec_field_ch)/spacing+1
            'rec_field_ch':     2,
            'spacing':         2,
            #'in_channel_dim':     # set to out_sfa_dim2 of the previous layer
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     8
        }),
        ('single.square', {
            #'in_sfa_dim':         set to upper_sfa_layer.output_dim in the code
            'dim_mid':     8,
            'dim_out':     36
        })
    ]
    params2_complete = [
        ('layer.square', {
            'bo':               6,
            'rec_field_ch':     4,
            'spacing':          1,
            'in_channel_dim':   1,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     8
        }),
        ('layer.square', {
            # <bo> = (bo-rec_field_ch)/spacing+1
            'rec_field_ch':     1,
            'spacing':         1,
            #'in_channel_dim':     # set to out_sfa_dim2 of the previous layer
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     8
        }),
        ('layer.square', {
            #'in_sfa_dim':          set to upper_sfa_layer.output_dim in the code
            'dim_mid':     8,
            'dim_out':     4
        })
    ]

    params1_square = [
        ('layer.square', {
            'bo':               30,
            'rec_field_ch':     14,
            'spacing':          4,
            'in_channel_dim':   1,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     8
        }),
        ('layer.square', {
            # <bo> = (bo-rec_field_ch)/spacing+1
            'rec_field_ch':     2,
            'spacing':         1,
            #'in_channel_dim':     # set to out_sfa_dim2 of the previous layer
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     4
        })
    ]
    params2_square = [
        ('layer.square', {
            'bo':               4,
            'rec_field_ch':     2,
            'spacing':          1,
            'in_channel_dim':   4,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     8
        }),
        ('single.square',  {
            #'in_sfa_dim':          set to upper_sfa_layer.output_dim in the code
            'dim_mid':     8,
            'dim_out':     4
        })
    ]


    params1_simple = [
        ('layer.square', {
            'bo':               30,
            'rec_field_ch':     15,
            'spacing':          5,
            'in_channel_dim':   1,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     8
        }),
        ('single.square', {
            #'in_sfa_dim':         set to upper_sfa_layer.output_dim in the code
            'dim_mid':     8,
            'dim_out':     8
        })
    ]
    params2_simple = [
        ('single.square', {
            'dim_in': 8,
            'dim_mid': 8,
            'dim_out': 4
        })
    ]

    params2_double = [
        ('single.square', {
            'dim_in': 48,
            'dim_mid': 32,
            'dim_out': 32
        }),
        ('single.square', {
            'dim_in': 32,
            'dim_mid': 16,
            'dim_out': 8
        })
    ]

--------------
Functions
--------------
"""

def make_single_layer(dim_in, dim_mid, dim_out) :
    """
    get a parameter list for an SFA module with one layer of a single quadratic batch SFA

    :param dim_in: dimensionality of input
    :param dim_mid: dimensionality of data before quadratic expansion
    :param dim_out: dimensionality of output
    :return: parameter list
    """
    params = {}
    for key, value in locals().items():
        if value is not None: params[key] = value

    del params['params']
    return [ ('single', params) ]

def make_layer_series(*numbers) :
    """
    get a parameter list for an SFA module with arbitrary number of layers of single quadratic batch SFA

    :param numbers: pass in numbers in groups of three: (dim_in, dim_mid, dim_out) for each layer
    :return: parameter list
    """
    if len(numbers) % 3 is not 0:
        raise ValueError('make_params must be used with a number of parameters '
                         'divisible by three: (in, mid, and out) for each layer')
    
    layers = []
    for i in range(0,len(numbers),3) :
        layers.append(('single', dict(dim_in=numbers[i], dim_mid=numbers[i+1], dim_out=numbers[i+2])))
    
    return layers

def make_jingnet(outdim):
    """
    get a parameter list for an SFA module that was used in Fang et al. 2018

    Fang, J., Rüther, N., Bellebaum, C., Wiskott, L., & Cheng, S. (2018).
    The Interaction between Semantic Representation and Episodic Memory.
    Neural Computation, 30(2), 293–332. https://doi.org/10.1162/neco_a_01044

    :param outdim: dimensionality of output
    :return: parameter list
    """
    return [
        ('layer.square', {
            'bo':               30, 
            'rec_field_ch':     15,
            'spacing':          3,
            'in_channel_dim':   1,
            'out_sfa_dim1':     48,
            'out_sfa_dim2':     32
        }),
        ('layer.square', {
            # <bo>=  (bo-rec_field_ch)/spacing+1 := 6
            'rec_field_ch':     4,
            'spacing':          2,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     32
        }),
        ('single.square', {
            'dim_in':      128,
            'dim_mid':     48,
            'dim_out':     outdim
        })
    ]
