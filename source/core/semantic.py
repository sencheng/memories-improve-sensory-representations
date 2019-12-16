# -*- coding: utf-8 -*-
"""
This module is used to construct, train, run and save SFA modules.

"""

import math
import pickle
import warnings

import mdp
import numpy as np

from . import tools

from .incsfa import trainer
from .incsfa import incsfa


################################ METHODS FOR BUILDING SFA MODULES ############################

def _sfa_flow_node_linear(dim_in, dim_out):
    """generate Flow Node consisting of SFA
    :param dim_in: input dimensionality of SFA.
    :param dim_out: output dimensionality of SFA.
    :returns: Flow Node consisting of SFA
    """
    sfa_node1 = mdp.nodes.SFANode(input_dim=dim_in, output_dim=dim_out)
    flow_node = mdp.hinet.FlowNode(mdp.Flow([sfa_node1]))
    return flow_node

def _switchboard_based_layer_linear( out_sfa_dim, bo, rec_field_ch, spacing, in_channel_dim):
    """Generate switchboard driving several SFA Flow Nodes that the input is distributed among.
    :param out_sfa_dim: output dimensionality of SFA
    :param bo: number of input channels along one side of square input field. For instance side length of input image.
    :param rec_field_ch: number of input channels along one side of a square receptive field. One SFA node is one receptive field.
    :param spacing: spacing between two adjacent receptive fields.
    :param in_channel_dim: number of dimensions in a single input channel.
    :returns: tuple (switchboard, SFA_layer)
    """
    switchboard = mdp.hinet.Rectangular2dSwitchboard(   in_channels_xy = (bo, bo),
                                                        field_channels_xy = (rec_field_ch, rec_field_ch),
                                                        field_spacing_xy =(spacing, spacing),
                                                        in_channel_dim = in_channel_dim)

    flow_node = _sfa_flow_node_linear(switchboard.out_channel_dim, out_sfa_dim)
    sfa_layer = mdp.hinet.CloneLayer(flow_node, n_nodes=switchboard.output_channels)

    return switchboard, sfa_layer

def _sfa_flow_node(dim_in, dim_mid, dim_out):
    """generate Flow Node consisting of SFA - Quadratic Expansion - SFA.
    :param dim_in: input dimensionality of first SFA.
    :param dim_mid: output dimensionality of first SFA.
    :param dim_out: output dimensionality of second SFA.
    :returns: Flow Node consisting of SFA - Quadratic Expansion - SFA
    """
    sfa_node1 = mdp.nodes.SFANode(input_dim=dim_in, output_dim=dim_mid)
    sfa2_node = mdp.nodes.QuadraticExpansionNode(input_dim=dim_mid)
    #noi_node = mdp.nodes.NoiseNode(input_dim = sfa2node.output_dim,noise_args=(0,sqrt(0.0005)))   #test#
    sfa_node2 = mdp.nodes.SFANode(input_dim=sfa2_node.output_dim, output_dim=dim_out)

    flow_node = mdp.hinet.FlowNode(mdp.Flow([sfa_node1, sfa2_node, sfa_node2]))
    #flownode = mdp.hinet.FlowNode(mdp.Flow([sfanode,sfa2node,noi_node,sfanode2])) #test#

    return flow_node

def _sfa_flow_node_onexp(dim_in, dim_out):
    """generate Flow Node consisting of SFA - Quadratic Expansion - SFA.
    :param dim_in: input dimensionality of first SFA.
    :param dim_mid: output dimensionality of first SFA.
    :param dim_out: output dimensionality of second SFA.
    :returns: Flow Node consisting of SFA - Quadratic Expansion - SFA
    """
    sfa2_node = mdp.nodes.QuadraticExpansionNode(input_dim=dim_in)

    #noi_node = mdp.nodes.NoiseNode(input_dim = sfa2node.output_dim,noise_args=(0,sqrt(0.0005)))   #test#
    sfa_node2 = mdp.nodes.SFANode(input_dim=sfa2_node.output_dim, output_dim=dim_out)

    flow_node = mdp.hinet.FlowNode(mdp.Flow([sfa2_node, sfa_node2]))
    #flownode = mdp.hinet.FlowNode(mdp.Flow([sfanode,sfa2node,noi_node,sfanode2])) #test#

    return flow_node

def _switchboard_based_layer( out_sfa_dim1, out_sfa_dim2, bo, rec_field_ch, spacing, in_channel_dim):
    """Generate switchboard driving several SFA Flow Nodes that the input is distributed among.
    :param out_sfa_dim1: output dimensionality of first SFA of each SFA Flow Node.
    :param out_sfa_dim2: output dimensionality of second SFA of each SFA Flow Node.
    :param bo: number of input channels along one side of square input field. For instance side length of input image.
    :param rec_field_ch: number of input channels along one side of a square receptive field. One SFA Flow node is one receptive field.
    :param spacing: spacing between two adjacent receptive fields.
    :param in_channel_dim: number of dimensions in a single input channel.
    :returns: tuple (switchboard, SFA_layer)
    """
    switchboard = mdp.hinet.Rectangular2dSwitchboard(   in_channels_xy = (bo, bo),
                                                        field_channels_xy = (rec_field_ch, rec_field_ch),
                                                        field_spacing_xy =(spacing, spacing),
                                                        in_channel_dim = in_channel_dim)

    flow_node = _sfa_flow_node(switchboard.out_channel_dim, out_sfa_dim1, out_sfa_dim2)
    sfa_layer = mdp.hinet.CloneLayer(flow_node, n_nodes=switchboard.output_channels)

    return switchboard, sfa_layer
    
#parallel versions================================
def _sfa_flow_node_parallel( in_sfa_dim, out_sfa_dim1, out_sfa_dim2):
    """parallel version of _SFA_flow_node. Only Flows and FlowNodes are parallelized.
    """
    sfa_node1 = mdp.nodes.SFANode(input_dim=in_sfa_dim, output_dim=out_sfa_dim1)
    sfa2_node = mdp.nodes.QuadraticExpansionNode(input_dim=out_sfa_dim1)
    #noi_node = mdp.nodes.NoiseNode(input_dim = sfa2node.output_dim,noise_args=(0,sqrt(0.0005)))   #test#
    sfa_node2 = mdp.nodes.SFANode(input_dim=sfa2_node.output_dim, output_dim=out_sfa_dim2)

    flow_node = mdp.parallel.ParallelFlowNode(mdp.parallel.ParallelFlow([sfa_node1, sfa2_node, sfa_node2]))
    #flownode = mdp.hinet.FlowNode(mdp.Flow([sfanode,sfa2node,noi_node,sfanode2])) #test#

    return flow_node    

def _switchboard_based_layer_parallel(self, out_sfa_dim1, out_sfa_dim2, bo, rec_field_ch, spacing, in_channel_dim):
    """parallel version of _switchboard_based_layer. Only Flows and FlowNodes are parallelized.
    """
    switchboard = mdp.hinet.Rectangular2dSwitchboard(   in_channels_xy = (bo, bo),
                                                        field_channels_xy = (rec_field_ch, rec_field_ch),
                                                        field_spacing_xy =(spacing, spacing),
                                                        in_channel_dim = in_channel_dim)

    flow_node = _sfa_flow_node_parallel(switchboard.out_channel_dim, out_sfa_dim1, out_sfa_dim2)
    sfa_layer = mdp.parallel.ParallelCloneLayer(flow_node, n_nodes=switchboard.output_channels)

    return switchboard, sfa_layer


""" Mapping from names of layer types --> methods for creating them """
_LOOKUP = {
    'layer.square' : _switchboard_based_layer,
    'layer.linear' : _switchboard_based_layer_linear,
    'layer.square.parallel':  _switchboard_based_layer_parallel,
    'single.square' : _sfa_flow_node,
    'single.onesquexp' : _sfa_flow_node_onexp,
    'single.linear' : _sfa_flow_node_linear,
    'single.square.parallel' : _sfa_flow_node_parallel
    # LOOKUP is not used for:
    # 'inc.linear'
    # 'inc.square'
    # ...
}


class ExampleSFANode(object):
    def __init__(self, **parms):
        """
        Any SFANode generated by :py:func:`build_module` has the same interface.
        However, this example class is not used as a base class, because the modules for batch SFA
        are imported from ``mdp``, while the incremental modules are defined in this module
        using the incremental SFA from :py:mod:`core.incsfa`.

        .. note:: This example class is not meant to be instantiated. It is just here for documentation
                  of the SFA module interface.

        :param parms: Some parameters. Initialization of the modules is taken care of
                      by :py:func:`build_module`.
        """
        pass

    def execute(self, x):
        """
        Run input through the trained SFA

        :param x: input sequence to execute SFA on
        :return: SFA features
        """
        return 0

    def train(self, x):
        """
        Train SFA

        :param x: Training sequence
        """
        pass

    def save(self, filename):
        """
        Pickle SFA module to file

        :param filename: Path to save module to
        """
        pass

    def __getitem__(self, key):
        """
        Functionality to index module. Useful if module consists of several layers.

        :param key: Index
        """
        return self.layers[key]


class incflow(object):
    def __init__(self, layer_list):
        self.layers = layer_list
        self.trained = False
        # print("Generated incFlow with layer_list: " + str(self.layers))
        # print("Layer_list has length {}".format(len(self.layers)))

    def execute(self, x):
        ex = x
        for lay in self.layers:
            ex = lay.execute(ex)
        return ex

    def train(self, x):
        ex = x
        for l, lay in enumerate(self.layers):
            # print("Training layer {}".format(l))
            lay.train(ex)
            ex = lay.execute(ex)
        self.trained = True

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def __getitem__(self, key):
        return self.layers[key]


class inctuple(object):
    def __init__(self, node, trainer):
        self._node = node
        self._trainer = trainer
        self.trained = False
        self.output_dim = self._node.output_dim

    def execute(self, x):
        return self._node.execute(x)

    def train(self, x):
        self._trainer.train(x)
        self.trained = True

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    # Overload getitem here. In streamlined sfa[-1].output_dim is accessed. Not-inc SFAs are Flows, so we just implement indexing here.
    def __getitem__(self, key):
        return [self._node][key]


class hierarchical_incsfa():
    def __init__(self, params):
        self.bo = params['bo']
        self.rec_field_ch = params['rec_field_ch']
        self.spacing = params['spacing']
        self.in_channel_dim = params['in_channel_dim']
        self.in_dim = self.rec_field_ch * self.rec_field_ch * self.in_channel_dim
        if 'out_sfa_dim' in params:    # we have linear sfa here
            self.out_sfa_dim = params['out_sfa_dim']
            _node = incsfa.IncSFANode(self.in_dim, self.in_dim, self.out_sfa_dim)
            _trainer = trainer.TrainerNode(_node)
            self.node = inctuple(_node, _trainer)
        else:   # we have quadratic expansion sfa here. we use SFA-QE-SFA structure here (SFA-SFA2) analogous to bSFA version instead of just SFA2 (see Varun's paper)
            self.out_sfa_dim1 = params['out_sfa_dim1']
            self.out_sfa_dim2 = params['out_sfa_dim2']
            _node1 = incsfa.IncSFANode(self.in_dim, self.in_dim, self.out_sfa_dim1)
            _trainer1 = trainer.TrainerNode(_node1)
            _node2 = incsfa.IncSFA2Node(self.out_sfa_dim1, self.out_sfa_dim1, self.out_sfa_dim2)
            _trainer2 = trainer.TrainerNode(_node2)
            tuple1 = inctuple(_node1, _trainer1)
            tuple2 = inctuple(_node2, _trainer2)
            self.node = incflow([tuple1, tuple2])
        self.next_bo = (self.bo-self.rec_field_ch)//self.spacing+1

    def execute(self,x):
        res_list = []
        for x_field in range(self.next_bo):  # iterate through all receptive fields = "move ROI on the input image/data"
            for y_field in range(self.next_bo):
                slice_begin = self.in_channel_dim*(self.bo*self.spacing*y_field+self.spacing*x_field)
                slice_end = slice_begin+self.in_channel_dim*self.rec_field_ch
                slices = []    # since one data point (first dimension of x is time, unless x is just one data point) is represented as one-dimensional array, we need to find the correct slice indices
                for isl in range(self.rec_field_ch):
                    slices.append([slice_begin + isl*self.in_channel_dim*self.bo, slice_end + isl*self.in_channel_dim*self.bo])
                data = np.concatenate([x[:,iin:iout] for iin, iout in slices],axis=1) if len(np.shape(x)) > 1 else np.concatenate([x[iin:iout] for iin, iout in slices])
                res_list.append(self.node.execute(data))
        return np.concatenate(res_list, axis=len(np.shape(res_list[0]))-1)

    def train(self, x):
        for x_field in range(self.next_bo):
            for y_field in range(self.next_bo):
                slice_begin = self.in_channel_dim * (self.bo * self.spacing * y_field + self.spacing * x_field)
                slice_end = slice_begin + self.in_channel_dim * self.rec_field_ch
                slices = []
                for isl in range(self.rec_field_ch):
                    slices.append([slice_begin + isl * self.in_channel_dim * self.bo, slice_end + isl * self.in_channel_dim * self.bo])
                data = np.concatenate([x[:, iin:iout] for iin, iout in slices], axis=1) if len(np.shape(x)) > 1 else np.concatenate([x[iin:iout] for iin, iout in slices])
                self.node.train(data)

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    # Overload getitem here. In streamlined sfa[-1].output_dim is accessed. Not-inc SFAs are Flows, so we just implement indexing here.
    def __getitem__(self, key):
        return [self.node][key]


def _make_network ( layer_params, last_dimension=None, silent_fix=False, flow_maker=mdp.Flow, eps=0.001):
    """ creates an SFA module from given parameters

    :param layer_params: parameters, represented as a list of tupples ('type', parameter_dict)
    :param last_dimension: the input dimension for this network if not explicitly in parameters
    :param silent_fix: silently smooth, by overwriting with correct params instead of throwing errors.
    :param flow_maker: to preserve whatever parallel functionality there once was, pass in  mdp.parallel.ParallelFlow.
    :returns: Flow node representing an SFA layer
    """
    layer_list = []
    # we are already keeping track of last_dimension from arguments;
    last_channel_dim = None
    
    def checkdim(descr, shouldbe, isnow):
        if isnow is None and shouldbe is None:
            raise ValueError('First dimension was undefined and there was no previous layer to calculate it from')

        if shouldbe is not None:        
            if (isnow is not None) and (isnow != shouldbe) and (not silent_fix) :
                raise ValueError('Mismatch dimensions; forced {} to {}, whereas {} was required from previous layer'.format(descr, isnow, shouldbe ))
            isnow = shouldbe        
        
        return isnow

    
    #LOWER LAYERS
    for idx, (flavour, params) in enumerate(layer_params):
        if flavour.startswith('layer'):
            params['in_channel_dim'] = checkdim('<in_channel_dim>', last_channel_dim, params.get('in_channel_dim'))
            desired_bo = None if last_dimension is None else math.sqrt(last_dimension/float(params['in_channel_dim']))
            params['bo'] = int(checkdim('<bo>', desired_bo, params.get('bo')))
            
            if not params['bo'] == int(params['bo']):  #needs to be integer
                raise ValueError('Calculated side length was not an integer in layer ' + str(idx)+"; "+str(params['bo']))
            
            if params['rec_field_ch'] > params['bo']:
                if silent_fix: params['rec_field_ch'] = params['bo']
                else: raise ValueError('rec_field_ch greater than bo in layer ' + str(idx))
            
            try:
                last_channel_dim = params['out_sfa_dim2']
            except:
                last_channel_dim = params['out_sfa_dim']
            potential_next_bo = (params['bo']-params['rec_field_ch'])/float(params['spacing'])+1
            last_dimension = (potential_next_bo**2)*last_channel_dim
            
            #create actual layer and add to list
            layer_maker = _LOOKUP[flavour]
            switchboard, sfa_layer = layer_maker(**params)   
            layer_list.extend([switchboard, sfa_layer])
            
        elif flavour.startswith('single') :
            #params['dim_in'] = int(checkdim('dim_in', last_dimension, params.get('dim_in')))

            try:
                if params['dim_mid'] > params['dim_in'] :
                    if silent_fix:
                        params['dim_mid'] = params['dim_in']
                    else:
                        raise ValueError('Middle dimension of sfa sandwich must not be greater than its input dimension.')
            except:
                pass
            
            flownode_maker = _LOOKUP[flavour]
            top_sfa_node = flownode_maker(**params)
            layer_list.append(top_sfa_node)

        elif flavour.startswith('inc.linear'):
            _node = incsfa.IncSFANode(params['dim_in'], params['dim_in'], params['dim_out'], eps=eps)
            _trainer = trainer.TrainerNode(_node)
            layer_list.append(inctuple(_node, _trainer))

        elif flavour.startswith('inc.onesquare'):
            _node = incsfa.IncSFA2Node(params['dim_in'], params['dim_in'], params['dim_out'], all_expand=False, eps=eps)
            _trainer = trainer.TrainerNode(_node)
            layer_list.append(inctuple(_node, _trainer))

        elif flavour.startswith('inc.square'):
            _node = incsfa.IncSFANode(params['dim_in'], params['dim_in'], params['dim_mid'], eps=eps)
            _trainer = trainer.TrainerNode(_node)
            _node2 = incsfa.IncSFA2Node(params['dim_mid'], params['dim_mid'], params['dim_out'], all_expand=False, eps=eps)
            _trainer2 = trainer.TrainerNode(_node2)
            layer_list.append(incflow([inctuple(_node, _trainer), inctuple(_node2, _trainer2)]))

        elif flavour.startswith('inc.onesquexp'):
            _node = incsfa.IncSFA2Node(params['dim_in'], params['dim_in'], params['dim_out'], all_expand=True, eps=eps)
            _trainer = trainer.TrainerNode(_node)
            layer_list.append(inctuple(_node, _trainer))

        elif flavour.startswith('inc.squexp'):
            _node = incsfa.IncSFANode(params['dim_in'], params['dim_in'], params['dim_mid'], eps=eps)
            _trainer = trainer.TrainerNode(_node)
            _node2 = incsfa.IncSFA2Node(params['dim_mid'], params['dim_mid'], params['dim_out'], all_expand=True, eps=eps)
            _trainer2 = trainer.TrainerNode(_node2)
            layer_list.append(incflow([inctuple(_node, _trainer), inctuple(_node2, _trainer2)]))

        elif flavour.startswith('inclayer'):
            layer_list.append(hierarchical_incsfa(params))


    if not flavour.startswith('inc'):
        network = flow_maker(layer_list)
        network.trained = False
    else:
        network = incflow(layer_list)
        network.trained = False
            
    return network

def get_d_values(SFA, get_all=False, data=None, norm=None, normparms=None):
    """returns delta-values of the given SFA module on training data. During training, the delta value of feature ouput
    on the training data is optimized. The resulting delta values are stored in the batch SFA nodes. In
    the incremental SFA nodes, delta values are not saved. To get delta values of incremental SFA, training data
    and if desired a normalizer with parameters have to be provided.

    .. note:: I am not sure whether this works for all SFA modules.

    :param SFA: SFA to get delta values from
    :param get_all: If False, return only d-values of top SFA of top node.
    :param data: if SFA is incremental, the original training data has to be provided here
    :param norm: if SFA is incremental and feature output is supposed to be normalized before calculating d-values,
                 pass normalizer here (usually :py:class:`core.streamlined.normalizer`)
    :param normparms: normalization parameters to use for the normalizer ``norm``
    :return: list of delta values. If ``get_all=True`` this can be a nested list that has the same structure as the
             provided SFA module.
    """

    # if not SFA.trained:
    #     raise Exception("Module has not been trained yet.")    #does not work somehow, no time to investigate

    d = []
    for el in SFA:
        if type(el) == mdp.hinet.FlowNode:
            d.append(el[-1].d)
        elif type(el) == mdp.hinet.CloneLayer:
            d.append(el[-1][-1].d)
        elif (type(el) == incsfa.IncSFANode or type(el) == incsfa.IncSFA2Node or type(el) == inctuple or type(el) == incflow) and data is not None:
            seq = exec_SFA(SFA, data)
            if norm is not None and normparms is not None:
                seq = norm(seq, normparms)(seq)
            d.append(tools.delta_diff(seq))
    if not get_all:
        return d[-1]
    return d  # may be wrong in case of incSFA


def train_SFA(SFA, input_sequence, scheduler=None):
    """Train SFA module with given index on given input. Wrapper to suppress warnings
    and manually set the trained flag. Can be easily done manually.

    :param SFA: SFA module to train
    :param input_sequence: Training data
    :param scheduler: Pass scheduler if modules were built with parallel nodes, otherwise leave out.
    """
    warnings.simplefilter("ignore")
    
    if scheduler:
        SFA.train(input_sequence,scheduler=scheduler)
    else:
        SFA.train(input_sequence)
        
    SFA.trained = True      

def load_SFA(filename):
    """
    Load a pickled SFA module

    :param filename: Path to the pickle file
    :return: SFA module
    """
    SFA = pickle.load(open(filename,"rb"))
    SFA.trained = True
    return SFA

def exec_SFA( SFA, input_sequence, scheduler=None):
    """
    Run sequences through SFA

    :param SFA: SFA module
    :param input_sequence: Sequences to run through SFA
    :param scheduler: Pass scheduler if modules were built with parallel nodes, otherwise leave out.
    :return: SFA features
    """
    warnings.simplefilter("ignore")
    return SFA.execute(input_sequence, scheduler=scheduler) if scheduler else SFA.execute(input_sequence)

def build_module(parameters, flow_maker=mdp.Flow, eps=0.001) :
    """
    Build an SFA module with given parameters

    :param parameters: SFA parameter list. See :py:mod:`core.semantic_params` for details.
    :param flow_maker: Leave out. Method how to pack batch SFA module together
    :param eps: Learning rate for incremental SFA
    :return: SFA module
    """
    """ Use this to build an individual module, and also save it. """
    net = _make_network(parameters, flow_maker=flow_maker, eps=eps)
    return net