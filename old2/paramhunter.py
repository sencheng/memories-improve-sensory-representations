"""
Searching for parameters that maximize the final performance of the object retrieval. 
These functions just run the program for various parameter sets and save results, so 
that they can be visualized later, with the exception of load_snippets, which actually
builds a matrix from loaded data. Here's a short list of some of the parameters
we're thinking of playing with. 

INPUT PARAMETERS:
    for both stage 1 and 2:
        snippet_length
        number_of_snippets
        interleaved 
        blank_frame
        glue
        font size / font
    
    motion \in {lissajous, random walk, random stroll}
        - speed of movement

SEMANTIC PARAMETERS:
    DIFFERENT NETWORK TOPOLOGIES
     SFA1 is [[bot, mid], [top]] OR [[bot, mid],[]] [[bot], [top]]
     SFA2 is [[bot, mid], [top]] OR [[bot], [top]] OR [[]set_to_default, [top1, top2]]
    
    Very bottom layer must have 
     -  bo, rec_field_ch, spacing, in_channel_dim, out_sfa_dim1, out_sfa_dim2
         (in_channel_dim, bo set automatically for mid layers)
    Top layers have
     - in_sfa_dim, out_sfa_dim1, out_sfa_dim2
         (in_sfa_dim set automatically for bottom top layer)

EPISODIC PARAMETERS:
    - retrieval_noise
"""
import errno
import numpy as np
import os
from mdp import FlowExceptionCR

from core import result,input_params,system_params
from core.streamlined import program, construct_SFA1


def GO(paramset, filename, sfa1=None) :
    #First, create directory if necessary.
    path = filename.rsplit('/',1)[0] 
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    try:
        rslt = program(paramset, sfa1)
        rslt.save_to_file(filename)
    except FlowExceptionCR as err:
        print err

############################### 1D GRID TESTS ##################################
def rotation_tests_stroll():
    OMEGAS = np.arange(-0.1, 0.1, 0.003)
     
    for o in OMEGAS:
        paraset = system_params.SysParamSet()
        paraset.input_params_default['movement_params'] = dict(d2x=0.010, d2t = o, dx_max=0.1,dt_max=10*o)
        paraset.input_params_default['movement_type'] = 'random_stroll'
        
        GO(paraset, 'results/rotation_tests/[{}]stroll.txt'.format(o))
        
            
def rotation_tests_lissajous():
    OMEGAS = np.arange(-0.1, 0.1, 0.003)
     
    for o in OMEGAS:
        paraset = system_params.SysParamSet()
        paraset.input_params_default['movement_params'] = dict(a= 2, b= 3, deltaX= 3.378, omega=o, step= 0.02)
        paraset.input_params_default['movement_type'] = 'lissajous'
        
        GO(paraset, 'results/rotation_tests/[{}]lissa.txt'.format(o))

def speed_tests_stroll():
    # Make a bunch of predefined but totally values that might be interesting and are roughly logarithmic~ish~maybe.
    SPEEDS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 3,4,5,6,7,8,9, 10]
     
    for s in SPEEDS:
        paraset = system_params.SysParamSet()
        paraset.input_params_default['movement_params'] = dict(d2x=0.010, d2t = 0.009, dx_max=0.1,dt_max=0.1,step=s)
        paraset.input_params_default['movement_type'] = 'random_stroll'
        
        GO(paraset, 'results/speed_tests/[{}]stroll.txt'.format(s))
            
def speed_tests_lissajous():
    # Make a bunch of predefined but totally values that might be interesting and are roughly logarithmic~ish~maybe.
    # Again. Copypasta obviously.
    SPEEDS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 3,4,5,6,7,8,9, 10]
     
    for s in SPEEDS :
        paraset = system_params.SysParamSet()
        paraset.input_params_default['movement_params'] = dict(a= 2, b= 3, deltaX= 3.378, omega=1, step= 0.02*s)
        paraset.input_params_default['movement_type'] = 'lissajous'
        
        GO(paraset, 'results/speed_tests/[{}]lissa.txt'.format(s))

def max_speed_tests_stroll():
    MAXSPEEDS = [0.01,0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1, 1.5, 2, 3]
    
    for s in MAXSPEEDS :
        paraset = system_params.SysParamSet()
        paraset.input_params_default['movement_params'] = dict(d2x=0.010, d2t = 0.009, dx_max=s, dt_max=0.1)
        paraset.input_params_default['movement_type'] = 'random_stroll'
        
        GO(paraset, 'results/maxspeed_tests/[{}]lissa.txt'.format(s))
            
def category_weight_tests():
    paraset = system_params.SysParamSet()
    sfa1 = construct_SFA1(paraset)
    
    for w in np.arange(0,10,0.01) :
        paraset.st2['memory']['category_weight'] = w
        GO(paraset, 'results/cat_tests/[{}]lissa.txt'.format(w), sfa1)
        
def movetype_tests():
    for q,m in enumerate([input_params.lissa, input_params.stroll, input_params.walk]):
        for i in range(20):
            paraset = system_params.SysParamSet()
            paraset.input_params_default = m
            GO(paraset, 'results/movetype_tests/[{},{}].txt'.format(q*10,i))

############################### 2D GRID TESTS ##################################

def motion_tests():
    movements = [
        ('random_stroll', dict(d2x=0.005, d2t = 0.009, dx_max=0.05,dt_max=0.1)),
        ('random_stroll', dict(d2x=0.010, d2t = 0.018, dx_max=0.1,dt_max=0.2)),
        ('random_stroll', dict(d2x=0.005, d2t = 0.0, dx_max=0.05,dt_max=0.0)),
        ('random_stroll', dict(d2x=0.010, d2t = 0.0, dx_max=0.1,dt_max=0.0)),
        ('lissajous', dict(a= 2, b= 3, deltaX= 3,omega= 1, step= 0.02)),
        ('lissajous', dict(a= 2, b= 3, deltaX= 3,omega= 1, step= 0.04)),
        ('lissajous', dict(a= 2, b= 3, deltaX= 3,omega= 0, step= 0.02)),
        ('lissajous', dict(a= 2, b= 3, deltaX= 3,omega= 0, step= 0.04)),
    ]
    
    for i,m in enumerate(movements):
        for j,n in enumerate(movements):
            paraset = system_params.SysParamSet()
            paraset.st1['movement_type'], paraset.st1['movement_params'] = m
            paraset.st2['movement_type'], paraset.st2['movement_params'] = n
            
            GO(paraset, 'results/move-tests/[{},{}].txt'.format(i,j))

def lissajous_ab_tests():
    A = [1,1.5,2.4,3,3.15,3.595,4]
     
    for a in A:
        for b in A:
            paraset = system_params.SysParamSet()
            paraset.input_params_default['movement_params'] = dict(a= a, b= b, deltaX= 3,omega= 1, step= 0.04)       
            paraset.input_params_default['movement_type'] = 'lissajous'
            
            GO(paraset, 'results/lissajous-ab-tests/[{},{}].txt'.format(a,b))
                
def snippet_length_tests(defaultparam,version=1) :
    L = [5,10,20,50,100,200,500,1000,2000]
    
    for l in L:
        for k in L:
            paraset = system_params.SysParamSet()
            paraset.st1['number_of_snippets'] = l
            paraset.st1['snippet_length'] = 10000/l
            
            paraset.st2['number_of_snippets'] = k
            paraset.st2['snippet_length'] = 10000/k
            
            paraset.input_params_default = getattr(input_params, defaultparam)
            GO(paraset, 'results/sniptests/v{}/[{},{}].{}.txt'.format(version,l,k,defaultparam))


            
def snip_and_tot_len_tests(defaultparam, version=1):
    pcts = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    lens = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    for l in lens:
        for k in (pcts):
            paraset = system_params.SysParamSet()
            
            num = int(l*k)
            if num < 1 or l/num < 1 :
                continue
            
            paraset.input_params_default = getattr(input_params, defaultparam)
            paraset.st2['number_of_snippets'] = num
            paraset.st2['snippet_length'] = l/num
            
            GO(paraset, 'results/sniptests/v{}/[{},{}]totlen.{}.txt'.format(version,num,l/num, defaultparam))

################################# END OF EXPERIMENTS ############################
# The next functions and the main code below are used instead for loading the
# data generated in the experiment functions above. 
#################################################################################

def load_matrix(filepattern, measure='me'):
    folder, pattern = filepattern.rsplit('/',1)
    rslts = {} # map (index tupple) --> result object
    
    prefindx = pattern.index('[')
    prefix = pattern[:prefindx]                       # require the beginning of filename to match this
    suffix = pattern[pattern.index(']', prefindx)+1:] # and end of the filename to match this
        
    index_sets = None
    
    for f in os.listdir(folder) :
        first = f.find('[')
        last = f.find(']', first)
        
        if first < 0 or last < 0 or not (f[:first] == prefix and f[last+1:] == suffix):
            continue
        
        indexstr = f[first+1 : last ]                       # mask off only indices, which live in square brackets
        indices = [float(q) for q in indexstr.split(',')]     # and this is a list of them
        rslts[indexstr] = result.load_from_file(folder+'/'+f)
                
        ni = len(indices)
        if index_sets is None:
            index_sets = [set() for i in range(ni)]
        elif ni != len(index_sets):
            raise ValueError('Different Files have different numbers of indices; {} and {}'.format(len(index_sets), ni))
        
        for i,idx in enumerate(indices):
            index_sets[i].add(idx)
    
    index_lists = [sorted(s) for s in index_sets]
    rslt_matrix = np.zeros([len(q) for q in index_lists], dtype=('object' if measure =='me' else 'float'))

    for indexkey in rslts.keys() : # Are you ready ?
        rslt_matrix[tuple(index_lists[i].index(float(inx)) for i,inx in enumerate(indexkey.split(',')))] = getattr(rslts[indexkey] ,measure)()
        # Okay. Idea = put result function <measure> applied to result in the proper index. This index
        # happens to be the tupple formed from the indices of each parameterindex in the sorted array.
    
    return rslt_matrix

def load_list(filepattern, measure='me') :
    folder, pattern = filepattern.rsplit('/',1)
    
    prefindx = pattern.index('[')
    prefix = pattern[:prefindx]                       # require the beginning of filename to match this
    suffix = pattern[pattern.index(']', prefindx)+1:] # and end of the filename to match this
    
    rslt_list = []
        
    for f in os.listdir(folder) :
        first = f.find('[')
        last = f.find(']', first)
        
        if first < 0 or last < 0 or not (f[:first] == prefix and f[last+1:] == suffix):
            continue
        
        indices = [float(q) for q in f[first+1 : last].split(',')]     
        rslt_list.append((indices, getattr(result.load_from_file(folder+'/'+f), measure)() ))
                
    return rslt_list    
    
def plotgrid(filepattern, measure='jump_count') :
    from matplotlib import pyplot
    
    XE = load_matrix(filepattern, measure+"_e").astype(float)
    XS = load_matrix(filepattern, measure+"_s").astype(float)
    X = XS-XE

    datmax, datmin = np.max(X), np.min(X)
    bound = max(abs(datmax), abs(datmin))*1.1
    
    pyplot.matshow(X, cmap='seismic', vmin=-bound, vmax= bound)
    pyplot.colorbar()
    pyplot.matshow(XS, cmap='Reds')
    pyplot.colorbar()
    pyplot.matshow(XE, cmap='Blues')
    pyplot.colorbar()
    
def plot1d(filepattern, measure='jump_count') :
    from matplotlib import pyplot

    XE = load_matrix(filepattern, measure+"_e").astype(float)
    XS = load_matrix(filepattern, measure+"_s").astype(float)
    X = XS-XE
    pyplot.figure()
    pyplot.plot(X)
    pyplot.plot(XS)
    pyplot.plot(XE)
    
def plot1d_coord(filepattern, measure="jump_count") :
    pass
#    thing = load_list(filepattern, measure+"_)
    
# IF we're running this file, we use system arguments to determine which to run
if __name__ == '__main__':
    import sys
    
    methods = {n:m for n,m in dict(locals()).iteritems() if hasattr(m, '__call__')}
    
    
    for arg in sys.argv[1:] : # for every argument (assumed to be function names)
        if arg[0] == '-' :    # ignore options for now
            continue
        
        argparts = arg.split(':')
        if argparts[0] in methods :
            methods[argparts[0]](*argparts[1].split(','))
        
        
      
##################################### SGD PARAMETER SEARCH #################################
#if __name__ == '__main__':
#    snippet_length_tests()
#    def inp_params_lists() :
#        pass
#        
#    def sem_params1_lists() :
#        bottom_params = {
#            '!bo':               [30], 
#            '!rec_field_ch':     [15,17,20,25,28],
#            '!spacing':          [ 5, 1, 1, 1, 1],
#            '!in_channel_dim':   [1],
#            '!out_sfa_dim1':     [8],
#            '!out_sfa_dim2':     [8]
#        }
#        middle_params = {
#            '!rec_field_ch':      [2],
#            '!spacing':           [1],
#            '!out_sfa_dim1':      [8],
#            '!out_sfa_dim2':      [4]
#        }    
#        top_params = {
#            #'in_sfa_dim':         set to upper_sfa_layer.output_dim in the code
#            '!out_sfa_dim1':     [16],
#            '!out_sfa_dim2':     [16]
#        }
#        
#        return [[bottom_params],[top_params]]
#    
#    RESULTS = []
#    for i in range(8):
#        try:
#            params = choices.select_uniform(system_params.SysParamSet().set_experiment(),i)
#            sempar = choices.select_uniform(sem_params1_lists(), i)
#            params.sem_params1 = (sempar, semantic.SQUARE)
#            rslt = program(params)
#            
#            RESULTS.append(rslt)
#        except FlowExceptionCR as err:
#            print err
#        except ValueError:
#            print 'invalid params'            