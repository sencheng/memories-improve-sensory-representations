# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:36:34 2016

@author: richaopf
"""
from core import semantic, episodic, sensory
from core import system_params, tools, result

import numpy as np
import random
import sys


def program(PARAMETERS):
    semantic_system = semantic.SemanticSystem()
    sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default)
    
    sfa2S = semantic_system.build_module(PARAMETERS.sem_params2S)
    sfa2E = semantic_system.build_module(PARAMETERS.sem_params2E)
    
    memtestS = episodic.EpisodicMemory(sfa2S[-1].output_dim, **PARAMETERS.st4['memtest'] )
    memtestE = episodic.EpisodicMemory(sfa2E[-1].output_dim, **PARAMETERS.st4['memtest'] )
    
    PARAMETERS.st2['movement_type'] = 'stillframe'
    PARAMETERS.st2["movement_params"] = dict()
    PARAMETERS.st2["number_of_snippets"] = 5000
    PARAMETERS.st2["snippet_length"] = 2
    PARAMETERS.st2["interleaved"] = True
    PARAMETERS.st2["blank_frame"] = False
    PARAMETERS.st2["glue"] = "random"
    PARAMETERS.st2['memory']['category_weight'] = 5
    PARAMETERS.st2['memory']['retrieval_noise'] = 0.2
    
    #    print "Generating input.. "
    testing_sequence, testing_categories, testing_latent = sensory_system.generate(**PARAMETERS.st2)
    selection_sequence, selection_categories, selection_latent, selection_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st2)
    
    nlat = len(selection_latent[0])
    
    SFA1_output = np.array([l[:4]+np.random.normal(0,0.01,4) if l is not None else [-1,]*(4) for l in selection_latent])
    memory   = episodic.EpisodicMemory( len(SFA1_output[0]), categories=True, **PARAMETERS.st2['memory'])
    
    #    print "storing+scenarios... "

    for ran in selection_ranges:
        liran = list(ran)
        memory.store_sequence(SFA1_output[liran], categories=selection_categories[liran])
    
     
    stage2_retrievals_list = []
    stage2_visual = []
    comparison_visual = []
    BLACK = np.ones((30,30))*255
    for j in range(PARAMETERS.st2['number_of_retrieval_trials']):
        cue_index = random.randint(0,len(SFA1_output)-1)
        retrieved_sequence, retrieved_indices = memory.retrieve_sequence(SFA1_output[cue_index], selection_categories[cue_index],return_indices=True)
        stage2_retrievals_list.append(retrieved_sequence)
        comparison_visual.extend(selection_sequence[cue_index:cue_index+PARAMETERS.st2['memory']['retrieval_length']])
        comparison_visual.extend([BLACK]*5)
        stage2_visual.extend(selection_sequence[retrieved_indices])
        stage2_visual.extend([BLACK]*5)
     
    stage2_retrieval_arr = np.concatenate(stage2_retrievals_list,axis=0)
    
    tools.compare_inputs([stage2_visual,comparison_visual],rate =50)    
           
    #    print "training SFA2s... "
    semantic.train_SFA(sfa2S, SFA1_output)    
    semantic.train_SFA(sfa2E, stage2_retrieval_arr)
    
    #=========STAGE 4: TESTING===============================================

    #    print "testing..."
    #run SFA chain for both cases
    st4_SFA1_output      = np.array([l[:4]+np.random.normal(0,0.01,4) if l is not None else [-1]*(4) for l in testing_latent])
    SFA2_output_simple   = sfa2S.execute(st4_SFA1_output)
    SFA2_output_episodic = sfa2E.execute(st4_SFA1_output) 
    
    result.plot_pairwise_feature_scatter(st4_SFA1_output)
    result.plot_pairwise_feature_scatter(SFA2_output_simple)
    result.plot_pairwise_feature_scatter(SFA2_output_episodic)
    result.show_plots()
     
    #storage
    memtestS.store_sequence(SFA2_output_simple)
    memtestE.store_sequence(SFA2_output_episodic)
     
    #retrieval
    retrieval_length = PARAMETERS.st4['memtest']['retrieval_length']
    retr_simple, jumps_simple, indices_simple, retr_episodic, jumps_episodic, indices_episodic, org_simple, org_episodic = [], [], [], [], [], [], [], []
    
    for i in range(PARAMETERS.st4['number_of_retrievals']):
        cue_index = random.randint(0,len(SFA2_output_simple)-retrieval_length)
        
        retr_simple_s, jumps_simple_s, indices_simple_s = memtestS.retrieve_sequence(cue_index , return_jumps=True, return_indices=True)
        retr_episodic_s, jumps_episodic_s, indices_episodic_s = memtestE.retrieve_sequence(cue_index, return_jumps=True, return_indices=True)
        org_simple_s = SFA2_output_simple[cue_index:cue_index+retrieval_length]
        org_episodic_s = SFA2_output_episodic[cue_index:cue_index+retrieval_length]
         
        retr_simple.append(retr_simple_s)
        jumps_simple.append(jumps_simple_s)
        indices_simple.append(indices_simple_s)
        retr_episodic.append(retr_episodic_s)
        jumps_episodic.append(jumps_episodic_s)
        indices_episodic.append(indices_episodic_s)
        org_simple.append(org_simple_s)
        org_episodic.append(org_episodic_s)
             
    d_values = [semantic.get_d_values(s) for s in (sfa2E, sfa2S)]
    
    print("d values sfa2S",d_values[1])
    print("d values sfa2E",d_values[0])
    
    selection_corr = tools.feature_latent_correlation(SFA1_output, selection_latent, selection_categories)
    testing_corr = tools.feature_latent_correlation(st4_SFA1_output, testing_latent, testing_categories)
    testing_corr_simple = tools.feature_latent_correlation(SFA2_output_simple, testing_latent, testing_categories)
    testing_corr_episodic = tools.feature_latent_correlation(SFA2_output_episodic, testing_latent, testing_categories)
    
    semantic_params = semantic_system.parameters
    result_obj = result.Result(PARAMETERS,locals())
     
    #     if PARAMETERS.st4['output_to_file']:
    #         result_obj.save_to_file(PARAMETERS.result_prefix + str(running_ind) + ".p")
    
    BINS=10
    
    outputs = [SFA1_output, st4_SFA1_output, SFA2_output_simple, SFA2_output_episodic]
    latents = [selection_latent, testing_latent, testing_latent, testing_latent]
    
    for out, lat in zip(outputs,latents):
        test_hist = result.histogram_2d(out, lat, BINS)
        autocorrs = result.spatial_autocorrelation(test_hist)
        tits = []
        for histo in test_hist:
            G,_,_ = result.cellGridnessScore(histo, len(histo)-1, 1, 0)
            tits.append("gridness: " + str(round(G,2)))
        result.plot_correlation(autocorrs, titles=tits,decimal_places=-1,axlabel=False)
            
    result_obj.plot_correlation()
    result.show_plots()
    # score the system parameters:
    return result_obj

if __name__ == '__main__':
    
    PARAMETERS = system_params.SysParamSet()
    PARAMETERS.sem_params2S = [
    ('single', {
        'dim_in':    4,
        'dim_mid':  4,
        'dim_out':  8
    }),
    ('single', {
        'dim_in':    8,
        'dim_mid':  8,
        'dim_out':  8
    })
]
    PARAMETERS.sem_params2E = [
    ('single', {
        'dim_in':    4,
        'dim_mid':  4,
        'dim_out':  8
    }),
    ('single', {
        'dim_in':    8,
        'dim_mid':  8,
        'dim_out':  8
    })
]

res = program(PARAMETERS)

result.barplot(res.jump_count_s(), res.jump_count_e(), "Jump count")
result.barplot(res.euclidean_distance_s(), res.euclidean_distance_e(), "Euclidean Distance")
result.show_plots()
