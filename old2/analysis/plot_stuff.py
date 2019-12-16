import sys
sys.path.append("..")
import result    #@UnresolvedImport
import numpy as np
from matplotlib import pyplot

#Several consecutive files can share one parameter set (FILES_PER_SET)
#STEP_SIZE=1 mean that every set is plotted, STEP_SIZE=2, f.i, means that only every 2nd set is plotted. This can be useful if sets share common parameters that repeat in a pattern.
FILE_INDEX = 0
SET_COUNT = 1
STEP_SIZE = 1
FILES_PER_SET = 1

PREFIX = "../results/result"     #where to save files and how to name them
SUFFIX = ".p"

CORRELATIONS = ["selection_corr","testing_corr","testing_corr_simple","testing_corr_episodic"]  #should be according to results files
CORR_MAX_FEATURES = 8   #maximum number of features to look at for correlation plot (if f.i. 32 features are plotted, single cell may be too small)

#gather input parameters
if __name__ == "__main__":
    if len(sys.argv) > 3:
        idx = sys.argv[1]
        cnt = sys.argv[2]
        stp = sys.argv[3]
    elif len(sys.argv) == 3:
        idx = sys.argv[1]
        cnt = sys.argv[2]
        stp = 1
    elif len(sys.argv) == 2:
        idx = sys.argv[1]
        cnt = 1
        stp = 1
    else:
        idx = FILE_INDEX     #if started from IDE
        cnt = SET_COUNT
        stp = STEP_SIZE
    set_size = FILES_PER_SET       #no possibility to enter from console yet.
#generate filename list to load
    filenames = []
    for a, a_ind in zip(range(idx,cnt*stp+idx,stp), range(0,cnt)):
        filenames.append([])      #append empty list for each requested set
        for b in range(set_size):
            filenames[a_ind].append(PREFIX + str(a*set_size+b) + SUFFIX)     #filenames for the set

#load results from files
    res_list = []
    for s, setlist in enumerate(filenames):
        res_list.append([])
        for filename in setlist:
            res_list[s].append(result.load_from_file(filename))

    #prepare plots for temporal error and for correlation (..c)
    f, axarr = pyplot.subplots(cnt, 2, sharex='col', sharey='row')
    fc, axarrc = pyplot.subplots(cnt, len(CORRELATIONS))
    if cnt >1:
        axarr_left = axarr[:,0]
        axarr_right = axarr[:,1]
    else:
        axarr_left = [axarr[0]]
        axarr_right = [axarr[1]]
        axarrc = np.array([axarrc])   #we access by indices, so make it a list, even if there is only one row of plots
    axarr_left[0].set_title('temporal error - simple')
    axarr_right[0].set_title('temporal error - episodic')
    f.canvas.set_window_title("temporal error")
    fc.canvas.set_window_title("correlation")
    for ax in axarrc[:,0]:
        ax.set_ylabel("Latent variable")
    for a, ax in enumerate(axarrc[0,:]):
        ax.set_xlabel("Feature number")
        ax.set_title([CORRELATIONS[a]])
    sequence_count = len(res_list[0][0].org_simple)
    
    #prepare arrays holding temporal error and jump values. To be filled in following iteration thorugh results files
    err_time_simple_l = []
    err_time_episodic_l = []
    jumps_simple_l = []
    jumps_episodic_l = []
    corr_l = []
    #================================================================================================================
    #ITERATE THROUGH RESULTS FILES
    for subindex, res_sublist in enumerate(res_list):
        jumps_simple_l.append([])
        jumps_episodic_l.append([])
        err_time_simple_l.append(np.zeros((set_size,sequence_count)))
        err_time_episodic_l.append(np.zeros((set_size,sequence_count)))
        corr_l.append([])
        for i, res in enumerate(res_sublist):
            #error
            for org_s, retr_s, org_e, retr_e in zip(res.org_simple, res.retr_simple,res.org_episodic, res.retr_episodic):
                err_time_simple_l[subindex][i] = err_time_simple_l[subindex][i] + np.sqrt(np.sum((org_s-retr_s)**2,axis=1))
                err_time_episodic_l[subindex][i] = err_time_episodic_l[subindex][i] + np.sqrt(np.sum((org_e-retr_e)**2,axis=1))
            #jumps
            js = np.array(res.jumps_simple)
            je = np.array(res.jumps_episodic)
            jumps_simple_l[subindex].append([])
            jumps_episodic_l[subindex].append([])
            jumps_simple_l[subindex][i].append(np.sum(np.abs(js)))
            jumps_episodic_l[subindex][i].append(np.sum(np.abs(je)))
            jumps_simple_l[subindex][i].append(np.sum(js[js>0]))
            jumps_episodic_l[subindex][i].append(np.sum(je[je>0]))
            jumps_simple_l[subindex][i].append(sum(np.abs(js[js<0])))
            jumps_episodic_l[subindex][i].append(np.sum(np.abs(je[je<0])))
            jumps_simple_l[subindex][i].append(len(np.nonzero(js)[0]))
            jumps_episodic_l[subindex][i].append(len(np.nonzero(je)[0]))
            
            #get feature-latent-correlation
            corr_l[subindex].append([])
            for p, var in enumerate(CORRELATIONS): 
                data = eval("res."+var)[:,:CORR_MAX_FEATURES]
                corr_l[subindex][i].append(data)
            #print data descriptions
        print("============== " + str(subindex) + " ==============")
        print(res_sublist[0].data_description)
        print("semantic mode: " + res_sublist[0].params.semantic_mode)
    #================================================================================================================
    #calculate mean of all the files per set. we get one result to plot per set
    jumps_simple = np.mean(jumps_simple_l,axis=1)
    jumps_episodic = np.mean(jumps_episodic_l,axis=1)
    err_time_simple = np.mean(err_time_simple_l, axis=1)
    err_time_episodic = np.mean(err_time_episodic_l, axis=1)
    #calculation of corr mean is not that easy, because different correlation matrices have different sizes
    corr_arr = np.array(corr_l)
    correl = []
    for set_index in range(cnt):
        correl.append([])
        for corr_index in range(len(CORRELATIONS)):
            correl[set_index].append(np.mean(corr_arr[set_index,:,corr_index], axis=0))  #manually calculate mean for each correlation measure of each set
    #copy every single value of array, otherwise when drawing matrix, error "image data could not convert to float" occured. Couldnt solve it.
    temp = []
    for sss in correl:
        temp.append([])
        for ttt in sss:
            temp[-1].append([])
            for uuu in ttt:
                temp[-1][-1].append([])
                for vvv in uuu:
                    temp[-1][-1][-1].append(vvv)
    #plot temporal error
    t = np.arange(sequence_count)
    for ii in range(cnt):
        axarr_left[ii].plot(t,err_time_simple[ii])
        axarr_right[ii].plot(t,err_time_episodic[ii])
    #plot correlation
    for data_index, corr_data in enumerate(temp):
        for p in range(len(CORRELATIONS)): 
            last_plot= axarrc[data_index,p].matshow(np.array(corr_data[p]),cmap=pyplot.cm.seismic, vmin=-1, vmax=1)     #@UndefinedVariable
            for (ii, jj), z in np.ndenumerate(corr_data[p]):
                #axarrc[i,p].text(jj, ii, '{:.2f}'.format(z), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                axarrc[data_index,p].text(jj, ii, '{:.2f}'.format(z), ha='center', va='center',color="white")
    fc.subplots_adjust(right=0.8)
    cbar_ax = fc.add_axes([0.85, 0.15, 0.05, 0.7])
    fc.colorbar(last_plot, cax=cbar_ax)
    #mean error
    err_simple = [0]*cnt
    err_episodic = [0]*cnt
    for j in range(cnt):
        err_simple[j] = np.mean(err_time_simple[j])
        err_episodic[j] = np.mean(err_time_episodic[j])
    
    #prepare bar plots for error and jumping
    f,axarr = pyplot.subplots(cnt, 3)
    if cnt == 1:
        axarr = [axarr]
    f.canvas.set_window_title("overall error and jumps")
        
    #creating bar plots
    for pl in range(cnt):
        axarr[pl][0].bar([0], [err_simple[pl]], width=0.35, color='r')
        axarr[pl][0].bar([0.35], [err_episodic[pl]], width=0.35, color='y')
        axarr[pl][0].set_title("err")
        axarr[pl][0].set_xticks([0.35])
        axarr[pl][0].set_xticklabels(["[s] [e]"])
        
        axarr[pl][1].bar([0,1,2], [jumps_simple[pl][0], jumps_simple[pl][1], jumps_simple[pl][2]], width=0.35, color='r')
        axarr[pl][1].bar([0.35,1.35,2.35], [jumps_episodic[pl][0], jumps_episodic[pl][1], jumps_episodic[pl][2]], width=0.35, color='y')
        axarr[pl][1].set_title("jump size")
        axarr[pl][1].set_xticks([0.35,1.35,2.35])
        axarr[pl][1].set_xticklabels(["[s] abs [e]","[s] pos [e]","[s] neg [e]"])
        
        axarr[pl][2].bar([0], [jumps_simple[pl][3]], width=0.35, color='r')
        axarr[pl][2].bar([0.35], [jumps_episodic[pl][3]], width=0.35, color='y')
        axarr[pl][2].set_title("jump count")
        axarr[pl][2].set_xticks([0.35])
        axarr[pl][2].set_xticklabels(["[s] [e]"])
        
    pyplot.show()