"""
Module containing the :py:class:`Result` class that makes it easy to save results
generated with :py:func:`core.streamlined.program`.

Also containing some functions that were at some point helpful to analyze results, like visualizing what
features code for or analyzing how much sequences differ,...

.. note::
   Some of the plotting functions do not show the plots automatically. In these cases ``pyplot.show()``
   or :py:func:`show_plots()` have to be called manually.

"""

from . import tools, semantic

import pickle
import numpy as np
from matplotlib import pyplot
import scipy
import math

CORRELATIONS = ["forming_corrY", "testing_corrY", "testing_corrZ_simple", "testing_corrZ_episodic"]
""" 
Constant which can be used by some functions to select attributes of result object and/or to set 
legends and titles of plots
"""

def load_from_file(filename):
    """
    Returns a Result object that is loaded from a pickle file

    :param filename: path to the file to load
    :returns: Result object
    """
    return pickle.load(open(filename,"rb"))

def corr_matrix_score(matrix):
    """
    Scores a correlation matrix by taking the average of the maximum elements of
    each row. This metric has flaws, but fewer than some others. This function
    was introduced to automatically rate performance of an SFA by one value.
    Using a linear regressor prediction and rating that is probably better.

    :param matrix: correlation matrix, as returned from :py:mod:`core.tools.feature_latent_correlation`.
    :return: scalar matrix score
    """
    total = 0
    for row in matrix:
        total += max(row)
        
    return total/len(matrix)

def levenshtein(s1, s2):
    """
    Calculates the levenshtein "edit distance" between sequences s1 and s2.

    :param s1: sequence 1
    :param s2: sequence 2
    :return: scalar edit distance
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (not np.array_equal(c1, c2))
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def euclidean(org,ret):
    """
    Calculates the Euclidean distance between sequences s1 and s2.

    :param org: sequence 1
    :param ret: sequence 2
    :return: scalar euclidean distance
    """
    return np.mean(np.sqrt(np.sum((org-ret[:len(org)])**2, axis=1)))

def distance(org_arr,retr_arr,fun):
    """
    Helper function to get the average distance between lists of sequences.

    :param org_arr: list or array of sequences, like ``[a1, b1, c1, d1]``
    :param retr_arr: list or array of sequences, like ``[a2, b2, c2, d2]``
    :param fun: function to evaluate when calculating distance between sequences
                This can for instance be :py:func:`levenshtein` or :py:func:`euclidean`.
    :return: scalar distance measure as ``mean(fun(a1, a2), fun(b1, b2), ... )``
    """
    dis = 0
    for org, ret in zip(org_arr, retr_arr):
        dis = dis + fun(org,ret)
    dis = dis/len(org_arr)
    return dis

def barplot(error1, error2, caption="err"):
    """
    Make a 2-bar barplot showing mean of supplied arrays with error bars (std)

    :param error1: first values as array
    :param error2: second values as array
    :param caption: plot title
    """
    std1 = np.std(error1)
    std2 = np.std(error2)
    mean1 = np.mean(error1)
    mean2 = np.mean(error2)
    _,ax = pyplot.subplots(1, 1)
    ax.bar([0], [mean1], width=0.35, color='r',yerr=std1)
    ax.bar([0.35], [mean2], width=0.35, color='y',yerr=std2)
    ax.set_title(caption)
    ax.set_xticks([0.35])
    ax.set_xticklabels(["[s] [e]"])
    
def barplot_singlevar(var, err=None, caption=CORRELATIONS):
    """
    Make several barplots with a single bar each(as subplots) with errorbars,
    one for every element in supplied * var*.

    :param var: array-like of mean values to plot
    :param err: array-like of stds. If None, stds are set to 0
    :param caption: array-like of subplot titles
    """
    if err is None:
        err = [0]*len(var)
    _,ax = pyplot.subplots(1, len(var))
    ran = np.arange(len(var[0]))
    for i, varvar in enumerate(var):
        ax[i].bar(ran, varvar, width=0.7, color='r',yerr=err[i])
        ax[i].set_title(caption[i])
        ax[i].set_xticks(ran+0.35)
        ax[i].set_xticklabels(ran.astype(str))
    
def plot_correlation(values,titles=None,nFeatures=0,decimal_places=2,axlabel=True,window_title=None):
    """
    Make a matrix plot for each correlation matrix in *values*. Additionally, numbers are printed
    on the matrix

    :param values: array-like of correlation matrices. Needs to fit to titles. If titles is None
                   it needs to fit to :py:data:`CORRELATIONS`.
    :param titles: array-like of plot titles (strings). If None it is set to :py:data:`CORRELATIONS`.
    :param nFeatures: Number of features to include in plot. If 0 or higher than actual number of features in
                      correlation matrix, all features are plotted
    :param decimal_places: How to round printed numbers
    :param axlabel: Whether or not to show axis label ("Latent variable" / "Feature number")
    :param window_title: String to print on window bar.
    """
    texts = CORRELATIONS if titles is None else titles
    corrs = values
    fc, axarrc = pyplot.subplots(1, len(texts))
    if window_title is not None:
        fc.canvas.set_window_title(window_title)
    if axlabel:
        for ax in axarrc:
            ax.set_ylabel("Latent variable")
    for a, ax in enumerate(axarrc):
        if axlabel:
            ax.set_xlabel("Feature number")
        ax.set_title([texts[a]])
    for p, val in enumerate(corrs):
        corr_array = val[:,:nFeatures] if nFeatures>0 else val
        last_plot= axarrc[p].matshow(np.abs(corr_array),cmap=pyplot.cm.Blues, vmin=0, vmax=1)     #@UndefinedVariable
        if decimal_places >= 0:
            for (ii, jj), z in np.ndenumerate(corr_array):
                formstring = '{:.0f}'
                #axarrc[i,p].text(jj, ii, '{:.2f}'.format(z), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                axarrc[p].text(jj, ii, formstring.format(z*10**decimal_places), ha='center', va='center',color="white")
    fc.subplots_adjust(right=0.8)
    cbar_ax = fc.add_axes([0.85, 0.15, 0.05, 0.7])
    fc.colorbar(last_plot, cax=cbar_ax)

def plot_corr_matrix(mat):
    """
    Plot a single correlation matrix. Numbers are shown as integers (%)

    :param mat: correlation matrix
    """
    pyplot.matshow(np.abs(mat), cmap=pyplot.cm.Blues, vmin=0, vmax=1)
    for (ii, jj), z in np.ndenumerate(mat):
        pyplot.text(jj, ii, '{:.0f}'.format(z*100), ha='center', va='center', color="white")
    pyplot.show()

def plot_pairwise_feature_space(feature_sequence):
    """
    Make a 2d histogram plot for each even pair of successive features. Similar to :py:func:`plot_pairwise_feature_scatter`
    but here data points are histogrammed (10 bins). For instance, if ``feature_sequence`` has 4 features,
    feature 0 and 1 are paired and feature 2 and 3 are paired. If the number of features is odd, the last feature is discarded.

    :param feature_sequence: sequence of features (what a surprise)
    """
    nfeatures = np.shape(feature_sequence)[1]
    f,axarr = pyplot.subplots(1,nfeatures//2)
    histlist = []
    for i in range(nfeatures//2):
        f1 = feature_sequence[:,i*2]
        f2 = feature_sequence[:,i*2+1]
        rangel = [[np.floor(np.min(f1)),np.ceil(np.max(f1))],[np.floor(np.min(f2)),np.ceil(np.max(f2))]]
        h, _, _ = np.histogram2d(f1, f2, bins=10, range=rangel)
        histlist.append(h)
        axc = axarr[i].matshow(histlist[-1],cmap="Greys")
        axarr[i].set_xticklabels(np.concatenate((np.array([0]),np.linspace(rangel[0][0], rangel[0][1], num=5))))
        axarr[i].set_yticklabels(np.concatenate((np.array([0]),np.linspace(rangel[1][0], rangel[1][1], num=5))))
        for (ii, jj), z in np.ndenumerate(histlist[-1]):
            axarr[i].text(jj, ii, int(z), ha='center', va='center',color="red")
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(axc, cax=cbar_ax)
    
def plot_pairwise_feature_scatter(feature_sequence):
    """
    Make a scatter plot for each even pair of successive features. For instance, if ``feature_sequence`` has 4 features,
    feature 0 and 1 are paired and feature 2 and 3 are paired. If the number of features is odd, e.g. 5, the second-to-last
    feature is duplicated such that (0,1), (2,3), (3,4) are paired.

    :param feature_sequence: sequence of features
    """
    nfeatures = np.shape(feature_sequence)[1]
    if nfeatures%2:
        f_s = np.insert(feature_sequence, nfeatures-1, feature_sequence[:,nfeatures-2], axis=1)   #if uneven number of features, insert second-to-last feature before last feature like [0,1,2,3,3,4]
        nfeatures += 1 
        print("new shape", np.shape(f_s))
    else:
        f_s = feature_sequence
    _,axarr = pyplot.subplots(1,nfeatures//2)
    if nfeatures == 2:
        axarr = [axarr]
    for i in range(nfeatures//2):
        f1 = f_s[:,i*2]
        f2 = f_s[:,i*2+1]
        #rangel = [[np.floor(np.min(f1)),np.ceil(np.max(f1))],[np.floor(np.min(f2)),np.ceil(np.max(f2))]]
        _ = axarr[i].scatter(f1,f2,s=2)
        #axarr[i].set_xticklabels(np.concatenate((np.array([0]),np.linspace(rangel[0][0], rangel[0][1], num=5))))
        #axarr[i].set_yticklabels(np.concatenate((np.array([0]),np.linspace(rangel[1][0], rangel[1][1], num=5))))
        #for (ii, jj), z in np.ndenumerate(histlist[-1]):
        #    axarr[i].text(jj, ii, int(z), ha='center', va='center',color="red")
    #f.subplots_adjust(right=0.8)
    #cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    #f.colorbar(axc, cax=cbar_ax)
    
def histogram_2d(feature_sequence, coordinate_sequence, bins=10, norm=True):
    """Computes a special 2d histogram of how the features code for x- and y- coordinate. Coordinates are binned and for each bin all
    corresponding feature values are summed. Values are normalized to [0,1] in the end if that option is not False

    :param feature_sequence: temporal sequence of feature vectors. One histogram is created for each feature.
    :param coordinate_sequence: temporal sequence of latent variable vectors, of which element 0 is x-coordinate and element 1 is y-coordinate
    :param bins: number of bins in x- and y-direction.
    :param norm: turn normalization on or off. If True, histogram values are normalized to [0,1] for each feature individually.
    :returns: list of histograms, one for each feature contained in feature_sequence elements. Each histogram is a <bins>x<bins> matrix.
    """
    coord_arr = np.array(coordinate_sequence)
    max_coord = max(np.max(coord_arr[:,0]),np.max(coord_arr[:,1]))
    min_coord = min(np.min(coord_arr[:,0]),np.min(coord_arr[:,1]))
    interv = float(max_coord-min_coord)
    bin_size = interv/bins
    
    hist = []
    for _ in range(len(feature_sequence[0])):      #for each feature
        hist.append(np.zeros((bins,bins)))
    
    for feat_vec, lat_vec in zip(feature_sequence, coordinate_sequence):
        for f, feat in enumerate(feat_vec):
            try:                 #absorbing none type elements from blank frames
                x = lat_vec[0]
            except:
                continue
            x_bin = int((x-min_coord)/bin_size)  #x-index of bin
            if x_bin >= bins*interv:
                print("interv", interv)
                print("x_bin", x_bin)
                print("bins*interv")
                print("================")
            x_bin = bins-1 if x_bin == bins else x_bin   #correct maximum value (since coordinates on closed interval [-1,1])
            y = lat_vec[1]
            y_bin = int((y-min_coord)/bin_size)  #y-index of bin
            y_bin = bins-1 if y_bin == bins else y_bin
            hist_feat = hist[f]
            
            hist_feat[y_bin,x_bin] = hist_feat[y_bin,x_bin]+feat
    if norm:
        for m in range(len(hist)):
            maxval = np.max(np.absolute(hist[m]))
            hist[m] = hist[m]/maxval
    return hist


def histogram_angle(feature_sequence, angle_sequence, bins=10, absolute = False):
    """Computes a special 2d histogram of how the features code for sin and cos of :math:`{\phi}` coordinate. Coordinates are binned and for each bin all
    corresponding feature values are summed. Values are normalized to [0,1] in the end.

    :param feature_sequence: temporal sequence of feature vectors. One histogram is created for each feature.
    :param coordinate_sequence: temporal sequence of latent variable vectors, of which element 2 is sin(:math:`{\phi}`) and element 3 is cos(:math:`{\phi}`)
    :param bins: number of bins.
    :returns: list of histograms, one for each feature contained in feature_sequence elements. Each histogram is a <bins>x<bins> matrix.
    """
    if absolute:
        fun = abs
    else:
        fun = float

    max_coord = np.pi
    min_coord = -np.pi
    interv = float(max_coord - min_coord)
    bin_size = interv / bins

    hist = []
    for _ in range(len(feature_sequence[0])):  # for each feature
        hist.append([])
        for _ in range(bins):
            hist[-1].append([])  #for each bin in each feature vector

    for idx, feat_vec, lat_vec in zip(range(len(feature_sequence)),feature_sequence, angle_sequence):
        for f, feat in enumerate(feat_vec):
            try:  # absorbing none type elements from blank frames
                x = lat_vec[0]
            except:
                continue
            y = lat_vec[1]
            angle = math.atan2(x,y)
            x_bin = int((angle - min_coord) / bin_size)  # x-index of bin
            if x_bin >= bins * interv:
                print("interv", interv)
                print("x_bin", x_bin)
                print("bins*interv")
                print("================")
            x_bin = bins - 1 if x_bin == bins else x_bin  # correct maximum value (since coordinates on closed interval [-1,1])
            hist_feat = hist[f]

            hist_feat[x_bin].append(fun(feat))   #if absolute = True, append absolute value here - some information is lost
    ret_mean = []
    ret_std = []
    for m in range(len(hist)):   #m - feature index
        feat_mean = np.zeros(bins)
        feat_std = np.zeros(bins)
        for bi in range(bins):      #bi - bin index
            feat_mean[bi] = np.sum(hist[m][bi])
            feat_std[bi] = np.std(hist[m][bi])
        maxval = np.max(np.absolute(feat_mean))
        feat_mean = feat_mean/maxval
        ret_mean.append(feat_mean)
        maxval = np.max(np.absolute(feat_std))
        feat_std = feat_std / maxval
        ret_std.append(np.sqrt(feat_std))
    return ret_mean, ret_std

def spatial_autocorrelation(histo):
    '''Computes a spatial autocorrelograms from a given list of 2d histograms (using scipy builtin)
    :params histo: list of data as 2d histograms or measurements in quantized spatial dimensions

    :returns: list of spatial autocorrelograms of data
    '''
    ret = []
    for hist in histo:
        ret.append(scipy.signal.correlate2d(hist,hist))     #@UndefinedVariable
    return ret


def show_plots():
    """
    Just calls pyplot.show(). Pretty useless, apart from not having to import pyplot maybe.
    """
    pyplot.show()
    
# def distance_monotone(dataY, lat, cat, M = 5000,weights=[1,1,1,1,0,0,0]):
#     latdists = tools.weighted_distance(lat[I1,:],lat[I2,:],weights=weights)
#     ydists = tools.weighted_distance(dataY[I1,:], dataY[I2,:])
#
#     complete = tools.completeness(lat[I1,:]) * tools.completeness(lat[I2,:])
#     samecat = cat[I1] == cat[I2]
#     diffcat = np.logical_not(samecat)
#
#     return (lambda IND : scipy.stats.spearmanr(latdists[IND], ydists[IND])[0])(np.logical_and(samecat, complete==1))

def plot_delta(res):
    """
    Make a bar plot with one bar for each delta value taken from the Result object. The accessed delta values are by default
    ``["sfa1", "forming_Y", "retrieved_Y", "sfa2S", "sfa2E", "testingY", "testingZ_S", "testingZ_E", "training_X", "forming_X", "testingX"]``.
    Can easily be changed in the code.

    :param res: object of :py:class:`Result`
    """
    d_keys = ["sfa1", "forming_Y", "retrieved_Y", "sfa2S", "sfa2E", "testingY", "testingZ_S", "testingZ_E", "training_X", "forming_X", "testingX"]
    # D_KEYS = ["sfa1", "forming_Y", "retrieved_Y", "sfa2S", "sfa2E", "testingY", "testingZ_S", "testingZ_E"]
    d_colors = ['r', 'b', 'b', 'r', 'r', 'b', 'k', 'k', 'g', 'g', 'g']
    f, ax = pyplot.subplots(1, 1, sharex=True, sharey=True)
    for i, (k, c) in enumerate(zip(d_keys, d_colors)):
        v = res.d_values[k]
        v_mean_16 = np.mean(np.sort(v)[:16])
        v_mean = np.mean(v)
        ax.bar(i, v_mean_16, width=0.6, color=c)
        ax.bar(i + 0.2, v_mean, width=0.2, color='y')
        ax.text(i + 0.3, v_mean_16, '{:.3f}'.format(v_mean_16), ha='center', va='bottom', color="black", rotation=90)
    tix = list(np.arange(len(d_keys)) + 0.5)
    ax.set_xticks(tix)
    ax.set_xticklabels(d_keys, rotation=70)
    ax.set_yscale("log", nonposy='clip')

class Result():
    def save_to_file(self, filename=None, description=None):
        """Saves the object into a pickle file

        :param filename: path to the file to save the object in. If not provided, save it
                         as unnamed.<ind>.result, where ind is read from the .ind file in the source folder.
                         If the .ind file was used, the number is incremented.
        """
        
        if filename is None :
            with open('./.ind','r+b') as f:
                index = int(f.readline())
                f.seek(0) # reset the file reader to position 0, to overwrite. Because numbers never become
                            # shorter in description, we don't need to worry that the old number remains.
                f.write(index+1)
                filename = 'unnamed.{}.result'.format(index)

        if description is not None:
            self.data_description = str(self.data_description) + str(description)
        pickle.dump(self, open(filename,"wb"))
    
    def __getstate__(self):
        return {key:getattr(self,key) for key in self.__dict__ if key in self.SAVABLE}
    
    def __init__(self, PARAMS, initdict, normalizer = None):
        """
        This class is for storing (and pickling) results that were generated with :py:func:`core.streamlined.program`,
        which returns a Result object.

        :param PARAMS: PARAMETERS object (:py:class:`core.system_params.SysParamSet`)
        :param initdict: dictionary possibly created with ``locals()``, containing all the system and result variables
        :param normalizer: None or reference to :py:class:`core.streamlined.normalizer` or another adequate normalization method.
                           If not None it is used to normalize input sequences before calculating delta values.
        """
        
        self.__dict__.update(initdict)
        
        self.params = PARAMS
        self.data_description = PARAMS.data_description
        try:
            self.dmat_dia = np.diagonal(self.memory.dmat)
        except:
            pass

        b = False
        if self.params.st4b is not None:
            b = True

        goal = self.params.program_extent
        S, E = 'S' in self.params.which, 'E' in self.params.which

        self.d_values = {'sfa1': semantic.get_d_values(self.sfa1, data=self.training_sequence, norm=normalizer, normparms=self.params.normalization)}
        try:
            if normalizer is not None:
                self.d_values['training_X'] = tools.delta_diff(normalizer(self.training_sequence, self.params.normalization)(self.training_sequence))
            else:
                self.d_values['training_X'] = tools.delta_diff(self.training_sequence)
        except:
            self.d_values['training_X'] = float('nan')
            pass
        if goal >= 2:
            self.forming_corrY = tools.feature_latent_correlation(self.forming_sequenceY, self.forming_latent, self.forming_categories)
            self.d_values['forming_lat'] = tools.delta_diff(self.forming_latent)
            self.d_values['forming_cat'] = tools.delta_diff(np.array(self.forming_categories)[:,None])
            try:
                if normalizer is not None:
                    self.d_values['forming_X'] = tools.delta_diff(normalizer(self.forming_sequenceX, self.params.normalization)(self.forming_sequenceX))
                else:
                    self.d_values['forming_X'] = tools.delta_diff(self.forming_sequenceX)
            except:
                self.d_values['forming_X'] = float('nan')
                pass
            self.d_values['forming_Y'] = tools.delta_diff(self.forming_sequenceY)
        if goal >= 3:
            if S: self.d_values['sfa2S'] = semantic.get_d_values(self.sfa2S, data=self.forming_sequenceY, norm=normalizer, normparms=self.params.normalization)
            if E: self.d_values['sfa2E'] = semantic.get_d_values(self.sfa2E, data = self.retrieved_sequence, norm=normalizer, normparms=self.params.normalization)
            if E: self.d_values['retrieved_Y'] = tools.delta_diff(self.retrieved_sequence)
            if self.params.st3['use_memory']:
                if E:
                    self.d_values['retrieved_lat'] = tools.delta_diff(self.ret_lat)
                    self.d_values['retrieved_cat'] = tools.delta_diff(self.ret_cat[:,None])
            else:
                if E:
                    self.d_values['retrieved_lat'] = self.d_values['forming_lat']
                    self.d_values['retrieved_cat'] = self.d_values['forming_cat']
        if goal >= 4:
            self.d_values['testing_lat'] = tools.delta_diff(self.testing_latent)
            self.d_values['testing_cat'] = tools.delta_diff(np.array(self.testing_categories)[:,None])
            if b:
                self.d_values['testing_latb'] = tools.delta_diff(self.testing_latentb)
                self.d_values['testing_catb'] = tools.delta_diff(np.array(self.testing_categoriesb)[:,None])
            if S:
                self.testing_corrZ_simple = tools.feature_latent_correlation(self.testing_sequenceZ_S, self.testing_latent, self.testing_categories)
                if b:
                    self.testing_corrZ_simpleb = tools.feature_latent_correlation(self.testing_sequenceZ_Sb, self.testing_latentb, self.testing_categoriesb)
                self.d_values['testingZ_S'] = tools.delta_diff(self.testing_sequenceZ_S)
                if b:
                    self.d_values['testingZ_Sb'] = tools.delta_diff(self.testing_sequenceZ_Sb)
            if E:
                self.testing_corrZ_episodic = tools.feature_latent_correlation(self.testing_sequenceZ_E, self.testing_latent, self.testing_categories)
                if b:
                    self.testing_corrZ_episodicb = tools.feature_latent_correlation(self.testing_sequenceZ_Eb, self.testing_latentb, self.testing_categoriesb)
                self.d_values['testingZ_E'] = tools.delta_diff(self.testing_sequenceZ_E)
                if b:
                    self.d_values['testingZ_Eb'] = tools.delta_diff(self.testing_sequenceZ_Eb)
            self.testing_corrY = tools.feature_latent_correlation(self.testing_sequenceY, self.testing_latent, self.testing_categories)
            if b:
                self.testing_corrYb = tools.feature_latent_correlation(self.testing_sequenceYb, self.testing_latentb, self.testing_categoriesb)
            if normalizer is not None:
                self.d_values['testingX'] = tools.delta_diff(normalizer(self.testing_sequenceX, self.params.normalization)(self.testing_sequenceX))
                if b:
                    self.d_values['testingXb'] = tools.delta_diff(normalizer(self.testing_sequenceXb, self.params.normalization)(self.testing_sequenceXb))
            else:
                self.d_values['testingX'] = tools.delta_diff(self.testing_sequenceX)
                if b:
                    self.d_values['testingXb'] = tools.delta_diff(self.testing_sequenceXb)
            self.d_values['testingY'] = tools.delta_diff(self.testing_sequenceY)
            if b:
                self.d_values['testingYb'] = tools.delta_diff(self.testing_sequenceYb)

        # saving certain parts of original data here

        self.training_latent = np.array(self.training_latent)[:, :4]
        trainYraw = semantic.exec_SFA(self.sfa1, self.training_sequence)
        self.trainingY = trainYraw[:, 4] if normalizer is None else normalizer(trainYraw, self.params.normalization)(trainYraw)[:, 4]

        if goal >= 2:
            self.forming_latent = np.array(self.forming_latent)[:, :4]
            self.formingY = self.forming_sequenceY
        if goal >= 3:
            self.retrieved_presequence = self.retrieved_presequence[:1000]
            self.retrievedY = self.retrieved_sequence
            formZraw = semantic.exec_SFA(self.sfa2S, self.forming_sequenceY)
            retrZraw = semantic.exec_SFA(self.sfa2E, self.retrieved_sequence)
            self.formingZ = formZraw[:, :4] if normalizer is None else normalizer(formZraw, self.params.normalization)(formZraw)[:, :4]
            self.retrievedZ = retrZraw[:, :4] if normalizer is None else normalizer(retrZraw, self.params.normalization)(retrZraw)[:, :4]

        if goal >= 4:
            self.testing_latent = np.array(self.testing_latent)[:,:4]
            self.testingY = self.testing_sequenceY[:,:4]
            self.testingZ_S = self.testing_sequenceZ_S[:,:4]
            self.testingZ_E = self.testing_sequenceZ_E[:, :4]

            if b:
                self.testing_latentb = np.array(self.testing_latentb)[:, :4]
                self.testingYb = self.testing_sequenceYb[:, :4]
                self.testingZ_Sb = self.testing_sequenceZ_Sb[:, :4]
                self.testingZ_Eb = self.testing_sequenceZ_Eb[:, :4]

        self.SAVABLE = ['params', 'dmat_dia',
                        'd_values',
                        'forming_corrY', 'testing_corrY', 'testing_corrZ_simple', 'testing_corrZ_episodic',
                        'data_description',
                        'opti_values', 'opti_values_test_s', 'opti_values_test_e',
                        'error_distances', 'error_types',
                        'retrieved_indices',
                        # 'retrieved_presequence',
                        # 'training_latent', 'training_categories', 'trainingY',
                        # 'forming_latent', 'forming_categories', 'formingY', 'retrievedY', 'formingZ', 'retrievedZ',
                        # 'testing_latent', 'testing_categories', 'testingY', 'testingZ_S', 'testingZ_E',
                        'sfa2S', 'sfa2E', 'learnerS', 'learnerE', 'whitener'
                        ]
        """
        This is an important attribute that must be changed as required. It controls which attributes are saved when
        pickling the object. By default it is set to::
        
           self.SAVABLE = ['params', 'dmat_dia',
                     'd_values',
                     'forming_corrY', 'testing_corrY', 'testing_corrZ_simple', 'testing_corrZ_episodic',
                     'data_description',
                     'opti_values', 'opti_values_test_s', 'opti_values_test_e',
                     'error_distances', 'error_types',
                     # 'retrieved_indices',
                     # 'retrieved_presequence',
                     # 'training_latent', 'training_categories', 'trainingY',
                     # 'forming_latent', 'forming_categories', 'formingY', 'retrievedY', 'formingZ', 'retrievedZ',
                     # 'testing_latent', 'testing_categories', 'testingY', 'testingZ_S', 'testingZ_E',
                     'sfa2S', 'sfa2E', 'learnerS', 'learnerE', 'whitener'
                     ]
           
        * *params* is the object of :py:class:`core.system_params.SysParamSet`
        * *dmat_dia* is the matrix of pattern-key distances in episodic memory
        * *...corr...* are matrices containing correlations between features and latent variables (including object identity)
        * *data_description* is a string describing the set of parameters
        * *opti_values* is only available if ``return_opti_values=True`` when calling 
          :py:func:`core.episodic.EpisodicMemory.retrieve_sequence` which is not usually the case in
          :py:func:`core.streamlined.program`.
        * *error_distances* and *error_types*: retrieval offsets and types of retrieval error. Only available if
          :py:data:`core.system_params.SysParamSet.st2` ``['memory']['return_err_values']`` is True.
        * *retrieved_indices* is a sequence of numbers that correspond to the indices of patterns in the original forming data
          in the order they were retrieved.
        * *retrieved_presequence* is a sequence that represents the sequence retrieved from episodic memory. While the
          retrieved sequence is in feature space of sfa1, *retrieved_presequence* is the sequence of the original input
          images that the retrieved feature patterns were extracted from. To make resulting pickle files easier to handle,
        * *sfa2S* and *sfa2E* are the trained SFA2 instances
        * *learnerS* and *learnerE* are the linear regressors trained to estimate latent variables and object identity
        * *whitener* is the normalizer trained on the output of SFA1 on forming data. An object of
          :py:class:`core.streamlined.normalizer`.
        
        *Y* or *Z* at the end of the variable means that it is the output of sfa1 or sfa2, respectively.
        *retrieved_presequence* is truncated to only the first 1000 frames and the feature sequences only contain the first
        4 features.
        """

        if self.params.st4['do_memtest']:
            self.SAVABLE.extend(['retr_simple', 'jumps_simple', 'indices_simple', 'org_simple',
                                'retr_episodic', 'jumps_episodic', 'indices_episodic', 'org_episodic'])
            if b:
                self.SAVABLE.extend(['retr_simpleb', 'jumps_simpleb', 'indices_simpleb', 'org_simpleb',
                                     'retr_episodicb', 'jumps_episodicb', 'indices_episodicb', 'org_episodicb'])

        if b:
            self.SAVABLE.extend(['testing_corrYb', 'testing_corrZ_simpleb', 'testing_corrZ_episodicb',
                                 'testing_latentb', 'testing_categoriesb', 'testingYb', 'testingZ_Sb', 'testingZ_Eb'])

        
    
    ############################## SCORING METRICS ##############################

    def jump_score_E(self):
        return np.sum(np.absolute(self.jumps_episodic))
    def jump_score_S(self):
        return np.sum(np.absolute(self.jumps_simple))
    def jump_score(self):
        return self.jump_score_S() -self.jump_score_E()
    
    def jump_count_E(self):
        return len(np.nonzero(self.jumps_episodic)[0])
    def jump_count_S(self):
        return len(np.nonzero(self.jumps_simple)[0])
    def jump_count(self):
        return self.jump_count_S() -self.jump_count_E()

    def edit_distance_S(self):
        return distance(self.org_simple, self.retr_simple,levenshtein)
    def edit_distance_E(self):
        return distance(self.org_episodic, self.retr_episodic,levenshtein)
    def edit_distance(self):
        return self.edit_distance_S() - self.edit_distance_E()
    
    def euclidean_distance_S(self):
        return distance(self.org_simple, self.retr_simple, euclidean)
    def euclidean_distance_E(self):
        return distance(self.org_episodic, self.retr_episodic, euclidean)
    def euclidean_distance(self):
        return self.euclidean_distance_S() - self.euclidean_distance_E()
    def me(self):
        return self
        
    ################## PLOT THINGS DIRECTLY FROM RESULT OBJECT ##################\
        
    def plot_hist1(self, featurename='testing_sequenceZ_E', fnumber=0) :
        pyplot.hist(getattr(self, featurename)[:,fnumber], alpha=0.5)
        
    def plot_sfa2inputs(self):
        tools.compare_inputs([self.retrieved_presequence, self.forming_sequenceX])

    def plot_episodic_perfect(self):
        tools.compare_inputs([self.retrieved_presequence, self.perfect_presequence])
    
    def plot_correlation(self, nFeatures=0, do_b=False):
        """
        Plots correlation matrices (SFA features vs latent variables including obejct identity) directly from result object

        :param nFeatures: Number of features to include in plot. If 0 or higher than actual number of features in
                          correlation matrix, all features are plotted
        :param do_b: Needed for recursion, do not change. Makes sure that correlations on second set of testing data (b),
                     if available, are also plotted.
        """
        if not do_b:
            CORRELATIONS = ["forming_corrY", "testing_corrY", "testing_corrZ_simple", "testing_corrZ_episodic"]
        else:
            CORRELATIONS = ["forming_corrY", "testing_corrYb", "testing_corrZ_simpleb", "testing_corrZ_episodicb"]
        fc, axarrc = pyplot.subplots(1, len(CORRELATIONS))
        for ax in axarrc:
            ax.set_ylabel("Latent variable")
        for a, ax in enumerate(axarrc):
            ax.set_xlabel("Feature number")
            ax.set_title([CORRELATIONS[a]])
        for p, val in enumerate(CORRELATIONS):
            corr_array = getattr(self, val)[:,:nFeatures] if nFeatures>0 else getattr(self,val)
            last_plot= axarrc[p].matshow(np.abs(corr_array),cmap=pyplot.cm.Blues, vmin=0, vmax=1)
            for (ii, jj), z in np.ndenumerate(corr_array):
               #axarrc[i,p].text(jj, ii, '{:.2f}'.format(z), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
               axarrc[p].text(jj, ii, '{:.0f}'.format(z*100), ha='center', va='center',color="white")
        # fc.subplots_adjust(right=0.8)
        # cbar_ax = fc.add_axes([0.85, 0.15, 0.05, 0.7])
        # fc.colorbar(last_plot, cax=cbar_ax)
        if self.params.st4b is not None and not do_b:
            self.plot_correlation(nFeatures, True)
        else:
            pyplot.show()

    def plot_delta(self):
        """
        Makes delta value barplot directly from Result object. Uses function
        :py:func:`plot_delta`

        """
        plot_delta(self)
        pyplot.show()

    
    def plot_scatter_dists(self, M=5000, weights=[1,1,1,1,0,0,0]):
        tools.distpicker(self.testing_sequenceX, self.testing_sequenceY, self.testing_latent,
                         self. testing_categories, M=M, weights=weights)