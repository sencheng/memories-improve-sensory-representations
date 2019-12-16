"""
This module can be used to get a quick overview on sets of results, that are available
as pickle files of :py:class:`core.result.Result` objects in one directory. This module is meant to be
used by **importing** it from a python or an ipython console. At the heart of this module
is a GUI window that shows a checkbox for every file of interest and has buttons to start
visualization functions. To select files of interest before creating the GUI window, use one of
the functions :py:func:`filter_by_file`, :py:func:`filter_by_condition`,
:py:func:`filter_by_indices`. To create the GUI window, call :py:func:`filegui`. If the GUI
is created without having selected files of interest, all files in the folder are selected.
Which folder the files are loaded from can be set in the source (:py:data:`RESPATH`) or before
filtering and creating the GUI by calling :py:func:`set_path`.

Example usage::

   import gridresults
   gridresult.filter_by_condition()
   gridresults.filegui()

This is what the GUI window looks like:

.. image:: ./img/gridres.png
   :width: 750px
   :alt: This is an alternative text
   :align: center

On the top, files can be checked and unchecked individually. Also, all files can be selected or
de-selected using the *all*-checkbox.

The buttons in the **green** box help to select sets of result files, as shown in the image.
The first spinbox is the index of the
first element to include, the second spinbox is the index of the first element to exclude and the
third spinbox is the stepsize. This is like the parameters to the *range* function in python.
When the desired numbers are entered, the checkbox can be used to select or deselect the respective
files. This can also be applied multiple times with different numbers.

The buttons in the **blue** box help to print parameters of the files on the console.

- **Parameters**: Print the parameters of all selected files separately
- **Compare**: Print only those parameters that are different between the selected files.
- **only st2**: If selected, only parameters from the dictionary *st2* are printed / compared.

The buttons in the **red** box help to compare delta values and feature-latent correlations between results files.
Three plot windows can be generated, one showing delta values of SFA output and the other two showing the quality of a linear
regressor to predict object coordinate and rotation angle. Values for SFA2S (simple scenario) and SFA2E (episodic scenario)
are plotted with separate lines.

- fist spinbox: The number determines how many subplots are generated on the plot windows. Once **Prep Graph** has been
                used, this number must not be changed before using **Show Graph**.
- **Prep Graph**: Plot the data into the current subplot. Delta values and prediction quality from all selected result files
                  are considered. Values are sorted on the x-axis by the variable given in **X-Axis Var**. Result windows
                  are not yet shown, because **Prep Graph** can be used multiple times to look at multiply result batches.
                  Next time **Prep Graph** is used, data is plotted into the next subplot. Wrap around is implemented such that
                  if **Prep Graph** is used more times than there are subplots, multiple plots go into one subplot. Plots are not
                  overwritten - the more oftern **Prep Graph** is used, the more lines you get.
- **Show Graph**: Show plot windows. After the windows are closed, all data is reset.
- **d count**: How many SFA features to consider. Delta value is averaged over all of these features (but see **indiv. d**) and
               regressor is trained on that many features.
- **indiv. d**: If checked, delta values are not averaged, but for all SFA feature an individual line is plotted.
- **Legend Text**: Label of the lines that are added by **Prep Graph**. It makes sense to change that every time **Prep Graph**
                   is used to be able to identify result batches. To differentiate between simple and episodic *_S* or *_E* are
                   appened, respectively.
- **X-Axis Var**: Variable to show on the x-axis of the plots.
                  Must be a member of the :py:class:`core.system_params.SysParamSet` object in the result files.
- **log x-axis**: Whether or not the x-axis has logarithmic scaling.

Finally, let's take a look at the buttons that are not in a colored frame:

- **d-val Grids**: Makes a plot window with a subplot for each selected result file. Plots
                   show delta values of data in every stage of processing and for every type
                   of input, in a matrix shape:

                   ================    ================    =============   ==============
                          .            Training            Forming         Testing
                   ================    ================    =============   ==============
                   **Input**           0.0                 0.0             0.0
                   **SFA1**            0.0                 0.0             0.0
                   **SFA2**            0.0                 0.0             0.0
                   ================    ================    =============   ==============
- **Histograms**: Makes a plot window with a subplot for each selected result file. Plots show
                  histograms of pattern-key distances in episodic memory. Also shows a line
                  at the percentile that is defined by the ``st2['memory']['smoothing_percentile']``
                  parameter of the result file. This was useful at some point for debugging purposes
                  but it most probably is not useful anymore.
- **Preseq**: Makes a plot window with a subplot for each selected result file. Plots show animations
              of the *retrieved_presequence* for each result. Using the function :py:func:`core.tools.compare_inputs`.
              See :py:data:`core.result.Result.SAVABLE`.
- **Features**: Visualizes latent variables and SFA features of data in every stage of processing and for
                every type of input, as subplots in a matrix shape (as above). If multiple result files
                are selected, features are visualized for just the first of those. Plots are zoomed in,
                user can navigate the region of interest with the arrow keys *right* and *left* and can
                change the zoom level with *up* and *down*.
- **res corrs**: Plots the feature-latent variable correlation matrices of the first selected res file.
                 Using the function :py:func:`core.result.Result.plot_correlation`.

"""

import re
import os
import pickle
import numpy as np
try:
    import tkinter as tk
except:
    import Tkinter as tk
try:
    from tkinter import messagebox
except:
    import tkMessageBox as messagebox
from matplotlib import pyplot
from matplotlib import gridspec
from core import tools, semantic, streamlined
import scipy.stats
import sys
import pdb

FILE = "gridconfig/resultparams.txt"
"""
Path to file used in :py:func:`filter_by_file`
"""

CFILE = "gridconfig/condition.txt"
"""
Path to file used in :py:func:`filter_by_condition`
"""

RES_PRE = "/local/results/"
"""
Path prefix so put before directory when calling :py:func:`set_path`
"""

RESPATH = RES_PRE + "reorder_o18b/"
"""
Default path to load result files from
"""

D_KEYS = ["sfa1", "forming_Y", "retrieved_Y", "sfa2S", "sfa2E", "testingY", "testingYb", "testingZ_S", "testingZ_E", "testingZ_Sb", "testingZ_Eb", "training_X", "forming_X", "testingX", "testingXb"]
"""
Keys to delta value dictionaries loaded from *Result* objects
"""

# D_KEYS = ["sfa1", "forming_Y", "retrieved_Y", "sfa2S", "sfa2E", "testingY", "testingZ_S", "testingZ_E"]
D_COLORS = ['r','b','b','r','r','b','b','k','k','k','k','g','g','g','g']
"""
Defines a color for every element in :py:data:`D_KEYS` (for bar plot)
"""

ZOOM_STEP = 100

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def dict_diff(d1, d2, prefix=None, exclude="data_description"):
    if not type(d1) == type(d2):
        return
    for k in d1:
        if k.startswith(exclude):
            continue
        v1 = d1[k]
        v2 = d2[k]
        if isinstance(v1, dict) and v1.keys() == v2.keys():
            if prefix:
                dict_diff(v1, v2, "{}['{}']".format(prefix,k))
                continue
            else:
                dict_diff(v1, v2, k)
                continue
        if not v1 == v2:
            if prefix:
                print("{}: {} -> {}".format("{}['{}']".format(prefix,k), str(v1), str(v2)))
            else:
                print("{}: {} -> {}".format(k, str(v1), str(v2)))

def difference_keys(dictlist):
    ret = []
    dict0 = dictlist[0]
    keys0 = dict0.keys()
    for dl in dictlist:
        if not dl.keys() == keys0:
            return None
    for k in keys0:
        if isinstance(dict0[k], dict):
            subret = difference_keys([di[k] for di in dictlist])
            # None means the keys are different, which means the entire dict should be regarded as different between the instances
            if subret is None:
                ret.append('["{}"]'.format(k))
            else:
                # This also works for an empty list, it just adds nothing
                for sr in subret:
                    ret.append('["{}"]{}'.format(k, sr))
        else:
            for d in dictlist:
                if not d[k] == dict0[k]:
                    ret.append('["{}"]'.format(k))
                    break
    return ret

def print_dict(d):
    for k,v in d.items():
        print("{}: {}".format(k, str(v)))

class FileGui:
    def call_histogram(self):
        obj.show_histograms(subselection=True)
    def call_d_val(self):
        obj.show_d_values(subselection=True)
    def grid_d_vals(self):
        obj.collect_d_values(subselection=True)
    def features(self):
        obj.plot_features(subselection=True)
    def call_parms(self):
        obj.show_params(only_st2=bool(self.check_st2.get()), subselection=True)
    def call_preseq(self):
        obj.show_presequences(subselection=True)
    def call_graph(self):
        obj.graph_stuff(int(self.subspin.get()), self.legvar.get(), self.xaxvar.get(), self.singledvar.get(), int(self.dcntspin.get()), subselection=True)
    def show_graph(self):
        obj.graph_show(bool(self.logvar.get()))
    def select_fun(self):
        st = int(self.selspin_start.get())
        en = int(self.selspin_end.get())
        stp = int(self.selspin_step.get())
        if st > en:
            en = st
            self.selspin_end.delete(0, "end")
            self.selspin_end.insert(0, str(en))
        val = obj.selselvar.get()
        for i in range(st, en+1, stp):
            obj.checkvars[i].set(val)
    def compare_parms(self):
        sell = []
        for e, el in enumerate(obj.checkvars):
            if el.get() == 1:
                sell.append(e)
        if len(sell) < 2:
            messagebox.showwarning("Bad selection", "Select two or more files")
            return
        only_st2 = self.check_st2.get()
        if len(sell) == 2:
            print("----------------------------")
            if only_st2:
                dict_diff(obj.selected_res[sell[0]].params.st2, obj.selected_res[sell[1]].params.st2)
            else:
                dict_diff(obj.selected_res[sell[0]].params.__dict__, obj.selected_res[sell[1]].params.__dict__)
            return
        else:
            subsel_res = [obj.selected_res[s] for s in sell]
            subsel_files = [obj.selected_files[s] for s in sell]
            if only_st2:
                difkeys = difference_keys([o.params.st2 for o in subsel_res])
            else:
                difkeys = difference_keys([o.params.__dict__ for o in subsel_res])
            if difkeys is None:
                obj.show_params(only_st2=bool(only_st2), subselection=True)
                return
            for sf, sr in zip(subsel_files, subsel_res):
                print("----------------------------")
                print(sf)
                for dk in difkeys:
                    if only_st2:
                        exec("print('{{}}: {{}}'.format(dk, sr.params.st2{}))".format(dk))
                    else:
                        exec ("print('{{}}: {{}}'.format(dk, sr.params.__dict__{}))".format(dk))
    def rescor(self):
        obj.rescor_plot(subselection=True)




    def __init__(self, obj):
        try:
            if obj.selected_files == [] or obj.selected_files is None:
                raise Exception
        except:
            print("Selecting all files")
            all_files = [st for st in os.listdir(RESPATH) if st.startswith("res") and st[::-1].startswith("p.")]
            all_files.sort(key=natural_keys)
            obj.selected_files = []
            obj.selected_res = []
            for fname in all_files:
                with open(RESPATH + fname, 'rb') as f:
                    res = pickle.load(f)
                obj.selected_files.append(fname)
                obj.selected_res.append(res)
        self.root = tk.Tk()
        self.root.title("Selected result files")
        cnt = len(obj.selected_files)
        cols = int(np.sqrt(cnt))
        rows = int(cnt/cols)+int(bool(cnt%cols))
        obj.checkvars = []
        for idx in range(cnt):
            rr = int(idx/cols)
            cc = idx%cols
            obj.checkvars.append(tk.IntVar())
            tk.Checkbutton(self.root, text=obj.selected_files[idx],variable=obj.checkvars[idx]).grid(row=rr, column=cc)
        def sel_all():
            setval = obj.selvar.get()
            for chvar in obj.checkvars:
                chvar.set(setval)
        obj.selvar = tk.IntVar()
        tk.Checkbutton(self.root, text="all", font="-size 9 -weight bold", variable=obj.selvar, command=sel_all).grid(row=rows, column=int(cols/2))
        tk.Button(self.root, text="d-val Grids", command=self.grid_d_vals).grid(row=rows + 1, column=0, pady=15)
        tk.Button(self.root, text="Histograms", command=self.call_histogram).grid(row=rows+1, column=1)
        tk.Button(self.root, text="Preseq", command=self.call_preseq).grid(row=rows + 1, column=2)
        tk.Button(self.root, text="Features", command=self.features).grid(row=rows + 1, column=3)
        self.check_st2 = tk.IntVar()
        tk.Button(self.root, text="Parameters", command=self.call_parms).grid(row=rows+1, column=4)
        tk.Checkbutton(self.root, text="only st2", variable=self.check_st2).grid(row=rows+1, column=5)
        tk.Button(self.root, text="Compare", command=self.compare_parms).grid(row=rows+1, column=6)
        self.check_st2.set(1)

        self.subspin = tk.Spinbox(self.root, from_=1, to=4, width=10)
        self.subspin.grid(row=rows + 2, column=0)
        tk.Button(self.root, text="Prep Graph", command=self.call_graph).grid(row=rows + 2, column=1)
        tk.Button(self.root, text="Show Graph", command=self.show_graph).grid(row=rows + 2, column=2)
        tk.Label(self.root, text="d count:").grid(row=rows + 2, column=3)
        self.dcntspin = tk.Spinbox(self.root, from_=1, to=16, width=10)
        self.dcntspin.delete(0, 'end')
        self.dcntspin.insert(0, 4)
        self.dcntspin.grid(row=rows + 2, column=4)
        self.singledvar = tk.IntVar()
        tk.Checkbutton(self.root, text="indiv. d", variable=self.singledvar).grid(row=rows + 2, column=5)
        tk.Button(self.root, text="res corrs", command=self.rescor).grid(row=rows + 2, column=6)

        tk.Label(self.root, text="Legend Text:").grid(row=rows + 3, column=0)
        self.legvar = tk.StringVar()
        tk.Entry(self.root, text="", textvariable=self.legvar, width=10).grid(row=rows + 3, column=1)
        tk.Label(self.root, text="X-Axis Var:").grid(row=rows + 3, column=2)
        self.xaxvar = tk.StringVar()
        self.xaxvar.set("st2['snippet_length']")
        tk.Entry(self.root, text="st2['snippet_length']", textvariable=self.xaxvar, width=10).grid(row=rows + 3, column=3)
        self.logvar = tk.IntVar()
        self.logvar.set(1)
        tk.Checkbutton(self.root, text="log x-axis", variable=self.logvar).grid(row=rows + 3, column=4)

        self.selspin_start = tk.Spinbox(self.root, from_=0, to=500, width=10)
        self.selspin_end = tk.Spinbox(self.root, from_=0, to=500, width=10)
        self.selspin_step = tk.Spinbox(self.root, from_=1, to=500, width=10)
        self.selspin_start.grid(row=rows + 4, column=0, pady = 15)
        self.selspin_end.grid(row=rows + 4, column=1)
        self.selspin_step.grid(row=rows + 4, column=2)
        obj.selselvar = tk.IntVar()
        tk.Checkbutton(self.root, text="Select", variable=obj.selselvar, command=self.select_fun).grid(row=rows + 4, column=3)

        self.root.mainloop()

class Gridres:
    def __init__(self):
        self.respath = RESPATH
        self.sub_index = -1
        self.sub_wrap = 0

    def filter_by_condition(self, ffile = CFILE):
        with open(ffile, 'r') as f:
            lines = [line.split('#')[0] for line in f if not line.startswith('#')]
        all_files = [st for st in os.listdir(self.respath) if st.startswith("res") and st[::-1].startswith("p.")]
        all_files.sort(key=natural_keys)
        self.selected_files = []
        self.selected_res = []
        for fname in all_files:
            with open(self.respath+fname, 'rb') as f:
                res = pickle.load(f)
            want_this = True
            for tst in lines:
                if not eval(tst):
                    want_this = False
                    break
            if want_this:
                self.selected_files.append(fname)
                self.selected_res.append(res)

    def filter_by_file(self, ffile = FILE):
        with open(ffile, 'r') as f:
            lines = [line.split("=") for line in f]
        teststrings = ["res.params.{} == {}".format(l[0], '='.join(l[1:]).split('#')[0]) for l in lines if len(l) > 1 and len(l[0]) > 4 and not l[0].startswith('#') and not "print" in l[1]]
        p_temp = [p[0] for p in lines if len(p) > 1 and len(p[0]) > 4 and not p[0].startswith('#') and "print" in p[1]]
        printstrings = []
        for st in p_temp:
            if '"' in st:
                printstrings.append("res.params.data_description += '\\n{}: ' + str(res.params.{})".format(st.split()[0], st))
            else:
                printstrings.append('res.params.data_description += "\\n{}: " + str(res.params.{})'.format(st.split()[0], st))


        all_files = [st for st in os.listdir(self.respath) if st.startswith("res") and st[::-1].startswith("p.")]
        all_files.sort(key=natural_keys)
        self.selected_files = []
        self.selected_res = []
        for fname in all_files:
            with open(self.respath+fname, 'rb') as f:
                res = pickle.load(f)
            want_this = True
            for tst in teststrings:
                if not eval(tst):
                    want_this = False
                    break
            if want_this:
                for prt in printstrings:
                    exec(prt)
                self.selected_files.append(fname)
                self.selected_res.append(res)

    def filter_by_indices(self, indices):
        all_files = [st for st in os.listdir(self.respath) if st.startswith("res") and st[::-1].startswith("p.")]
        all_files.sort(key=natural_keys)
        self.selected_files = []
        self.selected_res = []
        for ind in indices:
            fname = all_files[ind]
            with open(self.respath + fname, 'rb') as f:
                res = pickle.load(f)
            self.selected_files.append(fname)
            self.selected_res.append(res)

    def rescor_plot(self, subselection=False):
        sel_files = self.selected_files if not subselection else [self.selected_files[i] for i in range(len(self.selected_files)) if self.checkvars[i].get() == 1]
        for fname in sel_files[0:1]:
            with open(RESPATH + fname, 'rb') as f:
                res = pickle.load(f)
            res.plot_correlation()

    def graph_stuff(self, subs, legvar, xaxvar, singledvar, dcnt=16, subselection=False):

        # b: blue, g: green, r: red, c: cyan, m: magenta, y: yellow, k: black, w: white
        colors = ['b', 'r', 'g', 'k', 'y', 'm', 'c']
        linest = ['--', '-']

        def graph_close(evt):
            self.sub_index = -1
            self.sub_wrap = 0

        sel_files = self.selected_files if not subselection else [self.selected_files[i] for i in range(len(self.selected_files)) if self.checkvars[i].get() == 1]
        # sel_res = self.selected_res if not subselection else [self.selected_res[i] for i in range(len(self.selected_res)) if self.checkvars[i].get() == 1]
        cnt = len(sel_files)

        if self.sub_index == -1:
            self.graph_rows = int(np.sqrt(subs))
            self.graph_cols = int(subs / self.graph_rows) + int(bool(subs % self.graph_rows))
            self.graph_fd, self.graph_axd = pyplot.subplots(self.graph_rows, self.graph_cols, squeeze=False, sharex=True, sharey=True)
            self.graph_fxy, self.graph_axxy = pyplot.subplots(self.graph_rows, self.graph_cols, squeeze=False, sharex=True, sharey=True)
            self.graph_fphi, self.graph_axphi = pyplot.subplots(self.graph_rows, self.graph_cols, squeeze=False, sharex=True, sharey=True)
            if singledvar:
                self.graph_fsingxy, self.graph_axsingxy = pyplot.subplots(self.graph_rows, self.graph_cols, squeeze=False, sharex=True, sharey=True)
                self.graph_fsingxy.suptitle("xy correlation of indiv. features")
            self.graph_fd.suptitle("avg d-values")
            self.graph_fxy.suptitle("xy prediction accuracy")
            self.graph_fphi.suptitle("phi prediction accuracy")
            self.graph_fd.canvas.mpl_connect('close_event', graph_close)
            self.sub_index = 0

        if self.sub_index == self.graph_axd.shape[0]*self.graph_axd.shape[1]:
            self.sub_index = 0
            self.sub_wrap += 1
        ci = self.sub_index//self.graph_rows
        ri = self.sub_index % self.graph_rows
        color_sel = colors[self.sub_wrap]

        if xaxvar != "":
            x = []
            if xaxvar.startswith('"') or xaxvar.startswith("'"):
                xaxvar = "[" + xaxvar + "]"
            elif not xaxvar.startswith('.'):
                xaxvar = '.' + xaxvar
            # for res in sel_res:
            for fname in sel_files:
                with open(RESPATH + fname, 'rb') as f:
                    res = pickle.load(f)
                x.append(eval("res.params"+xaxvar))
        else:
            x = list(range(cnt))

        testing_data = np.load(RESPATH+"testing0.npz")
        seq2, cat2, lat2, _ = testing_data['testing_sequenceX'], testing_data['testing_categories'], testing_data['testing_latent'], testing_data['testing_ranges']

        axd = self.graph_axd[ri, ci]
        axxy = self.graph_axxy[ri, ci]
        axphi = self.graph_axphi[ri, ci]
        if singledvar:
            axsingxy = self.graph_axsingxy[ri,ci]
        dZ_S = []
        dZ_E = []
        xyZ_S = []
        xyZ_E = []
        phiZ_S = []
        phiZ_E = []
        xySing_S = []
        xySing_E = []
        for i in range(cnt):
            fname = sel_files[i]
            with open(RESPATH + fname, 'rb') as f:
                res = pickle.load(f)
            # res = sel_res[i]
            # vS = np.sort(res.d_values['testingZ_S'])[:dcnt]
            # vE = np.sort(res.d_values['testingZ_E'])[:dcnt]
            vS = res.d_values['testingZ_S'][:dcnt]
            vE = res.d_values['testingZ_E'][:dcnt]
            cS = np.max(np.abs(res.testing_corrZ_simple[:2,:dcnt]),axis=0)
            cE = np.max(np.abs(res.testing_corrZ_episodic[:2, :dcnt]), axis=0)
            if singledvar:
                for di in range(dcnt):
                    if i == 0:
                        dZ_S.append([])
                        dZ_E.append([])
                        xySing_S.append([])
                        xySing_E.append([])
                    dZ_S[di].append(vS[di])
                    dZ_E[di].append(vE[di])
                    xySing_S[di].append(cS[di])
                    xySing_E[di].append(cE[di])
            else:
                dZ_S.append(np.mean(vS) if isinstance(vS, np.ndarray) else vS)
                dZ_E.append(np.mean(vE) if isinstance(vE, np.ndarray) else vE)

            sfa1 = semantic.load_SFA(RESPATH + res.data_description + "train0.sfa")
            yy2 = semantic.exec_SFA(sfa1, seq2)
            yy2_w = streamlined.normalizer(yy2, res.params.normalization)(yy2)
            zz2S = semantic.exec_SFA(res.sfa2S, yy2_w)
            zz2S_w = streamlined.normalizer(zz2S, res.params.normalization)(zz2S)
            zz2E = semantic.exec_SFA(res.sfa2E, yy2_w)
            zz2E_w = streamlined.normalizer(zz2E, res.params.normalization)(zz2E)

            predictionS = res.learnerS.predict(zz2S_w)
            predictionE = res.learnerE.predict(zz2E_w)
            _, _, r_valueX_S, _, _ = scipy.stats.linregress(lat2[:, 0], predictionS[:, 0])
            _, _, r_valueY_S, _, _ = scipy.stats.linregress(lat2[:, 1], predictionS[:, 1])
            _, _, r_valueCosphi_S, _, _ = scipy.stats.linregress(lat2[:, 2], predictionS[:, 2])
            _, _, r_valueSinphi_S, _, _ = scipy.stats.linregress(lat2[:, 3], predictionS[:, 3])
            _, _, r_valueX_E, _, _ = scipy.stats.linregress(lat2[:, 0], predictionE[:, 0])
            _, _, r_valueY_E, _, _ = scipy.stats.linregress(lat2[:, 1], predictionE[:, 1])
            _, _, r_valueCosphi_E, _, _ = scipy.stats.linregress(lat2[:, 2], predictionE[:, 2])
            _, _, r_valueSinphi_E, _, _ = scipy.stats.linregress(lat2[:, 3], predictionE[:, 3])

            xyZ_S.append(np.mean((r_valueX_S, r_valueY_S)))
            phiZ_S.append(np.mean((r_valueCosphi_S, r_valueSinphi_S)))
            xyZ_E.append(np.mean((r_valueX_E, r_valueY_E)))
            phiZ_E.append(np.mean((r_valueCosphi_E, r_valueSinphi_E)))

            # cS = np.abs(res.testing_corrZ_simple)
            # # cS = tools.feature_latent_correlation(res.testingZ_S, res.testing_latent)
            # xyZ_S.append(np.mean([np.max(cS[0, :]), np.max(cS[1, :])]))
            # phiZ_S.append(np.mean([np.max(cS[2, :]), np.max(cS[3, :])]))
            # cE = np.abs(res.testing_corrZ_episodic)
            # # cE = tools.feature_latent_correlation(res.testingZ_E, res.testing_latent)
            # xyZ_E.append(np.mean([np.max(cE[0, :]), np.max(cE[1, :])]))
            # phiZ_E.append(np.mean([np.max(cE[2, :]), np.max(cE[3, :])]))
            if self.sub_wrap == 0:
                # testing_d = tools.delta_diff(res.testing_latent)
                testing_d = tools.delta_diff(lat2)
        lab_base = legvar if legvar != "" else sel_files[0]
        if self.sub_wrap == 0:
            xy_d = np.mean([testing_d[0], testing_d[1]])
            phi_d = np.mean([testing_d[2], testing_d[3]])
        if singledvar:
            for di in range(dcnt):
                axd.plot(x, dZ_S[di], label=lab_base + "_S", c=colors[self.sub_wrap], ls=linest[0])
                axd.plot(x, dZ_E[di], label=lab_base + "_E", c=colors[self.sub_wrap], ls=linest[1])
                axsingxy.plot(x, xySing_S[di], label=lab_base + "_S", c=colors[self.sub_wrap], ls=linest[0])
                axsingxy.plot(x, xySing_E[di], label=lab_base + "_E", c=colors[self.sub_wrap], ls=linest[1])
        else:
            axd.plot(x, dZ_S, label=lab_base + "_S", c=colors[self.sub_wrap], ls=linest[0])
            axd.plot(x, dZ_E, label=lab_base + "_E", c=colors[self.sub_wrap], ls=linest[1])
        if self.sub_wrap == 0:
            axd.plot(x, [xy_d]*len(x), label='XY', c=colors[-1], ls=linest[0])
            axd.plot(x, [phi_d] * len(x), label='PHI', c=colors[-1], ls=linest[1])
        axxy.plot(x, xyZ_S, label=lab_base + "_S", c=colors[self.sub_wrap], ls=linest[0])
        axxy.plot(x, xyZ_E, label=lab_base + "_E", c=colors[self.sub_wrap], ls=linest[1])
        axphi.plot(x, phiZ_S, label=lab_base + "_S", c=colors[self.sub_wrap], ls=linest[0])
        axphi.plot(x, phiZ_E, label=lab_base + "_E", c=colors[self.sub_wrap], ls=linest[1])
        # axsingxy.plot(x, np.mean(xySing_S,axis=0), label=lab_base + "_S", c=colors[self.sub_wrap], ls=linest[0])
        # axsingxy.plot(x, np.mean(xySing_E,axis=0), label=lab_base + "_E", c=colors[self.sub_wrap], ls=linest[1])

        self.sub_index += 1

    def graph_show(self, log=False):
        for ci in range(self.graph_cols):
            for ri in range(self.graph_rows):
                self.graph_axd[ri,ci].legend()
                self.graph_axxy[ri, ci].legend()
                self.graph_axphi[ri, ci].legend()
                try:
                    self.graph_axsingxy[ri, ci].legend()
                except:
                    pass
                if log:
                    self.graph_axd[ri, ci].set_xscale('log')
                    self.graph_axxy[ri, ci].set_xscale('log')
                    self.graph_axphi[ri, ci].set_xscale('log')
                    try:
                        self.graph_axsingxy[ri, ci].set_xscale('log')
                    except:
                        pass
        self.graph_fd.show()
        self.graph_fxy.show()
        self.graph_fphi.show()
        try:
            self.graph_fsingxy.show()
        except:
            pass

    def show_presequences(self, subselection=False):
        if "retrieved_presequence" in self.selected_res[0].__dict__:
            seqlist = self.selected_res if not subselection else [self.selected_res[i].retrieved_presequence for i in range(len(self.selected_res)) if self.checkvars[i].get() == 1]
            tools.compare_inputs(seqlist)
        else:
            messagebox.showwarning("Potato", "Retrieved Presequences are not included in the results-files.")

    def show_d_values(self, subselection=False):
        # DEPRECATED
        sel_files = self.selected_files if not subselection else [self.selected_files[i] for i in range(len(self.selected_files)) if self.checkvars[i].get() == 1]
        sel_res = self.selected_res if not subselection else [self.selected_res[i] for i in range(len(self.selected_res)) if self.checkvars[i].get() == 1]
        cnt = len(sel_files)
        cols = int(np.sqrt(cnt))
        rows = int(cnt / cols) + int(bool(cnt % cols))
        ax1 = pyplot.subplot(cols,rows,1)
        for fi, (fname, res) in enumerate(zip(sel_files,sel_res)):
            print(fname)
            print(res.params.data_description)
            print("")
            print("------------------------------------")
            dv = res.d_values
            if fi == 0:
                ax = ax1
            else:
                ax = pyplot.subplot(cols, rows, fi+1, sharex=ax1, sharey=ax1)
            ax.set_title(fname)
            for i, (k, c) in enumerate(zip(D_KEYS, D_COLORS)):
                v = dv[k]
                v_mean_16 = np.mean(np.sort(v)[:16]) if isinstance(v, np.ndarray) else v  # check if v is nan
                v_mean = np.mean(v)
                ax.bar(i, v_mean_16, width=0.6, color=c)
                ax.bar(i+0.2, v_mean, width=0.2, color='y')
                ax.text(i+0.3, v_mean_16, '{:.3f}'.format(v_mean_16), ha='center', va='bottom', color="black", rotation=90)
            ax.set_xticks([j + 0.3 for j in range(len(D_KEYS))])
            ax.set_xticklabels(D_KEYS, rotation=70)
        pyplot.show()

    def plot_features(self, subselection=False):

        global zoom_step
        zoom_step = ZOOM_STEP
        
        sel_files = self.selected_files if not subselection else [self.selected_files[i] for i in range(len(self.selected_files)) if self.checkvars[i].get() == 1]
        sel_res = self.selected_res if not subselection else [self.selected_res[i] for i in range(len(self.selected_res)) if self.checkvars[i].get() == 1]
        cnt = len(sel_files)

        fname = sel_files[0]
        res = sel_res[0]

        b = True
        try:
            if res.params.st4b is None:
                b = False
        except:
            b = False

        fig = pyplot.figure()
        fig.suptitle(fname)
        gs = gridspec.GridSpec(3,8) if b else gridspec.GridSpec(3,6)

        tit0 = fig.add_subplot(gs[0,0])
        tit0.text(0,0, "Input")
        tit0.axis('off')
        tit0.set_ylim(-1,1)
        tit1 = fig.add_subplot(gs[1, 0])
        tit1.text(0, 0, "SFA1")
        tit1.axis('off')
        tit1.set_ylim(-1, 1)
        tit2 = fig.add_subplot(gs[2, 0])
        tit2.text(0, 0, "SFA2")
        tit2.axis('off')
        tit2.set_ylim(-1, 1)

        try:
            train_len = len(res.trainingY)
            train_len = len(res.training_latent)
        except:
            pass
        ax00 = fig.add_subplot(gs[0,1])
        lin = []
        try:
            for i in range(4):
                lin.append(ax00.plot(range(train_len), res.training_latent[:,i])[0])
            lin.append(ax00.plot(range(train_len), res.training_categories)[0])
        except:
            pass
        ax10 = fig.add_subplot(gs[1,1])
        try:
            for i in range(4):
                ax10.plot(range(train_len), res.trainingY[:,i])
        except:
            pass
        ax00.set_title("Training")
        pyplot.figlegend(lin, ['1', '2', '3', '4', 'cat'], 'lower left')

        try:
            form_len = len(res.formingY)
            form_len = len(res.forming_latent)
        except:
            pass
        try:
            retr_len = len(res.retrievedY)
        except:
            pass
        ax01_2 = fig.add_subplot(gs[0,2:4])
        try:
            for i in range(4):
                ax01_2.plot(range(form_len), res.forming_latent[:,i])
            ax01_2.plot(range(form_len), res.forming_categories)
        except:
            pass
        ax11 = fig.add_subplot(gs[1,2])
        for i in range(4):
            ax11.plot(range(form_len), res.formingY[:,i])
        ax12 = fig.add_subplot(gs[1, 3])
        try:
            for i in range(4):
                ax12.plot(range(retr_len), res.retrievedY[:,i])
        except:
            pass
        ax21 = fig.add_subplot(gs[2, 2])
        for i in range(4):
            ax21.plot(range(form_len), res.formingZ[:,i])
        ax22 = fig.add_subplot(gs[2, 3])
        for i in range(4):
            ax22.plot(range(retr_len), res.retrievedZ[:,i])
        ax01_2.set_title("Forming")

        try:
            test_len = len(res.testingY)
            test_len = len(res.testing_latent)
        except:
            pass
        ax03_4 = fig.add_subplot(gs[0, 4:6])
        try:
            for i in range(4):
                ax03_4.plot(range(test_len), res.testing_latent[:,i])
            ax03_4.plot(range(test_len), res.testing_categories)
        except:
            pass
        ax13_4 = fig.add_subplot(gs[1, 4:6])
        for i in range(4):
            ax13_4.plot(range(test_len), res.testingY[:,i])
        ax23 = fig.add_subplot(gs[2, 4])
        for i in range(4):
            ax23.plot(range(test_len), res.testingZ_S[:,i])
        ax24 = fig.add_subplot(gs[2, 5])
        for i in range(4):
            ax24.plot(range(test_len), res.testingZ_E[:,i])
        ax03_4.set_title("Testing")

        if b:
            try:
                test_lenb = len(res.testingYb)
                test_lenb = len(res.testing_latentb)
            except:
                pass
            ax05_6 = fig.add_subplot(gs[0, 6:8])
            try:
                for i in range(4):
                    ax05_6.plot(range(test_lenb), res.testing_latentb[:,i])
                ax05_6.plot(range(test_lenb), res.testing_categoriesb)
            except:
                pass
            ax15_6 = fig.add_subplot(gs[1, 6:8])
            for i in range(4):
                ax15_6.plot(range(test_lenb), res.testingYb[:,i])
            ax25 = fig.add_subplot(gs[2, 6])
            for i in range(4):
                ax25.plot(range(test_lenb), res.testingZ_Sb[:,i])
            ax26 = fig.add_subplot(gs[2, 7])
            for i in range(4):
                ax26.plot(range(test_lenb), res.testingZ_Eb[:,i])
            ax05_6.set_title("Testing")

        ran = [0,zoom_step]
        axlist = [ax00, ax10, ax01_2, ax11, ax12, ax21, ax22, ax03_4, ax13_4, ax23, ax24]
        for anax in axlist:
            anax.set_xlim(ran)
        if b:
            axlistb = [ax05_6, ax15_6, ax25, ax26]
            for anaxb in axlistb:
                anaxb.set_xlim(ran)

        def key_event(e):
            global zoom_step
            ran = list(ax00.get_xlim())
            b = True
            try:
                _ = ax05_6
            except:
                b = False
                pass

            if e.key == "right":
                ran[0] += zoom_step
                ran[1] += zoom_step
            elif e.key == "left":
                ran[0] -= zoom_step
                ran[1] -= zoom_step
                if ran[0] < 0:
                    ran[0] = 0
                    ran[1] = zoom_step
            elif e.key == "up":
                if not zoom_step <= 25:
                    zoom_step -= 25
                    ran[1] -= 25
            elif e.key == "down":
                zoom_step += 25
                ran[1] += 25
            else:
                return

            axlist = [ax00, ax10, ax01_2, ax11, ax12, ax21, ax22, ax03_4, ax13_4, ax23, ax24]
            for anax in axlist:
                anax.set_xlim(ran)
            if b:
                axlistb = [ax05_6, ax15_6, ax25, ax26]
                for anaxb in axlistb:
                    anaxb.set_xlim(ran)

            fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', key_event)
        pyplot.show()

    def collect_d_values(self, subselection=False):

        FONT_AUTO = False
        FONT_SIZE = 12

        sel_files = self.selected_files if not subselection else [self.selected_files[i] for i in range(len(self.selected_files)) if self.checkvars[i].get() == 1]
        sel_res = self.selected_res if not subselection else [self.selected_res[i] for i in range(len(self.selected_res)) if self.checkvars[i].get() == 1]
        cnt = len(sel_files)
        cols = int(np.sqrt(cnt))
        rows = int(cnt / cols) + int(bool(cnt % cols))
        ax1 = pyplot.subplot(cols, rows, 1)
        pyplot.tight_layout()

        for fi, (fname, res) in enumerate(zip(sel_files, sel_res)):

            print(fname)
            try:
                print(res.params.data_description)
            except:
                print("No description")
            print("")
            print("------------------------------------")
            dv = res.d_values
            if fi == 0:
                ax = ax1
            else:
                ax = pyplot.subplot(cols, rows, fi+1, sharex=ax1, sharey=ax1)
            ax.set_title(fname)
            pyplot.axis('off')

            b = True
            try:
                if res.params.st4b is None:
                    b = False
            except:
                b = False
           
            dv16 = {}
            
            for i, (k, c) in enumerate(zip(D_KEYS, D_COLORS)):
                try:
                    v = dv[k]
                except:
                    v = np.nan    # if key does not exist (e.g. because results files are from before I introduced X d_values)
                v_mean_16 = np.mean(np.sort(v)[:16]) if isinstance(v, np.ndarray) else v
                dv16[k] =  v_mean_16

            cell_text1 = [[format(dv16["training_X"], '.2f'),format(dv16["forming_X"], '.2f'),format(dv16["testingX"], '.2f')]]
            cell_text2 = [[format(dv16["sfa1"], '.2f'),format(dv16["forming_Y"], '.2f'), format(dv16["retrieved_Y"], '.2f'), format(dv16["testingY"], '.2f')]]
            cell_text3 = [["-",format(dv16["sfa2S"], '.2f'), format(dv16["sfa2E"], '.2f'), format(dv16["testingZ_S"], '.2f'), format(dv16["testingZ_E"], '.2f')]]
            if b:
                cell_text1[0].append(format(dv16["testingXb"], '.2f'))
                cell_text2[0].append(format(dv16["testingYb"], '.2f'))
                cell_text3[0].extend([format(dv16["testingZ_Sb"], '.2f'), format(dv16["testingZ_Eb"], '.2f')])

            colNames1 = ['Training', 'Forming', 'Testing']
            if b:
                colNames1.append('Testingb')
            rowNames1 = ['Input']
            the_table = ax.table(cellText = cell_text1,rowLabels = rowNames1, colLabels = colNames1, cellLoc='center', bbox = [0, .75, 1, .2])
            the_table.auto_set_font_size(FONT_AUTO)
            the_table.set_fontsize(FONT_SIZE)
            cellDict=the_table.get_celld()
            cellDict[(0,0)].set_width(0.2)
            cellDict[(0,1)].set_width(0.4)
            cellDict[(0,2)].set_width(0.4)
            cellDict[(1,0)].set_width(0.2)
            cellDict[(1,1)].set_width(0.4)
            cellDict[(1,2)].set_width(0.4)
            if b:
                cellDict[(0, 3)].set_width(0.4)
                cellDict[(1, 3)].set_width(0.4)

            rowNames2 = ['SFA1']
            the_table = ax.table(cellText = cell_text2,rowLabels = rowNames2, cellLoc='center', bbox = [0, .65, 1, .1])
            the_table.auto_set_font_size(FONT_AUTO)
            the_table.set_fontsize(FONT_SIZE)
            cellDict=the_table.get_celld()
            cellDict[(0,0)].set_width(0.2)
            cellDict[(0,1)].set_width(0.2)
            cellDict[(0,2)].set_width(0.2)
            cellDict[(0,3)].set_width(0.4)
            if b:
                cellDict[(0, 4)].set_width(0.4)
            
            rowNames3 = ['SFA2']         
            the_table = ax.table(cellText = cell_text3, rowLabels = rowNames3, cellLoc='center', bbox = [0, .55, 1, .1])
            the_table.auto_set_font_size(FONT_AUTO)
            the_table.set_fontsize(FONT_SIZE)
            cellDict=the_table.get_celld()
            cellDict[(0,0)].set_width(0.2)
            cellDict[(0,1)].set_width(0.2)
            cellDict[(0,2)].set_width(0.2)
            cellDict[(0,3)].set_width(0.2)
            cellDict[(0,4)].set_width(0.2)
            if b:
                cellDict[(0, 5)].set_width(0.2)
                cellDict[(0, 6)].set_width(0.2)
            
            #Create colormap for below the table of values---------------------
            dv16_colors = dict(dv16)
                  
            for key in dv16_colors:
                if not np.isnan(dv16_colors[key]):
                    dv16_colors[key] = np.log(dv16_colors[key])
            
            #Arbitrary initialization
            dv16_min = 10000.
            dv16_max = -10000.
            
            #Find max and min values of dv_16 colors
            for key in dv16_colors:
                if dv16_colors[key] > dv16_max:
                    dv16_max = dv16_colors[key]
                if dv16_colors[key] < dv16_min:
                    dv16_min = dv16_colors[key]
            
            # Begins as a copy of dv16, but modifies itself to become values between 0 and 1
            for key in dv16_colors:
                dv16_colors[key] = dv16_colors[key] - dv16_min
                dv16_colors[key] = str(dv16_colors[key] / (dv16_max - dv16_min))

            for key in dv16_colors:
                dv16_colors[key] = str(1 - float(dv16_colors[key]))
            
            rowNames4 = ['Input']
            colors4 = [[]]
            cell_text4 = [[]]
            keylist = ["training_X", "forming_X", "testingX"]
            if b:
                keylist.append("testingXb")
            for k in keylist:
                try:
                    if dv16_colors[k] == "nan":
                        raise Exception
                    colors4[0].append(dv16_colors[k])
                    cell_text4[0].append("")
                except:
                    colors4[0].append("1")
                    cell_text4[0].append("N/A")
            # colors4 = [[dv16_colors["training_X"], dv16_colors["forming_X"], dv16_colors["testingX"]]]
            # cell_text4 = [['','','']] #To get around an old matplotlib bug
            the_table = ax.table(cellText = cell_text4, cellColours = colors4, rowLabels = rowNames4, cellLoc='center', bbox = [0, .45, 1, .1])
            the_table.auto_set_font_size(FONT_AUTO)
            the_table.set_fontsize(FONT_SIZE)
            cellDict=the_table.get_celld()
            cellDict[(0,0)].set_width(0.2)
            cellDict[(0,1)].set_width(0.4)
            cellDict[(0,2)].set_width(0.4)
            if b:
                cellDict[(0, 3)].set_width(0.4)
            
            rowNames5 = ['SFA1']
            colors5 = [[]]
            cell_text5 = [[]]
            keylist = ["sfa1", "forming_Y", "retrieved_Y", "testingY"]
            if b:
                keylist.append("testingYb")
            for k in keylist:
                try:
                    if dv16_colors[k] == "nan":
                        raise Exception
                    colors5[0].append(dv16_colors[k])
                    cell_text5[0].append("")
                except:
                    colors5[0].append("1")
                    cell_text5[0].append("N/A")
            # colors5 = [[dv16_colors["sfa1"],dv16_colors["forming_Y"],dv16_colors["retrieved_Y"],dv16_colors["testingY"]]]
            # cell_text5 = [['','','','']]
            the_table = ax.table(cellText = cell_text5, cellColours = colors5, rowLabels = rowNames5, cellLoc='center', bbox = [0, .35, 1, .1])
            the_table.auto_set_font_size(FONT_AUTO)
            the_table.set_fontsize(FONT_SIZE)
            cellDict=the_table.get_celld()
            cellDict[(0,0)].set_width(0.2)
            cellDict[(0,1)].set_width(0.2)
            cellDict[(0,2)].set_width(0.2)
            cellDict[(0,3)].set_width(0.4)
            if b:
                cellDict[(0, 4)].set_width(0.4)
            
            rowNames6 = ['SFA2']
            colors6 = [["1"]]
            cell_text6 = [["N/A"]]
            keylist = ["sfa2S", "sfa2E", "testingZ_S", "testingZ_E"]
            if b:
                keylist.extend(["testingZ_Sb", "testingZ_Eb"])
            for k in keylist:
                try:
                    if dv16_colors[k] == "nan":
                        raise Exception
                    colors6[0].append(dv16_colors[k])
                    cell_text6[0].append("")
                except:
                    colors6[0].append("1")
                    cell_text6[0].append("N/A")
            # colors6 = [["1",dv16_colors["sfa2S"],dv16_colors["sfa2E"],dv16_colors["testingZ_S"],dv16_colors["testingZ_E"]]]
            # cell_text6 = [["N/A",'','','','']]
            the_table = ax.table(cellText = cell_text6, cellColours = colors6, rowLabels = rowNames6, cellLoc='center', bbox = [0, .25, 1, .1])
            the_table.auto_set_font_size(FONT_AUTO)
            the_table.set_fontsize(FONT_SIZE)
            cellDict=the_table.get_celld()
            cellDict[(0,0)].set_width(0.2)
            cellDict[(0,1)].set_width(0.2)
            cellDict[(0,2)].set_width(0.2)
            cellDict[(0,3)].set_width(0.2)
            cellDict[(0,4)].set_width(0.2)
            if b:
                cellDict[(0, 5)].set_width(0.2)
                cellDict[(0, 6)].set_width(0.2)
            
            #newLine = plt.table(cellColours=cell_text,
                      #rowLabels=rows,
                      #rowColours=colors,
                      #colLabels=columns,
                      #loc='bottom')
            
        pyplot.show()

    def show_histograms(self, subselection=False):
        sel_files = self.selected_files if not subselection else [self.selected_files[i] for i in range(len(self.selected_files)) if self.checkvars[i].get() == 1]
        sel_res = self.selected_res if not subselection else [self.selected_res[i] for i in range(len(self.selected_res)) if self.checkvars[i].get() == 1]
        cnt = len(sel_files)
        cols = int(np.sqrt(cnt))
        rows = int(cnt / cols) + int(bool(cnt % cols))
        ax1 = pyplot.subplot(cols, rows, 1)
        for fi, (fname, res) in enumerate(zip(sel_files, sel_res)):
            print(fname)
            print(res.params.data_description)
            print("")
            print("------------------------------------")
            if fi == 0:
                ax = ax1
            else:
                ax = pyplot.subplot(cols, rows, fi + 1)
            ax.set_title(fname)
            dia = res.dmat_dia
            hst = ax.hist(dia, 100)
            maxy = np.max(hst[0])
            perc = res.params.st2['memory']['smoothing_percentile']
            x = np.percentile(dia, perc)
            ax.plot((x, x), (0, maxy), 'r')
            ax.text(x, maxy, "p={}".format(perc), ha = 'right', color = 'black')
        pyplot.show()
        
    def show_params(self, only_st2=True, subselection=False):
        sel_files = self.selected_files if not subselection else [self.selected_files[i] for i in range(len(self.selected_files)) if self.checkvars[i].get() == 1]
        sel_res = self.selected_res if not subselection else [self.selected_res[i] for i in range(len(self.selected_res)) if self.checkvars[i].get() == 1]
        for fname, res in zip(sel_files, sel_res):
            print(fname)
            print("")
            if only_st2:
                print_dict(res.params.st2)
            else:
                print_dict(res.params.__dict__)
            print("------------------------------------")

obj = Gridres()

def set_path(arg):
    """
    Set the path to load result files from. Using this function
    overwrites the default value :py:data:`RESPATH`.

    :param arg: path to be appended to :py:data:`RES_PRE`
    """
    if not arg[-1] == "/":
        arg += "/"
    obj.respath = RES_PRE + arg

def filter_by_file(ffile = FILE):
    """
    Select results files by the criteria in the given file
    (default :py:data:`FILE`). The file is a text file containing
    parameter definitions that can be evaluated as members of a
    :py:class:`core.system_params.SysParamSet` object. For instance,
    contents of the file could be::

       st2['movement_type'] = 'gaussian_walk'
       st2['movement_params'] = dict(dx=0.05, dt=0.05, step=5)

    This would select all results to which both these settings apply.
    Lines beginning with *#* are not evaluated.

    :param ffile: optional filepath
    """
    obj.filter_by_file(ffile)

def filter_by_condition(ffile = CFILE):
    """
    Select result files by the criteria in the given file
    (default :py:data:`CFILE`). The file is a text file containing conditions that
    can be evaluated in python. The result files are *res*, and numpy can be called
    with *np*. For instance, contents of the file could be::

       res.params.data_description[::-1].startswith("1")
       np.mean(np.sort(res.d_values["testingZ_S"][:16])) > np.mean(np.sort(res.d_values["testingZ_E"][:16]))
       res.params.st2["input_noise"] == 0.1

    This would select all results to which all three conditions apply.
    Lines beginning with *#* are not evaluated.

    :param ffile: optional filepath
    """
    obj.filter_by_condition(ffile)

def filter_by_indices(indices):
    """
    Select result files by their index. First, all results
    files in the folder are loaded and sorted by filename.
    Then the given indices are applied to sub-select

    :param indices: list of indices
    """
    obj.filter_by_indices(indices)

def show_d_values():
    obj.show_d_values()

def show_histograms():
    obj.show_histograms()

def show_params(only_st2=True):
    obj.show_params(only_st2=only_st2)

def collect_d_values():
    obj.collect_d_values()

def filegui():
    """
    Creates the GUI window. Checkboxes are created for all files in the folder
    specified by :py:data:`RESPATH` or :py:func:`set_path` or a subset of those
    if one of the functions :py:func:`filter_by_file`, :py:func:`filter_by_condition`,
    :py:func:`filter_by_indices` was called before.
    """
    gui = FileGui(obj)
