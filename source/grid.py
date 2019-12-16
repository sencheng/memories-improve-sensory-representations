"""
Module that was used to start large batches of simulations on a computing cluster running
Slurm Workload Manager. After results are ready :py:mod:`gridresults` can be used to get a first
look on the data, although for in-depth analyses composition of specific scripts is required.
Package :py:mod:`gridconfig` is used to set parameters for the simulation runs started with
*grid.py* and optionally selection criteria for result files to look at with :py:mod:`gridresults`.

When used with the Slurm Workload Manager the following command was run on the main cluster node::

   python grid.py [directory]

**[directory]** - folder in *../results/*. Needs to be created before running the command

The command writes python scripts (in the stated order) for generation of testing, training and forming input,
for training SFA1 and for executing the rest of the simulation run using :py:func:`core.streamlined.program`.
These python scripts are then executed on the cluster using the slurm command *sbatch* with the batch
file *batfile* in the source folder. The generated scripts include a polling mechanism such that a script only
starts if the required files have been generated already. For instance, training SFA1 needs training data
to be available and running a simulation needs forming data, testing data and a trained SFA1 instance to be
available.

"""

import gridconfig
import os
import sys
import time
import itertools
from core import system_params
import pdb
# import pyslurm
# from pwd import getpwuid, getpwnam

def testing_input_batch(subfolder = "grid"):
    ks = gridconfig.params.st4.keys()
    b = False
    if gridconfig.params.st4b is not None:
        b = True
        ksb = gridconfig.params.st4b.keys()
    if 0 == len(ks):
        empty = True
        it = iter([0])
    else:
        vals = gridconfig.params.st4.values()
        it = zip(*vals)
        empty = False
    if b:
        if 0 == len(ksb):
            emptyb = True
            itb = iter([0])
        else:
            valsb = gridconfig.params.st4b.values()
            itb = zip(*valsb)
            emptyb = False
    for i, vals in enumerate(it):
        outfile = open("testing_input{}.py".format(i), 'w')
        outfile.write("from core import system_params, sensory\n\n")
        outfile.write("import numpy as np\n\n")
        outfile.write("PARAMETERS = system_params.SysParamSet()\n")
        if not empty:
            d = dict(zip(ks, vals))
            outfile.write("PARAMETERS.st4.update(" + str(d) + ")\n")
        outfile.write("sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)\n")
        outfile.write("testing_sequenceX, testing_categories, testing_latent, testing_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st4)\n")
        outfile.write(
            "np.savez('../results/" + subfolder + "/testing{}.npz', testing_sequenceX=testing_sequenceX,"
                                                  " testing_categories=testing_categories, testing_latent=testing_latent, testing_ranges=testing_ranges)\n".format(i))
        outfile.close()
        os.system("sbatch batfile testing_input{}.py &".format(i))
        if b:
            outfileb = open("testing_input{}b.py".format(i), 'w')
            outfileb.write("from core import system_params, sensory\n\n")
            outfileb.write("import numpy as np\n\n")
            outfileb.write("PARAMETERS = system_params.SysParamSet()\n")
            if not emptyb:
                db = dict(zip(ksb, next(itb)))
                outfileb.write("PARAMETERS.st4b = dict(PARAMETERS.st4)\n")
                outfileb.write("PARAMETERS.st4b.update(" + str(db) + ")\n")
            outfileb.write("sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)\n")
            outfileb.write("testing_sequenceX, testing_categories, testing_latent, testing_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st4b)\n")
            outfileb.write(
                "np.savez('../results/" + subfolder + "/testing{}b.npz', testing_sequenceX=testing_sequenceX,"
                                                      " testing_categories=testing_categories, testing_latent=testing_latent, testing_ranges=testing_ranges)\n".format(i))
            outfileb.close()
            os.system("sbatch batfile testing_input{}b.py &".format(i))
    return i + 1


def training_input_batch(subfolder = "grid"):
    ks = gridconfig.params.st1.keys()
    if 0 == len(ks):
        empty = True
        it = iter([0])
    else:
        vals = gridconfig.params.st1.values()
        it = zip(*vals)
        empty = False
    first = True
    for i, vals in enumerate(it):
        if not first:
            os.system("sbatch batfile training_input{}.py &".format(i-1))
        else:
            first = False
        while True:
            try:
                outfile = open("training_input{}.py".format(i), 'w')
            except:
                time.sleep(1)
                continue
            break
        outfile.write("from core import system_params, sensory\n\n")
        outfile.write("import numpy as np\n\n")
        outfile.write("PARAMETERS = system_params.SysParamSet()\n")
        if not empty:
            d = dict(zip(ks, vals))
            outfile.write("PARAMETERS.st1.update(" + str(d) + ")\n")
        outfile.write("sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)\n")
        outfile.write("training_sequence, training_categories, training_latent = sensory_system.generate(**PARAMETERS.st1)\n")
        outfile.write("np.savez('../results/"+subfolder+"/training{}.npz', training_sequence=training_sequence, "
                                                        "training_categories=training_categories, training_latent=training_latent)\n".format(i))
        outfile.close()
    os.system("sbatch batfile training_input{}.py &".format(i))
    return i+1


def forming_input_batch(subfolder = "grid"):
    filecounter = 0
    st2 = gridconfig.params.st2
    if gridconfig.params.st2b is None:
        b = False
        st2b = st2
    else:
        st2b = gridconfig.params.st2b
        b = True
    if "snippet_length" in st2:
        leng = st2["snippet_length"]
    else:
        leng = [None] * len(st2["movement_type"])
    if "snippet_length" in st2b:
        lengb = st2b["snippet_length"]
    else:
        lengb = [None] * len(st2b["movement_type"])
    if "sequence" in st2:
        sqnc = st2["sequence"]
    else:
        sqnc = [-1] * len(leng)
    if "sequence" in st2b:
        sqncb = st2b["sequence"]
    else:
        sqncb = [-1] * len(lengb)
    if 'background_params' in st2:
        par = (st2["movement_type"], st2["movement_params"], leng, st2['background_params'], st2["number_of_snippets"], sqnc,
               st2b["movement_type"], st2b["movement_params"], lengb, st2b['background_params'], st2b["number_of_snippets"], sqncb)
    else:
        par = (st2["movement_type"], st2["movement_params"], leng, [-1]*len(leng), st2["number_of_snippets"], sqnc,
               st2b["movement_type"], st2b["movement_params"], lengb, [-1] * len(lengb), st2b["number_of_snippets"], sqncb)
    if not 'input_noise' in st2:
        st2["input_noise"] = [-1]
    if not 'input_noise' in st2b:
        st2b['input_noise'] = [-1]
    for mvtype, mvparams, snlen, bgr, nsnippets, seque, mvtypeb, mvparamsb, snlenb, bgrb, nsnippetsb, sequeb in zip(*par):
        for inp_noi, inp_noib in zip(st2["input_noise"], st2b["input_noise"]):
            outfile = open("forming_input"+str(filecounter)+".py", 'w')
            outfile.write("from core import system_params, sensory\n\n")
            outfile.write("import numpy as np\n\n")
            outfile.write("PARAMETERS = system_params.SysParamSet()\n")
            outfile.write("PARAMETERS.st2['movement_type'] = '"+mvtype+"'\n")
            outfile.write("PARAMETERS.st2['movement_params'] = " + str(mvparams) + "\n")
            outfile.write("PARAMETERS.st2['snippet_length'] = " + str(snlen) + "\n")
            outfile.write("PARAMETERS.st2['number_of_snippets'] = " + str(nsnippets) + "\n")
            if bgr != -1:
                outfile.write("PARAMETERS.st2['background_params'] = " + str(bgr) + "\n")
            if inp_noi != -1:
                outfile.write("PARAMETERS.st2['input_noise'] = " + str(inp_noi) + "\n")
            if seque != -1:
                outfile.write("PARAMETERS.st2['sequence'] = " + str(seque) + "\n")
            if b:
                outfile.write("PARAMETERS.st2b = dict(PARAMETERS.st2)\n")
                outfile.write("PARAMETERS.st2b['movement_type'] = '" + mvtypeb + "'\n")
                outfile.write("PARAMETERS.st2b['movement_params'] = " + str(mvparamsb) + "\n")
                outfile.write("PARAMETERS.st2b['snippet_length'] = " + str(snlenb) + "\n")
                outfile.write("PARAMETERS.st2b['number_of_snippets'] = " + str(nsnippetsb) + "\n")
                if bgrb != -1:
                    outfile.write("PARAMETERS.st2b['background_params'] = " + str(bgrb) + "\n")
                if inp_noib != -1:
                    outfile.write("PARAMETERS.st2b['input_noise'] = " + str(inp_noib) + "\n")
                if sequeb != -1:
                    outfile.write("PARAMETERS.st2b['sequence'] = " + str(sequeb) + "\n")
            outfile.write("sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)\n")
            if not b:
                outfile.write("forming_sequenceX, forming_categories, forming_latent, forming_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st2)\n")
            else:
                outfile.write("forming_sequenceXa, forming_categoriesa, forming_latenta, forming_rangesa = sensory_system.generate(fetch_indices=True, **PARAMETERS.st2)\n")
                outfile.write("forming_sequenceXb, forming_categoriesb, forming_latentb, forming_rangesb = sensory_system.generate(fetch_indices=True, **PARAMETERS.st2b)\n")
                outfile.write("forming_sequenceX = np.concatenate((forming_sequenceXa, forming_sequenceXb))\n")
                outfile.write("forming_categories = np.concatenate((forming_categoriesa, forming_categoriesb))\n")
                outfile.write("forming_latent = forming_latenta\n")
                outfile.write("forming_latent.extend(forming_latentb)\n")
                outfile.write("forming_ranges = forming_rangesa\n")
                outfile.write("forming_ranges.extend(forming_rangesb)\n")
            outfile.write("np.savez('../results/"+subfolder+"/forming"+str(filecounter)+".npz', forming_sequenceX=forming_sequenceX, forming_categories=forming_categories, forming_latent=forming_latent, forming_ranges=forming_ranges)\n")
            outfile.close()
            os.system("sbatch batfile forming_input"+str(filecounter)+".py &")
            filecounter += 1
    return filecounter


def sfa1_batch(training_file_count=1, subfolder = "grid"):
    for i, sfadef in enumerate(gridconfig.importlist):
        for j in range(training_file_count):
            sfa1_parms = eval("gridconfig."+sfadef).sfa1
            outfile = open(sfadef+"_train{}.py".format(j),'w')
            outfile.write("from core import system_params, sensory, semantic\n\n")
            outfile.write("import numpy as np\n")
            outfile.write("import time, os\n\n")
            outfile.write("while True:\n")
            outfile.write("    time.sleep(10)\n")
            outfile.write("    if os.path.isfile('../results/" + subfolder + "/training{}.npz'):\n".format(j))
            outfile.write("        break\n")
            outfile.write("time.sleep(10)\n")
            outfile.write("PARAMETERS = system_params.SysParamSet()\n")
            outfile.write("PARAMETERS.sem_params1 = " + str(sfa1_parms) + "\n")
            outfile.write("sfa1 = semantic.build_module(PARAMETERS.sem_params1)\n")
            outfile.write("training_data = np.load('../results/" + subfolder + "/training{}.npz')\n".format(j))
            outfile.write("training_sequence = training_data['training_sequence']\n")
            outfile.write("semantic.train_SFA(sfa1, training_sequence)\n")
            outfile.write("sfa1.save('../results/"+subfolder+"/" + sfadef + "train{}.sfa')\n".format(j))
            outfile.close()
            os.system("sbatch batfile " + sfadef+"_train{}.py &".format(j))

def main_loop(training_file_count=1, forming_file_count=1, testing_file_count=1, subfolder = "grid"):
    rescounter = 0
    st2 = gridconfig.params.st2
    outinfo = open("../results/"+subfolder+"/resinfo.txt", 'w')
    train_ks = gridconfig.params.st1.keys()
    if 0 == len(train_ks):
        train_empty = True
        it_list = dict(answers=[42]).values()
    else:
        train_vals = gridconfig.params.st1.values()
        it_list = list(train_vals)
        train_empty = False
    for i, sfadef in enumerate(gridconfig.importlist):
        train_it = zip(*it_list)
        for j, tvals in enumerate(train_it):
            formcounter = 0
            outinfo.write(sfadef+ "_train{}.py &".format(j) + " - " + str(rescounter) + "\n")
            sfa1_parms = eval("gridconfig." + sfadef).sfa1
            sfa2_parms = eval("gridconfig." + sfadef).sfa2
            if "snippet_length" not in st2:
                st2["snippet_length"] = [None] * len(st2["movement_type"])
            st2_copy = dict(st2)
            for key in ["movement_type", "movement_params", "snippet_length", "background_params", "number_of_snippets", "input_noise"]:
                try:
                    del st2_copy[key]
                except:
                    pass
            remain_keys = st2_copy.keys()
            if "memory" in remain_keys:
                mem_dict = dict(st2_copy["memory"])
                del st2_copy["memory"]
            else:
                mem_dict = dict()
            mem_keys = mem_dict.keys()
            st2_copy.update(mem_dict)     #save keys that belong to memory originally then put the memory into the st2 array, actually flattening dict
            remain_values = st2_copy.values()
            remain_keys = st2_copy.keys()
            if 'background_params' in st2:
                par = (st2["movement_type"], st2["movement_params"], st2["snippet_length"], st2['background_params'], st2["number_of_snippets"])
            else:
                par = (st2["movement_type"], st2["movement_params"], st2["snippet_length"], [-1] * len(st2["snippet_length"]), st2["number_of_snippets"])
            if not 'input_noise' in st2:
                st2["input_noise"] = [-1]
            for mvtype, mvparams, snlen, bgr, nsnippets in zip(*par):
                for inp_noi in st2["input_noise"]:
                    outinfo.write("    forming" + str(formcounter) + " - " + str(rescounter) + ":\n")
                    outinfo.write("    movement_type: '" + mvtype + "'\n")
                    outinfo.write("    movement_params: " + str(mvparams) + "\n")
                    outinfo.write("    snippet_length: " + str(snlen) + "\n")
                    if bgr != -1:
                        outinfo.write("    background_params: " + str(bgr) + "\n")
                    if inp_noi != -1:
                        outinfo.write("    input_noise: " + str(inp_noi) + "\n")
                    outinfo.write("    number_of_snippets: " + str(nsnippets) + "\n")
                    test_ks = gridconfig.params.st4.keys()
                    b = False
                    if gridconfig.params.st4b is not None:
                        b = True
                        test_ksb = gridconfig.params.st4b.keys()
                    if 0 == len(test_ks):
                        test_empty = True
                        test_it = iter([0])
                    else:
                        test_vals = gridconfig.params.st4.values()
                        test_it = zip(*test_vals)
                        test_empty = False
                    if b:
                        if 0 == len(test_ksb):
                            test_emptyb = True
                            test_itb = iter([0])
                        else:
                            test_valsb = gridconfig.params.st4b.values()
                            test_itb = zip(*test_valsb)
                            test_emptyb = False
                    testcounter = 0
                    for i_test, test_vals in enumerate(test_it):
                        if not test_empty:
                            dtest = dict(zip(test_ks, test_vals))
                        if b and not test_emptyb:
                            dtestb = dict(zip(test_ksb, next(test_itb)))
                        outinfo.write("        testing" + str(testcounter) + " - " + str(rescounter) + ":\n")
                        if "movement_type" in dtest.keys():
                            outinfo.write("        movement_type: '" + str(dtest["movement_type"]) + "'\n")
                        if "movement_params" in dtest.keys():
                            outinfo.write("        movement_params: " + str(dtest["movement_params"]) + "\n")
                        if "snippet_length" in dtest.keys():
                            outinfo.write("        snippet_length: " + str(dtest["snippet_length"]) + "\n")
                        if "background_params" in dtest.keys():
                            outinfo.write("        background_params: " + str(dtest["background_params"]) + "\n")
                        # ===============Now new parameters============
                        for vals in itertools.product(*remain_values):
                            if gridconfig.params.type_match and not train_empty and not test_empty and \
                                    (dict(zip(train_ks, tvals))['movement_type'] != mvtype or dtest['movement_type'] != mvtype):
                                break
                            if gridconfig.params.param_match and not train_empty and not test_empty and \
                                    (dict(zip(train_ks, tvals))['movement_params'] != mvparams or dtest['movement_params'] != mvparams):
                                break
                            outfile = open("main_exec_res" + str(rescounter) + ".py", 'w')
                            outfile.write("from core import streamlined, system_params, sensory, semantic\n")
                            outfile.write("import grid\n\n")
                            outfile.write("import numpy as np\n")
                            outfile.write("import time, os\n\n")
                            outfile.write("while True:\n")
                            outfile.write("    time.sleep(10)\n")
                            if_string = "    if os.path.isfile('../results/"+subfolder+"/" + sfadef + "train{}.sfa')".format(j)
                            if_string += " and os.path.isfile('../results/"+subfolder+"/forming" + str(formcounter) + ".npz')"
                            if_string += " and os.path.isfile('../results/"+subfolder+"/testing{}.npz')".format(i_test)
                            if_string += ":\n"
                            outfile.write(if_string)
                            outfile.write("        break\n")
                            outfile.write("time.sleep(20)\n")
                            outfile.write("PARAMETERS = system_params.SysParamSet()\n")
                            outfile.write("PARAMETERS.result_path = '../results/" + subfolder + "/'\n")
                            outfile.write("PARAMETERS.sem_params1 = " + str(sfa1_parms) + "\n")
                            outfile.write("PARAMETERS.sem_params2S = PARAMETERS.sem_params2E = " + str(sfa2_parms) + "\n")
                            if not train_empty:
                                d = dict(zip(train_ks, tvals))
                                outfile.write("PARAMETERS.st1.update(" + str(d) + ")\n")
                            outfile.write("PARAMETERS.st2['movement_type'] = '" + mvtype + "'\n")
                            outfile.write("PARAMETERS.st2['movement_params'] = " + str(mvparams) + "\n")
                            outfile.write("PARAMETERS.st2['number_of_snippets'] = " + str(nsnippets) + "\n")
                            outfile.write("PARAMETERS.st2['snippet_length'] = " + str(snlen) + "\n")
                            if bgr != -1:
                                outfile.write("PARAMETERS.st2['background_params'] = " + str(bgr) + "\n")
                            if inp_noi != -1:
                                outfile.write("PARAMETERS.st2['input_noise'] = " + str(inp_noi) + "\n")
                            for k, v in zip(remain_keys,vals):
                                if k in mem_keys:
                                    outfile.write("PARAMETERS.st2['memory']['" + k + "'] = " + str(v) + "\n")
                                else:
                                    outfile.write("PARAMETERS.st2['" + k + "'] = " + str(v) + "\n")
                            if not test_empty:
                                outfile.write("PARAMETERS.st4.update(" + str(dtest) + ")\n")
                            if b and not test_emptyb:
                                outfile.write("PARAMETERS.st4b = dict(PARAMETERS.st4)\n")
                                outfile.write("PARAMETERS.st4b.update(" + str(dtestb) + ")\n")
                            outfile.write("PARAMETERS.data_description = '" + sfadef + "'\n")
                            outfile.write("PARAMETERS.st3.update(" + str(gridconfig.params.st3) + ")\n")
                            outfile.write("training_data = np.load('../results/" + subfolder + "/training{}.npz')\n".format(j))
                            outfile.write("try:\n")
                            outfile.write("    forming_data = np.load('../results/"+subfolder+"/forming" + str(formcounter) + ".npz')\n")
                            outfile.write("except:\n")
                            outfile.write("    time.sleep(20)\n")
                            outfile.write("    forming_data = np.load('../results/" + subfolder + "/forming" + str(formcounter) + ".npz')\n")
                            outfile.write("testing_data = np.load('../results/"+subfolder+"/testing{}.npz')\n".format(i_test))
                            if b:
                                outfile.write("testing_datab = np.load('../results/" + subfolder + "/testing{}b.npz')\n".format(i_test))
                            outfile.write("training_sequence, training_categories, training_latent = training_data['training_sequence'], training_data['training_categories'], training_data['training_latent']\n")
                            outfile.write("forming_sequenceX, forming_categories, forming_latent, forming_ranges = forming_data['forming_sequenceX'], forming_data['forming_categories'], forming_data['forming_latent'], forming_data['forming_ranges']\n")
                            outfile.write("testing_sequenceX, testing_categories, testing_latent, testing_ranges = testing_data['testing_sequenceX'], testing_data['testing_categories'], testing_data['testing_latent'], testing_data['testing_ranges']\n")
                            if b:
                                outfile.write(
                                    "testing_sequenceXb, testing_categoriesb, testing_latentb, testing_rangesb = testing_datab['testing_sequenceX'], testing_datab['testing_categories'], testing_datab['testing_latent'], testing_datab['testing_ranges']\n")
                            outfile.write("sfa1 = semantic.load_SFA('../results/"+subfolder+"/" + sfadef + "train{}.sfa')\n".format(j))
                            if not b:
                                outfile.write("res = streamlined.program(PARAMETERS, sfa1, [[forming_sequenceX, forming_categories, forming_latent, forming_ranges], [testing_sequenceX, testing_categories, testing_latent]], [training_sequence, training_categories, training_latent], {})\n".format(rescounter))
                            else:
                                outfile.write(
                                    "res = streamlined.program(PARAMETERS, sfa1, [[forming_sequenceX, forming_categories, forming_latent, forming_ranges], [testing_sequenceX, testing_categories, testing_latent], [testing_sequenceXb, testing_categoriesb, testing_latentb]], [training_sequence, training_categories, training_latent], {})\n".format(rescounter))
                            outfile.write("res.save_to_file('../results/"+subfolder+"/res" + str(rescounter) + ".p')\n")
                            outfile.close()
                            os.system("sbatch batfile main_exec_res" + str(rescounter) + ".py &")
                            rescounter += 1
                        testcounter += 1
                    formcounter += 1
    outinfo.close()
    return rescounter

def poll_training_files(training_file_count=1, testing_file_count=1, subfolder="grid"):
    prepare_complete = False
    while not prepare_complete:
        prepare_complete = True
        for j in range(training_file_count):
            if not os.path.isfile("../results/"+subfolder+"/training{}.npz".format(j)):
                prepare_complete = False
                break
        if prepare_complete:
            for idx in range(testing_file_count):
                if not os.path.isfile("../results/" + subfolder + "/testing{}.npz".format(idx)):
                    prepare_complete = False
                    break
        if prepare_complete and gridconfig.params.st4b is not None:
            for idx in range(testing_file_count):
                if not os.path.isfile("../results/" + subfolder + "/testing{}b.npz".format(idx)):
                    prepare_complete = False
                    break
        time.sleep(10)  # sleep after test in case the file is created a moment before it is actually complete

def poll_files(training_file_count = 1, forming_file_count=1, subfolder="grid"):
    prepare_complete = False
    while not prepare_complete:
        prepare_complete = True
        for sfadef in gridconfig.importlist:
            for j in range(training_file_count):
                if not os.path.isfile("../results/"+subfolder+"/" + sfadef + "train{}.sfa".format(j)):
                    prepare_complete = False
                    break
        if prepare_complete:
            for idx in range(forming_file_count):
                if not os.path.isfile("../results/"+subfolder+"/forming" + str(idx) + ".npz"):
                    prepare_complete = False
                    break
        time.sleep(10)

def poll_results(res_file_count=1, subfolder="grid"):
    results_ready = False
    while not results_ready:
        results_ready = True
        for i in range(res_file_count):
            if not os.path.isfile("../results/"+subfolder+"/res"+str(i)+".p"):
                results_ready = False
        time.sleep(10)

def clean_up(subfolder="grid"):
    # os.system("rm testing_input*")
    # os.system("rm training_input*")
    # os.system("rm forming_input*")
    # os.system("rm *_train*")
    # os.system("rm main_exec_res*")
    # os.system("rm gridconfig/*.pyc")
    # os.system("rm *.pyc")
    os.system("cp -r gridconfig/sfadef* ../results/"+subfolder+"/")
    os.system("cp gridconfig/params.py ../results/"+subfolder+"/")
    os.system("cp gridconfig/params.py ../results/"+subfolder+"/resultparams.txt")

if __name__ == "__main__":
    av = sys.argv
    subfolder = "grid"
    if len(av)>1:
        subfolder = av[1]
    testing_file_count = testing_input_batch(subfolder)
    print("Started testing input batch, {} files".format(testing_file_count))
    time.sleep(2)
    training_file_count = training_input_batch(subfolder)
    print("Started training input batch, {} files".format(training_file_count))
    time.sleep(2)
    # poll_training_files(training_file_count, testing_file_count, subfolder)
    forming_file_count = forming_input_batch(subfolder)
    print("Started forming input batch, {} files".format(forming_file_count))
    time.sleep(2)
    sfa1_batch(training_file_count, subfolder)
    print("Started sfa1 batch")
    time.sleep(2)
    # poll for completion of all sfa trainings and all forming inputs
    # poll_files(training_file_count, forming_file_count, subfolder)
    # for each sfa1 / forming input combination train corresponding SFA2s with different memory parameters, save them and run testing on them, finally save result
    res_file_count = main_loop(training_file_count, forming_file_count, testing_file_count, subfolder)
    print("Started main batch, {} files".format(res_file_count))
    # poll for completion of results generation
    # poll_results(res_file_count, subfolder)
    time.sleep(2)
    clean_up(subfolder)
