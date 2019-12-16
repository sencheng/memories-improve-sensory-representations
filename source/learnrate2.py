"""
Script that executes an entire simulation (on a computing cluster running Slurm
Workload Manager) while not using the sequence storage implementation of episodic memory
to retrieve novel sequences but just for **repeated replay of the episodes**.
SFA modules for each of 40 training iterations and for different learning rates
are generated.

.. note::
   Note that this script does not use :py:func:`core.streamlined.program`, although it would
   be possible to do the same stuff with it by setting ``st3['use_memory'] = False`` and ``st3['inc_repeats_S'] = 40``.

``learnrate2.py`` does not take arguments, but parameters are defined in the source
(see below). These parameters are lists and every combination of elements is executed.
For instance, if all parameter lists have just one element, a single simulation run is started. If two of the lists have
two elements each, this would result in four simulation runs.

``learnrate2.py`` executes :py:mod:`learnrate_pre_ex2` and :py:mod:`learnrate_ex2` with the correct
arguments on the cluster using the slurm command *sbatch* with the batch files *batpre* and *batlearnrate*,
respectively in the source folder. :py:mod:`learnrate_ex2` includes a polling mechanism such that the execution only
starts if the required sfa1 files have been generated already by :py:mod:`learnrate_pre_ex2`.

"""

import os, time, itertools

# typelist = ['o14', 'o18', 'sin']
typelist = ['o18']
"""
Which SFA1 parameters to use. Available options as follows.

``'o14'``::

   sfa1_parms = [
      ('layer.linear', {
         'bo':               30,
         'rec_field_ch':     14,
         'spacing':          8,
         'in_channel_dim':   1,
         'out_sfa_dim':     32
      })
   ]

``'o18'``::

   sfa1_parms = [
      ('layer.linear', {
         'bo': 30,
         'rec_field_ch': 18,
         'spacing': 6,
         'in_channel_dim': 1,
         'out_sfa_dim': 32
      })
   ]

``'sin'``::

   sfa1_parms = [
      ('single.linear', {
         'dim_in': 900,
         'dim_out': 288
      })
   ]

"""

framelist = [50]
"""
How many snippets of what length to use for training SFA1. Available options as follows.

``50``::

    number_of_snippets = 50
    snippet_length = 50
    
``600``::

    nsnip = 600
    snlen = 100

"""
# rotlist = ['t', 'o']
rotlist = ['t']
"""
Whether to rotate the objects or not (in the input used for training SFA1).

If ``'t'``, objects are rotated, if ``'o'`` they are not. In the latter case,
a random fixed angle is chosen for each snippet.
"""

eps_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
"""
Learning rates for incremental SFA2.
"""

if __name__ == "__main__":

    for typ, fram, rot in itertools.product(typelist, framelist, rotlist):
        os.system("sbatch batpre learnrate_pre_ex2.py {} {} {} &".format(typ, fram, rot))
        # os.system("srun python learnrate_pre_ex2.py {} {} {} ".format(typ, fram, rot))

        # time.sleep(2)
        # PATH = "../results/lro_{}{}{}/".format(typ, fram, rot)
        # if not os.path.isfile(PATH + "sfa1.p"):
        #     print("In " + PATH + ", sfa1 was not created. Aborting.")
        #     continue

        time.sleep(2)

        for dim_id in [1]:
            for ei, eps in enumerate(eps_list):
                os.system("sbatch batlearnrate learnrate_ex2.py {} {} {} {} {} {} &".format(dim_id, ei, eps, typ, fram, rot))
                # os.system("srun python learnrate_ex2.py {} {} {} {} {} {} &".format(dim_id, ei, eps, typ, fram, rot))
