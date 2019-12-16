.. _getting_started:

Getting started
========================================

Here is some example code for how to use the framework.
The code is from the script ``examples.py`` in the source folder.

Import modules from the core package::

   from core import sensory, system_params, semantic, streamlined

Depending on what you want to do, you might need to import more of them.

Import modules that are useful for data processing::

   import numpy as np
   import sklearn.linear_model
   import scipy.stats

Generate a default collection of parameters::

   PARAMETERS = system_params.SysParamSet()

This object contains all the defaults for the parameters that are required to run all aspects
of the framework. More information on what parameters are available and how they can be used
can be found in :py:mod:`core.system_params`.

.. note::
    To execute an entire simulation and return a
    :py:class:`core.result.Result` object, the ``PARAMETERS`` object can be passed to the
    function :py:func:`core.streamlined.program`. It is a quite useful piece of code - to see what
    parameters can be adjusted for ``streamlined``, :py:mod:`core.system_params` is worth a look.

    We do not use ``streamlined`` here but execute the required steps manually to show how the basics work.

----------------
Generate input
----------------

All settings in the ``PARAMETERS`` object can be edited - we change some in order to generate
the input we want::

   PARAMETERS.st1["movement_type"] = "gaussian_walk"
   PARAMETERS.st1["movement_params"] = dict(dx=0.05, dt=0.05, step=5)
   PARAMETERS.st1["number_of_snippets"] = 200
   PARAMETERS.st1["snippet_length"] = 50

* ``movement_type`` determines what statistics the movement of the objects on the images follows.
  ``gaussian_walk`` is a random walk where difference values are drawn from a Gaussian
  (as opposed to a uniform distribution in ``random_walk``). Every movement
  type expects a dictionary of specific
* ``movement_params``, in the case of ``gaussian_walk`` those parameters are ``dx`` and ``dt``
  (standard deviations of the zero-mean Gaussians which position and rotation change,
  respectively, are drawn from as well as a speed factor ``step``).
* ``number_of_snippets`` determines the number of individual sequences in the dataset and
* ``snippet_length`` is the number of frames in each of the sequences.

More information on how to use the different input parameter settings can be found in
:py:mod:`core.input_params`.

Before we can generate input, we need to create a object of :py:class:`core.sensory.SensorySystem`
that does the job::

   sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

Default input parameters have to be passed. If ``save_input`` is true, previously generated
sequences can be recalled using the :py:func:`core.sensory.SensorySystem.recall` method. I do not know
whether that is useful.

We now generate the input sequences::

   sequence, categories, latents = sensys.generate(**PARAMETERS.st1)
   cat_arr = np.array(categories)
   lat_arr = np.array(latents)

The default input parameters can be overridden with a dictionary when calling
:py:func:`core.sensory.SensorySystem.generate`.
By default, ``generate`` returns a tuple. ``categories`` is the sequence of object identity
labels. ``latents`` is the sequence of object position and rotation
(x, y, cos(:math:`{\phi}`), sin(:math:`{\phi}`)).

--------------------------
Sensory representations
--------------------------

First we have to set parameters for the SFA. We can choose whether to use batch or incremental
SFA, whether to use a single node or a receptive field structure and whether to use a pipeline
of multiple SFA layers in one instance::

   sfa_parms = [
      ('layer.linear', {
      'bo': 30,
      'rec_field_ch': 18,
      'spacing': 6,
      'in_channel_dim': 1,
      'out_sfa_dim': 32
      })
   ]

An SFA parameter list consists of tuples of an SFA type definition (here ``layer.linear``) and
a dictionary with parameters that the chosen SFA type requires. For instance, ``layer.linear``
requires a slightly different parameter set than ``layer.square``. ``layer`` means that a
receptive field structure with batch SFA is used and ``square`` or ``linear`` determine whether
a quadratic expansion is used or not. Alternatives to ``layer`` are ``single`` (a single node
of batch SFA), ``inc`` (a single node of incremental SFA) and ``inclayer`` (a receptive field
structure with incremental SFA), for instance. See :py:mod:`core.semantic_params` for more
information.

Working with SFA is done using the module :py:mod:`core.semantic`. We generate an SFA object
using the given parameters::

   sfa = semantic.build_module(sfa_parms)

We train it on the input sequence that was previously generated::

   semantic.train_SFA(sfa, sequence)

Now we can run data through the SFA to generate slow features::

   y = semantic.exec_SFA(sfa, sequence)

--------------------------
Data processing
--------------------------

For some further analyses and processing, it might be helpful to whiten the SFA output.
The required methods are available as part of the module :py:mod:`core.streamlined`::

   whitener = streamlined.normalizer(y, PARAMETERS.normalization)
   y_w = whitener(y)

Executing a :py:class:`core.streamlined.normalizer` object performs a linear transformation. The transformation matrix
is determined at initialization of the object. Note that only when the data ``normalizer``
is trained and executed on are identical, the result is actually whitened.

Delta values are an important measure for quality of SFA features::

   deltas = tools.delta_diff(y_w)

The module :py:mod:`core.tools` also provides more functions that might be handy sometimes. For instance,
one calculated the Pearson correlation of all individual features with latent variables::

   tools.feature_latent_correlation(y, lat_arr, cat_arr)

To measure feature quality more sophisticatedly, a linear regressor can be trained
zo extract latent variables from SFA output::

  training_matrix = y
  target_matrix = np.append(latents, cat_arr[:, None], axis=1)
  learner = sklearn.linear_model.LinearRegression()
  learner.fit(training_matrix, target_matrix)

The regressor is then used to predict latent variables from input. The Pearson correlation
between the original latent variables and the prediction can be used as a measure of
feature quality::

   prediction = learner.predict(y)
   _, _, r_valueX, _, _ = scipy.stats.linregress(lat_arr[:, 0], prediction[:, 0])
   _, _, r_valueY, _, _ = scipy.stats.linregress(lat_arr[:, 1], prediction[:, 1])
   _, _, r_valueCOS, _, _ = scipy.stats.linregress(lat_arr[:, 2], prediction[:, 2])
   _, _, r_valueSIN, _, _ = scipy.stats.linregress(lat_arr[:, 3], prediction[:, 3])
   _, _, r_valueCAT, _, _ = scipy.stats.linregress(cat_arr, prediction[:, 4])

Obviously, the Pearson correlation can be calculated in a simpler way. The ``linregress`` is
used here because it can also return the parameters of a regression line through a scatter plot
of variable-prediction pairs for visualization purposes.

.. |SFAlo| replace:: SFA\ :sub:`lo`\
.. |SFAhi| replace:: SFA\ :sub:`hi`\

|

:ref:`Back to top <getting_started>`

:ref:`Return Home <mastertoc>`