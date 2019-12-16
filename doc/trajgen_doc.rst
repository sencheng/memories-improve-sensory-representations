.. _trajgen:

Module trajgen
========================================
|

-----------------
Mover functions
-----------------

These functions represent the different movement types that are available.
When using the framework, the parameter ``movement_type`` is a string containing the name
of the mover function and the parameter ``movement_params`` is a dictionary containing the
function parameters.

.. automodule:: core.trajgen
   :members: copy_traj, gaussian_walk, lissajous, random_rails, random_stroll, random_walk, sample, stillframe, timed_border_stroll, uniform

---------------------
Generation functions
---------------------

Usually it is not necessary to dig into those.

.. automodule:: core.trajgen
   :members: alternator, makeTextImage, makeTextImageTrimmed, makeTransform, movie, presentor, trajGenMaker, trim


|

:ref:`Back to top <trajgen>`

:ref:`Return Home <mastertoc>`
