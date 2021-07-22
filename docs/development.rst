Development
===========
To use the code without installing it, e.g. for development purposes, use the `env.sh` script to set the environment:

.. code-block:: bash

    source /path/to/SurfaceTopography/env.sh [python3]

Note that the parameter to `env.sh` specifies the Python interpreter for which the environment is set up. PyCo contains portions that need to be compiled, make sure to run

.. code-block:: bash

    python setup.py build

whenever any of the Cython (.pyx) sources are modified.

Please read :ref:`contributing` if you plan to contribute to this code.
