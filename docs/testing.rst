Testing
=======

To run the automated tests, go to the main source directory and execute the following:

.. code-block:: bash

    pytest

This runs serially and does not need MPI. Parallel tests require the `mpi`
extra (see :doc:`installation`) and are run with `mpirun`:

.. code-block:: bash

    mpirun -n 4 python3 -m pytest

The tests use communicators of size 1 and 4; tests for other sizes are
skipped. For development purposes you can go beyond the number of processors
of your computer using `mpirun -n 10 --oversubscribe`.
