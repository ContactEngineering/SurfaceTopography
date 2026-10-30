# Testing

## Running the tests

Install the package with the `test` extra (see [Installation](installation.md))
and run, from the root of the source tree,

```bash
python3 -m pytest
```

This runs serially and does not need MPI. `pytest.ini` collects `test_*.py`
under `SurfaceTopography`, `examples`, `maintenance` and `test`.

With the `mpi` extra installed, the suite also runs in parallel:

```bash
mpirun -n 4 python3 -m pytest
```

Tests use communicators of size 1 and 4; tests for other sizes are skipped.
To run more processes than your computer has cores, add `--oversubscribe`
(OpenMPI), e.g. `mpirun --oversubscribe -n 10 python3 -m pytest`.

CI runs the suite for Python 3.10 to 3.14, serially without mpi4py and muGrid,
and with the `mpi` extra (compiled against OpenMPI) on 1 and 4 processes.

## Fixtures

`test/conftest.py` defines the fixtures shared by all tests. The communicator
fixtures come from NuMPI's testing utilities
([NuMPI documentation](https://muspectre.github.io/NuMPI/testing/)):

| Fixture | Provides |
|---------|----------|
| `comm` | communicators of size 1 and 4; a size is skipped if fewer processes run |
| `maxcomm` | a communicator spanning all processes of the run |
| `file_format_examples` | path of `test/file_format_examples`, the corpus of example files for the readers |

Always take the communicator from a fixture instead of using
`MPI.COMM_WORLD`, and pass it explicitly to every object that accepts one, so
that the test runs for each communicator size:

```python
import numpy as np
from NuMPI.FFT import make_fft_engine
from SurfaceTopography import Topography


def test_domain_decomposed_mean(comm):
    nb_grid_pts = (64, 32)
    fft = make_fft_engine(nb_grid_pts, communicator=comm)
    heights = np.ones(fft.nb_subdomain_grid_pts)  # local part of the heights
    topography = Topography(heights, (1.0, 1.0), decomposition='subdomain',
                            nb_grid_pts=nb_grid_pts,
                            subdomain_locations=fft.subdomain_locations,
                            communicator=comm)
    assert np.isclose(topography.mean(), 1.0)
```

## Serial-only tests

Tests that only exercise serial functionality, or write files with fixed
names, are skipped on more than one process with NuMPI's `serial_only`
marker:

```python
from NuMPI.Testing.Fixtures import serial_only

pytestmark = serial_only  # all tests of this module


@serial_only  # a single test
def test_something_serial():
    ...
```

## File-format tests

Every reader has an example file in `test/file_format_examples` and tests in
`test/IO/`. The declarative readers are additionally checked by the
serialization and conformance tests described in
[Declarative readers](declarative-readers.md).
