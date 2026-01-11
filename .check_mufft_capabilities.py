import sys

import muGrid

mpi_required = sys.argv[1] in ["yes", "1"]

print('`muGrid` imported successfully')
print(f'muGrid version: {muGrid.__version__}')

if mpi_required:
    # Make sure that we have the parallel version running
    assert muGrid.has_mpi, "MPI support is required but muGrid.has_mpi is False"
    print('MPI support: enabled')
