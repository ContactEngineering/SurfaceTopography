#
# Copyright 2019-2020 Lars Pastewka
#           2019 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import tempfile

import numpy as np
from scipy.io.netcdf import netcdf_file

from NuMPI import MPI
from muFFT import FFT

from SurfaceTopography.IO import read_topography
from SurfaceTopography.IO.NC import NCReader
from SurfaceTopography.Generation import fourier_synthesis


def test_save_and_load(comm):
    nb_grid_pts = (128, 128)
    size = (3, 3)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1, unit='µm')

    fft = FFT(nb_grid_pts, communicator=comm, fft="mpi")
    fft.create_plan(1)
    dt = t.domain_decompose(fft.subdomain_locations,
                            fft.nb_subdomain_grid_pts,
                            communicator=comm)
    assert t.unit == 'µm'
    assert dt.unit == 'µm'
    assert t.info['unit'] == 'µm'
    assert dt.info['unit'] == 'µm'
    if comm.size > 1:
        assert dt.is_domain_decomposed

    # Save file
    dt.to_netcdf('parallel_save_test.nc')

    # Attempt to open full file on each MPI process
    t2 = read_topography('parallel_save_test.nc')

    assert t.physical_sizes == t2.physical_sizes
    assert t.unit == t2.unit
    assert t.info['unit'] == t2.info['unit']
    np.testing.assert_array_almost_equal(t.heights(), t2.heights())

    # Attempt to open file in parallel
    r = NCReader('parallel_save_test.nc', communicator=comm)

    assert r.channels[0].nb_grid_pts == nb_grid_pts

    t3 = r.topography(subdomain_locations=fft.subdomain_locations,
                      nb_subdomain_grid_pts=fft.nb_subdomain_grid_pts)

    assert t.physical_sizes == t3.physical_sizes
    assert t.unit == t3.unit
    assert t.info['unit'] == t3.info['unit']
    np.testing.assert_array_almost_equal(dt.heights(), t3.heights())

    assert t3.is_periodic

    comm.barrier()
    if comm.rank == 0:
        os.remove('parallel_save_test.nc')


def test_save_and_load_no_unit(comm_self):
    nb_grid_pts = (128, 128)
    size = (3, 3)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1)

    # Save file
    t.to_netcdf('no_unit.nc')

    t2 = read_topography('no_unit.nc')

    assert t.physical_sizes == t2.physical_sizes
    assert 'unit' not in t2.info
    np.testing.assert_array_almost_equal(t.heights(), t2.heights())

    os.remove('no_unit.nc')


def test_load_no_physical_sizes(comm_self):
    nb_grid_pts = (128, 128)
    size = (3, 3)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1)

    # Topographies always have physical size information, we need to create a
    # NetCDF file without any manually
    with netcdf_file('no_physical_sizes.nc', 'w') as nc:
        nc.createDimension('x', nb_grid_pts[0])
        nc.createDimension('y', nb_grid_pts[1])
        nc.createVariable('heights', 'f8', ('x', 'y'))
        nc.variables['heights'][...] = t.heights()

    # Attempt to open full file on each process
    # with pytest.raises(ValueError):
    #    # This raises an error because the physical sizes are not present
    #    t2 = read_topography('no_physical_sizes.nc')
    t2 = read_topography('no_physical_sizes.nc', physical_sizes=size)

    assert t.physical_sizes == t2.physical_sizes
    assert 'unit' not in t2.info
    np.testing.assert_array_almost_equal(t.heights(), t2.heights())

    os.remove('no_physical_sizes.nc')


def test_save_and_load_line_scan(comm_self):
    nb_grid_pts = (128,)
    size = (3,)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1)
    t.info['unit'] = 'µm'

    with tempfile.TemporaryDirectory() as d:
        tmpfn = f'{d}/line_scan.nc'

        # Save file
        t.to_netcdf(tmpfn)

        # Read file
        t2 = read_topography(tmpfn)

        assert t == t2


def test_save_and_load_line_scan_opd(file_format_examples, comm_self):
    t = read_topography(os.path.join(file_format_examples, 'opd3.opd'))
    with tempfile.TemporaryDirectory() as d:
        tmpfn = f'{d}/line_scan.nc'

        # Save file
        t.to_netcdf(tmpfn)

        # Read file
        t2 = read_topography(tmpfn)

        assert t == t2


if __name__ == '__main__':
    test_save_and_load(MPI.COMM_WORLD)
