#
# Copyright 2019-2022 Lars Pastewka
#           2019-2020 Antoine Sanner
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

import tempfile

import numpy as np
import pytest
from muFFT import FFT
from NuMPI import MPI
from scipy.io import netcdf_file

from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.IO import read_topography
from SurfaceTopography.IO.NC import NCReader

from .test_io import binary_example_file_list, explicit_physical_sizes


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="FIXME! This tests often stalls (randomly) on multiple MPI processes; disabling for now")
def test_save_and_load(maxcomm):
    nb_grid_pts = (128, 128)
    size = (3, 3)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1, unit='µm')

    fft = FFT(nb_grid_pts, communicator=maxcomm, engine="mpi")
    fft.create_plan(1)
    dt = t.domain_decompose(fft.subdomain_locations,
                            fft.nb_subdomain_grid_pts,
                            communicator=maxcomm)
    assert t.unit == 'µm'
    assert dt.unit == 'µm'
    if maxcomm.size > 1:
        assert dt.is_domain_decomposed

    # Save file
    dt.to_netcdf('parallel_save_test.nc')

    # Attempt to open full file on each MPI process
    t2 = read_topography('parallel_save_test.nc')

    assert t.physical_sizes == t2.physical_sizes
    assert t.unit == t2.unit
    np.testing.assert_array_almost_equal(t.heights(), t2.heights())

    # Attempt to open file in parallel
    r = NCReader('parallel_save_test.nc', communicator=maxcomm)

    assert r.channels[0].nb_grid_pts == nb_grid_pts

    t3 = r.topography(subdomain_locations=fft.subdomain_locations,
                      nb_subdomain_grid_pts=fft.nb_subdomain_grid_pts)

    assert t.physical_sizes == t3.physical_sizes
    assert t.unit == t3.unit
    np.testing.assert_array_almost_equal(dt.heights(), t3.heights())

    assert t3.is_periodic


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_save_and_load_no_unit():
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


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_load_no_physical_sizes():
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


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_save_and_load_line_scan():
    nb_grid_pts = (128,)
    size = (3,)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1, unit='µm')

    with tempfile.TemporaryDirectory() as d:
        tmpfn = f'{d}/line_scan.nc'

        # Save file
        t.to_netcdf(tmpfn)

        # Read file
        t2 = read_topography(tmpfn)

        assert t == t2


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize('fn', binary_example_file_list)
def test_save_and_load_binary_files(fn):
    if fn in explicit_physical_sizes:
        return
    t = read_topography(fn)
    with tempfile.TemporaryDirectory() as d:
        tmpfn = f'{d}/tmp.nc'

        # Save file
        t.to_netcdf(tmpfn)

        # Read file
        t2 = read_topography(tmpfn)

        # Check that the two topographies equal
        assert t == t2
