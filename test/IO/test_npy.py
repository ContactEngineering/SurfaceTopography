#
# Copyright 2019-2021, 2023 Lars Pastewka
#           2020 Michael Röttger
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

import os
import tempfile
import unittest

import NuMPI
import numpy as np
import pytest
from muGrid.Wrappers import FFTEngine
from NuMPI import MPI

from SurfaceTopography import open_topography
from SurfaceTopography import UniformLineScan
from SurfaceTopography.Exceptions import FileFormatMismatch, UnsupportedFormatFeature
from SurfaceTopography.IO.NPY import NPYReader, NPZReader, save_npy


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_save_and_load(file_format_examples):
    # sometimes the surface isn't transposed the same way when
    topography = open_topography(
        os.path.join(file_format_examples, 'di-4.di'), format="di").topography()

    with tempfile.TemporaryDirectory() as d:
        npyfile = os.path.join(d, 'test_save_and_load.npy')
        save_npy(npyfile, topography)

        loaded_topography = NPYReader(npyfile).topography(
            # nb_subdomain_grid_pts=topography.nb_grid_pts,
            # subdomain_locations=(0,0),
            physical_sizes=(1., 1.))

        np.testing.assert_allclose(loaded_topography.heights(),
                                   topography.heights())


@pytest.mark.skipif(
    NuMPI._has_mpi4py,
    reason="NuMPI is using MPI I/O which does not support Python streams")
@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_load_binary_stream(file_format_examples):
    with open(os.path.join(file_format_examples, 'example-2d.npy'), mode="rb") as f:
        loaded_topography = NPYReader(f).topography(
            # nb_subdomain_grid_pts=topography.nb_grid_pts,
            # subdomain_locations=(0,0),
            physical_sizes=(1., 1.))
        loaded_topography


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_save_and_load_np(file_format_examples):
    # sometimes the surface isn't transposed the same way when

    topography = open_topography(
        os.path.join(file_format_examples, 'di-4.di'),
        format="di").topography()

    with tempfile.TemporaryDirectory() as d:
        npyfile = os.path.join(d, 'test_save_and_load_np.npy')
        np.save(npyfile, topography.heights())

        loaded_topography = NPYReader(npyfile).topography(physical_sizes=(1., 1.))

        np.testing.assert_allclose(loaded_topography.heights(),
                                   topography.heights())


@pytest.fixture
def examplefile(comm, file_format_examples):
    fn = file_format_examples + "/workflowtest.npy"
    res = (128, 64)
    np.random.seed(1)
    data = np.random.random(res)
    data -= np.mean(data)
    if comm.rank == 0:
        np.save(fn, data)

    comm.barrier()
    return (fn, res, data)


@pytest.mark.parametrize("loader", [open_topography, NPYReader])
def test_reader(comm, loader, examplefile):
    fn, res, data = examplefile

    # Read metadata from the file and returns a UniformTopography Object
    fileReader = loader(fn, communicator=comm)
    fileReader.nb_grid_pts = fileReader.channels[0].nb_grid_pts

    assert fileReader.nb_grid_pts == res

    fftengine = FFTEngine(fileReader.nb_grid_pts, communicator=comm)

    top = fileReader.topography(
        subdomain_locations=fftengine.subdomain_locations,
        nb_subdomain_grid_pts=fftengine.nb_subdomain_grid_pts,
        physical_sizes=fileReader.nb_grid_pts)

    assert top.nb_grid_pts == fftengine.nb_domain_grid_pts
    assert top.nb_subdomain_grid_pts \
           == fftengine.nb_subdomain_grid_pts
    # or top.nb_subdomain_grid_pts == (0,0) # for FreeFFTElHS
    assert top.subdomain_locations == fftengine.subdomain_locations

    np.testing.assert_array_equal(top.heights(), data[top.subdomain_slices])

    # test that the slicing is what is expected
    nb_domain_grid_pts = fftengine.nb_domain_grid_pts
    nb_subdomain_grid_pts = fftengine.nb_subdomain_grid_pts
    subdomain_locations = fftengine.subdomain_locations

    fulldomain_field = np.arange(np.prod(nb_domain_grid_pts)
                                 ).reshape(nb_domain_grid_pts)

    np.testing.assert_array_equal(
        fulldomain_field[top.subdomain_slices],
        fulldomain_field[tuple([
            slice(subdomain_locations[i],
                  subdomain_locations[i]
                  + max(0, min(nb_domain_grid_pts[i]
                               - subdomain_locations[i],
                               nb_subdomain_grid_pts[i])))
            for i in range(len(nb_domain_grid_pts))])])


class npySurfaceTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.fn = os.path.join(self.d.name, "example{}.npy".format(MPI.COMM_WORLD.Get_rank()))
        self.res = (128, 64)
        np.random.seed(1)
        self.data = np.random.random(self.res)
        self.data -= np.mean(self.data)

        np.save(self.fn, self.data)

    def test_read(self):
        size = (2, 4)
        loader = NPYReader(self.fn, communicator=MPI.COMM_SELF)

        topo = loader.topography(physical_sizes=size)

        np.testing.assert_array_almost_equal(topo.heights(), self.data)

        # self.assertEqual(topo.info, loader.info)
        self.assertEqual(topo.physical_sizes, size)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_line_scan(tmp_path):
    fn = str(tmp_path / 'line.npy')
    np.save(fn, np.arange(5.))
    reader = NPYReader(fn, communicator=MPI.COMM_SELF)
    assert reader.default_channel.dim == 1
    assert reader.default_channel.nb_grid_pts == (5,)
    t = reader.topography(physical_sizes=(2.,))
    assert isinstance(t, UniformLineScan)
    np.testing.assert_allclose(t.heights(), np.arange(5.))


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("data", [np.ones((2, 3, 4)), np.ones((2, 3), dtype=complex)])
def test_unsupported_arrays(tmp_path, data):
    # Gwyddion's npyfile.c only imports two-dimensional real arrays;
    # SurfaceTopography additionally supports line scans
    fn = str(tmp_path / 'unsupported.npy')
    np.save(fn, data)
    with pytest.raises(UnsupportedFormatFeature):
        NPYReader(fn, communicator=MPI.COMM_SELF)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("dtype", ["<f4", ">f8", "<i2", ">u2", "f2"])
def test_dtypes(tmp_path, dtype):
    fn = str(tmp_path / 'dtype.npy')
    data = np.arange(12).reshape(4, 3).astype(dtype)
    np.save(fn, data)
    t = NPYReader(fn, communicator=MPI.COMM_SELF).topography(physical_sizes=(1., 1.))
    np.testing.assert_allclose(t.heights(), data.astype(float))


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
@pytest.mark.parametrize("save", [np.savez, np.savez_compressed])
def test_npz(tmp_path, save):
    fn = str(tmp_path / 'arrays.npz')
    a = np.arange(12.).reshape(4, 3)
    b = np.arange(20, dtype=np.int16).reshape(5, 4)
    save(fn, first=a, line=np.arange(3.), volume=np.ones((2, 2, 2)), second=b,
         label=np.array(['a', 'b']))

    reader = NPZReader(fn)
    assert [c.name for c in reader.channels] == ['first', 'second']
    assert reader.channels[0].nb_grid_pts == (4, 3)
    assert reader.channels[1].nb_grid_pts == (5, 4)
    t = reader.topography(physical_sizes=(1., 2.))
    np.testing.assert_allclose(t.heights(), a)
    t = reader.topography(channel_index=1, physical_sizes=(1., 2.))
    np.testing.assert_allclose(t.heights(), b)
    assert t.physical_sizes == (1., 2.)

    with open(fn, 'rb') as f:
        reader = NPZReader(f)
        np.testing.assert_allclose(reader.topography(physical_sizes=(1., 2.)).heights(), a)


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_npz_without_images(tmp_path, file_format_examples):
    fn = str(tmp_path / 'arrays.npz')
    np.savez(fn, line=np.arange(3.))
    with pytest.raises(FileFormatMismatch):
        NPZReader(fn)
    with pytest.raises(FileFormatMismatch):
        NPZReader(os.path.join(file_format_examples, 'example-2d.npy'))


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_npz_detection(tmp_path, file_format_examples):
    fn = str(tmp_path / 'arrays.npz')
    np.savez(fn, first=np.arange(12.).reshape(4, 3))
    assert open_topography(fn).format() == 'npz'
    # Other ZIP-based formats are still detected by their own readers
    for name, fmt in [('nmm-1.zip', 'nmm'), ('plux-1.plux', 'plux'), ('poir-1.poir', 'poir')]:
        assert open_topography(os.path.join(file_format_examples, name)).format() == fmt
