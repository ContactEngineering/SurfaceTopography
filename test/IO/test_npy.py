#
# Copyright 2019-2020 Lars Pastewka
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

import unittest
import tempfile

import numpy as np
import os

import pytest

from muFFT import FFT

from SurfaceTopography import open_topography
from SurfaceTopography.IO.NPY import NPYReader
from SurfaceTopography.IO.NPY import save_npy

from NuMPI import MPI

DATADIR = os.path.dirname(os.path.realpath(__file__))


def test_save_and_load(comm_self, file_format_examples):
    # sometimes the surface isn't transposed the same way when
    topography = open_topography(
        os.path.join(file_format_examples, 'di4.di'), format="di").topography()

    with tempfile.TemporaryDirectory() as d:
        npyfile = os.path.join(d, 'test_save_and_load.npy')
        save_npy(npyfile, topography)

        loaded_topography = NPYReader(npyfile, communicator=comm_self).topography(
            # nb_subdomain_grid_pts=topography.nb_grid_pts,
            # subdomain_locations=(0,0),
            physical_sizes=(1., 1.))

        np.testing.assert_allclose(loaded_topography.heights(),
                                   topography.heights())


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")
def test_load_binary(comm_self, file_format_examples):
    with open(os.path.join(file_format_examples, 'example-2d.npy'),
              mode="rb") as f:
        loaded_topography = NPYReader(f, communicator=comm_self).topography(
            # nb_subdomain_grid_pts=topography.nb_grid_pts,
            # subdomain_locations=(0,0),
            physical_sizes=(1., 1.))
        loaded_topography


def test_save_and_load_np(comm_self, file_format_examples):
    # sometimes the surface isn't transposed the same way when

    topography = open_topography(
        os.path.join(file_format_examples, 'di4.di'),
        format="di").topography()

    with tempfile.TemporaryDirectory() as d:
        npyfile = os.path.join(d, 'test_save_and_load_np.npy')
        np.save(npyfile, topography.heights())

        loaded_topography = NPYReader(npyfile, communicator=comm_self).topography(
            physical_sizes=(1., 1.))

        np.testing.assert_allclose(loaded_topography.heights(),
                                   topography.heights())


@pytest.fixture
def examplefile(comm):
    fn = DATADIR + "/workflowtest.npy"
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

    fftengine = FFT(nb_grid_pts=fileReader.nb_grid_pts,
                    fft="mpi",
                    communicator=comm)

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

    fulldomain_field = np.arange(np.prod(fftengine.nb_domain_grid_pts)
                                 ).reshape(fftengine.nb_domain_grid_pts)

    np.testing.assert_array_equal(
        fulldomain_field[top.subdomain_slices],
        fulldomain_field[tuple([
            slice(fftengine.subdomain_locations[i],
                  fftengine.subdomain_locations[i]
                  + max(0, min(fftengine.nb_domain_grid_pts[i]
                               - fftengine.subdomain_locations[i],
                               fftengine.nb_subdomain_grid_pts[i])))
            for i in range(len(fftengine.nb_domain_grid_pts))])])


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
