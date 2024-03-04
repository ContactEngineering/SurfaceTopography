#
# Copyright 2020-2024 Lars Pastewka
#           2021 Michael Röttger
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
import zipfile
from datetime import datetime

import numpy as np
import pytest
import yaml
from NuMPI import MPI
from numpy.testing import assert_allclose

import SurfaceTopography
from SurfaceTopography import (open_topography, read_container,
                               read_published_container, read_topography)
from SurfaceTopography.Container.IO import CEReader
from SurfaceTopography.Container.SurfaceContainer import \
    InMemorySurfaceContainer

from .test_io import binary_example_file_list, text_example_file_list

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read_just_uniform(file_format_examples):
    for c, in [
        read_container(f'{file_format_examples}/container-1.zip'),
        # read_published_container('https://contact.engineering/go/867nv/')  # Same a container-1.zip
        # TODO maybe this makes the web app stall when running MPI tests von Travis, further investigation needed
    ]:
        assert len(c) == 3

        assert not c[0].is_periodic
        assert not c[1].is_periodic
        assert not c[2].is_periodic

        assert c[0].nb_grid_pts == (500, 500)
        assert c[1].nb_grid_pts == (500, 500)
        assert c[2].nb_grid_pts == (500, 500)

        assert c[0].physical_sizes == (100, 100)
        assert c[1].physical_sizes == (10, 10)
        assert c[2].physical_sizes == (1, 1)

        assert c[0].info['unit'] == 'µm'
        assert c[1].info['unit'] == 'µm'
        assert c[2].info['unit'] == 'µm'


def test_read_mixed(file_format_examples):
    c, = read_container(f'{file_format_examples}/container-2.zip')

    assert len(c) == 3


def test_write(file_format_examples):
    t1 = read_topography(f'{file_format_examples}/di-1.di')
    t2 = read_topography(f'{file_format_examples}/opd-1.opd')
    t3 = read_topography(f'{file_format_examples}/matrix-2.txt')

    c = InMemorySurfaceContainer([t1, t2, t3])

    with tempfile.TemporaryFile() as fobj:
        c.to_zip(fobj)

        c2, = read_container(fobj)

        assert len(c2) == 3

        assert not c2[0].is_periodic
        assert not c2[1].is_periodic
        assert not c2[2].is_periodic

        assert c2[0].nb_grid_pts == t1.nb_grid_pts
        assert c2[1].nb_grid_pts == t2.nb_grid_pts
        assert c2[2].nb_grid_pts == t3.nb_grid_pts

        assert_allclose(c2[0].physical_sizes, t1.physical_sizes)
        assert_allclose(c2[1].physical_sizes, t2.physical_sizes)
        assert_allclose(c2[2].physical_sizes, t3.physical_sizes)

        assert c2[0].info['unit'] == t1.info['unit']
        assert c2[1].info['unit'] == t2.info['unit']
        assert c2[2].info['unit'] == t3.info['unit']


def test_periodic():
    container, = read_published_container("https://contact.engineering/go/v9qwe/")

    pristine = container[0]
    convoluted = container[1]
    assert pristine.is_periodic
    assert convoluted.is_periodic


@pytest.mark.parametrize('filenames', [binary_example_file_list, text_example_file_list])
def test_read_files_from_container(filenames):
    """BCRF and GWY file use np.fromfile to read data, which has issues when reading within a ZIP file"""
    with tempfile.TemporaryDirectory() as d:
        containerfn = f'{d}/container.zip'

        # Write container with raw data files
        with zipfile.ZipFile(containerfn, 'w') as z:
            topographies = []
            for filepath in filenames:
                _, fn = os.path.split(filepath)
                z.write(filepath, fn)
                topography = {
                    'datafile': {'original': fn}
                }

                reader = open_topography(filepath)
                if reader.default_channel.physical_sizes is None:
                    topography['size'] = (1.,) * reader.default_channel.dim

                topographies += [topography]

            metadata = {
                'versions': {'SurfaceTopography': SurfaceTopography.__version__},
                'surfaces': [{
                    'topographies': topographies
                }],
                'creation_time': str(datetime.now()),
            }
            z.writestr("meta.yml", yaml.dump(metadata))

        os.system(f'cp {containerfn} /Users/pastewka/Downloads/')

        r = CEReader(containerfn)
        c = r.container()
        for t in c:
            # This loop actually reads the files
            # The test is that file reading progresses without issues
            pass


def test_ce_container():
    surface, = read_published_container('https://doi.org/10.57703/ce-mg4cy')
    rms_heights = [t.rms_height_from_profile() for t in surface]
    # This is a regression test. The values are not checked for correctness.
    np.testing.assert_allclose(rms_heights,
                               [0.003949841631562571, 0.004032999294137858, 0.013820472252164628, 0.0217028756922634,
                                0.004851062064249796, 0.02123958101044809, 0.030065647090746887, 0.009013796742461624,
                                0.019992889576544735, 0.03571244942555436, 0.019188774584700925, 0.03026452140452409,
                                0.04009296334584872, 0.03591547319579487, 0.026463673998312873, 0.18154731113775002,
                                1.123743993604103, 0.078929223879882])
