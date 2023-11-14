#
# Copyright 2015-2016, 2019-2023 Lars Pastewka
#           2019-2020 Antoine Sanner
#           2020 Michael Röttger
#           2019 Kai Haase
#           2015-2016 Till Junge
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

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography.IO.MI import MIReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__))),
    'file_format_examples')


def test_read_header():
    file_path = os.path.join(DATADIR, 'mi-1.mi')

    loader = MIReader(file_path)

    # Like in Gwyddion, there should be 4 channels in total
    assert len(loader.channels) == 4
    assert [ch.name for ch in loader.channels] == ['Topography',
                                                   'Deflection',
                                                   'Friction', 'Friction']

    # Check if metadata has been read in correctly
    assert loader.channels[0].dim == 2
    assert loader.channels[0].nb_grid_pts == (256, 256)
    assert loader.channels[0].physical_sizes == (2e-05, 2e-05)
    assert loader.channels[0].info['raw_metadata']['DisplayOffset'] == '8.8577270507812517e-004'
    assert loader.channels[0].info['raw_metadata']['DisplayRange'] == '1.3109436035156252e-002'
    assert loader.channels[0].info['raw_metadata']['acqMode'] == 'Main'
    assert loader.channels[0].info['raw_metadata']['label'] == 'Topography'
    assert loader.channels[0].info['raw_metadata']['range'] == '2.9025000000000003e+000'
    assert loader.channels[0].info['raw_metadata']['direction'] == 'Trace'
    assert loader.channels[0].info['raw_metadata']['filter'] == '3rd_order'
    assert loader.channels[0].info['raw_metadata']['name'] == 'Topography'
    assert loader.channels[0].info['raw_metadata']['trace'] == 'Trace'
    assert loader.channels[0].info['unit'] == 'µm'
    assert loader.channels[0].unit == 'µm'

    assert loader.default_channel.index == 0
    assert loader.default_channel.nb_grid_pts == (256, 256)

    # Some metadata value
    assert loader.info['biasSample'] == 'TRUE'


def test_topography():
    file_path = os.path.join(DATADIR, 'mi-1.mi')

    loader = MIReader(file_path)

    topography = loader.topography()

    # Check one height value
    np.testing.assert_allclose(topography._heights[0, 0], -0.4986900329589844, rtol=1e-6)

    # Check out if metadata from global and the channel are both in the
    # result from channel metadata
    assert 'direction' in topography.info['raw_metadata'].keys()
    # From global metadata
    assert 'zDacRange' in topography.info['raw_metadata'].keys()

    # Check the value of one of the metadata
    assert topography.info['unit'] == 'µm'
