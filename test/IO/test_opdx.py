#
# Copyright 2016-2021, 2023 Lars Pastewka
#           2018-2020 Antoine Sanner
#           2018-2020 Michael Röttger
#           2019-2020 Kai Haase
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

import datetime
import os

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO.OPDx import OPDxReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'opdx-2.opdx')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    f = open(file_path, 'rb')
    read_topography(f)

    # Test is successful if it reaches end of function without raising an
    # exception


def test_read_header(file_format_examples):
    file_path = os.path.join(file_format_examples, 'opdx-2.opdx')

    loader = OPDxReader(file_path)

    channel_0, = loader.channels

    # Check if metadata has been read in

    # Default channel should be 0, 'Raw'
    assert loader.default_channel.index == 0

    #
    # Channel 0: Raw
    #
    assert channel_0.unit == 'µm'

    # .. mandatory keys
    assert channel_0.name == 'Height'
    assert channel_0.dim == 2
    np.testing.assert_allclose(channel_0.physical_sizes[1], 35.85522403809594)
    np.testing.assert_allclose(channel_0.physical_sizes[0], 47.81942809668896)
    assert channel_0.nb_grid_pts[1] == 960
    assert channel_0.nb_grid_pts[0] == 1280


def test_topography(file_format_examples):
    file_path = os.path.join(file_format_examples, 'opdx-2.opdx')

    with OPDxReader(file_path) as loader:
        assert loader.default_channel.index == 0

        topography = loader.default_channel.topography()

        # Check physical sizes
        np.testing.assert_allclose(topography.physical_sizes[0], 47.819, rtol=1e-5)
        np.testing.assert_allclose(topography.physical_sizes[1], 35.855, rtol=1e-5)

        # Check nb_grid_ptss
        assert topography.nb_grid_pts[0] == 1280
        assert topography.nb_grid_pts[1] == 960

        # Check unit
        assert topography.unit == 'µm'  # see GH 281

        # Check an entry in the metadata
        assert topography.info['acquisition_time'] == str(datetime.datetime(2018, 12, 5, 12, 53, 14))

        # Check a height value
        np.testing.assert_allclose(topography.heights()[0, 0], -7.731534)


def test_opdx_txt_consistency(file_format_examples):
    t_opdx = OPDxReader(os.path.join(file_format_examples, 'opdx-2.opdx')).topography()
    t_txt = read_topography(os.path.join(file_format_examples, 'opdx-2.txt'))
    assert abs(t_opdx.pixel_size[0] / t_opdx.pixel_size[1] - 1) < 1e-3
    assert abs(t_txt.pixel_size[0] / t_txt.pixel_size[1] - 1) < 1e-3

    ratio_ref = t_opdx.physical_sizes[1] / t_opdx.physical_sizes[0]

    assert (t_txt.physical_sizes[1] / t_txt.physical_sizes[
        0] - ratio_ref) / ratio_ref < 1e-3
    assert t_opdx.nb_grid_pts == t_txt.nb_grid_pts

    # opd file's heights are in µm, txt file's heights in m
    assert t_opdx.info['unit'] == 'µm'
    assert t_txt.info['unit'] == 'm'
    np.testing.assert_allclose(t_opdx.detrend().heights(),
                               t_txt.detrend().scale(1e6).heights(), rtol=1e-6,
                               atol=1e-3)

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.colorbar(ax.imshow(t_txt.scale(1e9).heights()))
        fig2, ax2 = plt.subplots()
        plt.colorbar(ax2.imshow(t_opdx.heights()))
        plt.show(block=True)


def test_opdx_txt_heights_lateral_consistency(file_format_examples):
    t_txt = read_topography(os.path.join(file_format_examples, 'opdx-2.txt'))

    assert t_txt.info["unit"] == "m"

    # the radius of the sphere should be 250 µm
    R = 250 * 1e-6

    rhoxx, rhoyy, rhoxy = t_txt.detrend(detrend_mode="curvature").curvatures

    assert (1 / rhoxx - R) / R < 0.01
    assert (1 / rhoyy - R) / R < 0.01


def test_opdx3(file_format_examples):
    r = OPDxReader(f'{file_format_examples}/opdx-3.opdx')
    t = r.topography()
    assert t.info['instrument']['name'] == 'Dektak Profiler'

    x, y = np.loadtxt(f'{file_format_examples}/opdx-3.txt', unpack=True)
    np.testing.assert_allclose(t.heights(), y * 1e6)
