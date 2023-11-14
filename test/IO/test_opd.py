#
# Copyright 2020-2021, 2023 Lars Pastewka
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
import pytest

from NuMPI import MPI

from SurfaceTopography.IO import open_topography
from SurfaceTopography.IO.OPD import OPDReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read_opd(file_format_examples):
    surface = OPDReader(os.path.join(file_format_examples, 'opd-1.opd')).topography()
    nx, ny = surface.nb_grid_pts
    assert nx == 640
    assert ny == 480
    sx, sy = surface.physical_sizes

    assert sx == pytest.approx(0.125909140)
    assert sy == pytest.approx(0.094431855)
    assert surface.is_uniform
    assert surface.height_scale_factor == pytest.approx(0.0005772949829101563)


def test_undefined_points(file_format_examples):
    t = OPDReader(os.path.join(file_format_examples, 'opd-2.opd')).topography()
    assert t.has_undefined_data


def test_reader(file_format_examples):
    reader = open_topography(os.path.join(file_format_examples, 'opd-1.opd'))
    assert len(reader.channels) == 1
    ch = reader.default_channel
    assert ch.physical_sizes == pytest.approx((0.125909140, 0.094431855))
    assert ch.height_scale_factor == pytest.approx(0.0005772949829101563)
