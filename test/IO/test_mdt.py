#
# Copyright 2026 Lars Pastewka
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

from SurfaceTopography.IO import detect_format, open_topography
from SurfaceTopography.IO.MDT import MDTReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


def test_detect_format(file_format_examples):
    assert (
        detect_format(os.path.join(file_format_examples, "mdt-1.mdt")) == "mdt"
    )


def test_channels(file_format_examples):
    reader = open_topography(os.path.join(file_format_examples, "mdt-1.mdt"))
    # The fixture contains a Height frame, a MAG frame whose z axis
    # carries a voltage unit (not a topography channel) and a text frame;
    # only the Height frame is reported
    assert len(reader.channels) == 1
    (ch,) = reader.channels
    assert ch.name == "Height"
    assert ch.dim == 2
    assert ch.nb_grid_pts == (64, 32)
    assert ch.physical_sizes == pytest.approx((2.5, 1.25))
    assert ch.unit == "µm"
    # The data is reported in nm and converted to the lateral unit
    assert ch.height_scale_factor == pytest.approx(1e-3)
    assert ch.info["acquisition_time"] == datetime.datetime(
        2015, 3, 24, 13, 49, 17
    )


def test_topography(file_format_examples):
    t = MDTReader(
        os.path.join(file_format_examples, "mdt-1.mdt")
    ).topography()
    assert t.is_uniform
    assert t.unit == "µm"
    nx, ny = t.nb_grid_pts
    assert (nx, ny) == (64, 32)
    assert t.physical_sizes == pytest.approx((2.5, 1.25))

    # The raster is stored row by row as int16; heights are
    # raw * z_step + z_offset in nm (the axis scales are float32 in the
    # file), converted to the lateral unit µm
    raw = ((np.arange(32 * 64) * 37) % 4001 - 2000).reshape(32, 64)
    z_step = float(np.float32(0.05))
    expected = (raw.T * z_step + 2.0) * 1e-3
    np.testing.assert_allclose(t.heights(), expected, rtol=1e-12)
