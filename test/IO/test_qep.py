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

from SurfaceTopography.Exceptions import MetadataAlreadyFixedByFile
from SurfaceTopography.IO import detect_format, open_topography
from SurfaceTopography.IO.QEP import QEPReader
from SurfaceTopography.NonuniformLineScan import NonuniformLineScan
from SurfaceTopography.UniformLineScanAndTopography import UniformLineScan

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


def _expected_measured():
    """Positions/heights of the mock's measured profile (see generator)."""
    n = 60
    steps = np.where(np.arange(n - 1) % 2 == 0, 0.0005, 0.0015)
    x = 10.0 + np.concatenate(([0.0], np.cumsum(steps)))
    z = 5.0 + 0.01 * np.sin(np.arange(n) / 7.0)
    valid = np.ones(n, dtype=bool)
    valid[[5, 20, 40]] = False  # flagged invalid, omitted by the reader
    return x[valid] - x[valid][0], z[valid]


def test_detect_format(file_format_examples):
    assert (
        detect_format(os.path.join(file_format_examples, "qep-1.qep")) == "qep"
    )


def test_channels(file_format_examples):
    reader = open_topography(os.path.join(file_format_examples, "qep-1.qep"))
    assert len(reader.channels) == 2
    measured, reference = reader.channels

    assert measured.name == "measured"
    assert measured.dim == 1
    # 60 points, 3 flagged invalid and omitted (grid is nonuniform)
    assert measured.nb_grid_pts == (57,)
    assert measured.physical_sizes == pytest.approx((0.0585,))
    assert measured.unit == "mm"
    assert not measured.is_uniform
    assert not measured.has_undefined_data

    assert reference.name == "reference"
    assert reference.dim == 1
    # Equidistant grid is kept intact; invalid points become undefined
    assert reference.nb_grid_pts == (25,)
    assert reference.physical_sizes == pytest.approx((1.25,))
    assert reference.is_uniform
    assert reference.has_undefined_data

    for ch in (measured, reference):
        assert ch.info["acquisition_time"] == datetime.datetime(
            2019, 8, 1, 10, 11, 12
        )
        assert ch.info["instrument"]["vendor"] == "Mahr"
        assert ch.info["instrument"]["name"] == "MockMachine"
        assert ch.info["instrument"]["parameters"]["tip_radius"] == {
            "value": 0.002,
            "unit": "mm",
        }


def test_measured_profile(file_format_examples):
    t = QEPReader(
        os.path.join(file_format_examples, "qep-1.qep")
    ).topography()
    assert isinstance(t, NonuniformLineScan)
    assert t.unit == "mm"
    x_expected, h_expected = _expected_measured()
    x, h = t.positions_and_heights()
    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(h, h_expected)


def test_reference_profile(file_format_examples):
    reader = QEPReader(os.path.join(file_format_examples, "qep-1.qep"))
    t = reader.topography(channel_index=1)
    assert isinstance(t, UniformLineScan)
    assert t.has_undefined_data
    h = t.heights()
    assert np.ma.count_masked(h) == 2
    assert h[3] is np.ma.masked and h[10] is np.ma.masked
    expected = 5.0 + 0.001 * np.arange(25)
    np.testing.assert_allclose(np.ma.filled(h, 0), np.where(
        np.isin(np.arange(25), [3, 10]), 0, expected
    ))
    assert t.physical_sizes == pytest.approx((1.25,))


def test_metadata_fixed_by_file(file_format_examples):
    reader = QEPReader(os.path.join(file_format_examples, "qep-1.qep"))
    with pytest.raises(MetadataAlreadyFixedByFile):
        reader.topography(physical_sizes=(1.0,))
    with pytest.raises(MetadataAlreadyFixedByFile):
        reader.topography(unit="µm")


def test_nonuniform_cannot_be_periodic(file_format_examples):
    reader = QEPReader(os.path.join(file_format_examples, "qep-1.qep"))
    with pytest.raises(ValueError):
        reader.topography(channel_index=0, periodic=True)
