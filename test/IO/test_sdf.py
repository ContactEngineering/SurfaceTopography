#
# Copyright 2025 Lars Pastewka
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

import io
import os
import struct

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import SDFReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_sdf_ascii_filestream(file_format_examples):
    file_path = os.path.join(file_format_examples, 'sdf-1.sdf')

    read_topography(file_path)

    with open(file_path, 'rb') as f:
        read_topography(f)


def test_sdf_ascii_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'sdf-1.sdf')

    r = SDFReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 7
    assert ny == 4

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 7.0, rtol=1e-3)
    np.testing.assert_allclose(sy, 4.0, rtol=1e-3)

    assert t.unit == 'µm'


def test_read_sdf_binary_filestream(file_format_examples):
    file_path = os.path.join(file_format_examples, 'sdf-2.sdf')

    read_topography(file_path)

    with open(file_path, 'rb') as f:
        read_topography(f)


def test_sdf_binary_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'sdf-2.sdf')

    r = SDFReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 2000
    assert ny == 1000

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 500.0, rtol=1e-3)
    np.testing.assert_allclose(sy, 250.0, rtol=1e-3)

    assert t.unit == 'µm'


def test_sdf_format_detection(file_format_examples):
    from SurfaceTopography.IO import detect_format

    # Test ASCII format detection
    file_path = os.path.join(file_format_examples, 'sdf-1.sdf')
    assert detect_format(file_path) == 'sdf'

    # Test binary format detection
    file_path = os.path.join(file_format_examples, 'sdf-2.sdf')
    assert detect_format(file_path) == 'sdf'


_SDF_DTYPES = {0: 'u1', 1: '<u2', 2: '<u4', 3: '<f4', 4: 'i1', 5: '<i2',
               6: '<i4', 7: '<f8'}
_SDF_MARKERS = {1: 2**16 - 1, 2: 2**32 - 1, 5: -2**15, 6: -2**31}


def _make_binary_sdf(data, data_type, compression=0):
    """Build a binary SDF file; `data` is in (ny, nx) row order"""
    ny, nx = data.shape
    header = b'bISO-1.0' + b'TEST'.ljust(10, b'\x00') + b'010120240000' + \
        b'010120240000' + struct.pack('<HHddddBBB', nx, ny, 1e-6, 2e-6, 1e-9,
                                      1e-9, compression, data_type, 0)
    return header + np.asarray(data, dtype=_SDF_DTYPES[data_type]).tobytes()


@pytest.mark.parametrize('data_type', range(8))
def test_sdf_binary_data_types(data_type):
    nx, ny = 3, 2
    data = np.arange(nx * ny).reshape(ny, nx) + 1
    expected_mask = np.zeros((ny, nx), dtype=bool)
    if data_type in _SDF_MARKERS:
        data[1, 1] = _SDF_MARKERS[data_type]
        expected_mask[1, 1] = True
    elif data_type in (3, 7):
        data = data.astype(float)
        data[1, 1] = np.nan
        expected_mask[1, 1] = True
    t = SDFReader(io.BytesIO(_make_binary_sdf(data, data_type))).topography()
    assert t.nb_grid_pts == (nx, ny)
    assert t.unit == 'µm'
    np.testing.assert_allclose(t.physical_sizes, (3.0, 4.0))
    h = np.ma.getmaskarray(t.heights())
    np.testing.assert_array_equal(h, expected_mask.T)
    np.testing.assert_allclose(t.heights()[~expected_mask.T],
                               1e-3 * data.T[~expected_mask.T])


def test_sdf_compressed_unsupported():
    from SurfaceTopography.Exceptions import UnsupportedFormatFeature
    data = np.ones((2, 2))
    with pytest.raises(UnsupportedFormatFeature):
        SDFReader(io.BytesIO(_make_binary_sdf(data, 7, compression=1)))


def test_sdf_ascii_uint16_marker():
    text = """aISO-1.0
ManufacID = TEST
CreateDate = 010120240000
ModDate = 010120240000
NumPoints = 2
NumProfiles = 2
Xscale = 1.0E-6
Yscale = 1.0E-6
Zscale = 1.0E-9
Zresolution = 1.0E-9
Compression = 0
DataType = 1
CheckType = 0
*
1 65535
3 4
*
"""
    t = SDFReader(io.BytesIO(text.encode('ascii'))).topography()
    np.testing.assert_array_equal(np.ma.getmaskarray(t.heights()),
                                  [[False, False], [True, False]])
    np.testing.assert_allclose(t.heights()[0], [1e-3, 3e-3])
