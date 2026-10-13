#
# Copyright 2020-2024 Lars Pastewka
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

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import AL3DReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'al3d-1.al3d')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_al3d_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'al3d-1.al3d')

    r = AL3DReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 200
    assert ny == 296

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 8.76054e-05, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.000129655992, rtol=1e-6)

    assert t.unit == 'm'
    assert t.info['instrument']['vendor'] == 'Alicona Imaging'
    assert t.info['instrument']['software'] == 'MeasureSuite 5.3.6'

    np.testing.assert_allclose(t.rms_height_from_area(), 7.688266102603082e-06, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 3.915731160953795e-06, rtol=1e-6)
    np.testing.assert_allclose(t.transpose().rms_height_from_profile(), 6.620846789152015e-06, rtol=1e-6)


def _al3d_tag(key, value):
    return key.encode('latin-1').ljust(20, b'\x00') + \
        value.encode('latin-1').ljust(30, b'\x00') + b'\r\n'


def _make_al3d(data, invalid, comment=''):
    """Build a minimal AL3D file; `data` is in (ny, nx) row order"""
    ny, nx = data.shape
    stride = (nx * 4 + 7) // 8 * 8 // 4  # rows are padded to 8 bytes
    tags = [('Cols', str(nx)), ('Rows', str(ny)),
            ('PixelSizeXMeter', '1e-06'), ('PixelSizeYMeter', '2e-06'),
            ('InvalidPixelValue', invalid), ('DepthImageOffset', None)]
    header_size = 17 + 52 * (2 + len(tags)) + 256
    buffer = b'AliconaImaging\x00\r\n' + _al3d_tag('Version', '1') + \
        _al3d_tag('TagCount', str(len(tags)))
    for key, value in tags:
        buffer += _al3d_tag(key, str(header_size) if value is None else value)
    buffer += comment.encode('latin-1').ljust(254, b'\x00') + b'\r\n'
    padded = np.zeros((ny, stride), dtype='<f4')
    padded[:, :nx] = data
    return buffer + padded.tobytes()


@pytest.mark.parametrize('invalid', ['3.000000028082e+15', '-1e+10', 'nan'])
def test_al3d_invalid_pixels_and_comment(invalid):
    nx, ny = 3, 4  # odd nx forces row padding
    data = np.arange(nx * ny, dtype=np.float32).reshape(ny, nx) * 1e-6
    invalid_value = np.float32(float(invalid))
    data[1, 2] = invalid_value
    data[3, 0] = np.nan
    buffer = _make_al3d(data, invalid, comment='Measured by me')
    t = AL3DReader(io.BytesIO(buffer)).topography()
    assert t.nb_grid_pts == (nx, ny)
    np.testing.assert_allclose(t.physical_sizes, (nx * 1e-6, ny * 2e-6))
    assert t.info['comment'] == 'Measured by me'
    assert t.has_undefined_data
    h = t.heights()
    expected_mask = np.zeros((nx, ny), dtype=bool)
    expected_mask[2, 1] = True
    expected_mask[0, 3] = True
    np.testing.assert_array_equal(h.mask, expected_mask)
    np.testing.assert_allclose(h[~expected_mask], data.T[~expected_mask])
