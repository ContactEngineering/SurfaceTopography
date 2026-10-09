#
# Copyright 2020-2023 Lars Pastewka
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
from SurfaceTopography.IO import BCRReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'bcrf-1.bcrf')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_bcr_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'bcrf-1.bcrf')

    r = BCRReader(file_path)
    assert len(r.channels) == 1

    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 960
    assert ny == 600

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 1777404, rtol=1e-6)
    np.testing.assert_allclose(sy, 1110878, rtol=1e-6)

    assert t.unit == 'nm'

    np.testing.assert_allclose(t.rms_height_from_area(), 24.560207201442292, rtol=1e-6)

    np.testing.assert_allclose(t.rms_height_from_profile(), 24.009754961959388, rtol=1e-6)
    np.testing.assert_allclose(t.transpose().rms_height_from_profile(), 24.134045799796326, rtol=1e-6)


def _make_bcr(fileformat, data, header_lines, header_size=2048,
              encoding='latin-1'):
    """Build a BCR file; `data` is in (ny, nx) row order"""
    ny, nx = data.shape
    header = f'fileformat = {fileformat}\n' + \
        f'xpixels = {nx}\nypixels = {ny}\n' + \
        ''.join(f'{line}\n' for line in header_lines)
    header = header.ljust(header_size).encode(encoding)
    return header + data.tobytes()


def test_bcrstm_without_headersize_and_units():
    """
    Old files have no `headersize` (the header is then 2048 characters),
    units default to nm, `bit2nm` scales integer data and 32767 marks void
    pixels (independent of the `voidpixels` key).
    """
    data = np.array([[1, 2, 3], [4, 32767, -6]], dtype='<i2')
    buffer = _make_bcr('bcrstm', data, ['xlength = 30', 'ylength = 20',
                                        'bit2nm = 0.5', 'intelmode = 1'])
    t = BCRReader(io.BytesIO(buffer)).topography()
    assert t.nb_grid_pts == (3, 2)
    np.testing.assert_allclose(t.physical_sizes, (30, 20))
    assert t.unit == 'nm'
    h = t.heights()
    np.testing.assert_array_equal(h.mask, (data == 32767).T)
    np.testing.assert_allclose(h[~h.mask], 0.5 * data.T[~h.mask])


def test_bcrstm_big_endian_zunit():
    data = np.array([[1, 2], [3, 4]], dtype='>i2')
    buffer = _make_bcr('bcrstm', data, [
        'xlength = 2', 'ylength = 2', 'xunit = um', 'yunit = um',
        'zunit = nm', 'bit2nm = 2', 'intelmode = 0'])
    t = BCRReader(io.BytesIO(buffer)).topography()
    assert t.unit == 'µm'
    assert not t.has_undefined_data
    np.testing.assert_allclose(t.heights(), 2e-3 * data.T)


@pytest.mark.parametrize('encoding, fileformat', [
    ('latin-1', 'bcrf'), ('utf-16-le', 'bcrf_unicode')])
def test_bcrf_void_pixels_and_bit2nm(encoding, fileformat):
    """
    Floating-point data is stored in `zunit` (`bit2nm` does not apply);
    values above 1.7e38 are void even without a `voidpixels` key.
    """
    data = np.array([[1.5, 3.4028235e38, 2.0], [np.nan, 4.0, 2e38]],
                    dtype='<f4')
    buffer = _make_bcr(fileformat, data, [
        'headersize = 2048', 'xlength = 3', 'ylength = 2', 'xunit = um',
        'yunit = um', 'zunit = nm', 'bit2nm = 10', 'intelmode = 1'],
        encoding=encoding)
    t = BCRReader(io.BytesIO(buffer)).topography()
    assert t.unit == 'µm'
    h = t.heights()
    expected_mask = np.array([[False, True, False], [True, False, True]]).T
    np.testing.assert_array_equal(h.mask, expected_mask)
    np.testing.assert_allclose(h[~expected_mask],
                               1e-3 * data.T[~expected_mask])
