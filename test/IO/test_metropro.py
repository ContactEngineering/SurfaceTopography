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

import os
import struct

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.Exceptions import UnsupportedFormatFeature
from SurfaceTopography.IO import MetroProReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'metropro-1.dat')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_metropro_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'metropro-1.dat')

    r = MetroProReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 640
    assert ny == 480

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.0007028812979115173, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.000527160973433638, rtol=1e-6)

    assert t.unit == 'm'
    assert t.info['instrument']['vendor'] == 'Zygo'

    np.testing.assert_allclose(t.rms_height_from_area(), 7.528822204734589e-08, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 7.524071e-08, rtol=1e-6)

    t = t.detrend('curvature')
    np.testing.assert_allclose(t.rms_height_from_area(), 3.911386124282179e-09, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 3.868313e-09, rtol=1e-6)


def test_metropro_little_endian_header_fields(file_format_examples):
    """
    The second part of the common header is stored little endian (as in
    Gwyddion's metropro.c). As big endian, `min_mod_pct` would decode to a
    denormal number.
    """
    r = MetroProReader(os.path.join(file_format_examples, 'metropro-1.dat'))
    raw = r.channels[0].info['raw_metadata']
    assert raw['min_mod_pct'] == 1.0
    # Big endian start of the header
    assert raw['light_level_pct'] == pytest.approx(6.0639353)
    assert raw['sys_serial2'] == raw['sys_serial'] == 59407


def _patched_metropro(file_format_examples, tmp_path, offset, fmt, value):
    with open(os.path.join(file_format_examples, 'metropro-1.dat'), 'rb') as f:
        buffer = bytearray(f.read())
    struct.pack_into(fmt, buffer, offset, value)
    file_path = tmp_path / 'patched.dat'
    file_path.write_bytes(bytes(buffer))
    return file_path


def test_metropro_unknown_lateral_resolution(file_format_examples, tmp_path):
    """A lateral resolution of zero means unknown physical sizes"""
    file_path = _patched_metropro(file_format_examples, tmp_path, 184, '>f', 0.0)
    r = MetroProReader(file_path)
    assert r.channels[0].physical_sizes is None
    t = r.topography(physical_sizes=(1e-3, 1e-3))
    assert t.physical_sizes == (1e-3, 1e-3)
    t_ref = MetroProReader(os.path.join(file_format_examples, 'metropro-1.dat')).topography()
    np.testing.assert_allclose(t.heights(), t_ref.heights())


def test_metropro_header_size_mismatch(file_format_examples, tmp_path):
    """The header size must match the header format"""
    file_path = _patched_metropro(file_format_examples, tmp_path, 6, '>I', 4096)
    with pytest.raises(UnsupportedFormatFeature):
        MetroProReader(file_path).channels
