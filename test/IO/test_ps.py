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

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography.IO import PSReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_ps_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'example_ps.tiff')

    r = PSReader(file_path)

    assert r.channels[0].name == 'Z Height'

    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 512
    assert ny == 512

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 1.0, rtol=1e-6)
    np.testing.assert_allclose(sy, 1.0, rtol=1e-6)

    assert t.unit == 'µm'

    np.testing.assert_allclose(t.rms_height_from_area(), 0.003933333499668988, rtol=1e-6)


def _patched_ps(file_format_examples, **fields):
    """
    Return the example file with modified fields of the Park Systems image
    header (`image_type`, `data_scale_factor`, `data_offset` and
    `data_unit`).
    """
    import io
    import struct

    import tifffile

    file_path = os.path.join(file_format_examples, 'example_ps.tiff')
    with tifffile.TiffFile(file_path) as tiff:
        header_offset = tiff.pages[0].tags[50435].valueoffset
    with open(file_path, 'rb') as f:
        buffer = bytearray(f.read())
    offsets = {'image_type': (0, '<I'), 'data_scale_factor': (228, '<d'), 'data_offset': (236, '<d')}
    for name, value in fields.items():
        if name == 'data_unit':
            buffer[header_offset + 244:header_offset + 260] = value.encode('utf-16-le').ljust(16, b'\0')
        else:
            offset, fmt = offsets[name]
            struct.pack_into(fmt, buffer, header_offset + offset, value)
    return io.BytesIO(bytes(buffer))


def _raw_ps_data(file_format_examples):
    """Raw data of the example file, in Gwyddion's row order"""
    import tifffile

    file_path = os.path.join(file_format_examples, 'example_ps.tiff')
    with tifffile.TiffFile(file_path) as tiff:
        data = tiff.pages[0].tags[50434].value
    raw = np.frombuffer(data, dtype='<f4').reshape(512, 512).astype(float)
    # Gwyddion flips the image vertically
    return raw[::-1, :]


def test_ps_heights_like_gwyddion(file_format_examples):
    """Heights and orientation as in Gwyddion's psia.c"""
    t = PSReader(os.path.join(file_format_examples, 'example_ps.tiff')).topography()
    gain = -12.227015495171795
    np.testing.assert_allclose(t.heights(), (gain * _raw_ps_data(file_format_examples)).T, rtol=1e-6)


def test_ps_scale_and_offset(file_format_examples):
    """Heights are `data_gain * (data_scale_factor * raw + data_offset)`"""
    gain = -12.227015495171795
    raw = _raw_ps_data(file_format_examples)

    t = PSReader(_patched_ps(file_format_examples, data_scale_factor=2.0, data_offset=0.5,
                             data_unit='nm')).topography()
    assert t.unit == 'µm'
    np.testing.assert_allclose(t.heights(), (gain * (2 * raw + 0.5) * 1e-3).T, rtol=1e-5)

    # A vanishing scale factor means unity; a missing unit means µm
    t = PSReader(_patched_ps(file_format_examples, data_scale_factor=0.0, data_unit='')).topography()
    np.testing.assert_allclose(t.heights(), (gain * raw).T, rtol=1e-6)


def test_ps_line_profile(file_format_examples):
    from SurfaceTopography.Exceptions import UnsupportedFormatFeature

    with pytest.raises(UnsupportedFormatFeature):
        PSReader(_patched_ps(file_format_examples, image_type=1))
