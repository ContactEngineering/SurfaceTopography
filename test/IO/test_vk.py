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
import struct
import zipfile

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import VKReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'vk4-1.vk4')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_vk3_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'vk3-1.vk3')

    r = VKReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1024
    assert ny == 768

    sx, sy = t.physical_sizes
    # Physical size follows the pixel convention (nb_pixels * pixel_size)
    np.testing.assert_allclose(sx, 705536000, rtol=1e-6)
    np.testing.assert_allclose(sy, 529152000, rtol=1e-6)

    assert t.unit == 'pm'
    assert t.info['instrument']['vendor'] == 'Keyence'

    np.testing.assert_allclose(t.rms_height_from_area(), 1223148.5774419378, rtol=1e-6)

    assert t.info['acquisition_time'].isoformat() == '2022-10-28T09:51:59+02:00'


def test_vk4_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'vk4-1.vk4')

    r = VKReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1024
    assert ny == 768

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 1397695488, rtol=1e-6)
    np.testing.assert_allclose(sy, 1048271616, rtol=1e-6)

    assert t.unit == 'pm'
    assert t.info['instrument']['vendor'] == 'Keyence'

    np.testing.assert_allclose(t.rms_height_from_area(), 54193042.85097, rtol=1e-6)

    assert str(t.info['acquisition_time']) == '2022-10-14 09:23:04+02:00'


def test_vk6_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'vk6-1.vk6')

    r = VKReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 2048
    assert ny == 1536

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 97216512, rtol=1e-6)
    np.testing.assert_allclose(sy, 72912384, rtol=1e-6)

    assert t.unit == 'pm'
    assert t.info['instrument']['vendor'] == 'Keyence'

    np.testing.assert_allclose(t.rms_height_from_area(), 1061663.7395845044, rtol=1e-6)

    assert t.info['acquisition_time'].isoformat() == '2022-10-23T12:13:10-04:00'


def _vk_false_color_image(heights):
    ny, nx = heights.shape
    return (struct.pack('<7I', nx, ny, 32, 0, nx * ny * 4, 0, 0) + bytes(768)
            + np.asarray(heights, dtype='<u4').tobytes())


def _synthetic_vk4(height_images):
    """
    Build a minimal VK4 file whose offset table points to the given height
    images (arrays of shape (ny, nx) or None for absent images).
    """
    header = b'VK4_' + bytes([0, 0, 0, 1]) + bytes(4)
    offset_table_size = 18 * 4
    # Measurement conditions: size followed by 75 32-bit entries
    conditions = np.zeros(76, dtype='<u4')
    conditions[0] = 304
    conditions[1:8] = [2024, 1, 2, 3, 4, 5, 60]  # date, UTC difference
    conditions[42] = 2000  # x length per pixel (pm)
    conditions[43] = 3000  # y length per pixel (pm)
    conditions[44] = 10  # z length per digit (pm)
    offset = len(header) + offset_table_size + conditions.nbytes
    offsets = []
    images = b''
    for heights in height_images:
        if heights is None:
            offsets += [0]
        else:
            offsets += [offset + len(images)]
            images += _vk_false_color_image(heights)
    # setting, color peak, color light, light 1-3, height 1-3, thumbnails (4),
    # assemble, line measure, line thickness, string data, reserved
    offset_table = struct.pack('<18I', 0, 0, 0, 0, 0, 0, *offsets, *([0] * 9))
    return header + offset_table + conditions.tobytes() + images


def test_vk4_multiple_height_images(tmp_path):
    ny, nx = 3, 4
    height1 = np.arange(nx * ny).reshape(ny, nx)
    height2 = 100 + height1
    height3 = 1000 + height1

    file_path = tmp_path / 'heights.vk4'
    file_path.write_bytes(_synthetic_vk4([height1, height2, height3]))
    r = VKReader(str(file_path))
    assert [c.name for c in r.channels] == ['Default', 'Height 2', 'Height 3']
    for c, heights in zip(r.channels, [height1, height2, height3]):
        assert c.nb_grid_pts == (nx, ny)
        np.testing.assert_allclose(c.physical_sizes, (nx * 2000, ny * 3000))
        assert c.info['acquisition_time'].isoformat() == '2024-01-02T03:04:05+01:00'
        t = r.topography(channel_index=c.index)
        assert t.unit == 'pm'
        np.testing.assert_allclose(t.heights(), 10 * heights.T)


def test_vk4_absent_height_images(tmp_path):
    # Only the first and third height images are present
    height1 = np.arange(6).reshape(2, 3)
    height3 = 7 * height1
    file_path = tmp_path / 'heights.vk4'
    file_path.write_bytes(_synthetic_vk4([height1, None, height3]))
    r = VKReader(str(file_path))
    assert [c.name for c in r.channels] == ['Default', 'Height 3']
    np.testing.assert_allclose(r.topography(channel_index=1).heights(), 10 * height3.T)


@pytest.mark.parametrize('filename', ['vk3-1.vk3', 'vk4-1.vk4', 'vk6-1.vk6'])
def test_vk_single_height_image(file_format_examples, filename):
    # The example files hold a single height image
    r = VKReader(os.path.join(file_format_examples, filename))
    assert [c.name for c in r.channels] == ['Default']


@pytest.mark.parametrize('magic', [b'VK6', b'VK7'])
def test_vk67_container(tmp_path, magic):
    # VK6 and VK7 files: magic, size of a BMP preview, the preview and a
    # ZIP archive that contains the VK4 byte stream as member `Vk4File`
    heights = np.arange(12).reshape(3, 4)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('Vk4File', _synthetic_vk4([heights, None, None]))
    bmp = b'BM' + struct.pack('<I', 64) + bytes(58)
    file_path = tmp_path / ('synthetic.' + magic.decode().lower())
    file_path.write_bytes(magic + struct.pack('<I', len(bmp)) + bmp + archive.getvalue())

    r = VKReader(str(file_path))
    assert [c.name for c in r.channels] == ['Default']
    np.testing.assert_allclose(r.topography().heights(), 10 * heights.T)
