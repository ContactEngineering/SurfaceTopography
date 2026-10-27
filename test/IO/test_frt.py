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

from SurfaceTopography import read_topography
from SurfaceTopography.IO import FRTReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'frt-1.frt')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_frt1_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'frt-1.frt')

    r = FRTReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 500
    assert ny == 500

    # import matplotlib.pyplot as plt
    # t.plot()
    # plt.show()

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.012, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.012, rtol=1e-6)

    assert t.unit == 'm'
    assert t.info['instrument']['vendor'] == 'FRT'

    np.testing.assert_allclose(t.rms_height_from_area(), 4.915785774480653e-06, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 1.951522762933369e-06, rtol=1e-6)

    t = t.detrend('curvature')
    np.testing.assert_allclose(t.rms_height_from_area(), 4.157523910393923e-06, rtol=1e-4)
    np.testing.assert_allclose(t.rms_height_from_profile(), 1.0968828309450876e-06, rtol=1e-4)

    assert t.has_undefined_data


def test_frt2_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'frt-2.frt')

    r = FRTReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 150
    assert ny == 300

    # import matplotlib.pyplot as plt
    # t.plot()
    # plt.show()

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 0.03, rtol=1e-6)
    np.testing.assert_allclose(sy, 0.06, rtol=1e-6)

    assert t.unit == 'm'
    assert t.info['instrument']['vendor'] == 'FRT'

    np.testing.assert_allclose(t.rms_height_from_area(), 8.826763181113768e-06, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 7.663174319074244e-06, rtol=1e-6)

    t = t.detrend('curvature')
    np.testing.assert_allclose(t.rms_height_from_area(), 8.688677385181484e-06, rtol=1e-4)
    np.testing.assert_allclose(t.rms_height_from_profile(), 7.627057775705951e-06, rtol=1e-4)

    assert t.has_undefined_data


@pytest.mark.parametrize('file_name, nx, ny', [('frt-1.frt', 500, 500), ('frt-2.frt', 150, 300)])
def test_frt_orientation(file_format_examples, file_name, nx, ny):
    """
    The first stored row is the bottom line of the image. Like Gwyddion's
    `microprof.c` (and the other readers), the reader returns it as the
    last y-index.
    """
    import struct
    file_path = os.path.join(file_format_examples, file_name)
    with open(file_path, 'rb') as f:
        buffer = f.read()
    # Locate the topography block (tag 0x000b, 32-bit size, version 1.00)
    size = nx * ny * 2
    pos = buffer.find(struct.pack('<HI', 0x000b, size)) + 6
    raw = np.frombuffer(buffer[pos:pos + size], dtype='<u2').reshape(ny, nx)

    t = FRTReader(file_path).topography()
    scale = t.info['raw_metadata']['0x6c']['height_scale_factor']
    h = t.heights()
    np.testing.assert_array_equal(h.mask, (raw == 1)[::-1].T)
    np.testing.assert_allclose(h[:, -1].filled(0), np.where(raw[0] == 1, 0, raw[0] * scale))
    np.testing.assert_allclose(h[:, 0].filled(0), np.where(raw[-1] == 1, 0, raw[-1] * scale))


@pytest.mark.parametrize('file_name, timestamp', [('frt-1.frt', 1678463840), ('frt-2.frt', 1687796545)])
def test_frt_acquisition_time(file_format_examples, file_name, timestamp):
    """The start of the measurement (block 0x0072) is the acquisition time"""
    import datetime
    r = FRTReader(os.path.join(file_format_examples, file_name))
    # The topography in the multi-image block duplicates block 0x000B
    assert len(r.channels) == 1
    t = r.topography()
    assert t.info['acquisition_time'] == datetime.datetime.fromtimestamp(timestamp)


def _frt_block(tag, payload):
    import struct
    return struct.pack('<HI', tag, len(payload)) + payload


def test_frt_multi_image_block_only():
    """
    Files without block 0x000B: height-like images of the multi-image block
    0x007D are reported as channels, other images (intensity) are not.
    """
    import io
    import struct
    nx, ny = 3, 2
    topo = np.array([[10, 11, 1], [13, 14, 15]], dtype='<u2')  # 1 is undefined
    intensity = np.full((ny, nx), 100, dtype='<u2')
    thickness = np.array([[-5, 6, 7], [8, 9, 10]], dtype='<i4')
    images = b''
    for data_type, data in [(0x0004, topo), (0x0002, intensity), (0x10000080 | 0x01000000, thickness)]:
        images += struct.pack('<IIII', data_type, nx, ny, data.dtype.itemsize * 8) + data.tobytes()
    blocks = [
        _frt_block(0x0066, struct.pack('<III', nx, ny, 16)),
        _frt_block(0x0067, struct.pack('<dddddI', 3e-3, 2e-3, 0, 0, 1, 1)),
        _frt_block(0x006c, struct.pack('<Id', 5, 1e-9)),
        _frt_block(0x007d, struct.pack('<HHHH', 0, 0, 0, 0) + images),
    ]
    buffer = b'FRTM_GLIDERV1.00' + struct.pack('<H', len(blocks)) + b''.join(blocks)

    r = FRTReader(io.BytesIO(buffer))
    assert [c.name for c in r.channels] == ['Topography (top sensor) 1', 'Thickness (bottom sensor) 2']

    t = r.topography(channel_index=0)
    assert t.nb_grid_pts == (nx, ny)
    np.testing.assert_allclose(t.physical_sizes, (3e-3, 2e-3))
    h = t.heights()
    np.testing.assert_array_equal(h.mask, (topo == 1)[::-1].T)
    np.testing.assert_allclose(h[~h.mask], (1e-9 * topo[::-1].T)[~h.mask])

    t = r.topography(channel_index=1)
    assert not t.has_undefined_data
    np.testing.assert_allclose(t.heights(), 1e-9 * thickness[::-1].T)
