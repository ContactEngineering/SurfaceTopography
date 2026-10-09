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

import datetime
import os
import struct

import numpy as np
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


def test_opd_metadata(file_format_examples):
    t = OPDReader(os.path.join(file_format_examples, 'opd-1.opd')).topography()
    assert t.info['acquisition_time'] == datetime.datetime(2015, 7, 7, 16, 19, 48)
    raw = t.info['raw_metadata']
    assert raw['Date'] == '7/7/2015'
    assert raw['Magnification'] == pytest.approx(50.322)
    assert raw['Wavelength'] == pytest.approx(577.29498)
    assert raw['SecArr_ID_0'] == 2108

    # Date is month/day/year
    t = OPDReader(os.path.join(file_format_examples, 'mnt-2.opd')).topography()
    assert t.info['acquisition_time'] == datetime.datetime(2025, 12, 15, 17, 14, 51)


def _opd_block(name, block_type, payload):
    return struct.pack('<16shlH', name.encode('latin1'), block_type, len(payload), 0), payload


def _opd_array(name, data, dtype):
    nx, ny = data.shape
    payload = struct.pack('<HHH', nx, ny, np.dtype(dtype).itemsize) + np.asarray(data, dtype=dtype).tobytes()
    return _opd_block(name, 3, payload)


def _write_opd(file_path, blocks):
    entries = [entry for entry, _ in blocks]
    payloads = [payload for _, payload in blocks]
    directory = struct.pack('<16shlH', b'Directory', 1, 24 * (len(blocks) + 1), 0xffff)
    with open(file_path, 'wb') as f:
        f.write(b'\x01\x00' + directory + b''.join(entries) + b''.join(payloads))


def test_opd_synthetic(tmp_path):
    """
    Height arrays named SAMPLE_DATA, 16-bit integer and unsigned byte data,
    the `Mult` divisor and intensity arrays (which are not read)
    """
    wavelength, mult, pixel_size = 600.0, 4, 0.002
    heights_int16 = np.array([[1, 2, 3], [-4, 32766, 6]], dtype='<i2')  # (nx, ny)
    heights_byte = np.array([[0, 200, 255], [1, 2, 3]], dtype='u1')
    file_path = tmp_path / 'synthetic.opd'
    _write_opd(file_path, [
        _opd_block('Wavelength', 7, struct.pack('<f', wavelength)),
        _opd_block('Mult', 6, struct.pack('<h', mult)),
        _opd_block('Pixel_size', 7, struct.pack('<f', pixel_size)),
        _opd_block('Title', 5, b'My title'),
        _opd_array('SAMPLE_DATA', heights_int16, '<i2'),
        _opd_array('Image', heights_byte, 'u1'),
        _opd_array('OPD', heights_byte, 'u1'),
        _opd_block('Date', 5, b'1/2/2003'),
        _opd_block('Time', 5, b'4:05:06'),
    ])

    r = OPDReader(file_path)
    assert [c.name for c in r.channels] == ['SAMPLE_DATA', 'OPD']
    scale = wavelength / mult * 1e-6
    for channel in r.channels:
        assert channel.nb_grid_pts == (2, 3)
        np.testing.assert_allclose(channel.physical_sizes, (2 * pixel_size, 3 * pixel_size), rtol=1e-6)
        assert channel.height_scale_factor == pytest.approx(scale)
        assert channel.info['raw_metadata']['Title'] == 'My title'
        assert channel.info['acquisition_time'] == datetime.datetime(2003, 1, 2, 4, 5, 6)

    t = r.topography(channel_index=0)
    heights = t.heights()
    mask = np.ma.getmaskarray(heights)
    # The y direction is flipped; 32766 marks undefined data
    assert mask.sum() == 1 and mask[1, 1]
    expected = np.flip(heights_int16.astype(float), 1) * scale
    np.testing.assert_allclose(heights[~mask], expected[~mask], rtol=1e-6)

    t = r.topography(channel_index=1)
    np.testing.assert_allclose(t.heights(), np.flip(heights_byte.astype(float), 1) * scale, rtol=1e-6)


def test_opd_without_pixel_size(tmp_path):
    file_path = tmp_path / 'synthetic.opd'
    _write_opd(file_path, [
        _opd_block('Wavelength', 7, struct.pack('<f', 600.0)),
        _opd_array('RAW_DATA', np.ones((2, 3), dtype='<f4'), '<f4'),
    ])
    r = OPDReader(file_path)
    assert r.channels[0].physical_sizes is None
    t = r.topography(physical_sizes=(1, 1))
    np.testing.assert_allclose(t.heights(), 600e-6)
