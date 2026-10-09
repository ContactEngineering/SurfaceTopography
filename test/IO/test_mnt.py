#
# Copyright 2023-2026 Lars Pastewka
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
import zlib

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.Exceptions import CorruptFile, MetadataAlreadyFixedByFile
from SurfaceTopography.IO import MNTReader, open_topography
from SurfaceTopography.IO.MNT import (
    _decode_block_array,
    _decode_cstring,
    _decode_text,
)

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


@pytest.mark.parametrize("filename", ["mnt-1.mnt", "mnt-2.mnt"])
def test_read_filestream(file_format_examples, filename):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, filename)

    read_topography(file_path)

    with open(file_path, 'rb') as f:
        read_topography(f)


def test_mnt_metadata(file_format_examples):
    r = MNTReader(os.path.join(file_format_examples, 'mnt-1.mnt'))
    assert len(r.channels) == 1

    channel = r.default_channel
    assert channel.name == 'A48_10x_P1'

    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 960
    assert ny == 600

    # Grid spacing (µm) times number of grid points
    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 960 * 1.853393117831074)
    np.testing.assert_allclose(sy, 600 * 1.8545542570951588)

    assert t.unit == 'µm'

    # The file has no mask of non-measured points
    assert not t.has_undefined_data

    # Height per count is 0.0001 nm, the origin is 50.63 µm; the counts
    # range from -12629352 to 9608235
    h = t.heights()
    np.testing.assert_allclose(h.min(), 50.631545534551144 - 12629352e-7)
    np.testing.assert_allclose(h.max(), 50.631545534551144 + 9608235e-7)

    raw_metadata = t.info['raw_metadata']
    assert raw_metadata['title'] == '1 : A48_10x_P1'
    # Matches `StudiableGUID` in the XML header
    assert raw_metadata['guid'] == '{0BCC7C65-C57D-4A91-AB2F-86B874DBBFE3}'
    assert raw_metadata['axes']['z']['min_count'] == -12629352
    assert raw_metadata['axes']['z']['max_count'] == 9608235

    software = raw_metadata['software']
    assert software['name'] == 'MountainsMap® Imaging Topography'
    assert software['version'] == '9.2.0.9994'
    assert software['build_date'] == '2022/05/13'
    # Serial number of the software, not of the instrument
    assert software['serial_number'] == 'DS-364280957'
    assert 'instrument' not in t.info
    assert software['operators'] == [
        'kDSOperatorSurfaceLSLevelling',
        'kDSOperatorSurfaceProfileExtraction',
        'kDSOperatorSurfaceFormRemoval',
    ]


def test_mnt2_metadata(file_format_examples):
    r = MNTReader(os.path.join(file_format_examples, 'mnt-2.mnt'))
    assert len(r.channels) == 1
    assert r.default_channel.name == 'Si_1mm_B16_1x_polish_clean_50x'

    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1280
    assert ny == 960

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 1280 * 0.0748801903682761)
    np.testing.assert_allclose(sy, 960 * 0.0748801903682761)

    assert t.unit == 'µm'

    # Non-measured points are stored in a separate mask
    assert t.has_undefined_data
    assert np.ma.getmaskarray(t.heights()).sum() == 45405

    software = t.info['raw_metadata']['software']
    assert software['name'] == 'MountainsMap® Expert'
    assert software['version'] == '10.3.3.10995'
    assert software['serial_number'] == 'DS-781758056'


def test_mnt2_matches_opd(file_format_examples):
    """mnt-2.mnt was created from mnt-2.opd"""
    mnt = read_topography(os.path.join(file_format_examples, 'mnt-2.mnt'))
    opd = read_topography(os.path.join(file_format_examples, 'mnt-2.opd'))
    opd = opd.to_unit(mnt.unit)

    np.testing.assert_allclose(mnt.physical_sizes, opd.physical_sizes)

    mnt_heights = mnt.heights()
    opd_heights = np.ma.masked_invalid(opd.heights())
    np.testing.assert_array_equal(
        np.ma.getmaskarray(mnt_heights), np.ma.getmaskarray(opd_heights)
    )
    # The MNT file stores the heights as counts of 0.23 nm
    np.testing.assert_allclose(
        mnt_heights.compressed(), opd_heights.compressed(), atol=2e-4
    )


def test_metadata_fixed_by_file(file_format_examples):
    r = MNTReader(os.path.join(file_format_examples, 'mnt-1.mnt'))
    with pytest.raises(MetadataAlreadyFixedByFile):
        r.topography(physical_sizes=(1, 1))
    with pytest.raises(MetadataAlreadyFixedByFile):
        r.topography(height_scale_factor=1)
    with pytest.raises(MetadataAlreadyFixedByFile):
        r.topography(unit='m')


@pytest.mark.parametrize(
    "raw,text",
    [
        (b"\x04m\x00m\x00", "mm"),
        (b"\x04\xb5\x00m\x00", "µm"),
        (b"\x04m\x00", "m"),
        (b"\x04D\x00S\x00-\x001\x002\x00\x00\x00", "DS-12"),
        (b"\x04", ""),
        # Not a text entry
        (b"\x05m\x00", None),
        (b"\x04m", None),
        (b"", None),
    ],
)
def test_decode_text(raw, text):
    assert _decode_text(raw) == text


@pytest.mark.parametrize("text", ["", "abc", "x" * 254, "y" * 255, "z" * 70000])
def test_decode_cstring(text):
    encoded = text.encode("utf-16-le")
    n = len(text)
    if n < 0xFF:
        header = b"\xff\xfe\xff" + bytes([n])
    elif n < 0xFFFF:
        header = b"\xff\xfe\xff\xff" + struct.pack("<H", n)
    else:
        header = b"\xff\xfe\xff\xff\xff\xff" + struct.pack("<I", n)
    assert _decode_cstring(header + encoded) == text


def _block_array(values, nb_per_block, order):
    """Encode an int32 array in blocks, stored in the given order."""
    data = values.astype("<i4").tobytes()
    blocks = []
    for offset in range(0, len(values), nb_per_block):
        chunk = data[4 * offset:4 * (offset + nb_per_block)]
        compressed = zlib.compress(chunk)
        blocks.append(
            struct.pack("<QII", offset, len(chunk) // 4, len(compressed)) + compressed
        )
    return struct.pack("<Q", len(data)) + b"".join(blocks[i] for i in order)


def test_decode_block_array():
    values = np.arange(-50, 50, dtype=np.int32)
    buffer, element_size = _decode_block_array(
        _block_array(values, 30, [2, 0, 3, 1]), len(values)
    )
    assert element_size == 4
    np.testing.assert_array_equal(np.frombuffer(buffer, "<i4"), values)

    # Missing block
    with pytest.raises(CorruptFile):
        _decode_block_array(_block_array(values, 30, [2, 0, 3]), len(values))
    # Wrong number of elements
    with pytest.raises(CorruptFile):
        _decode_block_array(_block_array(values, 30, [0, 1, 2, 3]), 99)


def test_not_an_mnt_file(file_format_examples):
    # Other OLE files and non-OLE files are rejected during format detection
    t = open_topography(os.path.join(file_format_examples, 'di-1.di'))
    assert t.format() != 'mnt'
