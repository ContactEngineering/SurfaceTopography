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
from SurfaceTopography.IO import JPKReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest",
)


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, "jpk-1.jpk")

    read_topography(file_path)

    with open(file_path, "r") as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_jpk1_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, "jpk-1.jpk")

    r = JPKReader(file_path)

    assert r.channels[0].name == "Height (measured, retrace)"

    t = r.topography(0)

    nx, ny = t.nb_grid_pts
    assert nx == 512
    assert ny == 376

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 5.0e-07, rtol=1e-6)
    np.testing.assert_allclose(sy, 3.5e-07, rtol=1e-6)

    assert t.unit == "m"
    assert t.info["instrument"]["vendor"] == "JPK Instruments"

    np.testing.assert_allclose(t.rms_height_from_area(), 2.144639e-08, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 3.495942e-09, rtol=1e-6)

    t = t.detrend("curvature")
    np.testing.assert_allclose(t.rms_height_from_area(), 3.54634e-09, rtol=1e-4)
    np.testing.assert_allclose(t.rms_height_from_profile(), 3.394314e-09, rtol=1e-4)


def test_jpk2_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, "jpk-2.jpk")

    r = JPKReader(file_path)

    assert r.channels[0].name == "Height"

    t = r.topography(0)

    nx, ny = t.nb_grid_pts
    assert nx == 512
    assert ny == 512

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 5.0e-06, rtol=1e-6)
    np.testing.assert_allclose(sy, 5.0e-06, rtol=1e-6)

    assert t.unit == "m"
    assert t.info["instrument"]["vendor"] == "JPK Instruments"

    np.testing.assert_allclose(t.rms_height_from_area(), 6.877323e-07, rtol=1e-6)
    np.testing.assert_allclose(t.rms_height_from_profile(), 5.946019e-07, rtol=1e-6)

    t = t.detrend("curvature")
    np.testing.assert_allclose(t.rms_height_from_area(), 3.448658e-07, rtol=1e-4)
    np.testing.assert_allclose(t.rms_height_from_profile(), 3.032367e-07, rtol=1e-4)


@pytest.mark.parametrize("filename", ["jpk-1.jpk", "jpk-2.jpk"])
def test_jpk_heights_like_gwyddion(file_format_examples, filename):
    """
    Heights are `ScalingMultiply * raw + ScalingOffset` of the default slot
    and the first scan line (first row of the TIFF raster) is at the bottom
    of the image, as in Gwyddion's jpkscan.c.
    """
    import tifffile

    file_path = os.path.join(file_format_examples, filename)
    r = JPKReader(file_path)
    with tifffile.TiffFile(file_path) as tiff:
        for channel in r.channels:
            metadata = channel.info["raw_metadata"]
            slot = metadata["Slots"][metadata["DefaultSlot"]]
            assert metadata["GridReflect"] == 0
            # Find the TIFF page of this channel
            (page,) = [
                p
                for p in tiff.pages
                if 0x8052 in p.tags
                and p.tags[0x8052].value == metadata["ChannelFancyName"]
                and p.tags[0x8051].value == metadata["ChannelRetrace"]
            ]
            raw = page.asarray().astype(float)
            expected = raw * slot["ScalingMultiply"] + slot["ScalingOffset"]
            # Gwyddion rows are the second index; row 0 is the last line
            expected = expected[::-1, :].T
            np.testing.assert_allclose(
                channel.topography().heights(),
                expected,
                rtol=1e-12,
                atol=1e-12 * np.abs(expected).max(),
            )


def _write_jpk(raw, reflect):
    """Write a minimal JPK image scan with a single height channel."""
    import io

    import tifffile

    f = io.BytesIO()
    global_tags = [
        (0x8003, "s", 0, "2024-01-02 03:04:05.000 UTC", True),  # StartDate
        (0x8042, "d", 1, 2e-6, True),  # GridULength
        (0x8043, "d", 1, 1e-6, True),  # GridVLength
        (0x8045, "I", 1, 0, True),  # GridReflect
    ]
    channel_tags = [
        (0x8045, "I", 1, int(reflect), True),  # GridReflect
        (0x8051, "I", 1, 0, True),  # ChannelRetrace
        (0x8052, "s", 0, "Height", True),  # ChannelFancyName
        (0x8080, "I", 1, 1, True),  # NrOfSlots
        (0x8081, "s", 0, "nominal", True),  # DefaultSlot
        (0x8090, "s", 0, "nominal", True),  # SlotName
        (0x80A2, "s", 0, "m", True),  # EncoderUnit
        (0x80A3, "s", 0, "LinearScaling", True),  # ScalingType
        (0x80A4, "d", 1, 1e-9, True),  # ScalingMultiply
        (0x80A5, "d", 1, 3e-6, True),  # ScalingOffset
    ]
    with tifffile.TiffWriter(f) as tiff:
        tiff.write(np.zeros((2, 2), dtype=np.uint8), extratags=global_tags)
        tiff.write(raw, extratags=channel_tags)
    f.seek(0)
    return f


@pytest.mark.parametrize("reflect", [False, True])
def test_jpk_grid_reflect(reflect):
    raw = np.arange(12, dtype=np.int32).reshape(3, 4)  # 3 lines, 4 points
    r = JPKReader(_write_jpk(raw, reflect))
    (channel,) = r.channels
    assert channel.nb_grid_pts == (4, 3)
    t = channel.topography()
    expected = (raw * 1e-9 + 3e-6).T
    if not reflect:
        expected = expected[:, ::-1]
    np.testing.assert_allclose(t.heights(), expected, rtol=1e-12)
