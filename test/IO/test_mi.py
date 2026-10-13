#
# Copyright 2015-2016, 2019-2023 Lars Pastewka
#           2019-2020 Antoine Sanner
#           2020 Michael Röttger
#           2019 Kai Haase
#           2015-2016 Till Junge
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

from SurfaceTopography.IO.MI import MIReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest",
)

DATADIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "file_format_examples"
)


def test_read_header():
    file_path = os.path.join(DATADIR, "mi-1.mi")

    loader = MIReader(file_path)

    # Like in Gwyddion, there should be 4 channels in total
    assert len(loader.channels) == 4
    assert [ch.name for ch in loader.channels] == [
        "Topography",
        "Deflection",
        "Friction",
        "Friction",
    ]

    # Check if metadata has been read in correctly
    assert loader.channels[0].dim == 2
    assert loader.channels[0].nb_grid_pts == (256, 256)
    # `xLength`/`yLength` are stored in meters (2e-05 m) but the channel unit
    # is µm, so the physical sizes must be reported as 20 µm
    np.testing.assert_allclose(loader.channels[0].physical_sizes, (20.0, 20.0))
    # Non-length channels (V) keep the lateral sizes in meters
    np.testing.assert_allclose(loader.channels[1].physical_sizes, (2e-05, 2e-05))
    assert (
        loader.channels[0].info["raw_metadata"]["DisplayOffset"]
        == "8.8577270507812517e-004"
    )
    assert (
        loader.channels[0].info["raw_metadata"]["DisplayRange"]
        == "1.3109436035156252e-002"
    )
    assert loader.channels[0].info["raw_metadata"]["acqMode"] == "Main"
    assert loader.channels[0].info["raw_metadata"]["label"] == "Topography"
    assert loader.channels[0].info["raw_metadata"]["range"] == "2.9025000000000003e+000"
    assert loader.channels[0].info["raw_metadata"]["direction"] == "Trace"
    assert loader.channels[0].info["raw_metadata"]["filter"] == "3rd_order"
    assert loader.channels[0].info["raw_metadata"]["name"] == "Topography"
    assert loader.channels[0].info["raw_metadata"]["trace"] == "Trace"
    assert loader.channels[0].unit == "µm"

    assert loader.default_channel.index == 0
    assert loader.default_channel.nb_grid_pts == (256, 256)

    # Some metadata value
    assert loader.info["biasSample"] == "TRUE"


def test_topography():
    file_path = os.path.join(DATADIR, "mi-1.mi")

    loader = MIReader(file_path)

    topography = loader.topography()

    # Check one height value
    np.testing.assert_allclose(
        topography.heights()[0, 0], -0.4986900329589844, rtol=1e-6
    )

    # Check out if metadata from global and the channel are both in the
    # result from channel metadata
    assert "direction" in topography.info["raw_metadata"].keys()
    # From global metadata
    assert "zDacRange" in topography.info["raw_metadata"].keys()

    # Check the value of one of the metadata
    assert topography.unit == "µm"
    assert "unit" not in topography.info


def _write_mi(raw, data_marker, scan_up):
    """Write a minimal MI image file with a single buffer."""
    import io

    ny, nx = raw.shape
    lines = [
        ("fileType", "Image"),
        ("dateAcquired", "Tue Feb 18 15:00:51 2014"),
        ("xPixels", str(nx)),
        ("yPixels", str(ny)),
        ("xLength", "2.0e-006"),
        ("yLength", "1.0e-006"),
        ("scanUp", scan_up),
        ("bufferLabel", "Topography"),
        ("bufferRange", "1.5"),
        ("bufferUnit", "um"),
        ("data", data_marker),
    ]
    header = "".join(f"{key:<14}{value}\n" for key, value in lines)
    dtype = "<i4" if data_marker == "BINARY_32" else "<i2"
    return io.BytesIO(header.encode("ascii") + raw.astype(dtype).tobytes())


@pytest.mark.parametrize("data_marker,type_range", [
    ("BINARY", 32768), ("", 32768), ("BINARY_32", 2147483648)])
@pytest.mark.parametrize("scan_up", ["TRUE", "FALSE"])
def test_mi_synthetic(data_marker, type_range, scan_up):
    """
    As in Gwyddion's mifile.c: an empty data marker means 16-bit data, and
    the last line of the buffer is the top of the image independent of the
    scan direction.
    """
    import datetime

    raw = np.arange(6).reshape(2, 3) - 3  # 2 lines with 3 points
    reader = MIReader(_write_mi(raw, data_marker, scan_up))
    (channel,) = reader.channels
    assert channel.nb_grid_pts == (3, 2)
    assert channel.unit == "µm"
    np.testing.assert_allclose(channel.physical_sizes, (2.0, 1.0))
    assert channel.info["acquisition_time"] == datetime.datetime(2014, 2, 18, 15, 0, 51)
    t = reader.topography()
    expected = (raw * 1.5 / type_range)[::-1, :].T
    np.testing.assert_allclose(t.heights(), expected)
