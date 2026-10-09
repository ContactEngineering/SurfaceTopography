#
# Copyright 2016-2021, 2023 Lars Pastewka
#           2018-2020 Antoine Sanner
#           2018-2020 Michael Röttger
#           2019-2020 Kai Haase
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

import datetime
import os
import struct

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO.OPDx import OPDxReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest",
)


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, "opdx-2.opdx")

    read_topography(file_path)

    with open(file_path, "r") as f:
        read_topography(f)

    f = open(file_path, "rb")
    read_topography(f)

    # Test is successful if it reaches end of function without raising an
    # exception


def test_read_header(file_format_examples):
    file_path = os.path.join(file_format_examples, "opdx-2.opdx")

    loader = OPDxReader(file_path)

    (channel_0,) = loader.channels

    # Check if metadata has been read in

    # Default channel should be 0, 'Raw'
    assert loader.default_channel.index == 0

    #
    # Channel 0: Raw
    #
    assert channel_0.unit == "µm"

    # .. mandatory keys
    assert channel_0.name == "Height"
    assert channel_0.dim == 2
    # The extents stored in the file span the distance between the first
    # and last pixel, i.e. (nb_grid_pts - 1) pixels
    np.testing.assert_allclose(channel_0.physical_sizes[1], 35.85522403809594 * 960 / 959)
    np.testing.assert_allclose(channel_0.physical_sizes[0], 47.81942809668896 * 1280 / 1279)
    assert channel_0.nb_grid_pts[1] == 960
    assert channel_0.nb_grid_pts[0] == 1280


def test_topography(file_format_examples):
    file_path = os.path.join(file_format_examples, "opdx-2.opdx")

    with OPDxReader(file_path) as loader:
        assert loader.default_channel.index == 0

        topography = loader.default_channel.topography()

        # Check physical sizes
        np.testing.assert_allclose(topography.physical_sizes[0], 47.8568, rtol=1e-5)
        np.testing.assert_allclose(topography.physical_sizes[1], 35.8926, rtol=1e-5)

        # Check nb_grid_ptss
        assert topography.nb_grid_pts[0] == 1280
        assert topography.nb_grid_pts[1] == 960

        # Check unit
        assert topography.unit == "µm"  # see GH 281
        assert topography.info["instrument"]["vendor"] == "Bruker"

        # Check an entry in the metadata
        assert (
            topography.info["acquisition_time"]
            == datetime.datetime(2018, 12, 5, 12, 53, 14)
        )

        # Check a height value
        np.testing.assert_allclose(topography.heights()[0, 0], -7.731534)


def test_opdx_txt_consistency(file_format_examples):
    t_opdx = OPDxReader(os.path.join(file_format_examples, "opdx-2.opdx")).topography()
    t_txt = read_topography(os.path.join(file_format_examples, "opdx-2.txt"))
    assert abs(t_opdx.pixel_size[0] / t_opdx.pixel_size[1] - 1) < 1e-3
    assert abs(t_txt.pixel_size[0] / t_txt.pixel_size[1] - 1) < 1e-3

    ratio_ref = t_opdx.physical_sizes[1] / t_opdx.physical_sizes[0]

    assert (
        t_txt.physical_sizes[1] / t_txt.physical_sizes[0] - ratio_ref
    ) / ratio_ref < 1e-3
    assert t_opdx.nb_grid_pts == t_txt.nb_grid_pts

    # opd file's heights are in µm, txt file's heights in m
    assert t_opdx.unit == "µm"
    assert t_txt.unit == "m"
    np.testing.assert_allclose(
        t_opdx.detrend().heights(),
        t_txt.detrend().scale(1e6).heights(),
        rtol=1e-6,
        atol=1e-3,
    )

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plt.colorbar(ax.imshow(t_txt.scale(1e9).heights()))
        fig2, ax2 = plt.subplots()
        plt.colorbar(ax2.imshow(t_opdx.heights()))
        plt.show(block=True)


def test_opdx_txt_heights_lateral_consistency(file_format_examples):
    t_txt = read_topography(os.path.join(file_format_examples, "opdx-2.txt"))

    assert t_txt.unit == "m"

    # the radius of the sphere should be 250 µm
    R = 250 * 1e-6

    rhoxx, rhoyy, rhoxy = t_txt.detrend(detrend_mode="curvature").curvatures

    assert (1 / rhoxx - R) / R < 0.01
    assert (1 / rhoyy - R) / R < 0.01


def test_opdx3(file_format_examples):
    r = OPDxReader(f"{file_format_examples}/opdx-3.opdx")
    t = r.topography()
    assert t.info["instrument"]["name"] == "Dektak Profiler"
    assert t.info["instrument"]["vendor"] == "Bruker"

    x, y = np.loadtxt(f"{file_format_examples}/opdx-3.txt", unpack=True)
    np.testing.assert_allclose(t.heights(), y * 1e6)


def test_read_anon_matrix(file_format_examples):
    """
    Some OPDx files (e.g. multi-line "stripe" scans) store the 2D height
    matrix using the anonymous matrix data type (0x45, DEKTAK_ANON_MATRIX)
    rather than the named matrix type (0x00, DEKTAK_MATRIX) that
    `opdx-1.opdx`/`opdx-2.opdx` use. Reading such a file used to raise
    "Don't know how to read type with id 69".

    `opdx-4.opdx` is `opdx-2.opdx` with its primary channel's Matrix item
    transplanted from the DEKTAK_MATRIX encoding to the DEKTAK_ANON_MATRIX
    encoding (same pixel data, repacked header, surrounding container
    lengths patched accordingly), so the two files must decode to the same
    topography.
    """
    t_named = OPDxReader(os.path.join(file_format_examples, "opdx-2.opdx")).topography()
    t_anon = OPDxReader(os.path.join(file_format_examples, "opdx-4.opdx")).topography()

    assert t_anon.nb_grid_pts == t_named.nb_grid_pts
    np.testing.assert_allclose(t_anon.physical_sizes, t_named.physical_sizes)
    assert t_anon.unit == t_named.unit
    np.testing.assert_allclose(t_anon.heights(), t_named.heights())


def test_opdx_pixel_size(file_format_examples):
    """
    The pixel size must match the `PixelSize` entry of the metadata; the
    extents span (nb_grid_pts - 1) pixels.
    """
    t = OPDxReader(os.path.join(file_format_examples, "opdx-2.opdx")).topography()
    pixel_size = t.info["raw_metadata"]["PixelSize"]
    assert pixel_size["unit"] == "µm"
    np.testing.assert_allclose(t.pixel_size, (pixel_size["value"], pixel_size["value"]))


def test_opdx3_positions(file_format_examples):
    """
    The positions of the line scan must match the explicit positions in
    the file (and in the text export): the extent spans the distance from the
    first to the last point.
    """
    r = OPDxReader(f"{file_format_examples}/opdx-3.opdx")
    t = r.topography()
    x, y = np.loadtxt(f"{file_format_examples}/opdx-3.txt", unpack=True)
    np.testing.assert_allclose(t.positions(), x * 1e6, rtol=1e-6, atol=1e-6)


def test_opdx_metadata(file_format_examples):
    t = OPDxReader(os.path.join(file_format_examples, "opdx-3.opdx")).topography()
    raw = t.info["raw_metadata"]
    assert raw["opdx_prefix"] == "/1D_Data/Raw"
    assert raw["MeasurementSettings/InstrumentName"] == "Dektak Profiler"
    assert raw["MeasurementSettings/SamplesToLog"] == 3001
    assert raw["MeasurementSettings/ScanLength"] == {"value": 50.0, "unit": "µm"}
    assert raw["Instrument/RestorablePosition"] is True
    assert raw["1D_Channels/Height"] == ["Raw"]
    assert "TimeStamp" not in raw  # Binary timestamps are not decoded


def _name(s):
    s = s.encode("utf-8")
    return struct.pack("<I", len(s)) + s


def _varlen(n):
    return b"\x04" + struct.pack("<I", n)


_TERMINATOR = _name("") + b"\x7f\xff\xff"


def _container(name, items):
    content = b"".join(items) + _TERMINATOR
    return _name(name) + b"\x7d" + _varlen(len(content)) + content


def _string(name, value):
    value = value.encode("utf-8")
    return _name(name) + b"\x12" + _varlen(len(value)) + value


def _string_list(name, values):
    content = b"".join(_name(v) for v in values)
    return _name(name) + b"\x42" + _name("StringList") + _varlen(len(content)) + content


def _quantity(name, value, symbol):
    content = struct.pack("<d", value) + _name(symbol) + _name(symbol)
    return _name(name) + b"\x13" + _varlen(len(content)) + content


def _uint32(name, value):
    return _name(name) + b"\x07" + struct.pack("<I", value)


def _matrix(name, data):
    ny, nx = data.shape
    content = struct.pack("<II", ny, nx) + np.asarray(data, dtype="<f4").tobytes()
    return (_name(name) + b"\x00" + _name("Matrix") + struct.pack("<I", 0) + _name("WykoData.dll")
            + _varlen(len(content)) + content)


def _channel_2d(name, data, extent_x, extent_y, scale):
    ny, nx = data.shape
    return _container(name, [
        _string("DataKind", "Height"),
        _uint32("Dimension1Points", ny),
        _uint32("Dimension2Points", nx),
        _quantity("Dimension1Extent", extent_y, "µm"),
        _quantity("Dimension2Extent", extent_x, "µm"),
        _quantity("DataScale", scale, "nm"),
        _matrix("Matrix", data),
    ])


def test_opdx_multiple_height_channels(tmp_path):
    """
    All 2D height channels are read (as in Gwyddion's dektakvca.c), the
    primary channel first; the data kind is detected from the presence of
    data if the `DataKind` entry is unknown.
    """
    data1 = np.arange(6, dtype=float).reshape(2, 3)  # (ny, nx)
    data2 = -np.arange(6, dtype=float).reshape(2, 3)
    file_path = tmp_path / "synthetic.opdx"
    file_path.write_bytes(
        b"VCA DATA\x01\x00\x00\x55"
        + _container("2D_Data", [
            _channel_2d("First", data2, 4.0, 1.0, 1.0),
            _channel_2d("Second", data1, 2.0, 1.0, 2.0),
        ])
        + _container("MetaData", [
            _string("DataKind", "Something Else"),
            _string("PrimaryData2D", "Second"),
            _container("2D_Channels", [_string_list("Height", ["First", "Second"])]),
        ])
        + _TERMINATOR
    )

    r = OPDxReader(file_path)
    assert len(r.channels) == 2
    primary, other = r.channels
    assert primary.info["raw_metadata"]["opdx_prefix"] == "/2D_Data/Second"
    assert other.info["raw_metadata"]["opdx_prefix"] == "/2D_Data/First"
    assert primary.nb_grid_pts == (3, 2)
    np.testing.assert_allclose(primary.physical_sizes, (2.0 * 3 / 2, 1.0 * 2 / 1))
    np.testing.assert_allclose(other.physical_sizes, (4.0 * 3 / 2, 1.0 * 2 / 1))

    t = r.topography()
    assert t.unit == "µm"
    np.testing.assert_allclose(t.heights(), data1.T * 2e-3)
    t = r.topography(channel_index=1)
    np.testing.assert_allclose(t.heights(), data2.T * 1e-3)
