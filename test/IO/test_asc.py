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

import gzip
import os
from test.test_topography import DATADIR

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import Topography, UniformLineScan
from SurfaceTopography.Exceptions import CorruptFile
from SurfaceTopography.IO import AscReader, open_topography

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


def test_example1():
    surf = AscReader(os.path.join(DATADIR, "matrix-1.txt")).topography()
    assert isinstance(surf, Topography)
    assert surf.nb_grid_pts == (1024, 1024)
    np.testing.assert_allclose(surf.physical_sizes[0], 2000)
    np.testing.assert_allclose(surf.physical_sizes[1], 2000)
    np.testing.assert_allclose(surf.rms_height_from_area(), 17.22950485567042)
    np.testing.assert_allclose(surf.rms_gradient(), 0.4560243831362324)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert surf.unit == "nm"


def test_example2():
    surf = AscReader(os.path.join(DATADIR, "matrix-2.txt")).topography()
    assert surf.nb_grid_pts == (650, 650)
    np.testing.assert_allclose(surf.physical_sizes[0], 0.0002404103)
    np.testing.assert_allclose(surf.physical_sizes[1], 0.0002404103)
    np.testing.assert_allclose(surf.rms_height_from_area(), 2.7722350402740072e-07)
    np.testing.assert_allclose(surf.rms_gradient(), 0.35152685030417763)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert surf.unit == "m"


def test_example3():
    surf = AscReader(os.path.join(DATADIR, "matrix-3.txt")).topography()
    assert surf.nb_grid_pts == (256, 256)
    np.testing.assert_allclose(surf.physical_sizes[0], 10e-6)
    np.testing.assert_allclose(surf.physical_sizes[1], 10e-6)
    np.testing.assert_allclose(surf.rms_height_from_area(), 3.5222918750198742e-08)
    np.testing.assert_allclose(surf.rms_gradient(), 0.19235602282848963)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert surf.unit == "m"


@pytest.mark.parametrize("fn", ["matrix-4.txt", "matrix-4.txt.gz"])
def test_example4(fn):
    if fn.endswith(".gz"):
        surf = AscReader(gzip.open(os.path.join(DATADIR, fn))).topography()
    else:
        surf = AscReader(os.path.join(DATADIR, fn)).topography()
    assert surf.nb_grid_pts == (75, 305)
    np.testing.assert_allclose(surf.physical_sizes[0], 2.773965e-05)
    np.testing.assert_allclose(surf.physical_sizes[1], 0.00011280791)
    np.testing.assert_allclose(surf.rms_height_from_area(), 1.1745891510991089e-07)
    np.testing.assert_allclose(surf.rms_height_from_profile(), 7.198047e-08)
    np.testing.assert_allclose(surf.rms_gradient(), 0.06776316911544318)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert surf.unit == "m"

    # test setting the physical_sizes
    with pytest.raises(AttributeError):
        surf.physical_sizes = 1, 2


def test_example5():
    r = AscReader(os.path.join(DATADIR, "matrix-5.txt"))
    assert r.default_channel.physical_sizes is None

    surf = r.topography(physical_sizes=(1, 2))
    assert isinstance(surf, Topography)
    assert surf.nb_grid_pts == (10, 10)
    assert surf.physical_sizes == (1, 2)
    np.testing.assert_allclose(surf.rms_height_from_area(), 1.0)
    assert surf.is_uniform
    assert not surf.is_reentrant
    assert "unit" not in surf.info

    # test setting the physical_sizes
    surf = AscReader(os.path.join(DATADIR, "matrix-5.txt")).topography(
        physical_sizes=(1, 2)
    )
    np.testing.assert_allclose(surf.physical_sizes[0], 1)
    np.testing.assert_allclose(surf.physical_sizes[1], 2)

    bw = surf.bandwidth()
    np.testing.assert_allclose(bw[0], 1.5 / 10)
    np.testing.assert_allclose(bw[1], 1.5)

    reader = AscReader(os.path.join(DATADIR, "matrix-5.txt"))
    assert reader.default_channel.physical_sizes is None


def test_example6():
    topography_file = open_topography(os.path.join(DATADIR, "not-yet-working-1.txt"))
    surf = topography_file.topography(physical_sizes=(1,))
    assert isinstance(surf, UniformLineScan)
    np.testing.assert_allclose(surf.heights(), [1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_wyko_matrix8(file_format_examples, filename="matrix-8.txt"):
    file_path = os.path.join(file_format_examples, filename)

    r = AscReader(file_path)
    assert len(r.channels) == 1

    t = r.topography()
    assert t.unit == "nm"
    np.testing.assert_allclose(t.physical_sizes, (950400, 1267200))
    # Note: about half of this file's data points are undefined; sums and
    # means are normalized by the number of *defined* points
    np.testing.assert_allclose(t.rms_height_from_area(), 7804.797789741755)


def test_single_column(file_format_examples, filename="single_column.txt"):
    r = open_topography(os.path.join(file_format_examples, filename))
    assert r.format() == "asc"
    assert r.default_channel.dim == 1
    assert r.default_channel.nb_grid_pts == (9,)
    assert r.default_channel.physical_sizes is None
    t = r.topography(physical_sizes=(2.3,))
    assert t.nb_grid_pts == (9,)
    assert t.physical_sizes == (2.3,)
    np.testing.assert_allclose(t.heights(), np.arange(9) + 1)


# The following tests use small synthetic files that follow the header
# conventions of the respective ASCII exports (as documented by Gwyddion's
# import modules `spip-asc.c`, `attocube.c`, `nova-asc.c`, `asciiexport.c`
# and `opdfile.c`).

_matrix_4x3 = "1 2 3 4\n5 6 7 8\n9 10 11 12\n"
# Heights in SurfaceTopography's (nx, ny) order: lines are y, columns are x
_heights_4x3 = np.arange(1, 13).reshape(3, 4).T


def _write(tmp_path, s, name="file.txt"):
    fn = tmp_path / name
    fn.write_text(s, encoding="utf-8")
    return str(fn)


def test_spip_nonsquare(tmp_path):
    # `x-pixels` is the number of values per line; this used to be
    # interpreted as number of lines, which broke nonsquare files
    fn = _write(
        tmp_path,
        "# File Format = ASCII\n"
        "# Created by SPIP 6.3.2.0 2014-12-12 11:16\n"
        "# x-pixels = 4\n"
        "# y-pixels = 3\n"
        "# x-length = 2000\n"
        "# y-length = 1500\n"
        "# x-offset = 0\n"
        "# y-offset = 0\n"
        "# z-unit = nm\n"
        "# voidpixels =0\n"
        "# Start of Data:\n" + _matrix_4x3,
    )
    r = AscReader(fn)
    assert r.default_channel.nb_grid_pts == (4, 3)
    t = r.topography()
    assert t.nb_grid_pts == (4, 3)
    np.testing.assert_allclose(t.physical_sizes, (2000, 1500))
    assert t.unit == "nm"
    np.testing.assert_allclose(t.heights(), _heights_4x3)


def test_spip_wrong_number_of_pixels(tmp_path):
    fn = _write(
        tmp_path,
        "# File Format = ASCII\n"
        "# x-pixels = 3\n"
        "# y-pixels = 4\n"
        "# x-length = 2000\n"
        "# y-length = 1500\n"
        "# z-unit = nm\n" + _matrix_4x3,
    )
    with pytest.raises(Exception):
        AscReader(fn)


@pytest.mark.parametrize("bit2nm", [1.0, 0.5])
def test_spip_bit2nm(tmp_path, bit2nm):
    # SPIP files without `z-unit` give heights in units of `Bit2nm`
    # nanometers; lateral sizes are always in nanometers. (This is also what
    # Gwyddion's SPIP export writes for height data.)
    fn = _write(
        tmp_path,
        "# File Format = ASCII\n"
        "# Created by Gwyddion 2.70\n"
        "# Original file: NONE\n"
        "# x-pixels = 4\n"
        "# y-pixels = 3\n"
        "# x-length = 400\n"
        "# y-length = 300\n"
        "# x-offset = 0\n"
        "# y-offset = 0\n"
        f"# Bit2nm = {bit2nm}\n"
        "# Start of Data:\n" + _matrix_4x3,
    )
    r = AscReader(fn)
    assert r.default_channel.unit == "nm"
    assert r.default_channel.height_scale_factor == bit2nm
    t = r.topography()
    assert t.unit == "nm"
    np.testing.assert_allclose(t.physical_sizes, (400, 300))
    np.testing.assert_allclose(t.heights(), bit2nm * _heights_4x3)


def test_spip_lateral_unit_is_nm(tmp_path):
    # Lateral sizes are in nm even if heights are given in a different unit
    fn = _write(
        tmp_path,
        "# File Format = ASCII\n"
        "# x-pixels = 4\n"
        "# y-pixels = 3\n"
        "# x-length = 4000\n"
        "# y-length = 3000\n"
        "# z-unit = um\n" + _matrix_4x3,
    )
    t = AscReader(fn).topography()
    assert t.unit == "um"
    np.testing.assert_allclose(t.physical_sizes, (4, 3))


def test_attocube_header(tmp_path):
    # Attocube (Daisy) ASCII files use colons and give explicit units; the
    # `x-unit` key used to crash the reader
    fn = _write(
        tmp_path,
        "# Daisy frame view snapshot\n"
        "# 2009-07-14T10:12:34\n"
        "# x-pixels: 4\n"
        "# y-pixels: 3\n"
        "# x-length: 8\n"
        "# y-length: 6\n"
        "# x-offset: 0\n"
        "# y-offset: 0\n"
        "# x-unit: um\n"
        "# y-unit: um\n"
        "# z-unit: nm\n"
        "# scanspeed: 2\n"
        "# display: Topography\n"
        "# Start of Data:\n" + _matrix_4x3,
    )
    r = AscReader(fn)
    assert r.default_channel.nb_grid_pts == (4, 3)
    t = r.topography()
    assert t.unit == "nm"
    np.testing.assert_allclose(t.physical_sizes, (8000, 6000))
    np.testing.assert_allclose(t.heights(), _heights_4x3)


def test_nova_header(tmp_path):
    # Nova ASCII export: pixel sizes `Scale X/Y` in units of `Unit X`
    fn = _write(
        tmp_path,
        "File Format = ASCII\n"
        "Created by Nova\n"
        "NX = 4\n"
        "NY = 3\n"
        "Scale X = 0.5\n"
        "Scale Y = 0.25\n"
        "Unit X = um\n"
        "Unit Data = nm\n"
        "Scale Data = 1\n"
        "DataScaleNeeded = no\n"
        "Start of Data :\n" + _matrix_4x3,
    )
    r = AscReader(fn)
    assert r.default_channel.nb_grid_pts == (4, 3)
    t = r.topography()
    assert t.unit == "nm"
    np.testing.assert_allclose(t.physical_sizes, (2000, 750))
    np.testing.assert_allclose(t.heights(), _heights_4x3)


@pytest.mark.parametrize(
    "channel,width,height,value_units",
    [
        ("Channel:", "Width:", "Height:", "Value units:"),
        ("Kanál:", "Šířka:", "Výška:", "Jednotky hodnot:"),
        ("Kanal:", "Breite:", "Höhe:", "Einheiten:"),
        ("Canal :", "Largeur :", "Hauteur :", "Unités :"),
        ("Canale:", "Larghezza:", "Altezza:", "unità valore:"),
        ("Canal:", "Anchura:", "Altura:", "Unidades de valor:"),
        ("チャネル：", "幅:", "高さ：", "値の単位:"),
        ("Канал:", "Ширина:", "Высота:", "Единицы измерения:"),
    ],
)
def test_gwyddion_export_localized(tmp_path, channel, width, height, value_units):
    # Gwyddion translates the header of its ASCII matrix export
    fn = _write(
        tmp_path,
        f"# {channel} Topo\n"
        f"# {width} 4.000 µm\n"
        f"# {height} 3.000 µm\n"
        f"# {value_units} nm\n" + _matrix_4x3,
    )
    r = AscReader(fn)
    assert len(r.channels) == 1
    assert r.default_channel.name == "Topo"
    t = r.topography()
    assert t.unit == "nm"
    np.testing.assert_allclose(t.physical_sizes, (4000, 3000))
    np.testing.assert_allclose(t.heights(), _heights_4x3)


def test_gwyddion_export_concatenated(tmp_path):
    # Gwyddion can concatenate the exports of several images into a single
    # file; each block has its own header. Non-height channels are skipped.
    fn = _write(
        tmp_path,
        "# Channel: Height\n"
        "# Width: 4.000 µm\n"
        "# Height: 3.000 µm\n"
        "# Value units: m\n" + _matrix_4x3 + "\n"
        "# Channel: Phase\n"
        "# Width: 4.000 µm\n"
        "# Height: 3.000 µm\n"
        "# Value units: deg\n" + _matrix_4x3 + "\n"
        "# Channel: Zoom\n"
        "# Width: 10.00 nm\n"
        "# Height: 20.00 nm\n"
        "# Value units: nm\n"
        "1 2 3\n4 5 6\n7 8 9\n10 11 12\n",
    )
    r = AscReader(fn)
    assert [c.name for c in r.channels] == ["Height", "Zoom"]

    c1, c2 = r.channels
    assert c1.nb_grid_pts == (4, 3)
    assert c1.unit == "m"
    np.testing.assert_allclose(c1.physical_sizes, (4e-6, 3e-6))
    assert c2.nb_grid_pts == (3, 4)
    assert c2.unit == "nm"
    np.testing.assert_allclose(c2.physical_sizes, (10, 20))

    t1 = r.topography(channel_index=0)
    np.testing.assert_allclose(t1.physical_sizes, (4e-6, 3e-6))
    assert t1.unit == "m"
    np.testing.assert_allclose(t1.heights(), _heights_4x3)
    t2 = r.topography(channel_index=1)
    np.testing.assert_allclose(t2.physical_sizes, (10, 20))
    assert t2.unit == "nm"
    np.testing.assert_allclose(t2.heights(), np.arange(1, 13).reshape(4, 3).T)


def test_gwyddion_export_non_height(tmp_path):
    fn = _write(
        tmp_path,
        "# Channel: Phase\n"
        "# Width: 4.000 µm\n"
        "# Height: 3.000 µm\n"
        "# Value units: V\n" + _matrix_4x3,
    )
    with pytest.raises(CorruptFile):
        AscReader(fn)


def _wyko_file(real_units):
    lines = [
        f"Wyko ASCII Data File Format 0\t{real_units}\t1",
        "X Size\t4",
        "Y Size\t3",
        "Block Name\tType\tLength\tValue",
        "Pixel_size\t7\t4\t0.002",
        "Wavelength\t7\t4\t632.8",
        "Mult\t7\t4\t2",
        "RAW_DATA\t3\t48\t",
        "1\t2\t3",
        "4\tBad\t6",
        "7\t8\t9",
        "10\t11\t12",
    ]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("real_units", [0, 1])
def test_wyko_real_units_flag(tmp_path, real_units):
    # The second flag after the Wyko magic indicates whether data is stored
    # in nanometers (1) or in units of the wavelength (0)
    fn = _write(tmp_path, _wyko_file(real_units))
    r = AscReader(fn)
    t = r.topography()
    assert t.unit == "nm"
    assert t.nb_grid_pts == (3, 4)
    np.testing.assert_allclose(t.physical_sizes, (6000, 8000))
    assert t.has_undefined_data
    expected = np.arange(1, 13).reshape(4, 3).T * (1 if real_units else 632.8 / 2)
    np.testing.assert_allclose(t.heights()[0], expected[0])
