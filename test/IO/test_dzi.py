#
# Copyright 2020-2021, 2023 Lars Pastewka
#           2019-2020 Antoine Sanner
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

import json
import os
import tempfile
import xml.etree.cElementTree as ET

import numpy as np
import pytest
from NuMPI.Testing.Fixtures import serial_only
from scipy.io import netcdf_file

from SurfaceTopography import Topography, read_topography
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.IO.DZI import generate_manifest, write_dzi

pytestmark = serial_only

DZI_NAMESPACE = "{http://schemas.microsoft.com/deepzoom/2008}"


def _tiles(nb_grid_pts, tile_size, overlap):
    """
    Expected tiles of a DZI pyramid, formulated on the grid of each level.
    Yields level, column, row, the stride between full-resolution points
    sampled at that level and the index ranges of the tile on the level grid.
    """
    width, height = nb_grid_pts
    max_level = int(np.ceil(np.log2(max(width, height))))
    for k, level in enumerate(range(max_level, -1, -1)):
        stride = 2**k
        level_width = -(-width // stride)
        level_height = -(-height // stride)
        for column in range(-(-level_width // tile_size)):
            for row in range(-(-level_height // tile_size)):
                i0 = max(column * tile_size - overlap, 0)
                i1 = min((column + 1) * tile_size + overlap, level_width)
                j0 = max(row * tile_size - overlap, 0)
                j1 = min((row + 1) * tile_size + overlap, level_height)
                yield level, column, row, stride, (i0, i1), (j0, j1)


def _read_tile(fn):
    """Raw (undecoded) values of a tile"""
    if fn.endswith(".npy"):
        return np.load(fn)
    with netcdf_file(fn, "r", mmap=False) as nc:
        return nc.variables["heights"][...].copy()


def test_write_xml():
    nx, ny = 1782, 1302
    t = fourier_synthesis((nx, ny), (1, 1), 0.8, rms_slope=0.1, unit="mm")
    sx, sy = t.physical_sizes
    with tempfile.TemporaryDirectory() as d:
        manifest = t.to_dzi("synthetic", d)
        assert os.path.exists(f"{d}/synthetic_files")
        for i in range(12):
            assert os.path.exists(f"{d}/synthetic_files/{i}")
        root = ET.parse(open(f"{d}/synthetic.xml")).getroot()
        assert root.attrib["TileSize"] == "256"
        assert root.attrib["Overlap"] == "1"
        assert root.attrib["Format"] == "jpg"
        assert (
            root.attrib["ColorbarTitle"] == "Height (µm)"
        )  # The writer decided to use µm, not mm
        assert root.attrib["Colormap"] == "viridis"
        assert root[0].attrib["Width"] == f"{nx}"
        assert root[0].attrib["Height"] == f"{ny}"
        np.testing.assert_allclose(float(root[1].attrib["Width"]), 1000 * nx / sx)
        np.testing.assert_allclose(float(root[1].attrib["Height"]), 1000 * ny / sy)
        np.testing.assert_allclose(float(root[2].attrib["Minimum"]), 1000 * t.min())
        np.testing.assert_allclose(float(root[2].attrib["Maximum"]), 1000 * t.max())

        assert set(manifest) == set(
            [
                f"{d}/synthetic.xml",
                f"{d}/synthetic_files/11/0_0.jpg",
                f"{d}/synthetic_files/11/0_1.jpg",
                f"{d}/synthetic_files/11/0_2.jpg",
                f"{d}/synthetic_files/11/0_3.jpg",
                f"{d}/synthetic_files/11/0_4.jpg",
                f"{d}/synthetic_files/11/0_5.jpg",
                f"{d}/synthetic_files/11/1_0.jpg",
                f"{d}/synthetic_files/11/1_1.jpg",
                f"{d}/synthetic_files/11/1_2.jpg",
                f"{d}/synthetic_files/11/1_3.jpg",
                f"{d}/synthetic_files/11/1_4.jpg",
                f"{d}/synthetic_files/11/1_5.jpg",
                f"{d}/synthetic_files/11/2_0.jpg",
                f"{d}/synthetic_files/11/2_1.jpg",
                f"{d}/synthetic_files/11/2_2.jpg",
                f"{d}/synthetic_files/11/2_3.jpg",
                f"{d}/synthetic_files/11/2_4.jpg",
                f"{d}/synthetic_files/11/2_5.jpg",
                f"{d}/synthetic_files/11/3_0.jpg",
                f"{d}/synthetic_files/11/3_1.jpg",
                f"{d}/synthetic_files/11/3_2.jpg",
                f"{d}/synthetic_files/11/3_3.jpg",
                f"{d}/synthetic_files/11/3_4.jpg",
                f"{d}/synthetic_files/11/3_5.jpg",
                f"{d}/synthetic_files/11/4_0.jpg",
                f"{d}/synthetic_files/11/4_1.jpg",
                f"{d}/synthetic_files/11/4_2.jpg",
                f"{d}/synthetic_files/11/4_3.jpg",
                f"{d}/synthetic_files/11/4_4.jpg",
                f"{d}/synthetic_files/11/4_5.jpg",
                f"{d}/synthetic_files/11/5_0.jpg",
                f"{d}/synthetic_files/11/5_1.jpg",
                f"{d}/synthetic_files/11/5_2.jpg",
                f"{d}/synthetic_files/11/5_3.jpg",
                f"{d}/synthetic_files/11/5_4.jpg",
                f"{d}/synthetic_files/11/5_5.jpg",
                f"{d}/synthetic_files/11/6_0.jpg",
                f"{d}/synthetic_files/11/6_1.jpg",
                f"{d}/synthetic_files/11/6_2.jpg",
                f"{d}/synthetic_files/11/6_3.jpg",
                f"{d}/synthetic_files/11/6_4.jpg",
                f"{d}/synthetic_files/11/6_5.jpg",
                f"{d}/synthetic_files/10/0_0.jpg",
                f"{d}/synthetic_files/10/0_1.jpg",
                f"{d}/synthetic_files/10/0_2.jpg",
                f"{d}/synthetic_files/10/1_0.jpg",
                f"{d}/synthetic_files/10/1_1.jpg",
                f"{d}/synthetic_files/10/1_2.jpg",
                f"{d}/synthetic_files/10/2_0.jpg",
                f"{d}/synthetic_files/10/2_1.jpg",
                f"{d}/synthetic_files/10/2_2.jpg",
                f"{d}/synthetic_files/10/3_0.jpg",
                f"{d}/synthetic_files/10/3_1.jpg",
                f"{d}/synthetic_files/10/3_2.jpg",
                f"{d}/synthetic_files/9/0_0.jpg",
                f"{d}/synthetic_files/9/0_1.jpg",
                f"{d}/synthetic_files/9/1_0.jpg",
                f"{d}/synthetic_files/9/1_1.jpg",
                f"{d}/synthetic_files/8/0_0.jpg",
                f"{d}/synthetic_files/7/0_0.jpg",
                f"{d}/synthetic_files/6/0_0.jpg",
                f"{d}/synthetic_files/5/0_0.jpg",
                f"{d}/synthetic_files/4/0_0.jpg",
                f"{d}/synthetic_files/3/0_0.jpg",
                f"{d}/synthetic_files/2/0_0.jpg",
                f"{d}/synthetic_files/1/0_0.jpg",
                f"{d}/synthetic_files/0/0_0.jpg",
            ]
        )

        assert manifest == generate_manifest(
            "synthetic",
            t.nb_grid_pts,
            format="jpg",
            meta_format="xml",
            root_directory=d,
        )


def test_write_json():
    nx, ny = 1324, 871
    t = fourier_synthesis((nx, ny), (1.3, 1.2), 0.8, rms_slope=0.1, unit="mm")
    sx, sy = t.physical_sizes
    with tempfile.TemporaryDirectory() as d:
        manifest = t.to_dzi("synthetic", d, meta_format="json")
        assert os.path.exists(f"{d}/synthetic_files")
        for i in range(12):
            assert os.path.exists(f"{d}/synthetic_files/{i}")
        with open(f"{d}/synthetic.json", "r") as f:
            meta = json.load(f)
        meta = meta["Image"]
        assert meta["TileSize"] == 256
        assert meta["Overlap"] == 1
        assert meta["Format"] == "jpg"
        assert (
            meta["ColorbarTitle"] == "Height (µm)"
        )  # The writer decided to use µm, not mm
        assert meta["Colormap"] == "viridis"
        assert meta["Size"]["Width"] == nx
        assert meta["Size"]["Height"] == ny
        np.testing.assert_allclose(meta["PixelsPerMeter"]["Width"], 1000 * nx / sx)
        np.testing.assert_allclose(meta["PixelsPerMeter"]["Height"], 1000 * ny / sy)
        np.testing.assert_allclose(meta["ColorbarRange"]["Minimum"], t.min() * 1000)
        np.testing.assert_allclose(meta["ColorbarRange"]["Maximum"], t.max() * 1000)

        assert set(manifest) == set(
            [
                f"{d}/synthetic.json",
                f"{d}/synthetic_files/0/0_0.jpg",
                f"{d}/synthetic_files/1/0_0.jpg",
                f"{d}/synthetic_files/10/0_0.jpg",
                f"{d}/synthetic_files/10/0_1.jpg",
                f"{d}/synthetic_files/10/1_0.jpg",
                f"{d}/synthetic_files/10/1_1.jpg",
                f"{d}/synthetic_files/10/2_0.jpg",
                f"{d}/synthetic_files/10/2_1.jpg",
                f"{d}/synthetic_files/11/0_0.jpg",
                f"{d}/synthetic_files/11/0_1.jpg",
                f"{d}/synthetic_files/11/0_2.jpg",
                f"{d}/synthetic_files/11/0_3.jpg",
                f"{d}/synthetic_files/11/1_0.jpg",
                f"{d}/synthetic_files/11/1_1.jpg",
                f"{d}/synthetic_files/11/1_2.jpg",
                f"{d}/synthetic_files/11/1_3.jpg",
                f"{d}/synthetic_files/11/2_0.jpg",
                f"{d}/synthetic_files/11/2_1.jpg",
                f"{d}/synthetic_files/11/2_2.jpg",
                f"{d}/synthetic_files/11/2_3.jpg",
                f"{d}/synthetic_files/11/3_0.jpg",
                f"{d}/synthetic_files/11/3_1.jpg",
                f"{d}/synthetic_files/11/3_2.jpg",
                f"{d}/synthetic_files/11/3_3.jpg",
                f"{d}/synthetic_files/11/4_0.jpg",
                f"{d}/synthetic_files/11/4_1.jpg",
                f"{d}/synthetic_files/11/4_2.jpg",
                f"{d}/synthetic_files/11/4_3.jpg",
                f"{d}/synthetic_files/11/5_0.jpg",
                f"{d}/synthetic_files/11/5_1.jpg",
                f"{d}/synthetic_files/11/5_2.jpg",
                f"{d}/synthetic_files/11/5_3.jpg",
                f"{d}/synthetic_files/2/0_0.jpg",
                f"{d}/synthetic_files/3/0_0.jpg",
                f"{d}/synthetic_files/4/0_0.jpg",
                f"{d}/synthetic_files/5/0_0.jpg",
                f"{d}/synthetic_files/6/0_0.jpg",
                f"{d}/synthetic_files/8/0_0.jpg",
                f"{d}/synthetic_files/9/0_0.jpg",
                f"{d}/synthetic_files/9/1_0.jpg",
                f"{d}/synthetic_files/7/0_0.jpg",
            ]
        )

        assert manifest == generate_manifest(
            "synthetic",
            t.nb_grid_pts,
            format="jpg",
            meta_format="json",
            root_directory=d,
        )


def test_write_netcdf():
    nx, ny = 1324, 871
    t = fourier_synthesis((nx, ny), (1.3, 1.2), 0.8, rms_slope=0.1, unit="mm")
    sx, sy = t.physical_sizes
    with tempfile.TemporaryDirectory() as d:
        manifest = t.to_dzi("synthetic", d, format="nc", meta_format="json")
        assert os.path.exists(f"{d}/synthetic_files")
        for i in range(12):
            assert os.path.exists(f"{d}/synthetic_files/{i}")
        with open(f"{d}/synthetic.json", "r") as f:
            meta = json.load(f)
        meta = meta["Image"]
        assert meta["TileSize"] == 256
        assert meta["Overlap"] == 1
        assert meta["Format"] == "nc"
        assert (
            meta["ColorbarTitle"] == "Height (µm)"
        )  # The writer decided to use µm, not mm
        assert meta["Colormap"] == "viridis"
        assert meta["Size"]["Width"] == nx
        assert meta["Size"]["Height"] == ny
        np.testing.assert_allclose(meta["PixelsPerMeter"]["Width"], 1000 * nx / sx)
        np.testing.assert_allclose(meta["PixelsPerMeter"]["Height"], 1000 * ny / sy)
        np.testing.assert_allclose(meta["ColorbarRange"]["Minimum"], t.min() * 1000)
        np.testing.assert_allclose(meta["ColorbarRange"]["Maximum"], t.max() * 1000)

        assert set(manifest) == set(
            [
                f"{d}/synthetic.json",
                f"{d}/synthetic_files/0/0_0.nc",
                f"{d}/synthetic_files/1/0_0.nc",
                f"{d}/synthetic_files/10/0_0.nc",
                f"{d}/synthetic_files/10/0_1.nc",
                f"{d}/synthetic_files/10/1_0.nc",
                f"{d}/synthetic_files/10/1_1.nc",
                f"{d}/synthetic_files/10/2_0.nc",
                f"{d}/synthetic_files/10/2_1.nc",
                f"{d}/synthetic_files/11/0_0.nc",
                f"{d}/synthetic_files/11/0_1.nc",
                f"{d}/synthetic_files/11/0_2.nc",
                f"{d}/synthetic_files/11/0_3.nc",
                f"{d}/synthetic_files/11/1_0.nc",
                f"{d}/synthetic_files/11/1_1.nc",
                f"{d}/synthetic_files/11/1_2.nc",
                f"{d}/synthetic_files/11/1_3.nc",
                f"{d}/synthetic_files/11/2_0.nc",
                f"{d}/synthetic_files/11/2_1.nc",
                f"{d}/synthetic_files/11/2_2.nc",
                f"{d}/synthetic_files/11/2_3.nc",
                f"{d}/synthetic_files/11/3_0.nc",
                f"{d}/synthetic_files/11/3_1.nc",
                f"{d}/synthetic_files/11/3_2.nc",
                f"{d}/synthetic_files/11/3_3.nc",
                f"{d}/synthetic_files/11/4_0.nc",
                f"{d}/synthetic_files/11/4_1.nc",
                f"{d}/synthetic_files/11/4_2.nc",
                f"{d}/synthetic_files/11/4_3.nc",
                f"{d}/synthetic_files/11/5_0.nc",
                f"{d}/synthetic_files/11/5_1.nc",
                f"{d}/synthetic_files/11/5_2.nc",
                f"{d}/synthetic_files/11/5_3.nc",
                f"{d}/synthetic_files/2/0_0.nc",
                f"{d}/synthetic_files/3/0_0.nc",
                f"{d}/synthetic_files/4/0_0.nc",
                f"{d}/synthetic_files/5/0_0.nc",
                f"{d}/synthetic_files/6/0_0.nc",
                f"{d}/synthetic_files/7/0_0.nc",
                f"{d}/synthetic_files/8/0_0.nc",
                f"{d}/synthetic_files/9/0_0.nc",
                f"{d}/synthetic_files/9/1_0.nc",
            ]
        )

        assert manifest == generate_manifest(
            "synthetic",
            t.nb_grid_pts,
            format="nc",
            meta_format="json",
            root_directory=d,
        )

        # Try reading one of the files
        fn = f"{d}/synthetic_files/11/4_1.nc"
        t = read_topography(fn)
        assert t.nb_grid_pts == (258, 258)
        # The physical size is now the actual physical size of the tile, not the pixel size
        # Level 11, scale_factor 1
        # sx = 1300, sy = 1200 (in um)
        # nx = 1324, ny = 871
        dx = 1300 / 1324
        dy = 1200 / 871
        np.testing.assert_allclose(t.physical_sizes, (258 * dx, 258 * dy), rtol=1e-6)

        # We now check the NetCDF file directly to verify the offsets and coordinates
        from scipy.io import netcdf_file
        with netcdf_file(fn, "r") as nc:
            assert nc.variables["x"].unit == b"um"
            assert nc.variables["y"].unit == b"um"
            assert nc.variables["heights"].unit == b"um"

            # Level 11, scale_factor 1, column 4, row 1
            # left = (4 * 256 - 1) * 1 = 1023
            # bottom = (1 * 256 - 1) * 1 = 255
            ox = 1023 / 1324 * 1300
            oy = 255 / 871 * 1200
            np.testing.assert_allclose(nc.variables["x"][0], ox)
            np.testing.assert_allclose(nc.variables["y"][0], oy)
            np.testing.assert_allclose(
                nc.variables["x"][...], ox + np.arange(258) * dx
            )
            np.testing.assert_allclose(
                nc.variables["y"][...], oy + np.arange(258) * dy
            )


def test_write_npy():
    nx, ny = 1324, 871
    t = fourier_synthesis((nx, ny), (1.3, 1.2), 0.8, rms_slope=0.1, unit="mm")
    sx, sy = t.physical_sizes
    with tempfile.TemporaryDirectory() as d:
        manifest = t.to_dzi("synthetic", d, format="npy", meta_format="json")
        assert os.path.exists(f"{d}/synthetic_files")
        for i in range(12):
            assert os.path.exists(f"{d}/synthetic_files/{i}")
        with open(f"{d}/synthetic.json", "r") as f:
            meta = json.load(f)
        meta = meta["Image"]
        assert meta["TileSize"] == 256
        assert meta["Overlap"] == 1
        assert meta["Format"] == "npy"
        assert (
            meta["ColorbarTitle"] == "Height (µm)"
        )  # The writer decided to use µm, not mm
        assert meta["Colormap"] == "viridis"
        assert meta["Size"]["Width"] == nx
        assert meta["Size"]["Height"] == ny
        np.testing.assert_allclose(meta["PixelsPerMeter"]["Width"], 1000 * nx / sx)
        np.testing.assert_allclose(meta["PixelsPerMeter"]["Height"], 1000 * ny / sy)
        np.testing.assert_allclose(meta["ColorbarRange"]["Minimum"], t.min() * 1000)
        np.testing.assert_allclose(meta["ColorbarRange"]["Maximum"], t.max() * 1000)

        assert set(manifest) == set(
            [
                f"{d}/synthetic.json",
                f"{d}/synthetic_files/0/0_0.npy",
                f"{d}/synthetic_files/1/0_0.npy",
                f"{d}/synthetic_files/10/0_0.npy",
                f"{d}/synthetic_files/10/0_1.npy",
                f"{d}/synthetic_files/10/1_0.npy",
                f"{d}/synthetic_files/10/1_1.npy",
                f"{d}/synthetic_files/10/2_0.npy",
                f"{d}/synthetic_files/10/2_1.npy",
                f"{d}/synthetic_files/11/0_0.npy",
                f"{d}/synthetic_files/11/0_1.npy",
                f"{d}/synthetic_files/11/0_2.npy",
                f"{d}/synthetic_files/11/0_3.npy",
                f"{d}/synthetic_files/11/1_0.npy",
                f"{d}/synthetic_files/11/1_1.npy",
                f"{d}/synthetic_files/11/1_2.npy",
                f"{d}/synthetic_files/11/1_3.npy",
                f"{d}/synthetic_files/11/2_0.npy",
                f"{d}/synthetic_files/11/2_1.npy",
                f"{d}/synthetic_files/11/2_2.npy",
                f"{d}/synthetic_files/11/2_3.npy",
                f"{d}/synthetic_files/11/3_0.npy",
                f"{d}/synthetic_files/11/3_1.npy",
                f"{d}/synthetic_files/11/3_2.npy",
                f"{d}/synthetic_files/11/3_3.npy",
                f"{d}/synthetic_files/11/4_0.npy",
                f"{d}/synthetic_files/11/4_1.npy",
                f"{d}/synthetic_files/11/4_2.npy",
                f"{d}/synthetic_files/11/4_3.npy",
                f"{d}/synthetic_files/11/5_0.npy",
                f"{d}/synthetic_files/11/5_1.npy",
                f"{d}/synthetic_files/11/5_2.npy",
                f"{d}/synthetic_files/11/5_3.npy",
                f"{d}/synthetic_files/2/0_0.npy",
                f"{d}/synthetic_files/3/0_0.npy",
                f"{d}/synthetic_files/4/0_0.npy",
                f"{d}/synthetic_files/5/0_0.npy",
                f"{d}/synthetic_files/6/0_0.npy",
                f"{d}/synthetic_files/7/0_0.npy",
                f"{d}/synthetic_files/8/0_0.npy",
                f"{d}/synthetic_files/9/0_0.npy",
                f"{d}/synthetic_files/9/1_0.npy",
            ]
        )

        assert manifest == generate_manifest(
            "synthetic",
            t.nb_grid_pts,
            format="npy",
            meta_format="json",
            root_directory=d,
        )


@pytest.mark.parametrize("format", ["nc", "npy"])
@pytest.mark.parametrize("masked", [True, False])
def test_undefined_points_are_nan(format, masked):
    nx, ny = 300, 200
    tile_size, overlap = 64, 1
    h = np.random.default_rng(0).normal(size=(nx, ny))
    undefined = np.zeros_like(h, dtype=bool)
    undefined[100:140, 50:57] = True
    undefined[299, 199] = True
    if masked:
        # Values under the mask would dominate the range if not ignored
        h[undefined] = 1e6
        data = np.ma.masked_array(h, mask=undefined)
    else:
        h[undefined] = np.nan
        data = h
    with tempfile.TemporaryDirectory() as d:
        write_dzi(
            data,
            "undefined",
            (3, 2),
            "um",
            root_directory=d,
            tile_size=tile_size,
            overlap=overlap,
            format=format,
            meta_format="json",
        )
        with open(f"{d}/undefined.json", "r") as f:
            meta = json.load(f)["Image"]
        np.testing.assert_allclose(
            meta["ColorbarRange"]["Minimum"], h[~undefined].min()
        )
        np.testing.assert_allclose(
            meta["ColorbarRange"]["Maximum"], h[~undefined].max()
        )
        for level, column, row, stride, (i0, i1), (j0, j1) in _tiles(
            (nx, ny), tile_size, overlap
        ):
            tile = _read_tile(f"{d}/undefined_files/{level}/{column}_{row}.{format}")
            expected = undefined[::stride, ::stride][i0:i1, j0:j1]
            np.testing.assert_array_equal(np.isnan(tile), expected)
            assert np.all(tile[~expected] >= 0)
            assert np.all(tile[~expected] <= 1)


def test_undefined_points_image():
    h = np.random.default_rng(0).normal(size=(100, 80))
    mask = np.zeros_like(h, dtype=bool)
    mask[10:20, 10:20] = True
    with tempfile.TemporaryDirectory() as d:
        manifest = write_dzi(
            np.ma.masked_array(h, mask=mask), "image", (1, 1), "um",
            root_directory=d, format="png",
        )
        root = ET.parse(open(f"{d}/image.xml")).getroot()
        assert "FormatVersion" not in root.attrib
        np.testing.assert_allclose(float(root[2].attrib["Minimum"]), h[~mask].min())
        assert all(os.path.exists(fn) for fn in manifest)


def test_netcdf_default_version():
    h = np.random.default_rng(0).normal(size=(100, 80))
    with tempfile.TemporaryDirectory() as d:
        write_dzi(h, "v1", (1, 1), "um", root_directory=d, format="nc")
        root = ET.parse(open(f"{d}/v1.xml")).getroot()
        assert root.attrib["FormatVersion"] == "1"
        assert root.find(f"{DZI_NAMESPACE}Encoding") is None
        with netcdf_file(f"{d}/v1_files/7/0_0.nc", "r", mmap=False) as nc:
            heights = nc.variables["heights"]
            assert heights.data.dtype == np.dtype(">f8")
            assert not hasattr(heights, "scale_factor")
            np.testing.assert_allclose(
                heights[...], (h - h.min()) / (h.max() - h.min())
            )


def test_netcdf_unknown_version():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            write_dzi(
                np.zeros((4, 4)), "v3", (1, 1), "um", root_directory=d,
                format="nc", nc_version=3,
            )
        # The version is ignored for other formats
        for format in ["png", "npy"]:
            write_dzi(
                np.zeros((4, 4)), format, (1, 1), "um", root_directory=d,
                format=format, nc_version=3,
            )
            root = ET.parse(open(f"{d}/{format}.xml")).getroot()
            assert "FormatVersion" not in root.attrib


@pytest.mark.parametrize("meta_format", ["xml", "json"])
@pytest.mark.parametrize("offset", [0, 1e6])
def test_netcdf_version2_roundtrip(meta_format, offset):
    nx, ny = 300, 200
    tile_size, overlap = 64, 1
    h = offset + np.random.default_rng(0).normal(size=(nx, ny))
    h[17, 23] = np.nan
    mn, mx = np.nanmin(h), np.nanmax(h)
    scale_factor = mx - mn
    add_offset = (mx + mn) / 2
    with tempfile.TemporaryDirectory() as d:
        manifest = write_dzi(
            h,
            "v2",
            (3, 2),
            "um",
            root_directory=d,
            tile_size=tile_size,
            overlap=overlap,
            format="nc",
            meta_format=meta_format,
            nc_version=2,
        )
        assert manifest == generate_manifest(
            "v2",
            (nx, ny),
            root_directory=d,
            tile_size=tile_size,
            format="nc",
            meta_format=meta_format,
        )

        if meta_format == "xml":
            root = ET.parse(open(f"{d}/v2.xml")).getroot()
            assert root.attrib["FormatVersion"] == "2"
            np.testing.assert_allclose(float(root[2].attrib["Minimum"]), mn)
            np.testing.assert_allclose(float(root[2].attrib["Maximum"]), mx)
            encoding = root.find(f"{DZI_NAMESPACE}Encoding").attrib
            assert float(encoding["ScaleFactor"]) == scale_factor
            assert float(encoding["AddOffset"]) == add_offset
        else:
            with open(f"{d}/v2.json", "r") as f:
                meta = json.load(f)["Image"]
            assert meta["FormatVersion"] == 2
            np.testing.assert_allclose(meta["ColorbarRange"]["Minimum"], mn)
            np.testing.assert_allclose(meta["ColorbarRange"]["Maximum"], mx)
            assert meta["Encoding"]["ScaleFactor"] == scale_factor
            assert meta["Encoding"]["AddOffset"] == add_offset

        for level, column, row, stride, (i0, i1), (j0, j1) in _tiles(
            (nx, ny), tile_size, overlap
        ):
            expected = h[::stride, ::stride][i0:i1, j0:j1]
            fn = f"{d}/v2_files/{level}/{column}_{row}.nc"
            with netcdf_file(fn, "r", mmap=False) as nc:
                heights = nc.variables["heights"]
                assert heights.data.dtype == np.dtype(">f4")
                # Attributes must be float64 for decoding to float64
                assert heights.scale_factor.dtype == np.float64
                assert heights.add_offset.dtype == np.float64
                assert heights.scale_factor == scale_factor
                assert heights.add_offset == add_offset
                v = heights[...].astype(float)
            assert np.all(np.abs(v[np.isfinite(v)]) <= 0.5)
            np.testing.assert_array_equal(np.isnan(v), np.isnan(expected))
            np.testing.assert_allclose(
                v * scale_factor + add_offset,
                expected,
                rtol=0,
                atol=scale_factor * 2**-24,
            )


def test_netcdf_version2_flat():
    h = np.full((100, 80), 3.5)
    with tempfile.TemporaryDirectory() as d:
        write_dzi(h, "flat", (1, 1), "um", root_directory=d, format="nc", nc_version=2)
        with netcdf_file(f"{d}/flat_files/7/0_0.nc", "r", mmap=False) as nc:
            heights = nc.variables["heights"]
            assert heights.scale_factor == 1
            assert heights.add_offset == 3.5
            np.testing.assert_array_equal(heights[...], 0)


def test_netcdf_version2_decoding():
    xr = pytest.importorskip("xarray")
    nx, ny = 1324, 871
    t = fourier_synthesis((nx, ny), (1.3, 1.2), 0.8, rms_slope=0.1, unit="mm")
    # Large offset relative to the range of heights
    t = Topography(t.heights() + 1000 * np.ptp(t.heights()), t.physical_sizes, unit="mm")
    with tempfile.TemporaryDirectory() as d:
        t.to_dzi("synthetic", d, format="nc", nc_version=2)
        fn = f"{d}/synthetic_files/11/4_1.nc"
        with netcdf_file(fn, "r", mmap=False) as nc:
            unit = nc.variables["heights"].unit.decode()
            scale_factor = nc.variables["heights"].scale_factor
        expected = t.to_unit(unit).heights()[1023:1281, 255:513]

        # Generic NetCDF tools apply the CF attributes
        with xr.open_dataset(fn) as ds:
            assert ds.heights.dtype == np.float64
            np.testing.assert_allclose(
                ds.heights.values, expected, rtol=0, atol=scale_factor * 2**-24
            )

        # ...and so does our own reader
        tile = read_topography(fn)
        np.testing.assert_allclose(
            tile.heights(), expected, rtol=0, atol=scale_factor * 2**-24
        )


@pytest.mark.parametrize("overlap", [0, 1, 2])
@pytest.mark.parametrize(
    "nb_grid_pts",
    [(256, 128), (300, 200), (65, 1)],  # Multiple of tile size or not
)
def test_overlap(overlap, nb_grid_pts):
    nx, ny = nb_grid_pts
    sx, sy = 3.0, 2.0
    tile_size = 64
    h = np.random.default_rng(0).normal(size=(nx, ny))
    with tempfile.TemporaryDirectory() as d:
        manifest = write_dzi(
            h,
            "overlap",
            (sx, sy),
            "um",
            root_directory=d,
            tile_size=tile_size,
            overlap=overlap,
            format="nc",
            meta_format="json",
        )
        with open(f"{d}/overlap.json", "r") as f:
            assert json.load(f)["Image"]["Overlap"] == overlap

        # The overlap does not change the list of files
        assert manifest == generate_manifest(
            "overlap",
            (nx, ny),
            root_directory=d,
            tile_size=tile_size,
            format="nc",
            meta_format="json",
        )

        expected_tiles = list(_tiles((nx, ny), tile_size, overlap))
        assert len(manifest) == len(expected_tiles) + 1
        for level, column, row, stride, (i0, i1), (j0, j1) in expected_tiles:
            fn = f"{d}/overlap_files/{level}/{column}_{row}.nc"
            with netcdf_file(fn, "r", mmap=False) as nc:
                heights = nc.variables["heights"][...].copy()
                x = nc.variables["x"][...].copy()
                y = nc.variables["y"][...].copy()
            assert heights.shape == (i1 - i0, j1 - j0)
            np.testing.assert_allclose(
                heights,
                ((h - h.min()) / (h.max() - h.min()))[::stride, ::stride][
                    i0:i1, j0:j1
                ],
            )
            # Coordinates are those of the full-resolution points sampled at
            # this level, at tile borders and at the image edge alike
            np.testing.assert_allclose(x, np.arange(i0, i1) * stride * sx / nx)
            np.testing.assert_allclose(y, np.arange(j0, j1) * stride * sy / ny)


@pytest.mark.parametrize("meta_format", ["xml", "json"])
@pytest.mark.parametrize(
    "detrend_modes", [[], ["height"], ["curvature"], ["height", "center"]]
)
def test_detrend_mode(meta_format, detrend_modes):
    t = fourier_synthesis((100, 80), (1, 1), 0.8, rms_slope=0.1, unit="mm")
    for detrend_mode in detrend_modes:
        t = t.detrend(detrend_mode=detrend_mode)
    # A decorator after the detrending step does not hide it
    t = t.scale(2)
    with tempfile.TemporaryDirectory() as d:
        t.to_dzi("detrended", d, meta_format=meta_format)
        if meta_format == "xml":
            meta = ET.parse(open(f"{d}/detrended.xml")).getroot().attrib
        else:
            with open(f"{d}/detrended.json", "r") as f:
                meta = json.load(f)["Image"]
    if detrend_modes:
        assert meta["Detrend"] == detrend_modes[-1]
    else:
        assert "Detrend" not in meta


def test_detrend_mode_write_dzi():
    with tempfile.TemporaryDirectory() as d:
        write_dzi(np.zeros((4, 4)), "plain", (1, 1), "um", root_directory=d)
        assert "Detrend" not in ET.parse(open(f"{d}/plain.xml")).getroot().attrib
        write_dzi(
            np.zeros((4, 4)),
            "detrended",
            (1, 1),
            "um",
            root_directory=d,
            detrend_mode="curvature",
        )
        root = ET.parse(open(f"{d}/detrended.xml")).getroot()
        assert root.attrib["Detrend"] == "curvature"
