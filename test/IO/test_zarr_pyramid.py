#
# Copyright 2026 Lars Pastewka
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
import math
import zipfile

import numpy as np
import pytest
from NuMPI.Testing.Fixtures import serial_only

from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.IO.ZarrPyramid import write_zarr_pyramid

pytestmark = serial_only

zarr = pytest.importorskip("zarr", minversion="3")


@pytest.fixture
def write(tmp_path):
    """Write a pyramid and open it with zarr"""
    stores = []

    def _write(data, physical_sizes=(1.0, 1.0), unit="um", **kwargs):
        fn = tmp_path / f"pyramid{len(stores)}.ozx"
        write_zarr_pyramid(data, fn, physical_sizes, unit, **kwargs)
        store = zarr.storage.ZipStore(fn, mode="r")
        stores.append(store)
        return fn, zarr.open_group(store, mode="r")

    yield _write
    for store in stores:
        store.close()


def _heights(array):
    """Physical heights of a level"""
    attrs = array.attrs
    return attrs["add_offset"] + attrs["scale_factor"] * array[...].astype(float)


def _expected_level(data, stride, downsample):
    """Level of the pyramid, on the (nx, ny) grid of `data`"""
    if downsample == "nearest":
        return data[::stride, ::stride]
    nx, ny = data.shape
    out = np.full((-(-nx // stride), -(-ny // stride)), np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            block = data[i * stride : (i + 1) * stride, j * stride : (j + 1) * stride]
            if np.isfinite(block).any():
                out[i, j] = np.nanmean(block)
    return out


@pytest.mark.parametrize("downsample", ["mean", "nearest"])
@pytest.mark.parametrize(
    "nb_grid_pts,tile_size,shard_tiles",
    [
        ((75, 41), 16, 2),  # several shards, partial chunks and shards
        ((64, 64), 16, 4),  # exact multiples
        ((10, 7), 16, 4),  # smaller than a tile: a single level
        ((33, 17), 8, 16),  # one shard per level
    ],
)
def test_levels(write, nb_grid_pts, tile_size, shard_tiles, downsample):
    rng = np.random.default_rng(1)
    data = 1000 + rng.standard_normal(nb_grid_pts)
    data[3:9, 2:5] = np.nan
    mn, mx = np.nanmin(data), np.nanmax(data)
    _, root = write(
        data, tile_size=tile_size, shard_tiles=shard_tiles, downsample=downsample
    )

    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    # Coarsen until the level fits into a single tile
    nb_levels = 1
    while max(math.ceil(n / 2 ** (nb_levels - 1)) for n in nb_grid_pts) > tile_size:
        nb_levels += 1
    assert [d["path"] for d in datasets] == [str(i) for i in range(nb_levels)]

    for level in range(nb_levels):
        array = root[str(level)]
        assert array.dtype == np.float32
        assert array.attrs["scale_factor"] == pytest.approx(mx - mn)
        assert array.attrs["add_offset"] == pytest.approx((mx + mn) / 2)
        assert np.nanmax(np.abs(array[...])) <= 0.5 + 1e-6
        heights = _heights(array)
        expected = _expected_level(data, 2**level, downsample).T
        assert heights.shape == expected.shape
        np.testing.assert_array_equal(np.isnan(heights), np.isnan(expected))
        np.testing.assert_allclose(
            heights, expected, rtol=0, atol=1e-6 * (mx - mn), equal_nan=True
        )


def test_metadata(write):
    nx, ny = 70, 40
    sx, sy = 7.0, 2.0
    data = np.random.default_rng(2).standard_normal((nx, ny))
    fn, root = write(
        data,
        physical_sizes=(sx, sy),
        tile_size=16,
        shard_tiles=2,
        colorbar_title="Height (µm)",
        cmap="inferno",
        detrend_mode="curvature",
    )

    ome = root.attrs["ome"]
    assert ome["version"] == "0.5"
    (multiscale,) = ome["multiscales"]
    assert multiscale["axes"] == [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    assert multiscale["type"] == "mean"
    dx, dy = sx / nx, sy / ny
    for level, dataset in enumerate(multiscale["datasets"]):
        stride = 2**level
        transformations = dataset["coordinateTransformations"]
        assert transformations[0]["type"] == "scale"
        np.testing.assert_allclose(
            transformations[0]["scale"], [stride * dy, stride * dx]
        )
        if level == 0:
            assert len(transformations) == 1
        else:
            # Block means are located at the center of their block
            assert transformations[1]["type"] == "translation"
            np.testing.assert_allclose(
                transformations[1]["translation"],
                [(stride - 1) / 2 * dy, (stride - 1) / 2 * dx],
            )

    assert root.attrs["zoomdata"] == {
        "version": 1,
        "unit": "um",
        "colorbarRange": {"minimum": data.min(), "maximum": data.max()},
        "colormap": "inferno",
        "downsample": "mean",
        "colorbarTitle": "Height (µm)",
        "detrend": "curvature",
    }

    nb_levels = len(multiscale["datasets"])
    assert nb_levels == 4  # 70 -> 35 -> 18 -> 9
    for level in range(nb_levels):
        array = root[str(level)]
        assert array.shape == (-(-ny // 2**level), -(-nx // 2**level))
        assert array.chunks == (16, 16)
        # Shards are no larger than needed for the level
        assert array.shards == tuple(min(2, -(-n // 16)) * 16 for n in array.shape)
        assert np.isnan(array.fill_value)
        assert array.metadata.dimension_names == ("y", "x")
        assert array.attrs["units"] == "um"

    # RFC-9: metadata first, root first, uncompressed entries, comment
    with zipfile.ZipFile(fn) as zf:
        names = zf.namelist()
        assert names[: nb_levels + 1] == ["zarr.json"] + [
            f"{level}/zarr.json" for level in range(nb_levels)
        ]
        assert all("/c/" in name for name in names[nb_levels + 1 :])
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
        assert json.loads(zf.comment) == {
            "ome": {
                "version": "0.5",
                "zipFile": {"centralDirectory": {"jsonFirst": True}},
            }
        }


def test_nearest_has_no_translation(write):
    data = np.random.default_rng(3).standard_normal((40, 40))
    _, root = write(data, tile_size=8, downsample="nearest")
    (multiscale,) = root.attrs["ome"]["multiscales"]
    assert multiscale["type"] == "nearest"
    for dataset in multiscale["datasets"]:
        assert [t["type"] for t in dataset["coordinateTransformations"]] == ["scale"]


def test_undefined_shards_are_not_stored(write):
    nx, ny = 64, 32
    data = np.random.default_rng(4).standard_normal((nx, ny))
    # The shard x in [0, 32) is undefined
    data[:32, :] = np.nan
    fn, root = write(data, tile_size=8, shard_tiles=4)
    with zipfile.ZipFile(fn) as zf:
        assert [n for n in zf.namelist() if n.startswith("0/c/")] == ["0/c/0/1"]
    np.testing.assert_array_equal(np.isnan(root["0"][...]), np.isnan(data.T))


def test_masked_array(write):
    data = np.ma.masked_array(
        np.arange(12.0).reshape(4, 3), mask=np.zeros((4, 3), dtype=bool)
    )
    data.mask[1, 2] = True
    _, root = write(data, tile_size=4)
    heights = _heights(root["0"])
    assert np.isnan(heights[2, 1])
    assert np.isfinite(heights).sum() == 11
    np.testing.assert_allclose(heights[0, 0], 0.0, atol=1e-6)


def test_flat_data(write):
    _, root = write(np.full((5, 5), 3.0), tile_size=4)
    array = root["0"]
    assert array.attrs["scale_factor"] == 1.0
    assert array.attrs["add_offset"] == 3.0
    np.testing.assert_array_equal(array[...], 0)


def test_reproducible(tmp_path):
    data = np.random.default_rng(5).standard_normal((50, 30))
    write_zarr_pyramid(data, tmp_path / "a.ozx", (1, 1), "nm", tile_size=8)
    write_zarr_pyramid(data, tmp_path / "b.ozx", (1, 1), "nm", tile_size=8)
    assert (tmp_path / "a.ozx").read_bytes() == (tmp_path / "b.ozx").read_bytes()


def test_errors(write):
    with pytest.raises(ValueError, match="downsampling"):
        write(np.zeros((4, 4)), downsample="median")
    with pytest.raises(ValueError, match="no defined points"):
        write(np.full((4, 4), np.nan))
    with pytest.raises(ValueError, match="two-dimensional"):
        write(np.zeros(4))
    with pytest.raises(ValueError, match="positive"):
        write(np.zeros((4, 4)), tile_size=0)


def test_topography(tmp_path):
    t = fourier_synthesis((100, 60), (1.0, 0.6), 0.8, rms_slope=0.1, unit="mm")
    t = t.detrend("curvature")
    fn = tmp_path / "topography.ozx"
    t.to_zarr_pyramid(fn, tile_size=32)
    with zarr.storage.ZipStore(fn, mode="r") as store:
        root = zarr.open_group(store, mode="r")
        zoomdata = root.attrs["zoomdata"]
        # The writer converts the heights to µm
        assert zoomdata["unit"] == "um"
        assert zoomdata["colorbarTitle"] == "Height (µm)"
        assert zoomdata["detrend"] == "curvature"
        np.testing.assert_allclose(zoomdata["colorbarRange"]["minimum"], 1000 * t.min())
        np.testing.assert_allclose(zoomdata["colorbarRange"]["maximum"], 1000 * t.max())
        (multiscale,) = root.attrs["ome"]["multiscales"]
        np.testing.assert_allclose(
            multiscale["datasets"][0]["coordinateTransformations"][0]["scale"],
            [1000 * 0.6 / 60, 1000 * 1.0 / 100],
        )
        array = root["0"]
        np.testing.assert_allclose(
            _heights(array),
            1000 * t.heights().T,
            atol=1e-6 * array.attrs["scale_factor"],
        )
