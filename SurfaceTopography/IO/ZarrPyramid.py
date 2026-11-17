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

#
# Zoomable pyramid of a height map in a single file: a zipped OME-Zarr
# (.ozx), i.e. a Zarr v3 hierarchy with OME-Zarr 0.5 multiscales metadata in
# a ZIP archive. Specifications:
# https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
# https://ngff.openmicroscopy.org/0.5/
# https://ngff.openmicroscopy.org/rfc/9/ (zipped OME-Zarr)
#
# The hierarchy is written with zarr into a temporary directory and then
# packed into the ZIP archive in the order RFC-9 recommends, with all
# metadata at the start, which zarr's ZipStore does not control.
#

import json
import math
import os
import shutil
import tempfile
import zipfile

import numpy as np
from matplotlib import colormaps

from ..HeightContainer import UniformTopographyInterface
from ..Support.UnitConversion import mangle_length_unit_ascii
from .DZI import _coarsen, _export_unit_and_detrend

OME_VERSION = "0.5"

# Version of the "zoomdata" attributes of the root group
ZOOMDATA_VERSION = 1

# Fixed time stamp of the ZIP entries, so that equal data gives equal files
_DATE_TIME = (1980, 1, 1, 0, 0, 0)

# Units of OME-Zarr axes are UDUNITS-2 names
_OME_LENGTH_UNITS = {
    "km": "kilometer",
    "m": "meter",
    "mm": "millimeter",
    "um": "micrometer",
    "nm": "nanometer",
    "A": "angstrom",
    "pm": "picometer",
    "fm": "femtometer",
}


def _level_shapes(nb_grid_pts, tile_size):
    """
    Shapes (ny, nx) of the levels, from full resolution to the first level
    that fits into a single tile
    """
    nx, ny = nb_grid_pts
    shapes = [(ny, nx)]
    while max(shapes[-1]) > tile_size:
        ny, nx = shapes[-1]
        shapes += [(math.ceil(ny / 2), math.ceil(nx / 2))]
    return shapes


def _levels(data, nb_levels, downsample):
    """Yield the levels of `data`, transposed to (ny, nx)"""
    yield data.T
    if downsample == "mean":
        # Sums and counts of defined points within the block of each point
        defined = np.isfinite(data)
        sums = np.where(defined, data, 0)
        counts = defined.astype(np.int32)
    for level in range(1, nb_levels):
        if downsample == "mean":
            sums, counts = _coarsen(sums), _coarsen(counts)
            level_data = np.divide(
                sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0
            )
        else:
            stride = 2**level
            level_data = data[::stride, ::stride]
        yield level_data.T


def _zip_order(name):
    """
    RFC-9 order of the ZIP entries: the root zarr.json, then the other
    zarr.json breadth-first, then the shards
    """
    parts = name.split("/")
    is_metadata = parts[-1] == "zarr.json"
    return (
        not is_metadata,
        len(parts),
        [int(p) if p.isdigit() else p for p in parts],
    )


def _pack(directory, fobj):
    """Pack the Zarr hierarchy in `directory` into an RFC-9 ZIP archive"""
    names = []
    for path, _, files in os.walk(directory):
        for fn in files:
            names += [os.path.relpath(os.path.join(path, fn), directory)]
    names = sorted((name.replace(os.sep, "/") for name in names), key=_zip_order)
    with zipfile.ZipFile(fobj, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=_DATE_TIME)
            info.compress_type = zipfile.ZIP_STORED
            with (
                open(os.path.join(directory, name), "rb") as src,
                zf.open(info, "w", force_zip64=True) as dst,
            ):
                shutil.copyfileobj(src, dst)
        zf.comment = json.dumps(
            {
                "ome": {
                    "version": OME_VERSION,
                    "zipFile": {"centralDirectory": {"jsonFirst": True}},
                }
            }
        ).encode()


def write_zarr_pyramid(
    data,
    fobj,
    physical_sizes,
    unit,
    tile_size=256,
    shard_tiles=16,
    downsample="mean",
    compression_level=6,
    colorbar_title=None,
    cmap=None,
    detrend_mode=None,
):
    """
    Write a two-dimensional array as a zoomable pyramid into a single file,
    a zipped OME-Zarr (customarily with the suffix `.ozx`). The file can be
    read with HTTP range requests, so a web viewer needs a single URL for
    the whole pyramid.

    The file holds a Zarr v3 hierarchy with one float32 array per level,
    path '0' at full resolution and each further level coarsened by a
    factor of two, down to the first level that fits into a single tile.
    The arrays have dimensions (y, x), i.e. x varies fastest. They are cut
    into chunks of `tile_size` x `tile_size` points, which are compressed
    with gzip and grouped into shards of up to `shard_tiles` x `shard_tiles`
    chunks. The root group carries OME-Zarr 0.5 multiscales metadata with
    the pixel size of each level, and an attribute `zoomdata` with the data
    range, the unit, the color bar title and color map, the detrend mode
    and the downsampling method.

    Values are stored as `v = (h - add_offset) / scale_factor` in
    [-0.5, 0.5], with `add_offset = (max + min) / 2` and
    `scale_factor = max - min`, which spends the float32 mantissa on the
    range of the data. `scale_factor`, `add_offset` and `units` are CF
    attributes of every array. Undefined points (masked entries or NaN) are
    NaN, the fill value of the arrays; chunks and shards without defined
    points are not stored.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional array of shape (nx, ny). May be a masked array.
    fobj : str or file-like object
        File to write to.
    physical_sizes : tuple of floats
        Linear physical sizes of the two-dimensional array.
    unit : str
        Length unit of physical sizes and of the data.
    tile_size : int, optional
        Size of the chunks, i.e. the tiles of a viewer. (Default: 256)
    shard_tiles : int, optional
        Number of chunks along each side of a shard. (Default: 16)
    downsample : str, optional
        How coarse levels are obtained, 'mean' or 'nearest', see
        :func:`SurfaceTopography.IO.DZI.write_dzi`. The OME-Zarr metadata
        places a block mean at the center of its block and a sampled point
        at the point itself. (Default: 'mean')
    compression_level : int, optional
        gzip compression level of the chunks. (Default: 6)
    colorbar_title : str, optional
        Title of the color bar suggested to viewers. (Default: None)
    cmap : str or colormap, optional
        Color map suggested to viewers. (Default: None, i.e. matplotlib's
        default color map)
    detrend_mode : str, optional
        How the data was detrended, e.g. 'height' or 'curvature'.
        (Default: None)
    """
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "Writing zoomable pyramids requires zarr. Install it with "
            "`pip install SurfaceTopography[zarr]`."
        )

    if downsample not in ("nearest", "mean"):
        raise ValueError(f"Unknown downsampling method {downsample}.")
    if tile_size < 1 or shard_tiles < 1:
        raise ValueError("Tile size and number of tiles per shard must be positive.")
    if np.ndim(data) != 2:
        raise ValueError("Data must be two-dimensional.")

    # Undefined points are carried through as NaN
    data = np.ma.filled(np.ma.asarray(data, dtype=float), np.nan)
    if not np.isfinite(data).any():
        raise ValueError("Data has no defined points.")

    # Rescale heights to [-0.5, 0.5]
    mx, mn = float(np.nanmax(data)), float(np.nanmin(data))
    # A perfectly flat topography would otherwise divide by zero
    height_scale = mx - mn if mx > mn else 1.0
    height_offset = (mx + mn) / 2
    data = (data - height_offset) / height_scale

    unit = mangle_length_unit_ascii(unit)
    nx, ny = data.shape
    sx, sy = physical_sizes
    dx, dy = sx / nx, sy / ny

    shapes = _level_shapes(data.shape, tile_size)
    shard_shapes = [
        tuple(min(shard_tiles, math.ceil(n / tile_size)) * tile_size for n in shape)
        for shape in shapes
    ]

    axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    if unit in _OME_LENGTH_UNITS:
        for axis in axes:
            axis["unit"] = _OME_LENGTH_UNITS[unit]
    datasets = []
    for level in range(len(shapes)):
        stride = 2**level
        transformations = [{"type": "scale", "scale": [stride * dy, stride * dx]}]
        if downsample == "mean" and stride > 1:
            # A block mean represents the center of its block
            shift = (stride - 1) / 2
            transformations += [
                {"type": "translation", "translation": [shift * dy, shift * dx]}
            ]
        datasets += [{"path": str(level), "coordinateTransformations": transformations}]

    zoomdata = {
        "version": ZOOMDATA_VERSION,
        "unit": unit,
        "colorbarRange": {"minimum": mn, "maximum": mx},
        "colormap": colormaps.get_cmap(cmap).name,
        "downsample": downsample,
    }
    if colorbar_title is not None:
        zoomdata["colorbarTitle"] = colorbar_title
    if detrend_mode is not None:
        zoomdata["detrend"] = detrend_mode

    multiscale = {
        "name": "heights",
        "axes": axes,
        "datasets": datasets,
        "type": downsample,
    }
    array_attributes = {
        "scale_factor": height_scale,
        "add_offset": height_offset,
        "units": unit,
    }

    with tempfile.TemporaryDirectory() as directory:
        root = zarr.create_group(
            zarr.storage.LocalStore(directory),
            zarr_format=3,
            attributes={
                "ome": {"version": OME_VERSION, "multiscales": [multiscale]},
                "zoomdata": zoomdata,
            },
        )
        levels = _levels(data, len(shapes), downsample)
        for level, (level_data, shard_shape) in enumerate(zip(levels, shard_shapes)):
            root.create_array(
                str(level),
                data=level_data.astype(np.float32),
                chunks=(tile_size, tile_size),
                shards=shard_shape,
                compressors=zarr.codecs.GzipCodec(level=compression_level),
                fill_value=np.nan,
                attributes=array_attributes,
                dimension_names=("y", "x"),
                # Chunks and shards without defined points are not stored
                config={"write_empty_chunks": False},
            )
        _pack(directory, fobj)


def write_topography_zarr_pyramid(
    self: UniformTopographyInterface,
    fobj,
    tile_size=256,
    shard_tiles=16,
    downsample="mean",
    compression_level=6,
    cmap=None,
):
    """
    Write topography as a zoomable pyramid into a single file, a zipped
    OME-Zarr (customarily with the suffix `.ozx`), see
    :func:`write_zarr_pyramid`.

    The heights are converted to a unit suited to their range. If the
    pipeline of the topography contains a detrending step, its
    `detrend_mode` is recorded.

    Parameters
    ----------
    self : :obj:`Topography`
        Topography to export.
    fobj : str or file-like object
        File to write to.
    tile_size : int, optional
        Size of the chunks, i.e. the tiles of a viewer. (Default: 256)
    shard_tiles : int, optional
        Number of chunks along each side of a shard. (Default: 16)
    downsample : str, optional
        How coarse levels are obtained, 'mean' or 'nearest'.
        (Default: 'mean')
    compression_level : int, optional
        gzip compression level of the chunks. (Default: 6)
    cmap : str or colormap, optional
        Color map suggested to viewers. (Default: None)
    """
    t, unit, detrend_mode = _export_unit_and_detrend(self)
    write_zarr_pyramid(
        t.heights(),
        fobj,
        t.physical_sizes,
        unit,
        tile_size=tile_size,
        shard_tiles=shard_tiles,
        downsample=downsample,
        compression_level=compression_level,
        colorbar_title=f"Height ({unit})",
        cmap=cmap,
        detrend_mode=detrend_mode,
    )


UniformTopographyInterface.register_function(
    "to_zarr_pyramid", write_topography_zarr_pyramid
)
