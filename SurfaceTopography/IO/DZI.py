#
# Copyright 2021, 2023-2024 Lars Pastewka
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
# Deep Zoom Image File Format (DZI) - The DZI file format is described in detail here:
# https://github.com/openseadragon/openseadragon/wiki/The-DZI-File-Format
# https://docs.microsoft.com/en-us/previous-versions/windows/silverlight/dotnet-windows-silverlight/cc645077(v=vs.95)
#

import json
import math
import os
import xml.etree.cElementTree as ET
from typing import Tuple

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy.io import netcdf_file

from ..HeightContainer import UniformTopographyInterface
from ..Support.JSON import ExtendedJSONEncoder
from ..Support.UnitConversion import (
    get_unit_conversion_factor,
    mangle_length_unit_ascii,
    suggest_length_unit_for_data,
)
from ..Uniform.Detrending import DetrendedUniformTopography


def _coarsen(a):
    """Sum 2x2 blocks of `a`; a trailing odd row or column is summed alone"""
    nx, ny = a.shape
    a = np.pad(a, ((0, nx % 2), (0, ny % 2)))
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).sum(axis=(1, 3))


def write_dzi(
    data: np.ndarray,
    name: str,
    physical_sizes: Tuple[float, float],
    unit: str,
    root_directory: str = ".",
    tile_size: int = 256,
    overlap: int = 1,
    format: str = "jpg",
    meta_format: str = "xml",
    colorbar_title=None,
    cmap=None,
    nc_version: int = 1,
    detrend_mode: str = None,
    downsample: str = "mean",
    **kwargs,
):
    """
    Write generic numpy array to a Deep Zoom Image file. This can for example
    be used to create a zoomable topography with OpenSeadragon
    (https://openseadragon.github.io/).

    Additional keyword parameters are passed to Pillow's `save` function.

    Undefined points (masked entries or NaN in `data`) are written as NaN to
    'npy' and 'nc' tiles and rendered in the color map's "bad" color in
    image tiles. They are ignored when determining the data range.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional array containing the data. May be a masked array.
    name : str
        Name of the exported file. This is used as a prefix. The output filter
        creates the file `name`.xml that contains the metadata and a directory
        `name`_files that contains the rendered image files at different levels.
    physical_sizes : tuple of floats
        Linear physical sizes of the two-dimensional array.
    unit : str
        Length units of physical sizes and of the data. Recorded as `Unit` in
        the descriptor.
    root_directory : str, optional
        Root directory where to place `name`.xml and `name`_files.
        (Default: '.')
    tile_size : int, optional
        Size of individual tiles. (Default: 256)
    overlap : int, optional
        Overlap of tiles. (Default: 1)
    format : str, optional
        Data output format. Note that PNG files have seams at the boundary between
        tiles. Use 'npy' to output raw data in the native numpy format.
        Use 'nc' to output raw data as NetCDF files. (Default: 'jpg')
    meta_format : str, optional
        Format for metadata information (the DZI file), can be 'xml' or
        'json'. (Default: 'xml')
    colorbar_title : str, optional
        Additional title for the color bar that is dumped into the DZI file.
        Ignored if format is 'npy' or 'nc'.
        (Default: None)
    cmap : str or colormap, optional
        Color map for rendering the topography. Ignored if format is 'npy' or
        'nc'. (Default: None)
    nc_version : int, optional
        Version of the NetCDF tile format, recorded as `FormatVersion` in the
        descriptor. Version 1 stores float64 data rescaled to [0, 1] using
        the `ColorbarRange` of the descriptor. Version 2 stores float32 data
        `v = (h - add_offset) / scale_factor` in [-0.5, 0.5], with
        `add_offset = (max + min) / 2` and `scale_factor = max - min`;
        both are written as CF attributes of `heights` in every tile and to
        `Encoding` in the descriptor. Ignored unless format is 'nc'.
        (Default: 1)
    detrend_mode : str, optional
        How the data was detrended, e.g. 'height' or 'curvature'. Recorded as
        `Detrend` in the descriptor if given. (Default: None)
    downsample : str, optional
        How coarse levels are obtained, recorded as `Downsample` in the
        descriptor. 'mean' averages the defined points of each n x n block,
        which avoids aliasing in overviews of maps with fine structure; a
        block without defined points is undefined. Blocks at the right and
        top edge are truncated by the image boundary and average fewer
        points. 'nearest' takes every n-th point, so each coarse value is a
        measured value. In both cases a coarse value is located at the first
        point of its block, so that tiles stay on a regular grid.
        (Default: 'mean')

    Returns
    -------
    manifest : list of str
        List with names of files created during write operation
    """
    if format != "nc":
        nc_version = None
    elif nc_version not in (1, 2):
        raise ValueError(f"Unknown NetCDF tile format version {nc_version}.")
    if downsample not in ("nearest", "mean"):
        raise ValueError(f"Unknown downsampling method {downsample}.")

    def write_data(fn, subdata, pixel_physical_sizes, physical_offsets):
        if format == "npy":
            # We write the raw data in the native numpy format
            np.save(fn, subdata)
        elif format == "nc":
            # We write the raw data in NetCDF-3 format
            nx, ny = subdata.shape
            dx, dy = pixel_physical_sizes
            ox, oy = physical_offsets
            nc = netcdf_file(fn, "w")
            nc.createDimension("x", nx)
            nc.createDimension("y", ny)
            if nc_version == 2:
                heights_var = nc.createVariable("heights", "f4", ("x", "y"))
                # Attributes must be float64; scipy writes Python floats as
                # float32, and CF decoding returns the attributes' type
                heights_var.scale_factor = np.float64(height_scale)
                heights_var.add_offset = np.float64(height_offset)
            else:
                heights_var = nc.createVariable("heights", "f8", ("x", "y"))
            heights_var[...] = subdata
            heights_var.unit = mangle_length_unit_ascii(unit)
            x_var = nc.createVariable("x", "f8", ("x",))
            x_var.length = nx * dx
            x_var.unit = mangle_length_unit_ascii(unit)
            x_var[...] = ox + np.arange(nx) * dx
            y_var = nc.createVariable("y", "f8", ("y",))
            y_var.length = ny * dy
            y_var.unit = mangle_length_unit_ascii(unit)
            y_var[...] = oy + np.arange(ny) * dy
        else:
            # Convert to image and save
            colors = (cmap(subdata.T) * 255).astype(np.uint8)
            # Remove alpha channel before writing
            Image.fromarray(colors[:, :, :3]).save(fn, **kwargs)

    cmap = colormaps.get_cmap(cmap)

    # Image size
    full_width, full_height = width, height = data.shape

    # Compute pixels per meter
    sx, sy = physical_sizes
    fac = get_unit_conversion_factor(unit, "m")
    pixels_per_meter_width = width / (fac * sx)
    pixels_per_meter_height = height / (fac * sy)

    # Undefined points are carried through as NaN; the color map renders
    # them in its "bad" color, just like masked entries
    data = np.ma.filled(np.ma.asarray(data, dtype=float), np.nan)

    # Rescale heights to [0, 1], or to [-0.5, 0.5] for NetCDF version 2
    mx, mn = np.nanmax(data), np.nanmin(data)
    # A perfectly flat topography would otherwise divide by zero and produce
    # an all-NaN pyramid
    height_scale = mx - mn if mx > mn else 1
    height_offset = (mx + mn) / 2 if nc_version == 2 else mn
    data = (data - height_offset) / height_scale

    # Write configuration file
    if meta_format == "xml":
        fn = os.path.join(root_directory, name + ".xml")
        root = ET.Element(
            "Image",
            TileSize=str(tile_size),
            Overlap=str(overlap),
            Format=format,
            Unit=mangle_length_unit_ascii(unit),
            Colormap=cmap.name,
            xmlns="http://schemas.microsoft.com/deepzoom/2008",
        )
        if colorbar_title is not None:
            root.set("ColorbarTitle", colorbar_title)
        if nc_version is not None:
            root.set("FormatVersion", str(nc_version))
        if detrend_mode is not None:
            root.set("Detrend", detrend_mode)
        root.set("Downsample", downsample)
        ET.SubElement(root, "Size", Width=str(width), Height=str(height))
        ET.SubElement(
            root,
            "PixelsPerMeter",
            Width=str(pixels_per_meter_width),
            Height=str(pixels_per_meter_height),
        )
        ET.SubElement(root, "ColorbarRange", Minimum=str(mn), Maximum=str(mx))
        if nc_version == 2:
            ET.SubElement(
                root,
                "Encoding",
                ScaleFactor=str(height_scale),
                AddOffset=str(height_offset),
            )
        os.makedirs(root_directory, exist_ok=True)
        ET.ElementTree(root).write(fn, encoding="utf-8", xml_declaration=True)
    elif meta_format == "json":
        fn = os.path.join(root_directory, name + ".json")
        with open(fn, "w") as f:
            image_dict = {
                "xmlns": "http://schemas.microsoft.com/deepzoom/2008",
                "Format": format,
                "Unit": mangle_length_unit_ascii(unit),
                "Overlap": overlap,
                "TileSize": tile_size,
                "Size": {"Width": width, "Height": height},
                "PixelsPerMeter": {
                    "Width": pixels_per_meter_width,
                    "Height": pixels_per_meter_height,
                },
                "Colormap": cmap.name,
                "ColorbarRange": {"Minimum": mn, "Maximum": mx},
            }
            if colorbar_title is not None:
                image_dict.update({"ColorbarTitle": colorbar_title})
            if nc_version is not None:
                image_dict.update({"FormatVersion": nc_version})
            if detrend_mode is not None:
                image_dict.update({"Detrend": detrend_mode})
            image_dict.update({"Downsample": downsample})
            if nc_version == 2:
                image_dict.update(
                    {
                        "Encoding": {
                            "ScaleFactor": height_scale,
                            "AddOffset": height_offset,
                        }
                    }
                )
            json.dump({"Image": image_dict}, f, cls=ExtendedJSONEncoder)
    else:
        raise ValueError(f"Unknown metadata format {meta_format}.")
    manifest = [fn]

    # Determine number of levels
    max_level = math.ceil(math.log2(max(width, height)))

    # Loop over levels and write tiles
    root_directory = os.path.join(root_directory, name + "_files")
    os.makedirs(root_directory, exist_ok=True)
    scale_factor = 1
    level_data = data
    if downsample == "mean":
        # Sums and counts of defined points within the block of each point
        defined = np.isfinite(data)
        sums = np.where(defined, data, 0)
        counts = defined.astype(np.int32)
    for level in range(max_level, -1, -1):
        level_root_directory = os.path.join(root_directory, str(level))
        os.makedirs(level_root_directory, exist_ok=True)

        columns = math.ceil(width / tile_size)
        rows = math.ceil(height / tile_size)

        # Loop over all tiles
        for column in range(columns):
            for row in range(rows):
                # File name for this tile
                fn = os.path.join(level_root_directory, f"{column}_{row}.{format}")

                # Determine image section of this tile on the grid of this
                # level
                left = max(column * tile_size - overlap, 0)
                bottom = max(row * tile_size - overlap, 0)
                right = min((column + 1) * tile_size + overlap, width)
                top = min((row + 1) * tile_size + overlap, height)

                write_data(
                    fn,
                    level_data[left:right, bottom:top],
                    (sx / full_width * scale_factor, sy / full_height * scale_factor),
                    (
                        left * scale_factor / full_width * sx,
                        bottom * scale_factor / full_height * sy,
                    ),
                )
                manifest += [fn]

        width = math.ceil(width / 2)
        height = math.ceil(height / 2)
        scale_factor *= 2
        if downsample == "mean":
            sums, counts = _coarsen(sums), _coarsen(counts)
            level_data = np.divide(
                sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0
            )
        else:
            level_data = data[::scale_factor, ::scale_factor]

    return manifest


def write_topography_dzi(
    self: UniformTopographyInterface,
    name: str,
    root_directory: str = ".",
    tile_size: int = 256,
    overlap: int = 1,
    format: str = "jpg",
    meta_format: str = "xml",
    cmap=None,
    nc_version: int = 1,
    downsample: str = "mean",
    **kwargs,
):
    """
    Write topography to a Deep Zoom Image file. This can for example be used
    to create a zoomable topography with OpenSeadragon
    (https://openseadragon.github.io/).

    Additional keyword parameters are passed to Pillow's `save` function.

    If the pipeline of the topography contains a detrending step, its
    `detrend_mode` is recorded as `Detrend` in the descriptor.

    Parameters
    ----------
    self : :obj:`Topography`
        Topography to export.
    name : str
        Name of the exported file. This is used as a prefix. The output filter
        creates the file `name`.xml that contains the metadata and a directory
        `name`_files that contains the rendered image files at different levels.
    root_directory : str, optional
        Root directory where to place `name`.xml and `name`_files.
        (Default: '.')
    tile_size : int, optional
        Size of individual tiles. (Default: 256)
    overlap : int, optional
        Overlap of tiles. (Default: 1)
    format : str, optional
        Data output format. Note that PNG files have seams at the boundary
        between tiles. Use 'npy' to output raw data in the native numpy
        format. Use 'nc' to output raw data as NetCDF files.
        (Default: 'jpg')
    meta_format : str, optional
        Format for metadata information (the DZI file), can be 'xml' or
        'json'. (Default: 'xml')
    cmap : str or colormap, optional
        Color map for rendering the topography. Ignored if format is 'npy' or
        'nc'. (Default: None)
    nc_version : int, optional
        Version of the NetCDF tile format, see :func:`write_dzi`. Ignored
        unless format is 'nc'. (Default: 1)
    downsample : str, optional
        How coarse levels are obtained, 'mean' or 'nearest', see
        :func:`write_dzi`. (Default: 'mean')

    Returns
    -------
    filenames : list of str
        List with names of files created during write operation
    """
    # Get reasonable unit
    ideal_height_unit = suggest_length_unit_for_data(
        "linear", self.heights(), self.unit
    )
    t = self.to_unit(ideal_height_unit)
    # The last detrending step in the pipeline determines the exported data
    detrend_mode = None
    for stage in self.pipeline():
        if isinstance(stage, DetrendedUniformTopography):
            detrend_mode = stage.detrend_mode
    return write_dzi(
        t.heights(),
        name,
        t.physical_sizes,
        ideal_height_unit,
        root_directory=root_directory,
        tile_size=tile_size,
        overlap=overlap,
        format=format,
        meta_format=meta_format,
        colorbar_title=f"Height ({ideal_height_unit})",
        cmap=cmap,
        nc_version=nc_version,
        detrend_mode=detrend_mode,
        downsample=downsample,
        **kwargs,
    )


def generate_manifest(
    name: str,
    nb_grid_pts: Tuple[int, int],
    root_directory: str = ".",
    tile_size: int = 256,
    format: str = "jpg",
    meta_format: str = "xml",
):
    """
    Generate the file manifest, i.e. the list of files, that would be generated when
    writing the DZI.

    There is no `overlap` argument: the overlap changes the extent of each
    tile, but not the number of tiles and hence not the list of files.

    Parameters
    ----------
    name : str
        Name of the exported file. This is used as a prefix. The output filter
        creates the file `name`.xml that contains the metadata and a directory
        `name`_files that contains the rendered image files at different levels.
    nb_grid_pts : tuple of ints
        Number of grid points.
    root_directory : str, optional
        Root directory where to place `name`.xml and `name`_files.
        (Default: '.')
    tile_size : int, optional
        Size of individual tiles. (Default: 256)
    format : str, optional
        Data output format. Note that PNG files have seams at the boundary between
        tiles. Use 'npy' to output raw data in the native numpy format.
        Use 'nc' to output raw data as NetCDF files. (Default: 'jpg')
    meta_format : str, optional
        Format for metadata information (the DZI file), can be 'xml' or
        'json'. (Default: 'xml')

    Returns
    -------
    manifest : list of str
        List with names of files created during write operation
    """

    # Image size
    width, height = nb_grid_pts

    # Write configuration file
    if meta_format == "xml":
        fn = os.path.join(root_directory, name + ".xml")
    elif meta_format == "json":
        fn = os.path.join(root_directory, name + ".json")
    else:
        raise ValueError(f"Unknown metadata format {meta_format}.")
    manifest = [fn]

    # Determine number of levels
    max_level = math.ceil(math.log2(max(width, height)))

    # Loop over levels and write tiles
    root_directory = os.path.join(root_directory, name + "_files")
    for level in range(max_level, -1, -1):
        level_root_directory = os.path.join(root_directory, str(level))

        columns = math.ceil(width / tile_size)
        rows = math.ceil(height / tile_size)

        # Loop over all tiles
        for column in range(columns):
            for row in range(rows):
                # File name for this tile
                fn = os.path.join(level_root_directory, f"{column}_{row}.{format}")

                manifest += [fn]

        width = math.ceil(width / 2)
        height = math.ceil(height / 2)

    return manifest


UniformTopographyInterface.register_function("to_dzi", write_topography_dzi)
