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

import numpy as np
from matplotlib import colormaps
from numpyencoder import NumpyEncoder
from PIL import Image
from scipy.io import netcdf_file

from ..HeightContainer import UniformTopographyInterface
from ..Support.UnitConversion import (get_unit_conversion_factor,
                                      suggest_length_unit_for_data)


def write_dzi(data, name, physical_sizes, unit, root_directory='.', tile_size=256, overlap=1, format='jpg',
              meta_format='xml', colorbar_title=None, cmap=None, **kwargs):
    """
    Write generic numpy array to a Deep Zoom Image file. This can for example
    be used to create a zoomable topography with OpenSeadragon
    (https://openseadragon.github.io/).

    Additional keyword parameters are passed to Pillow's `save` function.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional array containing the data.
    name : str
        Name of the exported file. This is used as a prefix. Output filter
        create the file `name`.xml that contains the metadata and a directory
        `name`_files that contains the rendered image files at different levels.
    physical_sizes : tuple of floats
        Linear physical sizes of the two-dimensional array.
    unit : str
        Length units of physical sizes.
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

    Returns
    -------
    manifest : list of str
        List with names of files created during write operation
    """

    def write_data(fn, subdata, physical_sizes):
        if format == 'npy':
            # We write the raw data in the native numpy format
            np.save(fn, subdata)
        elif format == 'nc':
            # We write the raw data in NetCDF-3 format
            nx, ny = subdata.shape
            sx, sy = physical_sizes
            nc = netcdf_file(fn, 'w')
            nc.createDimension('x', nx)
            nc.createDimension('y', ny)
            heights_var = nc.createVariable('heights', 'f8', ('x', 'y'))
            heights_var[...] = subdata
            x_var = nc.createVariable('x', 'f8', ('x',))
            x_var.length = sx
            x_var[...] = np.arange(nx) / nx * sx
            y_var = nc.createVariable('y', 'f8', ('y',))
            y_var.length = sy
            y_var[...] = np.arange(ny) / ny * sy
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
    fac = get_unit_conversion_factor(unit, 'm')
    pixels_per_meter_width = width / (fac * sx)
    pixels_per_meter_height = height / (fac * sy)

    # Get heights and rescale to interval 0, 1
    mx, mn = data.max(), data.min()
    data = (data - mn) / (mx - mn)

    # Write configuration file
    if meta_format == 'xml':
        fn = os.path.join(root_directory, name + '.xml')
        root = ET.Element('Image', TileSize=str(tile_size), Overlap=str(overlap), Format=format, Colormap=cmap.name,
                          xmlns='http://schemas.microsoft.com/deepzoom/2008')
        if colorbar_title is not None:
            root.set('ColorbarTitle', colorbar_title)
        ET.SubElement(root, 'Size', Width=str(width), Height=str(height))
        ET.SubElement(root, 'PixelsPerMeter', Width=str(pixels_per_meter_width), Height=str(pixels_per_meter_height))
        ET.SubElement(root, 'ColorbarRange', Minimum=str(mn), Maximum=str(mx))
        os.makedirs(root_directory, exist_ok=True)
        ET.ElementTree(root).write(fn, encoding='utf-8', xml_declaration=True)
    elif meta_format == 'json':
        fn = os.path.join(root_directory, name + '.json')
        with open(fn, 'w') as f:
            image_dict = {
                'xmlns': 'http://schemas.microsoft.com/deepzoom/2008',
                'Format': format,
                'Overlap': overlap,
                'TileSize': tile_size,
                'Size': {
                    'Width': width,
                    'Height': height
                },
                'PixelsPerMeter': {
                    'Width': pixels_per_meter_width,
                    'Height': pixels_per_meter_height
                },
                'Colormap': cmap.name,
                'ColorbarRange': {
                    'Minimum': mn,
                    'Maximum': mx
                }
            }
            if colorbar_title is not None:
                image_dict.update({'ColorbarTitle': colorbar_title})
            json.dump({'Image': image_dict}, f, cls=NumpyEncoder)
    else:
        raise ValueError(f'Unknown metadata format {meta_format}.')
    manifest = [fn]

    # Determine number of levels
    max_level = math.ceil(math.log2(max(width, height)))

    # Loop over levels and write tiles
    root_directory = os.path.join(root_directory, name + '_files')
    os.makedirs(root_directory, exist_ok=True)
    scale_factor = 1
    for level in range(max_level, -1, -1):
        level_root_directory = os.path.join(root_directory, str(level))
        os.makedirs(level_root_directory, exist_ok=True)

        columns = math.ceil(width / tile_size)
        rows = math.ceil(height / tile_size)

        # Loop over all tiles
        for column in range(columns):
            for row in range(rows):
                # File name for this tile
                fn = os.path.join(level_root_directory, f'{column}_{row}.{format}')

                # Determine image section of this tile
                left = (column * tile_size - overlap) * scale_factor
                bottom = (row * tile_size - overlap) * scale_factor

                right = ((column + 1) * tile_size + overlap) * scale_factor
                top = ((row + 1) * tile_size + overlap) * scale_factor

                if left < 0:
                    left = 0
                if bottom < 0:
                    bottom = 0
                if right > full_width - 1:
                    right = full_width - 1
                if top > full_height - 1:
                    top = full_height - 1

                write_data(fn, data[left:right:scale_factor, bottom:top:scale_factor],
                           (sx / full_width * scale_factor, sy / full_height * scale_factor))
                manifest += [fn]

        width = math.ceil(width / 2)
        height = math.ceil(height / 2)
        scale_factor *= 2

    return manifest


def write_topography_dzi(self, name, root_directory='.', tile_size=256, overlap=1, format='jpg', meta_format='xml',
                         cmap=None, **kwargs):
    """
    Write topography to a Deep Zoom Image file. This can for example be used
    to create a zoomable topography with OpenSeadragon
    (https://openseadragon.github.io/).

    Additional keyword parameters are passed to Pillow's `save` function.

    Parameters
    ----------
    self : :obj:`Topography`
        Topogaphy to export
    name : str
        Name of the exported file. This is used as a prefix. Output filter
        create the file `name`.xml that contains the metadata and a directory
        `name`_files that contains the rendered image files at different levels.
    root_directory : str
        Root directory where to place `name`.xml and `name`_files.
    tile_size : int, optional
        Size of individual tiles. (Default: 256)
    overlap : int, optional
        Overlap of tiles. (Default: 1)
    format : str, optional
        Image format. Note that PNG files have seems at the boundary between
        tiles. (Default: jpg)
    meta_format : str, optional
        Format for metadata information (the DZI file), can be 'xml' or
        'json'. (Default: 'xml')
    cmap : str or colormap, optional
        Color map for rendering the topography. (Default: None)

    Returns
    -------
    filenames : list of str
        List with names of files created during write operation
    """
    # Get reasonable unit
    ideal_height_unit = suggest_length_unit_for_data('linear', self.heights(), self.unit)
    t = self.to_unit(ideal_height_unit)
    return write_dzi(t.heights(), name, t.physical_sizes, ideal_height_unit, root_directory=root_directory,
                     tile_size=tile_size, overlap=overlap, format=format, meta_format=meta_format,
                     colorbar_title=f'Height ({ideal_height_unit})', cmap=cmap, **kwargs)


UniformTopographyInterface.register_function('to_dzi', write_topography_dzi)
