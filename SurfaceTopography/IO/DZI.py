#
# Copyright 2021 Lars Pastewka
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

import math
import os
import xml.etree.cElementTree as ET

import numpy as np
from matplotlib import cm
from PIL import Image

from ..HeightContainer import UniformTopographyInterface


def write_dzi(self, name, root_directory='.', tile_size=256, overlap=1, format='jpg', cmap=None):
    """
    Write topography to a Deep Zoom Image file. This can for example be used
    to create a zoomable topography with OpenSeadragon
    (https://openseadragon.github.io/).

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
    cmap : str or colormap, optional
        Color map for rendering the topography. (Default: None)
    """
    cmap = cm.get_cmap(cmap)

    # Image size
    full_width, full_height = width, height = self.nb_grid_pts

    # Get heights and rescale to interval 0, 1
    heights = self.heights()
    mx, mn = self.max(), self.min()
    heights = (heights - mn) / (mx - mn)

    # Write configuration XML file
    root = ET.Element('Image', TileSize=str(tile_size), Overlap=str(overlap), Format=format,
                      xmlns='http://schemas.microsoft.com/deepzoom/2008')
    ET.SubElement(root, 'Size', Width=str(width), Height=str(height))
    os.makedirs(root_directory, exist_ok=True)
    ET.ElementTree(root).write(os.path.join(root_directory, name + '.xml'), encoding='utf-8', xml_declaration=True)

    # Determine number of levels
    max_level = math.ceil(math.log2(max(width, height)))

    # Loop over levels and write tiles
    root_directory = os.path.join(root_directory, name + '_files')
    os.makedirs(root_directory, exist_ok=True)
    step = 1
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
                left = (column * tile_size - overlap) * step
                bottom = (row * tile_size - overlap) * step

                right = ((column + 1) * tile_size + overlap) * step
                top = ((row + 1) * tile_size + overlap) * step

                if left < 0:
                    left = 0
                if bottom < 0:
                    bottom = 0
                if right > full_width - 1:
                    right = full_width - 1
                if top > full_height - 1:
                    top = full_height - 1

                # Convert to image and save
                colors = (cmap(heights[left:right:step, bottom:top:step].T) * 255).astype(np.uint8)
                # Remove alpha channel before writing
                Image.fromarray(colors[:, :, :3]).save(fn)

        width = math.ceil(width / 2)
        height = math.ceil(height / 2)
        step *= 2


UniformTopographyInterface.register_function('to_dzi', write_dzi)
