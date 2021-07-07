#
# Copyright 2021 Lars Pastewka
#           2021 Michael Röttger
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

"""
Reader for Keyence ZON files.
"""

# Thanks to @mcmalburg (https://github.com/mcmalburg) for reverse engineering the
# format. See discussion here https://github.com/gabeguss/Keyence/issues/2

import numpy as np
from numpy.lib.stride_tricks import as_strided
from struct import unpack
from zipfile import ZipFile

import defusedxml.ElementTree as ElementTree

from ..UniformLineScanAndTopography import Topography

from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo, MetadataAlreadyFixedByFile

# The files within ZON (zip) files are named using UUIDs. Some of these
# UUIDs are fixed and contain the same information in each of these files.

# This file contains height data
HEIGHT_DATA_UUID = '4cdb0c75-5706-48cc-a9a1-adf395d609ae'

# This contains information on unit conversion
UNIT_UUID = '686613b8-27b5-4a29-8ffc-438c2780873e'

# This contains an inventory of *image* data
INVENTORY_UUID = '772e6d38-40aa-4590-85d3-b041fa243570'


def _read_array(f, dtype=np.dtype('<i4')):
    """
    Read binary array contained in a ZON archive

    Arguments
    ---------
    f : file object
        Stream to read array from
    dtype : numpy.dtype, optional
        Data type of individual element.
        (Default: 32-bit integer)
    """
    width, height, element_size, row_bytes = unpack('iiii', f.read(16))
    if element_size % dtype.itemsize != 0:
        raise ValueError(f'File report element size of {element_size} bytes, '
                         f'but requested data type requires {dtype.itemsize} bytes.')
    if row_bytes % dtype.itemsize != 0:
        raise ValueError(f'File reports {row_bytes} bytes per row, but this is not an integer multiple of the data '
                         f'type of size {dtype.itemsize} bytes.')
    raw_data = np.frombuffer(f.read(element_size * height * (row_bytes // dtype.itemsize)), dtype)

    nb_entries = element_size // dtype.itemsize
    if nb_entries == 1:
        array_data = as_strided(raw_data, shape=(width, height), strides=(dtype.itemsize, row_bytes))
    else:
        array_data = as_strided(raw_data, shape=(width, height, nb_entries),
                                strides=(dtype.itemsize, element_size, row_bytes))
    return array_data


class ZONReader(ReaderBase):
    _format = 'zon'
    _name = 'Keyence ZON'
    _description = '''
This reader open ZON files that are written by some Keyence instruments.
'''

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self._file_path = file_path

        # ZON files are ZIP files with a header. The header contains a
        # thumbnail of the measurement and we are not really interested
        # in that one. Python's ZipFile automatically skips that header.

        self._channels = []
        with OpenFromAny(self._file_path, 'rb') as f:
            with ZipFile(f, 'r') as z:
                # Parse unit information
                root = ElementTree.parse(z.open(UNIT_UUID)).getroot()
                meter_per_pixel = float(root.find('XYCalibration').find('MeterPerPixel').text)
                meter_per_unit = float(root.find('ZCalibration').find('MeterPerUnit').text)

                self._orig_height_scale_factor = meter_per_unit

                # Parse height data information
                # Header consists of four int32, followed by image data
                width, height, element_size = unpack('iii', z.open(HEIGHT_DATA_UUID).read(12))
                assert element_size == 4
                self._channels += [
                    ChannelInfo(self, 0, name='default', dim=2, nb_grid_pts=(width, height),
                                physical_sizes=(width * meter_per_pixel, height * meter_per_pixel),
                                height_scale_factor=self._orig_height_scale_factor, unit='m',
                                info={'data_uuid': HEIGHT_DATA_UUID,
                                      'meter_per_pixel': meter_per_pixel,
                                      'meter_per_unit': meter_per_unit})]

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None, height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None, nb_subdomain_grid_pts=None):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        channel_info = self._channels[channel_index]
        physical_sizes = self._check_physical_sizes(physical_sizes, channel_info.physical_sizes)

        info.update(channel_info.info)

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')
        unit = channel_info.unit

        with OpenFromAny(self._file_path, 'rb') as f:
            # Read image data
            with ZipFile(f, 'r') as z:
                with z.open(channel_info.info['data_uuid']) as f:
                    height_data = _read_array(f)

        topo = Topography(height_data, physical_sizes, unit=unit, info=info, periodic=periodic)

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        return topo.scale(self._orig_height_scale_factor)
