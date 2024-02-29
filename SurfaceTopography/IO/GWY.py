#
# Copyright 2022-2024 Lars Pastewka
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
# Reference information and implementations:
# http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/gwyfile.c
#

import os
import re
from struct import calcsize, unpack

import numpy as np

from ..Exceptions import FileFormatMismatch, MetadataAlreadyFixedByFile
from ..Support.UnitConversion import get_unit_conversion_factor, is_length_unit
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase


def _read_null_terminated_string(f):
    return b''.join(iter(lambda: f.read(1), b'\x00')).decode('utf-8')


def _gwy_read_atomic(f, atomic_type):
    atomic_size = calcsize('<' + atomic_type)
    value, = unpack('<' + atomic_type, f.read(atomic_size))
    if atomic_type == 'b':  # booleans are special
        return value != 0
    return value


_gwy_atomic_readers = {
    'b': lambda f, **kwargs: _gwy_read_atomic(f, 'b'),
    'c': lambda f, **kwargs: _gwy_read_atomic(f, 'c'),
    'i': lambda f, **kwargs: _gwy_read_atomic(f, 'i'),
    'q': lambda f, **kwargs: _gwy_read_atomic(f, 'q'),
    'd': lambda f, **kwargs: _gwy_read_atomic(f, 'd'),
    's': lambda f, **kwargs: _read_null_terminated_string(f)
}


def _gwy_read_array(f, atomic_type, skip_arrays=False):
    offset = f.tell()
    nb_items, = unpack('<L', f.read(4))
    type = np.dtype(atomic_type)
    if skip_arrays:
        # Skip reading this data
        f.seek(type.itemsize * nb_items, os.SEEK_CUR)
        return {'offset': offset, 'type': atomic_type}  # If we skip reading the array, return the file offset
    else:
        return np.frombuffer(f.read(nb_items * type.itemsize), dtype=type)


def _gwy_read_string_array(f):
    nb_items, = unpack('<L', f.read(4))
    return [_read_null_terminated_string(f) for i in range(nb_items)]


_gwy_array_readers = {
    'C': lambda f, **kwargs: _gwy_read_array(f, 'c', **kwargs),
    'I': lambda f, **kwargs: _gwy_read_array(f, 'i', **kwargs),
    'Q': lambda f, **kwargs: _gwy_read_array(f, 'q', **kwargs),
    'D': lambda f, **kwargs: _gwy_read_array(f, 'd', **kwargs),
    'S': lambda f, **kwargs: _gwy_read_string_array(f)
}


def _gwy_read_component(f, skip_arrays=False):
    """
    Read a single component from a GWY file.

    Parameters
    ----------
    f : stream-like
        The file stream to read from.
    skip_arrays : bool, optional
        Skip reading arrays to avoid reading image data.
        (Default: False)

    Returns
    -------
    data : dict
        Dictionary containing the decoded data.
    """
    name = _read_null_terminated_string(f)
    type = f.read(1).decode('ascii')
    return {name: _gwy_readers[type](f, skip_arrays=skip_arrays)}


def _gwy_read_object(f, skip_arrays=False):
    """
    Read a single object from a GWY file.

    Parameters
    ----------
    f : stream-like
        The file stream to read from.
    skip_arrays : bool, optional
        Skip reading arrays to avoid reading image data.
        (Default: False)

    Returns
    -------
    data : dict
        Dictionary containing the decoded data.
    """
    name = _read_null_terminated_string(f)
    size, = unpack('<L', f.read(4))
    start = f.tell()
    data = {}
    while f.tell() < start + size:
        data.update(_gwy_read_component(f, skip_arrays=skip_arrays))
    return {name: data}


def _gwy_read_object_array(f):
    nb_items = unpack('<L', f.read(4))
    return [_gwy_read_object(f) for i in range(nb_items)]


_gwy_readers = {
    **_gwy_atomic_readers,
    **_gwy_array_readers,
    'o': _gwy_read_object,
    'O': _gwy_read_object_array
}


class GWYReader(ReaderBase):
    _format = 'gwy'
    _mime_types = ['application/x-gwyddion-spm']
    _file_extensions = ['gwy']

    _name = 'Gwyddion'
    _description = '''
This reader imports the native file format of the open-source SPM
visualization and analysis software Gwyddion.
'''

    _MAGIC = b'GWYP'

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Detect file magic
            magic = f.read(4)
            if magic != self._MAGIC:
                raise FileFormatMismatch('File magic does not match. This is not Gwyddion file.')

            # Read native metadata dictionary
            gwy = _gwy_read_object(f, skip_arrays=True)
            self._metadata = gwy['GwyContainer']

            # Construct channels
            self._channels = []
            for key, value in self._metadata.items():
                if key.endswith('/data'):
                    index = int(re.match(r'\/([0-9])\/data', key)[1])
                    data = value['GwyDataField']

                    # It's not height data if 'si_unit_z' is missing.
                    if 'si_unit_z' in data:
                        # Get number of grid points
                        nb_grid_pts = [data['xres']]
                        if 'yres' in data:
                            nb_grid_pts += [data['yres']]

                        # Get physical sizes
                        physical_sizes = [data['xreal']]
                        if 'yreal' in data:
                            physical_sizes += [data['yreal']]

                        assert len(nb_grid_pts) == len(physical_sizes)

                        xyunit = data['si_unit_xy']['GwySIUnit']['unitstr']
                        zunit = data['si_unit_z']['GwySIUnit']['unitstr']

                        if is_length_unit(zunit):
                            # This is height data!
                            self._channels += [ChannelInfo(
                                self,
                                len(self._channels),
                                name=self._metadata[f'/{index}/data/title'],
                                dim=len(nb_grid_pts),
                                nb_grid_pts=tuple(nb_grid_pts),
                                physical_sizes=tuple(physical_sizes),
                                unit=xyunit,
                                height_scale_factor=get_unit_conversion_factor(zunit, xyunit),
                                periodic=False,
                                uniform=True,
                                info={key: value
                                      for key, value in self._metadata.items()
                                      if key.startswith(f'/{index}/')},
                                tags={'data': data['data'], 'index': index}
                            )]

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        if channel_index is None:
            channel_index = self._default_channel_index

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        channel = self._channels[channel_index]
        with OpenFromAny(self.file_path, 'rb') as f:
            nx, ny = channel.nb_grid_pts
            f.seek(channel.tags['data']['offset'])
            height_data = _gwy_read_array(f, channel.tags['data']['type']).reshape((ny, nx)).T[:, ::-1]

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          channel.physical_sizes,
                          unit=channel.unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(channel.height_scale_factor)
