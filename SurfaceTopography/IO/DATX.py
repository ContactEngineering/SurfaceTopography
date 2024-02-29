#
# Copyright 2023-2024 Lars Pastewka
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

# Reference information and implementations:
# https://gist.github.com/g-s-k/ccffb1e84df065a690e554f4b40cfd3a

import h5py
import numpy as np

from ..Exceptions import (CorruptFile, FileFormatMismatch,
                          MetadataAlreadyFixedByFile, UnsupportedFormatFeature)
from ..Support.UnitConversion import (get_unit_conversion_factor,
                                      mangle_length_unit_utf8)
from ..UniformLineScanAndTopography import Topography
from .Reader import ChannelInfo, ReaderBase


class DATXReader(ReaderBase):
    _format = 'datx'
    _mime_types = ['application/x-hdf']
    _file_extensions = ['datx']

    _name = 'Zygo DATX'
    _description = '''
Import filter for Zygo DATX, an HDF5-based format.
    '''  # noqa: E501

    def __init__(self, fobj):
        self._fobj = fobj
        if callable(fobj):
            fobj = fobj()
        try:
            with h5py.File(fobj, 'r') as h5:
                # Check if this can be a datx file
                if 'MetaData' not in h5.keys():
                    raise FileFormatMismatch('Cannot read Zygo DATX. This is an HDF5 file, but the `MetaData` toplevel '
                                             'key is missing.')

                # Read file structure
                raw_metadata = {}
                for category, key, value in h5['/MetaData']:
                    category = category.decode('ascii')
                    key = key.decode('ascii')
                    value = value.decode('ascii')
                    if category not in raw_metadata:
                        raw_metadata[category] = {}
                    raw_metadata[category][key] = value

                # Turn into nested dictionary
                def recursive_replace(raw_metadata, context):
                    metadata = {}
                    for key, value in context.items():
                        if value.startswith('{') and value.endswith('}'):
                            metadata[key] = recursive_replace(raw_metadata, raw_metadata[value])
                        else:
                            metadata[key] = value
                    return metadata

                self._metadata = recursive_replace(raw_metadata, raw_metadata['Root'])

                # Get the path for the actual measurement with the HDF5 file
                self._surface_path = self._metadata['Measurement']['Surface']['Path']

                # Grid points
                self._nb_grid_pts = h5[self._surface_path].shape

                # Value for missing data points
                self._no_data, = h5[self._surface_path].attrs['No Data']

                # Unit
                self._unit, = h5[self._surface_path].attrs['Unit']
                self._unit = mangle_length_unit_utf8(self._unit)

                # Conversion
                x_converter, = h5[self._surface_path].attrs['X Converter']
                y_converter, = h5[self._surface_path].attrs['Y Converter']
                z_converter, = h5[self._surface_path].attrs['Z Converter']

                # Interpret converters and throw error if unsupported
                category, unit, values = x_converter
                if category != b'LateralCat' and unit != b'Pixels':
                    raise UnsupportedFormatFeature('DATX reader only supports `LateralCat` with `Pixels` unit for '
                                                   'X converter.')
                physical_sizes_x = self._nb_grid_pts[0] * values[1]  # in units of meters!

                category, unit, values = y_converter
                if category != b'LateralCat' and unit != b'Pixels':
                    raise UnsupportedFormatFeature('DATX reader only supports `LateralCat` with `Pixels` unit for '
                                                   'Y converter.')
                physical_sizes_y = self._nb_grid_pts[1] * values[1]  # in units of meters!

                category, height_unit, values = z_converter
                if category != b'HeightCat':
                    raise UnsupportedFormatFeature('DATX reader only supports `LateralCat` for Z converter.')
                height_unit = mangle_length_unit_utf8(height_unit.decode('ascii'))

                if height_unit != self._unit:
                    raise CorruptFile(f"Two conflicting height units found: '{height_unit}' and '{self._unit}'.")

                meters_to_unit = get_unit_conversion_factor('m', self._unit)
                self._physical_sizes = (physical_sizes_x * meters_to_unit, physical_sizes_y * meters_to_unit)

        except OSError:
            # This is not an HDF5 file
            raise FileFormatMismatch('Cannot read Zygo DATX. The file is not an HDF5 container.')

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',  # There is only a single channel
                            dim=2,
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            uniform=True,
                            unit=self._unit,
                            height_scale_factor=1,  # Data is in natural heights
                            info={'raw_metadata': self._metadata})]

    def topography(self, channel_index=None, physical_sizes=None, height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None, nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        if channel_index is None:
            channel_index = self._default_channel_index

        if channel_index != 0:
            raise RuntimeError('Channel index must be zero. (DATX files only have a single height channel.)')

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        _info = {'raw_metadata': self._metadata}
        _info.update(info)

        fobj = self._fobj
        if callable(fobj):
            fobj = fobj()
        with h5py.File(fobj, 'r') as h5:
            raw_data = np.array(h5[self._surface_path])
            mask = raw_data == self._no_data
            if mask.sum() > 0:
                # We need to mask this array
                raw_data = np.ma.masked_array(raw_data, mask=mask)
            t = Topography(raw_data, self._physical_sizes, unit=self._unit, info=_info,
                           periodic=periodic)

        return t.scale(1)
