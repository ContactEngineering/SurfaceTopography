#
# Copyright 2023 Lars Pastewka
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
#
#

import numpy as np
from zipfile import ZipFile, BadZipFile

import dateutil.parser
import xmltodict

from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography


class PLUXReader(ReaderBase):
    _format = 'plux'
    _mime_types = ['application/x-sensofarx-spm']
    _file_extensions = ['plux']

    _name = 'Sensorfar XML SPM'
    _description = '''
This reader imports Sensofar's XML SPM file format.
'''

    # Data type of the binary container
    _DTYPE = np.dtype('f4')

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self._file_path = file_path
        with OpenFromAny(self._file_path, 'rb') as f:
            try:
                with ZipFile(f, 'r') as z:
                    try:
                        index_xml = z.read('index.xml')
                    except OSError as exc:
                        # This appears not to be a PLUX
                        raise FileFormatMismatch("ZIP file does not have 'index.xml'.") from exc

                    try:
                        recipe_xml = z.read('recipe.txt')
                    except OSError as exc:
                        # This appears not to be a PLUX
                        raise FileFormatMismatch("ZIP file does not have 'recipe.txt'.") from exc

                index_metadata = xmltodict.parse(index_xml)
                recipe_metadata = xmltodict.parse(recipe_xml)

                # Get main metadata dictionaries
                index_metadata = index_metadata['xml']
                recipe_metadata = recipe_metadata['xml']

                # Grid points
                nb_grid_pts_x = int(index_metadata['GENERAL']['IMAGE_SIZE_X'])
                nb_grid_pts_y = int(index_metadata['GENERAL']['IMAGE_SIZE_Y'])

                # Physical size
                physical_size_x = float(index_metadata['GENERAL']['FOV_X']) * nb_grid_pts_x
                physical_size_y = float(index_metadata['GENERAL']['FOV_Y']) * nb_grid_pts_y

                # Extract measurement date and raw metadata information
                info = {
                    'acquisition_time': str(dateutil.parser.parse(index_metadata['GENERAL']['DATE'])),
                    'raw_metadata': {
                        'index': index_metadata,
                        'recipe': recipe_metadata
                    }
                }

                # Extract instrument information
                for key, value in index_metadata['INFO'].items():
                    if isinstance(value, dict) and value['NAME'] == 'Device':
                        info['instrument'] = {'name': value['VALUE']}

                # Extract data layers
                self._channels = []
                layer = 0
                while f'LAYER_{layer}' in index_metadata.keys():
                    layer_info = index_metadata[f'LAYER_{layer}']
                    height_filename = layer_info['FILENAME_Z']

                    height_file_info = z.getinfo(height_filename)
                    expected_file_size = nb_grid_pts_x * nb_grid_pts_y * self._DTYPE.itemsize
                    if height_file_info.file_size != expected_file_size:
                        raise CorruptFile(f'Data file {height_filename} contains {height_file_info.file_size} bytes, '
                                          f'but the image size of {nb_grid_pts_x} x {nb_grid_pts_y} pixels requires '
                                          f'{expected_file_size} bytes.')

                    self._channels += [ChannelInfo(
                        self,
                        layer,  # channel index
                        name=f'Layer {layer}',
                        dim=2,
                        nb_grid_pts=(nb_grid_pts_x, nb_grid_pts_y),
                        physical_sizes=(physical_size_x, physical_size_y),
                        height_scale_factor=1,
                        uniform=True,
                        periodic=False,
                        unit='µm',  # Everything seems to be in micrometers
                        info=info,
                        tags={'height_filename': height_filename}
                    )]

                    layer += 1

            except BadZipFile as exc:
                # This is not an PLUX since it is not a ZIP file
                raise FileFormatMismatch('This is not a ZIP file.') from exc

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

        if channel_index != self._default_channel_index:
            raise RuntimeError(f'There is only a single channel. Channel index must be {self._default_channel_index}.')

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        channel = self.channels[channel_index]

        _info = channel._info.copy()
        _info.update(info)

        with OpenFromAny(self._file_path, 'rb') as f:
            with ZipFile(f, 'r') as z:
                nx, ny = channel.nb_grid_pts
                raw_data = z.read(channel.tags['height_filename'])
                height_data = np.frombuffer(raw_data, count=np.prod(channel.nb_grid_pts), dtype=self._DTYPE) \
                    .reshape(ny, nx).T

        if np.sum(np.isnan(height_data)) > 0:
            height_data = np.ma.masked_array(height_data, mask=np.isnan(height_data))

        topo = Topography(
            height_data,
            channel.physical_sizes,
            unit=channel.unit,
            info=_info,
            periodic=False if periodic is None else periodic)
        return topo.scale(channel.height_scale_factor)
