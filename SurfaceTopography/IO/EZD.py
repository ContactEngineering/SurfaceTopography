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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/ezdfile.c
#

import datetime

import numpy as np

from .common import OpenFromAny
from ..Exceptions import FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import is_length_unit

from .Reader import ReaderBase, ChannelInfo


###

class EZDReader(ReaderBase):
    _format = 'ezd'
    _mime_types = ['application/x-nanosurf-spm']
    _file_extensions = ['ezd', 'nid']

    _name = 'NanoSurf easyScan data file'
    _description = '''
NanoSurf easyScan data file with typical file extension .ezd/.nid
'''

    _MAGIC = '[DataSet]\r\n'
    _DATA_MAGIC = b'#!'

    def __init__(self, file_path):
        """
        Load NanoSurf easyScan data files.

        Arguments
        ---------
        file_path : filename or file object
             File or data stream to open.
        """
        self._file_path = file_path

        # The start of the file is textual with metadata; we need to parse it
        with OpenFromAny(self._file_path, 'rb') as fobj:
            metadata = {}
            line = fobj.readline()
            if line == self._MAGIC:
                raise FileFormatMismatch('This is not a NanoSurf easyScan data file')

            section_name = 'DataSet'
            section_metadata = {}
            p = fobj.read(2)
            while p and not p[:len(self._DATA_MAGIC)] == self._DATA_MAGIC:
                line = (p + fobj.readline()).decode('latin-1').strip()
                if len(line) > 0:
                    if line.startswith('[') and line.endswith(']'):
                        # This starts a new section
                        metadata[section_name] = section_metadata
                        # Store new section header
                        section_name = line[1:-1]
                        section_metadata = {}
                    else:
                        key, value = line.split('=', 1)
                        section_metadata[key] = value

                # Store file position where binary data blocks start
                self._start_of_data = fobj.tell() + 2

                p = fobj.read(2)
            metadata[section_name] = section_metadata

        # Acquisition time
        acquisition_time = \
            str(datetime.datetime.combine(
                datetime.datetime.strptime(metadata['DataSet-Info']['Date'], r"%d-%m-%Y").date(),
                datetime.datetime.strptime(metadata['DataSet-Info']['Time'], r"%H:%M:%S").time()
            ))

        # Turn metadata in channel information
        self._channels = []
        global_metadata = metadata['DataSet']
        nb_groups = int(global_metadata['GroupCount'])
        channel_index = 0

        # Offset of data block
        offset = 0

        # Loop over all groups and all channels
        for i in range(nb_groups):
            nb_datasets = int(global_metadata[f'Gr{i}-Count'])
            for j in range(nb_datasets):
                # Entries may be missing
                try:
                    section_name = global_metadata[f'Gr{i}-Ch{j}']
                    dataset_metadata = metadata[section_name]
                except KeyError:
                    dataset_metadata = None

                # Missing entry? Continue with next one
                if dataset_metadata is None:
                    continue

                if dataset_metadata['SaveMode'] != 'Binary':
                    raise NotImplementedError("Cannot handle data set with 'SaveMode' other than 'Binary'.")

                # Compute size of data block
                nb_grid_pts = (int(dataset_metadata['Points']), int(dataset_metadata['Lines']))
                byte_order = '<' if dataset_metadata['SaveOrder'] == 'Intel' else '>'
                data_type = 'i' if dataset_metadata['SaveSign'] == 'Signed' else 'u'
                nb_bytes = int(dataset_metadata['SaveBits']) // 8
                dtype = np.dtype(f'{byte_order}{data_type}{nb_bytes}')

                if is_length_unit(dataset_metadata['Dim2Unit']):
                    name = f"{dataset_metadata['Frame']} ({dataset_metadata['Dim2Name']})"
                    physical_sizes = (float(dataset_metadata['Dim0Range']), float(dataset_metadata['Dim1Range']))
                    self._channels += [
                        ChannelInfo(self,
                                    channel_index,  # channel index
                                    name=name,
                                    dim=2,
                                    nb_grid_pts=nb_grid_pts,
                                    physical_sizes=physical_sizes,
                                    uniform=True,
                                    unit=dataset_metadata['Dim2Unit'],
                                    height_scale_factor=float(dataset_metadata['Dim2Range']) / 2 ** (8 * nb_bytes),
                                    info={
                                        'acquisition_time': acquisition_time,
                                        'raw_metadata': dataset_metadata
                                    },
                                    tags={
                                        'dtype': dtype,
                                        'offset': offset,
                                        'height_offset': float(dataset_metadata['Dim2Min'])  # Currently unused
                                    })
                    ]
                    channel_index += 1

                offset += np.prod(nb_grid_pts) * dtype.itemsize
        self._metadata = metadata

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={}, periodic=False,
                   subdomain_locations=None, nb_subdomain_grid_pts=None):

        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError(
                'This reader does not support MPI parallelization.')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        channel = self._channels[channel_index]
        with OpenFromAny(self._file_path, 'rb') as fobj:
            sx, sy = self._check_physical_sizes(physical_sizes,
                                                channel.physical_sizes)

            nx, ny = channel.nb_grid_pts

            offset = channel.tags['offset']
            dtype = channel.tags['dtype']

            fobj.seek(self._start_of_data + offset)
            rawdata = fobj.read(nx * ny * dtype.itemsize)
            unscaleddata = np.frombuffer(rawdata, count=nx * ny, dtype=dtype).reshape(nx, ny)

        # internal information from file
        _info = channel.info.copy()
        _info.update(info)

        # it is not allowed to provide extra `physical_sizes` here:
        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        # the orientation of the heights is modified in order to match
        # the image of gwyddion when plotted with imshow(t.heights().T)
        # or pcolormesh(t.heights().T) for origin in lower left and
        # with inverted y axis (cartesian coordinate system)

        topography = Topography(np.fliplr(unscaleddata.T), physical_sizes=(sx, sy), unit=channel.unit, info=_info,
                                periodic=periodic)
        if height_scale_factor is None:
            height_scale_factor = channel.height_scale_factor
        elif channel.height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')
        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
