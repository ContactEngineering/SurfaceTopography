#
# Copyright 2019-2021 Lars Pastewka
#           2019-2021 Michael Röttger
#           2019 Kai Haase
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

import numpy as np

from igor.binarywave import load as loadibw

from ..UniformLineScanAndTopography import Topography

from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo, MetadataAlreadyFixedByFile


class IBWReader(ReaderBase):
    _format = 'ibw'
    _name = 'Igor binary wave'
    _description = '''
Igor binary wave is a container format of the
[Igor Pro](https://www.wavemetrics.com/products/igorpro/programming)
language. This format is used by AFMs from Asylum Research (now Oxford
Instruments) to store topography information. This format contains information
on the physical size of the topography map as well as its units.
'''

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        with OpenFromAny(file_path, 'rb') as f:
            file = loadibw(f)

        if file['version'] != 5:
            raise RuntimeError('Only IBW version 5 is supported!')

        self.data = file['wave']

        # the first two labels are of x and y axis. we cannot read those
        self._channel_names = [channel.decode() for channel in
                               self.data['labels'][2][1:]]
        self._default_channel = 0

        #
        # Read physical sizes from file, since all data
        # has already been loaded, we can calculate it here
        #
        height_data = self.data['wData']

        # TODO is it always like this?
        assert len(height_data.shape) == 3, \
            "We expect all channels being coded in to one wave and all are " \
            "2D. This is not true somehow, cannot proceed."
        nx, ny, num_channels = height_data.shape

        # ensure that there are not too many channel names
        self._channel_names = self._channel_names[:num_channels]

        # add channel names for all channels without name
        no_name_idx = 1
        while len(self._channel_names) < num_channels:
            self._channel_names.append("no name ({})".format(no_name_idx))
            no_name_idx += 1

        #
        # Decode units
        #
        def decode_unit_entry(u):
            return u.tobytes().partition(b'\0')[0].decode('latin-1')

        data_unit = decode_unit_entry(self.data['wave_header']['dataUnits'])
        x_unit = decode_unit_entry(self.data['wave_header']['dimUnits'][0])
        y_unit = decode_unit_entry(self.data['wave_header']['dimUnits'][1])

        # the following is not necessary, we could handle different units by
        # rescaling, however I'll leave it like this for now, print some
        # message so we know what to do if a file occurs which does not fulfill
        # this assumption
        assert data_unit == x_unit == y_unit, \
            "So far, data units and dimension units must be all the same. " +\
            "data unit: '{}', x unit: '{}', y unit: '{}'".format(data_unit,
                                                                 x_unit,
                                                                 y_unit)

        # An empty unit should be None
        if data_unit == '':
            data_unit = None

        self._data_unit = data_unit

        #
        # Decode sizes
        #
        sfA = self.data['wave_header']['sfA']
        self._physical_sizes = (nx * sfA[0], ny * sfA[1])
        # Comment in C header file on these fields: Index value for element e
        # of dimension d = sfA[d]*e + sfB[d]. sfB is left out here, because we
        # are interested in the width and height, not the absolute offsets.

        #
        # Build channel information
        #
        self._channels = [
            ChannelInfo(self, i, name=cn, dim=2, nb_grid_pts=(nx, ny), physical_sizes=self._physical_sizes,
                        unit=self._data_unit, height_scale_factor=1)
            for i, cn in enumerate(self._channel_names)]

        # Shall we use the channel names in order to assign a unit as Gwyddion
        # does?

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        if channel_index is None:
            channel_index = self._default_channel_index

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError(
                'This reader does not support MPI parallelization.')

        height_data = self.data['wData']
        height_data = np.fliplr(height_data[:, :, channel_index].copy())

        if physical_sizes is None:
            physical_sizes = self._physical_sizes
        else:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        topo = Topography(height_data, physical_sizes, unit=self._data_unit, info=info, periodic=periodic)
        # we could pass the data units here, but they dont seem to be always
        # correct for all channels?!

        channel = self._channels[channel_index]
        if channel.height_scale_factor is not None:
            if height_scale_factor is not None:
                raise MetadataAlreadyFixedByFile('height_scale_factor')
            height_scale_factor = channel.height_scale_factor

        if height_scale_factor is not None:
            topo = topo.scale(height_scale_factor)

        return topo
