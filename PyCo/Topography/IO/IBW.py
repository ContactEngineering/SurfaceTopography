#
# Copyright 2019 Lars Pastewka
#           2019 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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

from PyCo.Topography.IO.Reader import ReaderBase, CorruptFile
from PyCo.Topography import Topography
from igor.binarywave import load as loadibw


class IBWReader(ReaderBase):

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):

        # depending from where this function is called, file_path might already be a filestream
        already_open = False
        if not hasattr(file_path, 'read'):
            f = open(file_path, "rb")
        else:
            already_open = True
            if str(type(file_path)) == "<class '_io.TextIOWrapper'>":
                # file was opened without the 'b' option, so read its buffer to get the binary data
                f = file_path.buffer  
            else:
                f = file_path

        # This catches and closes on its own
        try:
            file = loadibw(f)
        except Exception:
            if not f.closed:
                if not already_open:
                   f.close()
            raise RuntimeError('Invalid file format.')


        if file['version'] != 5:
            raise RuntimeError('Only IBW version 5 supported!')

        self.data = file['wave']

        # the first two labels are of x and y axis. we cannot read those
        self._channels = [channel.decode() for channel in self.data['labels'][2][1:]]
        self._default_channel = 0

    @property
    def channels(self):
        return [dict(name=self._channels[i],
                     dim=2,
                     unit=self.data['wave_header']['dataUnits'][i].decode()) for i in range(len(self._channels))]


    def topography(self, channel=None, physical_sizes=None, height_scale_factor=None, info={},
                   periodic=False, subdomain_locations=None, nb_subdomain_grid_pts=None):

        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        height_data = self.data['wData']

        if channel is None:
            channel = self._default_channel

        height_data = height_data[:, :, channel].copy()

        z_unit = self.data['wave_header']['dataUnits'][0].decode('latin-1')
        xy_unit = self.data['wave_header']['dimUnits'][0][0].decode('latin-1')

        assert z_unit == xy_unit

        sfA = self.data['wave_header']['sfA']
        nx, ny = height_data.shape

        if physical_sizes is None:
            physical_sizes = (nx * sfA[0], ny * sfA[1])
        
        surface = Topography(height_data, physical_sizes, info={**info, **dict(unit=z_unit)}, periodic=periodic)

        if height_scale_factor is not None:
           surface = surface.scale(height_scale_factor)

        return surface  