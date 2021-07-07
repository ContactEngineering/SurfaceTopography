#
# Copyright 2019-2021 Lars Pastewka
#           2021 Michael Röttger
#           2019 Antoine Sanner
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

from ..UniformLineScanAndTopography import Topography
from .Reader import ReaderBase, ChannelInfo


class MatReader(ReaderBase):
    _format = 'mat'
    _name = 'MATLAB'
    _description = '''
Imports topography data stored in MATLAB workspace files. The reader
automatically extracts all 2D arrays stored in the file and interprets those
as height information. Matlab files do not store units or physical sizes.
These need to be manually provided by the user.
    '''

    def __init__(self, fobj):
        """
        Reads a surface profile from a Matlab file and presents in in a
        SurfaceTopography-conformant manner.

        All two-dimensional arrays present in the matlab data file are
        returned.

        Parameters
        ----------
        fobj: filename or file object
             File to read.
        """
        from scipy.io import loadmat

        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'rb')
            close_file = True
        try:
            data = loadmat(fobj)
            self._channels = []
            self._height_data = []
            for key, value in data.items():
                is_2d_array = False
                try:
                    nx, ny = value.shape
                    is_2d_array = True
                except (AttributeError, ValueError):
                    pass
                if is_2d_array:
                    channel_info = ChannelInfo(self,
                                               len(self._channels),
                                               name=key,
                                               dim=len(value.shape),
                                               nb_grid_pts=value.shape)
                    # no height scale factor given in mat file

                    self._channels.append(channel_info)
                    self._height_data.append(value)
        finally:
            if close_file:
                fobj.close()

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

        info = info.copy()
        info['data_source'] = self.channels[channel_index].name

        topography = Topography(
            self._height_data[channel_index], physical_sizes=self._check_physical_sizes(physical_sizes), unit=unit,
            info=info, periodic=periodic)

        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)

        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
