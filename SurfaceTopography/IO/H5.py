#
# Copyright 2019-2021 Lars Pastewka
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


class H5Reader(ReaderBase):
    _format = 'h5'
    _name = 'Hierarchical data format (HDF5)'
    _description = '''
Import filter for [HDF5](https://support.hdfgroup.org/HDF5/) files provided
within the contact mechanics challenge. The reader looks for a two-dimensional
array named `surface`. HDF5 files do not store units or physical sizes. These
need to be manually provided by the user.

The original contact mechanics challenge data can be downloaded
[here](https://www.lmp.uni-saarland.de/index.php/research-topics/contact-mechanics-challenge-announcement/).
    '''  # noqa: E501

    def __init__(self, fobj):
        self._h5 = None
        import h5py
        self._h5 = h5py.File(fobj, 'r')

    def close(self):
        if self._h5 is not None:
            self._h5.close()

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=len(self._h5['surface'].shape),
                            nb_grid_pts=self._h5['surface'].shape)]

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, info={}, periodic=False,
                   subdomain_locations=None, nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError(
                'This reader does not support MPI parallelization.')
        if channel_index is None:
            channel_index = self._default_channel_index
        if channel_index != 0:
            raise RuntimeError('HDF5 reader only supports a single channel')
        size = self._check_physical_sizes(physical_sizes)
        t = Topography(self._h5['surface'][...], size, info=info,
                       periodic=periodic)
        if height_scale_factor is not None:
            t = t.scale(height_scale_factor)
        return t

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
