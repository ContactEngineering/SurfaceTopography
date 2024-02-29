#
# Copyright 2018-2024 Lars Pastewka
#           2018-2021 Michael Röttger
#           2019-2020 Antoine Sanner
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

"""
SurfaceTopography profile from file input
"""

import numpy as np

from ..UniformLineScanAndTopography import Topography
from .common import CHANNEL_NAME_INFO_KEY
from .Reader import ChannelInfo, ReaderBase


def binary(func):
    def func_wrapper(fobj, *args, **kwargs):
        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'rb')
            close_file = True
        try:
            retvals = func(fobj, *args, **kwargs)
        finally:
            if close_file:
                fobj.close()
        return retvals

    return func_wrapper


def make_wrapped_reader(reader_func, class_name='WrappedReader', format=None, mime_types=None, file_extensions=None,
                        name=None, description=None):
    class WrappedReader(ReaderBase):
        """
        emulates the new implementation of the readers
        """

        _format = format
        _mime_types = mime_types
        _file_extensions = file_extensions

        _name = name
        _description = description

        def __init__(self, fobj):
            self._fobj = fobj
            self._file_position = 0
            if callable(fobj):
                fobj = fobj()
            elif hasattr(fobj, 'tell'):
                self._file_position = fobj.tell()
            self._topography = reader_func(fobj)
            if CHANNEL_NAME_INFO_KEY in self._topography.info:
                self._channel_name = self._topography.info[CHANNEL_NAME_INFO_KEY]
                del self._topography.info[CHANNEL_NAME_INFO_KEY]
            else:
                self._channel_name = "Default"

        @property
        def channels(self):
            try:
                height_scale_factor = self._topography.height_scale_factor
            except AttributeError:
                height_scale_factor = None
                # None means: Not available in file

            return [ChannelInfo(
                self, 0,
                name=self._channel_name,
                dim=self._topography.dim,
                unit=self._topography.unit,
                uniform=self._topography.is_uniform,
                undefined_data=self._topography.has_undefined_data,
                info=self._topography.info,
                nb_grid_pts=self._topography.nb_grid_pts,
                physical_sizes=self._topography.physical_sizes,
                height_scale_factor=height_scale_factor)]

        def topography(self, channel_index=None, physical_sizes=None,
                       height_scale_factor=None, unit=None, info={}, periodic=False,
                       subdomain_locations=None, nb_subdomain_grid_pts=None):
            if channel_index is None:
                channel_index = self._default_channel_index

            if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
                raise RuntimeError(
                    'This reader does not support MPI parallelization.')

            if channel_index != 0:
                raise RuntimeError('Reader supports only a single channel 0.')

            physical_sizes = self._check_physical_sizes(physical_sizes, self._topography.physical_sizes)

            # Open file (if necessary)
            fobj = self._fobj
            if callable(fobj):
                fobj = fobj()

            # Rewind to position where the data is. Otherwise this method
            # cannot be called twice.
            if hasattr(fobj, 'seek'):
                fobj.seek(self._file_position)

            # Read again, but this time with physical_sizes and unit set (if not
            # specified in file)
            reader_kwargs = dict(height_scale_factor=height_scale_factor,
                                 unit=unit, info=info.copy(), periodic=periodic)
            if self._topography.physical_sizes is None:
                # file does not have physical sizes
                reader_kwargs['physical_sizes'] = physical_sizes
                # otherwise we won't add the argument, because that is not allowed any more

            return reader_func(fobj, **reader_kwargs)

        channels.__doc__ = ReaderBase.channels.__doc__
        topography.__doc__ = ReaderBase.topography.__doc__

    WrappedReader.__name__ = class_name
    return WrappedReader


@binary
def read_hgt(fobj, physical_sizes=None, height_scale_factor=None, unit=None, info={},
             periodic=False):
    """
    Read Shuttle Radar SurfaceTopography Mission (SRTM) topography data
    (.hgt extension).

    Keyword Arguments:
    fobj -- filename or file object
    """
    fobj.seek(0, 2)
    fsize = fobj.tell()
    fobj.seek(0)

    dim = int(np.sqrt(fsize / 2))
    if dim * dim * 2 != fsize:
        raise RuntimeError(
            'File physical_sizes of {0} bytes does not match file '
            'physical_sizes for a map of dimension {1}x{1}.'.format(fsize, dim))
    dtype = np.dtype('>i2')
    data = np.frombuffer(fobj.read(dim * dim * dtype.itemsize), dtype=dtype).reshape((dim, dim))

    if physical_sizes is None:
        topography = Topography(data, physical_sizes=tuple(float(x) for x in data.shape), unit=unit, info=info,
                                periodic=periodic)
    else:
        topography = Topography(data, physical_sizes=physical_sizes, unit=unit, info=info, periodic=periodic)
    if height_scale_factor is not None:
        topography = topography.scale(height_scale_factor)
    return topography


HGTReader = make_wrapped_reader(
    read_hgt, class_name="HGTReader", format='hgt', mime_types=['application/octet-stream'], file_extensions=['hgt'],
    name='NASA shuttle radar topography mission', description='''
Data format of the NASA shuttle radar topography mission that recorded the '
earths topography. More information can be found
[here](https://www2.jpl.nasa.gov/srtm/).
                                ''')
