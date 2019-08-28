#
# Copyright 2019 Lars Pastewka
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

import re

import numpy as np

from PyCo.Topography import Topography

from .FromFile import get_unit_conversion_factor, height_units, mangle_height_unit
from .Reader import ReaderBase


###

class DIReader(ReaderBase):
    def __init__(self, fobj):
        """
        Load Digital Instrument's Nanoscope files.

        Arguments
        ---------
        fobj : filename or file object
             File or data stream to open.
        """
        self._fobj = fobj
        close_file = False
        if not hasattr(fobj, 'read'):
            fobj = open(fobj, 'rb')
            close_file = True
        try:
            parameters = []
            section_name = None
            section_dict = {}

            l = fobj.readline().decode('latin-1').strip()
            while l and l.lower() != r'\*file list end':
                if l.startswith('\\*'):
                    if section_name is not None:
                        parameters += [(section_name, section_dict)]
                    section_name = l[2:].lower()
                    section_dict = {}
                elif l.startswith('\\'):
                    s = l[1:].split(': ', 1)
                    try:
                        key, value = s
                    except ValueError:
                        key, = s
                        value = ''
                    section_dict[key.lower()] = value.strip()
                else:
                    raise IOError("Header line '{}' does not start with a slash."
                                  "".format(l))
                l = fobj.readline().decode('latin-1').strip()
            parameters += [(section_name, section_dict)]

            self._info = {}
            self._channels = []
            self._offsets = []
            self._default_channel = 0

            scanner = {}
            i = 0
            for n, p in parameters:
                if n == 'scanner list' or n == 'ciao scan list':
                    scanner.update(p)
                elif n == 'ciao image list':
                    image_data_key = re.match(r'^S \[(.*?)\] ',
                                              p['@2:image data']).group(1)

                    nx = int(p['samps/line'])
                    ny = int(p['number of lines'])

                    s = p['scan size'].split(' ', 2)
                    sx = float(s[0])
                    sy = float(s[1])

                    xy_unit = mangle_height_unit(s[2])
                    offset = int(p['data offset'])
                    self._offsets.append(offset)

                    length = int(p['data length'])
                    elsize = int(p['bytes/pixel'])
                    if elsize != 2:
                        raise IOError("Don't know how to handle {} bytes per pixel "
                                      "data.".format(elsize))
                    if nx * ny * elsize != length:
                        raise IOError('Data block physical_sizes differs from extend of surface.')

                    scale_re = re.match(
                        r'^V \[(.*?)\] \(([0-9\.]+) (.*)\/LSB\) (.*) '
                        r'(.*)', p['@2:z scale'])
                    quantity = scale_re.group(1).lower()
                    hard_scale = float(scale_re.group(4)) / 65536
                    hard_unit = scale_re.group(5)

                    s = scanner['@' + quantity].split()
                    if s[0] != 'V' or len(s) < 2:
                        raise ValueError('Malformed Nanoscope DI file.')
                    soft_scale = float(s[1])

                    height_unit = None
                    hard_to_soft = 1.0
                    if len(s) > 2:
                        # Check units
                        height_unit, soft_unit = s[2].split('/')
                        hard_to_soft = get_unit_conversion_factor(hard_unit,
                                                                  soft_unit)
                        if hard_to_soft is None:
                            raise ValueError("Units for hard (={}) and soft (={}) "
                                             "scale differ for '{}'. Don't know how "
                                             "to handle this.".format(hard_unit,
                                                                      soft_unit,
                                                                      image_data_key))
                    if height_unit in height_units:
                        height_unit = mangle_height_unit(height_unit)
                        if xy_unit != height_unit:
                            fac = get_unit_conversion_factor(xy_unit, height_unit)
                            sx *= fac
                            sy *= fac
                            xy_unit = height_unit
                        unit = height_unit

                        # default channel is 0 or the first channel where height_unit
                        # is a length
                        if self._default_channel == 0:
                            self._default_channel = i
                    else:
                        unit = (xy_unit, height_unit)

                    channel_dict = {}
                    channel_dict["name"] = image_data_key
                    channel_dict["dim"] = 2
                    channel_dict["nb_grid_pts"] = (nx, ny)
                    channel_dict["physical_sizes"] = (sx, sy)
                    channel_dict["unit"] = unit
                    # channel_dict["height_unit"] = height_unit
                    # channel_dict["soft_unit"] = soft_unit
                    # channel_dict["hard_unit"] = hard_unit
                    # channel_dict["xy_unit"] = xy_unit
                    channel_dict["height_scale_factor"] = hard_scale * hard_to_soft * soft_scale

                    self._channels.append(channel_dict)
                    i += 1
        finally:
            if close_file:
                fobj.close()

    @property
    def channels(self):
        return self._channels

    def topography(self, channel=None, physical_sizes=None, height_scale_factor=None, info={},
                   subdomain_locations=None, nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')
        close_file = False
        if not hasattr(self._fobj, 'read'):
            fobj = open(self._fobj, 'rb')
            close_file = True
        else:
            fobj = self._fobj

        if channel is None:
            channel = self.default_channel

        channel_dict = self._channels[channel]
        sx, sy = self._check_physical_sizes(physical_sizes, channel_dict["physical_sizes"])

        nx, ny = channel_dict["nb_grid_pts"]

        offset = self._offsets[channel]
        dtype = np.dtype('<i2')

        ###################################

        fobj.seek(offset)
        rawdata = fobj.read(nx * ny * dtype.itemsize)
        unscaleddata = np.frombuffer(rawdata, count=nx * ny,
                                     dtype=dtype).reshape(nx, ny)

        # internal informations from file
        _info = dict(unit=channel_dict["unit"], data_source=channel_dict["name"])
        _info.update(info)

        surface = Topography(unscaleddata.T, (sx, sy), info=_info)
        if height_scale_factor is None:
            height_scale_factor = channel_dict["height_scale_factor"]
        surface = surface.scale(height_scale_factor)

        if close_file:
            fobj.close()

        return surface

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
