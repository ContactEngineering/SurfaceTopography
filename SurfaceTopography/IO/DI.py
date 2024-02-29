#
# Copyright 2019-2023 Lars Pastewka
#           2020-2021 Michael Röttger
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

#
# The DI file format is described in detail here:
# http://www.physics.arizona.edu/~smanne/DI/software/fileformats.html
#

import re
import dateutil.parser

import numpy as np

from .common import OpenFromAny
from ..Exceptions import CorruptFile, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor, is_length_unit, mangle_length_unit_utf8

from .Reader import ReaderBase, ChannelInfo


###

class DIReader(ReaderBase):
    _format = 'di'
    _mime_types = ['application/x-nanoscope-iii-spm']
    _file_extensions = ['spm', '001', '002', '003', '004', '005']

    _name = 'Bruker Dimension, Veeco Nanoscope, Digital Instruments Nanoscope'
    _description = '''
Digital Instruments Nanoscope (also Veeco Nanoscope and Bruker Dimension)
files typically have a three-digit number as the file extension (.001, .002, .003, ...).
Newer versions of this file format have the extension .spm. This format contains
information on the physical size of the topography map as well as its units.
The reader supports V4.3 and later version of the format.
'''

    def __init__(self, file_path):
        """
        Load Digital Instrument's Nanoscope files.

        Arguments
        ---------
        file_path : filename or file object
             File or data stream to open.
        """
        self._file_path = file_path
        with OpenFromAny(self._file_path, 'rb') as fobj:
            # Get file size
            pos = fobj.tell()
            fobj.seek(0, 2)
            file_size = fobj.tell()
            fobj.seek(pos)

            parameters = []
            section_name = None
            section_dict = {}

            L = fobj.readline().decode('latin-1').strip()
            while L and L.lower() != r'\*file list end':
                if L.startswith('\\*'):
                    if section_name is not None:
                        parameters += [(section_name, section_dict)]
                    new_section_name = L[2:].lower()
                    if section_name is None:
                        if new_section_name != 'file list':
                            raise IOError("Header must start with the "
                                          "'File list' section.")
                    section_name = new_section_name
                    section_dict = {}
                elif L.startswith('\\'):
                    if section_name is None:
                        raise IOError('Encountered key before section '
                                      'header.')
                    s = L[1:].split(': ', 1)
                    try:
                        key, value = s
                    except ValueError:
                        key, = s
                        value = ''
                    section_dict[key.lower()] = value.strip()
                else:
                    raise IOError("Header line does not start with a slash.")
                L = fobj.readline().decode('latin-1').strip()
            if section_name is None:
                raise IOError('No sections found in header.')
            parameters += [(section_name, section_dict)]

            self._channels = []
            self._offsets = []

            scanner = {}
            equipment = {}
            info = {}
            for n, p in parameters:
                if n == 'file list':
                    if 'date' in p:
                        info['acquisition_time'] = str(dateutil.parser.parse(p['date']))
                elif n == 'scanner list' or n == 'ciao scan list':
                    scanner.update(p)
                elif n == 'equipment list':
                    equipment.update(p)
                elif n == 'ciao image list':
                    image_data_key = re.match(r'^S \[(.*?)\] ',
                                              p['@2:image data']).group(1)

                    info['raw_metadata'] = p

                    nx = int(p['samps/line'])
                    ny = int(p['number of lines'])

                    s = p['scan size'].split(' ', 2)
                    sx = float(s[0])
                    sy = float(s[1])

                    xy_unit = mangle_length_unit_utf8(s[2])
                    offset = int(p['data offset'])
                    self._offsets.append(offset)

                    length = int(p['data length'])
                    elsize = int(p['bytes/pixel'])
                    binary_scale = 1
                    if elsize == 4:
                        binary_scale = 1 / 65536  # Rescale 32-bit integer to a 16-bit range
                    elif elsize != 2:
                        raise IOError(f"Don't know how to handle {elsize} bytes per pixel data.")
                    if nx * ny * elsize > length:
                        raise IOError(f'File reports a data block of length {length}, but computing the size of the '
                                      f'data block from {nx} x {ny} grid points and the per-pixel storage of {elsize} '
                                      f'bytes yields a larger value of {nx * ny * elsize}.')

                    scale_re = re.match(
                        r'^V (?:\[(.*?)\]|)\s*\(([0-9\.]+) (.*)\/LSB\) (.*) (.*)', p['@2:z scale'])
                    quantity = scale_re.group(1)
                    if quantity is not None:
                        quantity = quantity.lower()
                    hard_scale = float(scale_re.group(4)) / 65536
                    hard_unit = scale_re.group(5)

                    if quantity is None:
                        s = []
                        soft_scale = 1.0
                    else:
                        s = scanner['@' + quantity].split()
                        if s[0] != 'V' or len(s) < 2:
                            raise IOError('Malformed Nanoscope DI file.')
                        soft_scale = float(s[1])

                    height_unit = None
                    hard_to_soft = 1.0
                    if len(s) > 2:
                        # Check units
                        height_unit, soft_unit = s[2].split('/')
                        hard_to_soft = get_unit_conversion_factor(hard_unit, soft_unit)
                        if hard_to_soft is None:
                            raise RuntimeError(
                                "Units for hard (={}) and soft (={}) "
                                "scale differ for '{}'. Don't know how "
                                "to handle this.".format(hard_unit,
                                                         soft_unit,
                                                         image_data_key))
                    # We only report channels with height information
                    height_scale_factor = None
                    if is_length_unit(height_unit):
                        height_unit = mangle_length_unit_utf8(height_unit)
                        if xy_unit != height_unit:
                            fac = get_unit_conversion_factor(xy_unit, height_unit)
                            sx *= fac
                            sy *= fac
                        unit = height_unit

                        height_scale_factor = hard_scale * hard_to_soft * soft_scale * binary_scale
                    else:
                        unit = (xy_unit, height_unit)

                    if 'microscope' in equipment:
                        info['instrument'] = {'name': equipment['microscope']}
                    elif 'description' in equipment:
                        info['instrument'] = {'name': equipment['description']}

                    self._channels += [
                        ChannelInfo(self,
                                    len(self._channels),
                                    name=image_data_key,
                                    dim=2,
                                    nb_grid_pts=(nx, ny),
                                    physical_sizes=(sx, sy),
                                    height_scale_factor=height_scale_factor,
                                    periodic=False,
                                    uniform=True,
                                    unit=unit,
                                    info=info,
                                    tags={'elsize': elsize})
                    ]

                    # We seek to the end of the data buffer, this should not raise an exception
                    if offset + nx * ny * elsize > file_size:
                        raise CorruptFile('File is not large enough to contain all data buffers.')

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

        with OpenFromAny(self._file_path, 'rb') as fobj:
            channel = self._channels[channel_index]

            if unit is not None:
                raise MetadataAlreadyFixedByFile('unit')

            sx, sy = self._check_physical_sizes(physical_sizes,
                                                channel.physical_sizes)

            nx, ny = channel.nb_grid_pts

            offset = self._offsets[channel_index]
            elsize = channel.tags['elsize']
            if elsize == 2:
                dtype = np.dtype('<i2')
            elif elsize == 4:
                dtype = np.dtype('<i4')
            else:
                raise IOError(f"Don't know how to handle {elsize} bytes per pixel data.")

            assert elsize == dtype.itemsize

            ###################################

            fobj.seek(offset)
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

        surface = Topography(np.fliplr(unscaleddata.T), physical_sizes=(sx, sy), unit=channel.unit, info=_info,
                             periodic=periodic)
        if height_scale_factor is None:
            height_scale_factor = channel.height_scale_factor
        elif channel.height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')
        if height_scale_factor is not None:
            surface = surface.scale(height_scale_factor)

        return surface

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
