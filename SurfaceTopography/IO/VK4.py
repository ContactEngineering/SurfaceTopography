#
# Copyright 2022 Lars Pastewka
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

from struct import unpack

from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile

MAGIC = b'VK4_'
MAGIC0 = b'\x00\x00\x00\x00'

KEYENCE4_HEADER_SIZE = 12
KEYENCE4_OFFSET_TABLE_SIZE = 72
KEYENCE4_MEASUREMENT_CONDITIONS_MIN_SIZE = 304
KEYENCE4_ASSEMBLY_INFO_SIZE = 16
KEYENCE4_ASSEMBLY_CONDITIONS_SIZE = 8
KEYENCE4_ASSEMBLY_HEADERS_SIZE = KEYENCE4_ASSEMBLY_INFO_SIZE + KEYENCE4_ASSEMBLY_CONDITIONS_SIZE
KEYENCE4_ASSEMBLY_FILE_SIZE = 532
KEYENCE4_TRUE_COLOR_IMAGE_MIN_SIZE = 20
KEYENCE4_FALSE_COLOR_IMAGE_MIN_SIZE = 796
KEYENCE4_LINE_MEASUREMENT_LEN = 1024
KEYENCE4_LINE_MEASUREMENT_SIZE = 18440

KEYENCE4_NORMAL_FILE = 0
KEYENCE4_ASSEMBLY_FILE = 1
KEYENCE4_ASSEMBLY_FILE_UNICODE = 2


class VK4Reader(ReaderBase):
    _format = 'vk4'
    _name = 'Keyence VK4'
    _description = '''
File format of the Keyence laser conformal microscope.
'''

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Check VK4 file magic
            if f.read(len(MAGIC)) != MAGIC:   # 'VK4_'
                raise ValueError('File magic does not match. This is not a Keyence VK4 file.')
            f.read(4)  # dll version
            if f.read(len(MAGIC0)) != MAGIC0:  # All zeros
                raise ValueError('File magic does not match. This is not a Keyence VK4 file.')

            # Offset table
            setting, color_peak, color_light = unpack('<III', f.read(12))
            light = unpack('<III', f.read(12))
            self.height = unpack('<III', f.read(12))
            color_peak_thumbnail, color_thumbnail, light_thumbnail, height_thumbnail, assemble, line_measure, \
                line_thickness, string_data, reserved = unpack('<IIIIIIIII', f.read(9*4))

            # Measurement conditions
            size, year, month, day, hour, minute, second, diff_utc_by_minutes = unpack('<IIIIIIIi', f.read(8*4))
#            image_attributes, user_interface_mode, color_composite_mode, num_layer, run_mode, peak_mode, sharpening_level
#            speed, distance, pitch, optical_zoom, num_line, line0_pos, reserved1[3], lens_mag, pmt_gain_mode, pmt_gain
#            pmt_offset, nd_filter, reserved2, persist_count, shutter_speed_mode, shutter_speed, white_balance_mode
#            white_balance_red, white_balance_blue, camera_gain, plane_compensation, xy_length_unit, z_length_unit
#            xy_decimal_place, z_decimal_place, x_length_per_pixel, y_length_per_pixel, z_length_per_digit
#            reserved3[5], light_filter_type, reserved4, gamma_reverse, gamma, gamma_offset, ccd_bw_offset, numerical_aperture
#            head_type, pmt_gain2, omit_color_image, lens_id, light_lut_mode, light_lut_in0, light_lut_out0, light_lut_in1
#            light_lut_out1, light_lut_in2, light_lut_out2, light_lut_in3, light_lut_out3, light_lut_in4, light_lut_out4
#            upper_position, lower_position, light_effective_bit_depth, height_effective_bit_depth;

            # Height data
            f.seek(self.height[0])
            width, height, bit_depth = unpack('<III', f.read(12))

            compression, byte_size = unpack('<II', f.read(8))
            palette_range_min, palette_range_max = unpack('<II', f.read(8))

            if byte_size != width*height*bit_depth//8:
                raise CorruptFile('Reported size does not match image dimensions.')

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        if channel_index is None:
            channel_index = self._default_channel_index

        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI '
                               'parallelization.')

        #channel_info = self._channels[channel_index]

        raise NotImplementedError

    @property
    def channels(self):
        return self._channels
