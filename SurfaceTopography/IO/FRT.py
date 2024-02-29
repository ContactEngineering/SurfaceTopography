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

import os

#
# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/microprof.c
#

from struct import unpack

import numpy as np

from .binary import decode
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile, UnsupportedFormatFeature
from ..UniformLineScanAndTopography import Topography


class FRTReader(ReaderBase):
    _format = 'frt'
    _mime_types = ['application/x-microprof']
    _file_extensions = ['frt']

    _name = 'Microprof FRT'
    _description = '''
This reader imports MicroProf FRT profilometry data.
'''

    _MAGIC = 'FRTM_GLIDERV'
    _VERSION_LEN = 4  # magic is followed by 4 chars of version information

    _UNDEFINED_DATA = 1.0  # undefined data is marked by the value 1

    _block_structures = {
        0x0066: [
            ('nb_grid_pts_x', 'I'),
            ('nb_grid_pts_y', 'I'),
            ('bytes_per_pixel', 'I'),
        ],
        0x0067: [
            ('physical_size_x', 'd'),
            ('physical_size_y', 'd'),
            ('offset_x', 'd'),
            ('offset_y', 'd'),
            ('factor_y', 'd'),
            ('scan_direction', 'I'),
        ],
        0x0068: [
            ('xspeed', 'd'),
            ('yspeed', 'd'),
            ('override_speed', 'I'),
            ('check_sensor_error', 'I'),
            ('scan_back_meas', 'I'),
            ('sensor_delay', 'I'),
            ('sensor_error_time', 'I'),
        ],
        0x0069: [
            ('range_unit_type', 'I'),
            ('offset_unit_type', 'I'),
            ('xspeed_unit_type', 'I'),
            ('yspeed_unit_type', 'I'),
        ],
        0x006a: [
            ('step_xcount', 'I'),
            ('step_ycount', 'I'),
            ('xstep', 'd'),
            ('ystep', 'd'),
            ('step_delay', 'I'),
            ('back_scan_step', 'I'),
        ],
        0x006b: [
            ('wait_at_start_of_line', 'I'),
            ('display_start_box', 'I'),
            ('do_hysteresis_corr', 'I'),
            ('back_scan_delay', 'I'),
        ],
        0x006c: [
            ('meas_range', 'I'),
            ('height_scale_factor', 'd'),
        ],
        0x006d: [
            ('zrange', 'd'),
            ('use_percentage', 'd'),
            ('display_correction', 'I'),
            ('palette_type', 'I'),
            ('display_size', 'I'),
            ('autorange', 'I'),
        ],
        0x006e: [
            ('sensor_type', 'I'),
            ('xytable_type', 'I'),
            ('ztable_type', 'I'),
        ],
        0x006f: [
            ('do_integrate', 'I'),
            ('integrate_over', 'I'),
            ('sensor_was_piezo', 'I'),
            ('sensor_was_full', 'I'),
        ],
        0x0070: [
            ('first_valid', 'I'),
            ('last_valid', 'I'),
        ],
        0x0071: [
            ('zoffset', 'd'),
        ],
        0x0072: [
            ('meas_started', 'I'),
            ('meas_ended', 'I'),
            ('meas_time', 'I'),
        ],
        0x0073: [
            ('dio_type', 'I'),
        ],
        0x0074: [
            ('dllver1', 'I'),
            ('dllver2', 'I'),
        ],
        0x0075: [
            ('nb_values', 'I'),
            ('is_applied', 'I'),
            ('do_drift_corr_scan', 'I'),
            ('data_available', 'I'),
            ('line_not_row', 'I'),
        ],
        0x0076: [
            ('xstart', 'd'),
            ('ystart', 'd'),
            ('xend', 'd'),
            ('yend', 'd'),
        ],
        0x0079: [
            ('xdispoffset', 'd'),
            ('ydispoffset', 'd'),
        ],
        0x007a: [
            ('meas_rate', 'I'),
            ('min_intensity', 'I'),
        ],
        0x007b: [
            ('sensor_subtype', 'I'),
            ('xytable_subtype', 'I'),
        ],
        0x007c: [
            ('speed_control', 'I')
        ],
        0x007e: [
            ('max_xrange', 'd'),
            ('max_yrange', 'd')
        ],
        0x007f: [
            ('calibration', '255s'),
            ('is_calibrated', '?'),
        ],  # check
        0x0080: [
            ('is_z_motor_ctrl_on', 'I'),
        ],
        0x0081: [
            ('nb_layers', 'I'),
            ('range1', 'd'),
            ('range_rest', 'd')
        ],
        0x0082: [
            ('motion_type', 'I')
        ],
        0x0083: [
            ('data_type', 'I')
        ],
        0x0084: [
            ('use_std_schichthohe', 'I')
        ],
        0x0085: [
            ('volt_range', 'I'),
            ('val_channel', 'I'),
            ('int_channel', 'I'),
            ('val_range', 'd'),
            ('int_range', 'I'),
            ('min_valid_val', 'd'),
            ('max_valid_val', 'd'),
            ('min_valid_intens', 'd'),
            ('max_valid_intens', 'd'),
            ('unit1', '16s'),
            ('unit2', '16s'),
            ('unit3', '16s'),
            ('unit4', '16s'),
            ('unit5', '16s'),
            ('unit6', '16s'),
            ('unit7', '16s'),
            ('unit8', '16s'),
            ('selected_unit', 'I')
        ],
        0x0086: [
            ('product_id', 'H'),
            ('series_no', 'H'),
        ],
        0x0087: [
            ('use_frt_offset', 'I'),
        ],
        0x0088: [
            ('volt_range', 'I'),
            ('val_channel', 'I'),
            ('int_channel', 'I'),
            ('int_range', 'I'),
            ('min_valid_val', 'd'),
            ('max_valid_val', 'd'),
            ('min_valid_intens', 'd'),
            ('max_valid_intens', 'd'),
            ('unit1', '16s'),
            ('unit2', '16s'),
            ('unit3', '16s'),
            ('unit4', '16s'),
            ('unit5', '16s'),
            ('unit6', '16s'),
            ('unit7', '16s'),
            ('unit8', '16s'),
            ('selected_unit', 'I'),
            ('min_valid_unit_value', 'd'),
            ('max_valid_unit_value', 'd'),
        ],
        0x0089: [
            ('auto_approach', 'I'),
            ('auto_retract', 'I'),
        ],
        0x008a: [
            ('zmotor_drive_allowed', 'I'),
            ('zmotor_drive_way', 'd'),
        ],
        0x008b: [
            ('do_wait', 'I')
        ],
        0x008c: [
            ('tv_range', 'd'),
            ('tv_offset', 'd'),
            ('set_tv_offset', 'B'),
            ('set_tv_automatic', 'B'),
            ('tv_range_percent', 'f'),
        ],
        0x008d: [
            ('meas_mode', 'I'),
            ('height_edit', 'd'),
            ('topo_edit', 'd'),
            ('pref_mode', 'I'),
            ('freq_edit', 'd'),
            ('hf_edit', 'I'),
            ('nf_edit', 'I'),
            ('phase_edit', 'd'),
            ('nf_mode', 'I'),
            ('topo_scale', 'd'),
        ],
        0x008e: [
            ('serial_number', 'T'),
            ('day', 'B'),
            ('month', 'B'),
            ('year', 'H'),
            ('was_created', 'I'),
            ('nb_values', 'I'),
        ],
        0x008f: [
            ('tracking_mode_activated', 'I'),
        ],
        0x0090: [
            ('despike_do_it', 'I'),
            ('despike_threshold', 'd'),
            ('filter_meas_do_it', 'I'),
            ('filter_meas_type', 'I'),
            ('filter_meas_param', 'd'),
            ('tip_simul_do_it', 'I'),
            ('tip_simul_angle', 'd'),
            ('tip_simul_radius', 'd'),
        ],
        0x0091: [
            ('topography', 'I'),
            ('differential', 'I'),
            ('topo_edit', 'd'),
            ('height_edit', 'd'),
            ('topo_scale', 'd'),
            ('nb_subblocks', 'I'),
        ],
        # 0x0092:
        0x0093: [
            ('invalid_values', 'I'),
            ('lower_values', 'I'),
            ('upper_values', 'I'),
        ],
        0x0094: [
            ('min_teach', 'd'),
            ('max_teach', 'd'),
            ('min_norm_teach', 'I'),
            ('max_norm_teach', 'I'),
            ('name_of_teach', 'T'),
        ],
        0x0095: [
            ('thickness_mode', 'I'),
            ('kind_of_thickness', 'I'),
            ('refractive_index', 'd'),
        ],
        0x0096: [
            ('thickness_lints_on', 'I'),
            ('low_limit', 'd'),
            ('high_limit', 'd'),
        ],
        0x0097: [
            ('laser_power', 'I'),
            ('laser_power_fine', 'I'),
            ('laser_frequency', 'I'),
            ('intensity', 'I'),
            ('min_valid_intens', 'I'),
        ],
        0x0098: [
            ('meas_z_position', 'd'),
        ],
        0x0099: [
            ('is_dual_scan', 'I'),
            ('scan_frequency', 'd'),
            ('duty', 'f'),
        ],
        0x009a: [
            ('is_ttv', 'I'),
            ('meas_rate2', 'I'),
            ('intensity2', 'I'),
            ('zoffsets1', 'd'),
            ('zoffsets2', 'd'),
            ('scale1', 'd'),
            ('scale2', 'd'),
        ],
        0x009b: [
            ('is_roundness', 'I'),
            ('is_sample_used', 'I'),
            ('radius', 'd'),
            ('max_xrange', 'd'),
            ('max_yrange', 'd'),
        ],
        0x009c: [
            ('do_despike', 'I'),
            ('do_interpolate', 'I'),
        ],
        0x009d: [
            ('subtract_sinus', 'I'),
        ],
        0x009e: [
            ('layer_info', 'I'),
            ('fit_threshold', 'd'),
        ],
        0x009f: [
            ('textlen', 'H'),
            # There is a string 'zunit' here
        ],
        0x00a0: [
            ('brightness', 'H'),
            ('eval_method', 'H'),
            ('focus', 'H'),
            ('gain', 'H'),
            ('meas_zrange', 'H'),
            ('objective', 'H'),
            ('shutter', 'H'),
            ('zresolution', 'd'),
        ],
        0x00a1: [
            ('min_quality', 'H'),
            ('focus', 'd'),
        ],
        0x00a2: [
            ('volt_range', 'I'),
            ('val_channel', 'I'),
            ('int_channel', 'I'),
            ('int_range', 'I'),
            ('min_valid_val', 'd'),
            ('max_valid_val', 'd'),
            ('min_valid_intens', 'd'),
            ('max_valid_intens', 'd'),
            ('unit1', '16s'),
            ('unit2', '16s'),
            ('unit3', '16s'),
            ('unit4', '16s'),
            ('unit5', '16s'),
            ('unit6', '16s'),
            ('unit7', '16s'),
            ('unit8', '16s'),
            ('selected_unit', 'I'),
            ('min_valid_unit_value', 'd'),
            ('max_valid_unit_value', 'd'),
        ],
        0x00a3: [
            ('cfm_objective', 'H'),
            ('cfm_shutter', 'H'),
            ('start_pos', 'd'),
            ('end_pos', 'd'),
            ('cfm_zresolution', 'd'),
            ('lower_reflect_threshold', 'd'),
            ('upper_reflect_threshold', 'd'),
        ],
        0x00a4: [
            ('angle', 'd'),
            ('I_zfb', 'd'),
            ('P_zfb', 'd'),
            ('retract_time', 'd'),
            ('xoffset', 'd'),
            ('yoffset', 'd'),
            ('zgain', 'd'),
        ],
        0x00a5: [
            ('external_timing', 'I'),
        ],
        0x00a6: [
            ('textlen', 'I'),
            # There is a string 'objective_name' here
        ],
        0x00a9: [
            ('xaxis_subtracted', 'I'),
            ('yaxis_subtracted', 'I'),
        ],
        0x00aa: [
            ('sensor_ini_path', '259s'),
            ('start_pos', 'd'),
            ('end_pos', 'd'),
            ('zspeed', 'd'),
            ('presampling_zlength', 'd'),
            ('postsampling_zlength', 'd'),
            ('pos_after_zscan', 'I'),
            ('preprocessor', 'I'),
            ('postprocessor', 'I'),
        ],
        # 0x00ac user name and description
        0x00ad: [
            ('nb_subblocks', 'I'),
        ],
        0x00ae: [
            ('signal', 'I'),
            ('filter', 'I'),
            ('reference_type', 'I'),
            ('layer_stack_id', 'I'),
            ('reference_material_id', 'gint32'),
            ('reference_constant', 'd'),
            ('material_thickness', 'd'),
        ],
        0x00af: [
            ('auto_focus', 'I'),
            ('auto_brightness', 'I'),
            ('focus_search_length', 'd'),
            ('max_brightness', 'I'),
            ('move_back_after_meas', 'I'),
            ('move_back_below_scan_range', 'I'),
        ]
    }

    _subblocks_0x0091 = [
        ('active', 'I'),
        ('frequency', 'f'),
        ('ac_dB', 'f'),
        ('low_pass', 'f'),
        ('high_pass', 'f'),
        ('out_gain', 'f'),
        ('pre_gain', 'f'),
    ]

    _bytes_per_pixel_to_dtype = {
        16: np.dtype('<H'),
        32: np.dtype('<i')
    }

    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Check file magic
            magic = f.read(len(self._MAGIC)).decode('ascii')
            if magic != self._MAGIC:
                raise FileFormatMismatch
            self._version = f.read(self._VERSION_LEN).decode('ascii')
            if self._version != '1.00' and self._version != '1.01':
                raise UnsupportedFormatFeature

            # Read file metadata
            self._metadata = {}
            nb_blocks, = unpack('<H', f.read(2))  # Not sure what's in here
            for i in range(nb_blocks):
                if self._version == '1.00':
                    data = f.read(6)
                    if len(data) < 6:
                        break  # we are at the end of the file
                    block_id, block_size = unpack('<HI', data)
                else:
                    data = f.read(8)
                    if len(data) < 8:
                        break  # we are at the end of the file
                    block_id, block_size = unpack('<HQ', data)

                if block_id in self._block_structures.keys():
                    meta, size = decode(f, self._block_structures[block_id], return_size=True)

                    if block_id == 0x0075:
                        # There is a data block at the end of this section
                        meta['data_offset'] = f.tell()
                        f.seek(block_size - size, os.SEEK_CUR)
                        size = block_size
                    elif block_id == 0x0091:
                        # There are subblocks at the end
                        subblocks = []
                        for i in range(meta['nb_subblocks']):
                            _meta, _size = decode(f, self._subblocks_0x0091, return_size=True)
                            subblocks += [_meta]
                            size += _size
                        meta['subblocks'] = subblocks
                    elif block_id == 0x0094:
                        # Skip rest of this block which appears to be empty
                        f.seek(block_size - size, os.SEEK_CUR)
                        size = block_size

                    assert size == block_size  # Check that we read the correct size
                elif block_id == 0x0065 or block_id == 0x0077:
                    meta = {'text': f.read(block_size).decode('ascii').strip('\x00')}
                else:
                    # For all other block, store size and offset
                    meta = {'block_size': block_size, 'block_offset': f.tell()}
                    f.seek(block_size, os.SEEK_CUR)

                self._metadata[hex(block_id)] = meta

            # Read and fill channel information
            self._channels = []

            # Topography
            if '0xb' in self._metadata.keys():
                # Data block size and offset
                tags = self._metadata['0xb'].copy()
                tags['dtype'] = self._bytes_per_pixel_to_dtype[self._metadata['0x66']['bytes_per_pixel']]

                # Idiot check
                nb_grid_pts = (self._metadata['0x66']['nb_grid_pts_x'],
                               self._metadata['0x66']['nb_grid_pts_y'])
                assert tags['block_size'] == tags['dtype'].itemsize * np.prod(nb_grid_pts)

                # Construct channel info
                self._channels += [
                    ChannelInfo(
                        self,
                        0,
                        name='default',
                        dim=2,
                        nb_grid_pts=nb_grid_pts,
                        physical_sizes=(self._metadata['0x67']['physical_size_x'],
                                        self._metadata['0x67']['physical_size_y']),
                        unit='m',
                        height_scale_factor=self._metadata['0x6c']['height_scale_factor'],  # All units m
                        periodic=False,
                        uniform=True,
                        info={'raw_metadata': self._metadata},
                        tags=tags
                    )
                ]
            else:
                raise CorruptFile('File does not appear to contain topography data.')

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

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        channel = self._channels[channel_index]
        with OpenFromAny(self.file_path, 'rb') as f:
            nb_grid_pts_x, nb_grid_pts_y = channel.nb_grid_pts
            f.seek(channel.tags['block_offset'])
            dtype = channel.tags['dtype']
            height_data = np.frombuffer(f.read(np.prod(channel.nb_grid_pts) * dtype.itemsize), dtype=dtype) \
                .reshape((nb_grid_pts_y, nb_grid_pts_x)).T
            height_data = np.ma.masked_array(height_data, mask=height_data == self._UNDEFINED_DATA)

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          channel.physical_sizes,
                          unit=channel.unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(channel.height_scale_factor)
