#
# Copyright 2023-2025 Lars Pastewka
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
Reader for MicroProf FRT profilometry data.

The FRT format uses TLV (Tag-Length-Value) encoding:
- Version 1.00: tag (uint16) + size (uint32)
- Version 1.01: tag (uint16) + size (uint64)

Reference information and implementations:
https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/microprof.c
"""

import numpy as np

from ..Exceptions import CorruptFile, FileFormatMismatch, UnsupportedFormatFeature
from ..UniformLineScanAndTopography import Topography
from .binary import (
    BinaryStructure,
    LayoutWithTrailingData,
    TextBuffer,
    TLVContainer,
    Validate,
)
from .common import OpenFromAny
from .Reader import ChannelInfo, CompoundLayout, DeclarativeReaderBase, For


def _create_tlv_parser(context):
    """Create version-appropriate TLV parser based on header version."""
    version = context.header['version']
    nb_blocks = context.header['nb_blocks']

    # Select size format based on version
    if version == '1.00':
        size_format = '<I'  # uint32
    elif version == '1.01':
        size_format = '<Q'  # uint64
    else:
        raise UnsupportedFormatFeature(f'Unsupported FRT version: {version}')

    # Subblock structure for 0x0091
    subblock_structure = BinaryStructure([
        ('active', 'I'),
        ('frequency', 'f'),
        ('ac_dB', 'f'),
        ('low_pass', 'f'),
        ('high_pass', 'f'),
        ('out_gain', 'f'),
        ('pre_gain', 'f'),
    ])

    # Block definitions using binary.py layout classes
    block_structures = {
        0x0065: TextBuffer('text'),
        0x0066: BinaryStructure([
            ('nb_grid_pts_x', 'I'),
            ('nb_grid_pts_y', 'I'),
            ('bytes_per_pixel', 'I'),
        ], name='grid_info'),
        0x0067: BinaryStructure([
            ('physical_size_x', 'd'),
            ('physical_size_y', 'd'),
            ('offset_x', 'd'),
            ('offset_y', 'd'),
            ('factor_y', 'd'),
            ('scan_direction', 'I'),
        ], name='physical_params'),
        0x0068: BinaryStructure([
            ('xspeed', 'd'),
            ('yspeed', 'd'),
            ('override_speed', 'I'),
            ('check_sensor_error', 'I'),
            ('scan_back_meas', 'I'),
            ('sensor_delay', 'I'),
            ('sensor_error_time', 'I'),
        ], name='speed_params'),
        0x0069: BinaryStructure([
            ('range_unit_type', 'I'),
            ('offset_unit_type', 'I'),
            ('xspeed_unit_type', 'I'),
            ('yspeed_unit_type', 'I'),
        ], name='unit_types'),
        0x006a: BinaryStructure([
            ('step_xcount', 'I'),
            ('step_ycount', 'I'),
            ('xstep', 'd'),
            ('ystep', 'd'),
            ('step_delay', 'I'),
            ('back_scan_step', 'I'),
        ], name='step_params'),
        0x006b: BinaryStructure([
            ('wait_at_start_of_line', 'I'),
            ('display_start_box', 'I'),
            ('do_hysteresis_corr', 'I'),
            ('back_scan_delay', 'I'),
        ], name='scan_options'),
        0x006c: BinaryStructure([
            ('meas_range', 'I'),
            ('height_scale_factor', 'd'),
        ], name='height_scale'),
        0x006d: BinaryStructure([
            ('zrange', 'd'),
            ('use_percentage', 'd'),
            ('display_correction', 'I'),
            ('palette_type', 'I'),
            ('display_size', 'I'),
            ('autorange', 'I'),
        ], name='display_params'),
        0x006e: BinaryStructure([
            ('sensor_type', 'I'),
            ('xytable_type', 'I'),
            ('ztable_type', 'I'),
        ], name='hardware_types'),
        0x006f: BinaryStructure([
            ('do_integrate', 'I'),
            ('integrate_over', 'I'),
            ('sensor_was_piezo', 'I'),
            ('sensor_was_full', 'I'),
        ], name='integration_params'),
        0x0070: BinaryStructure([
            ('first_valid', 'I'),
            ('last_valid', 'I'),
        ], name='valid_range'),
        0x0071: BinaryStructure([
            ('zoffset', 'd'),
        ], name='z_offset'),
        0x0072: BinaryStructure([
            ('meas_started', 'I'),
            ('meas_ended', 'I'),
            ('meas_time', 'I'),
        ], name='timing'),
        0x0073: BinaryStructure([
            ('dio_type', 'I'),
        ], name='dio_type'),
        0x0074: BinaryStructure([
            ('dllver1', 'I'),
            ('dllver2', 'I'),
        ], name='dll_version'),
        0x0075: LayoutWithTrailingData('drift_corr', [
            ('nb_values', 'I'),
            ('is_applied', 'I'),
            ('do_drift_corr_scan', 'I'),
            ('data_available', 'I'),
            ('line_not_row', 'I'),
        ]),
        0x0076: BinaryStructure([
            ('xstart', 'd'),
            ('ystart', 'd'),
            ('xend', 'd'),
            ('yend', 'd'),
        ], name='scan_range'),
        0x0077: TextBuffer('text_77'),
        0x0079: BinaryStructure([
            ('xdispoffset', 'd'),
            ('ydispoffset', 'd'),
        ], name='display_offset'),
        0x007a: BinaryStructure([
            ('meas_rate', 'I'),
            ('min_intensity', 'I'),
        ], name='meas_rate'),
        0x007b: BinaryStructure([
            ('sensor_subtype', 'I'),
            ('xytable_subtype', 'I'),
        ], name='hardware_subtypes'),
        0x007c: BinaryStructure([
            ('speed_control', 'I'),
        ], name='speed_control'),
        0x007e: BinaryStructure([
            ('max_xrange', 'd'),
            ('max_yrange', 'd'),
        ], name='max_range'),
        0x007f: BinaryStructure([
            ('calibration', '255s'),
            ('is_calibrated', '?'),
        ], name='calibration'),
        0x0080: BinaryStructure([
            ('is_z_motor_ctrl_on', 'I'),
        ], name='z_motor_ctrl'),
        0x0081: BinaryStructure([
            ('nb_layers', 'I'),
            ('range1', 'd'),
            ('range_rest', 'd'),
        ], name='layer_info'),
        0x0082: BinaryStructure([
            ('motion_type', 'I'),
        ], name='motion_type'),
        0x0083: BinaryStructure([
            ('data_type', 'I'),
        ], name='data_type'),
        0x0084: BinaryStructure([
            ('use_std_schichthohe', 'I'),
        ], name='std_schichthohe'),
        0x0085: BinaryStructure([
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
            ('selected_unit', 'I'),
        ], name='sensor_params_85'),
        0x0086: BinaryStructure([
            ('product_id', 'H'),
            ('series_no', 'H'),
        ], name='product_info'),
        0x0087: BinaryStructure([
            ('use_frt_offset', 'I'),
        ], name='frt_offset'),
        0x0088: BinaryStructure([
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
        ], name='sensor_params_88'),
        0x0089: BinaryStructure([
            ('auto_approach', 'I'),
            ('auto_retract', 'I'),
        ], name='auto_approach'),
        0x008a: BinaryStructure([
            ('zmotor_drive_allowed', 'I'),
            ('zmotor_drive_way', 'd'),
        ], name='zmotor_drive'),
        0x008b: BinaryStructure([
            ('do_wait', 'I'),
        ], name='do_wait'),
        0x008c: BinaryStructure([
            ('tv_range', 'd'),
            ('tv_offset', 'd'),
            ('set_tv_offset', 'B'),
            ('set_tv_automatic', 'B'),
            ('tv_range_percent', 'f'),
        ], name='tv_params'),
        0x008d: BinaryStructure([
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
        ], name='meas_mode'),
        0x008e: BinaryStructure([
            ('serial_number', 'T'),
            ('day', 'B'),
            ('month', 'B'),
            ('year', 'H'),
            ('was_created', 'I'),
            ('nb_values', 'I'),
        ], name='serial_info'),
        0x008f: BinaryStructure([
            ('tracking_mode_activated', 'I'),
        ], name='tracking_mode'),
        0x0090: BinaryStructure([
            ('despike_do_it', 'I'),
            ('despike_threshold', 'd'),
            ('filter_meas_do_it', 'I'),
            ('filter_meas_type', 'I'),
            ('filter_meas_param', 'd'),
            ('tip_simul_do_it', 'I'),
            ('tip_simul_angle', 'd'),
            ('tip_simul_radius', 'd'),
        ], name='filter_params'),
        0x0091: BinaryStructure([
            ('topography', 'I'),
            ('differential', 'I'),
            ('topo_edit', 'd'),
            ('height_edit', 'd'),
            ('topo_scale', 'd'),
            ('nb_subblocks', 'I'),
            For(lambda ctx: ctx['nb_subblocks'], subblock_structure, name='subblocks'),
        ], name='topo_params'),
        0x0093: BinaryStructure([
            ('invalid_values', 'I'),
            ('lower_values', 'I'),
            ('upper_values', 'I'),
        ], name='value_stats'),
        0x0094: LayoutWithTrailingData('teach_params', [
            ('min_teach', 'd'),
            ('max_teach', 'd'),
            ('min_norm_teach', 'I'),
            ('max_norm_teach', 'I'),
            ('name_of_teach', 'T'),
        ]),
        0x0095: BinaryStructure([
            ('thickness_mode', 'I'),
            ('kind_of_thickness', 'I'),
            ('refractive_index', 'd'),
        ], name='thickness_mode'),
        0x0096: BinaryStructure([
            ('thickness_lints_on', 'I'),
            ('low_limit', 'd'),
            ('high_limit', 'd'),
        ], name='thickness_limits'),
        0x0097: BinaryStructure([
            ('laser_power', 'I'),
            ('laser_power_fine', 'I'),
            ('laser_frequency', 'I'),
            ('intensity', 'I'),
            ('min_valid_intens', 'I'),
        ], name='laser_params'),
        0x0098: BinaryStructure([
            ('meas_z_position', 'd'),
        ], name='meas_z_position'),
        0x0099: BinaryStructure([
            ('is_dual_scan', 'I'),
            ('scan_frequency', 'd'),
            ('duty', 'f'),
        ], name='dual_scan'),
        0x009a: BinaryStructure([
            ('is_ttv', 'I'),
            ('meas_rate2', 'I'),
            ('intensity2', 'I'),
            ('zoffsets1', 'd'),
            ('zoffsets2', 'd'),
            ('scale1', 'd'),
            ('scale2', 'd'),
        ], name='ttv_params'),
        0x009b: BinaryStructure([
            ('is_roundness', 'I'),
            ('is_sample_used', 'I'),
            ('radius', 'd'),
            ('max_xrange', 'd'),
            ('max_yrange', 'd'),
        ], name='roundness'),
        0x009c: BinaryStructure([
            ('do_despike', 'I'),
            ('do_interpolate', 'I'),
        ], name='despike_interp'),
        0x009d: BinaryStructure([
            ('subtract_sinus', 'I'),
        ], name='subtract_sinus'),
        0x009e: BinaryStructure([
            ('layer_info', 'I'),
            ('fit_threshold', 'd'),
        ], name='layer_fit'),
        0x009f: BinaryStructure([
            ('textlen', 'H'),
        ], name='zunit_len'),
        0x00a0: BinaryStructure([
            ('brightness', 'H'),
            ('eval_method', 'H'),
            ('focus', 'H'),
            ('gain', 'H'),
            ('meas_zrange', 'H'),
            ('objective', 'H'),
            ('shutter', 'H'),
            ('zresolution', 'd'),
        ], name='camera_params'),
        0x00a1: BinaryStructure([
            ('min_quality', 'H'),
            ('focus', 'd'),
        ], name='quality_focus'),
        0x00a2: BinaryStructure([
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
        ], name='sensor_params_a2'),
        0x00a3: BinaryStructure([
            ('cfm_objective', 'H'),
            ('cfm_shutter', 'H'),
            ('start_pos', 'd'),
            ('end_pos', 'd'),
            ('cfm_zresolution', 'd'),
            ('lower_reflect_threshold', 'd'),
            ('upper_reflect_threshold', 'd'),
        ], name='cfm_params'),
        0x00a4: BinaryStructure([
            ('angle', 'd'),
            ('I_zfb', 'd'),
            ('P_zfb', 'd'),
            ('retract_time', 'd'),
            ('xoffset', 'd'),
            ('yoffset', 'd'),
            ('zgain', 'd'),
        ], name='feedback_params'),
        0x00a5: BinaryStructure([
            ('external_timing', 'I'),
        ], name='external_timing'),
        0x00a6: BinaryStructure([
            ('textlen', 'I'),
        ], name='objective_name_len'),
        0x00a9: BinaryStructure([
            ('xaxis_subtracted', 'I'),
            ('yaxis_subtracted', 'I'),
        ], name='axis_subtracted'),
        0x00aa: BinaryStructure([
            ('sensor_ini_path', '259s'),
            ('start_pos', 'd'),
            ('end_pos', 'd'),
            ('zspeed', 'd'),
            ('presampling_zlength', 'd'),
            ('postsampling_zlength', 'd'),
            ('pos_after_zscan', 'I'),
            ('preprocessor', 'I'),
            ('postprocessor', 'I'),
        ], name='zscan_params'),
        0x00ad: BinaryStructure([
            ('nb_subblocks', 'I'),
        ], name='subblock_count'),
        0x00ae: BinaryStructure([
            ('signal', 'I'),
            ('filter', 'I'),
            ('reference_type', 'I'),
            ('layer_stack_id', 'I'),
            ('reference_material_id', 'i'),  # gint32 = signed int32
            ('reference_constant', 'd'),
            ('material_thickness', 'd'),
        ], name='signal_params'),
        0x00af: BinaryStructure([
            ('auto_focus', 'I'),
            ('auto_brightness', 'I'),
            ('focus_search_length', 'd'),
            ('max_brightness', 'I'),
            ('move_back_after_meas', 'I'),
            ('move_back_below_scan_range', 'I'),
        ], name='auto_focus'),
    }

    return TLVContainer(
        block_structures,
        name='blocks',
        size_format=size_format,
        count=nb_blocks
    )


class FRTReader(DeclarativeReaderBase):
    _format = 'frt'
    _mime_types = ['application/x-microprof']
    _file_extensions = ['frt']

    _name = 'Microprof FRT'
    _description = '''
This reader imports MicroProf FRT profilometry data.
'''

    _MAGIC = 'FRTM_GLIDERV'
    _UNDEFINED_DATA = 1.0  # undefined data is marked by the value 1

    _bytes_per_pixel_to_dtype = {
        16: np.dtype('<H'),
        32: np.dtype('<i')
    }

    _file_layout = CompoundLayout([
        BinaryStructure([
            ('magic', '12s', Validate(_MAGIC, FileFormatMismatch)),
            ('version', '4s'),
            ('nb_blocks', 'H'),
        ], name='header'),
        _create_tlv_parser,  # Callable that returns version-appropriate TLVContainer
    ])

    def _validate_metadata(self):
        """Validate that required blocks are present."""
        blocks = self._metadata.get('blocks', {})
        if 0x000b not in blocks:
            raise CorruptFile('File does not appear to contain topography data.')

    def _filter_raw_bytes(self, obj):
        """Recursively filter out _raw bytes from metadata for JSON serialization.

        Also converts int keys to hex strings for JSON compatibility.
        """
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k == '_raw' or isinstance(v, bytes):
                    continue
                # Convert int keys to hex strings for JSON compatibility
                key = hex(k) if isinstance(k, int) else k
                result[key] = self._filter_raw_bytes(v)
            return result
        elif isinstance(obj, list):
            return [self._filter_raw_bytes(item) for item in obj]
        elif isinstance(obj, bytes):
            return None  # Filter out bytes
        else:
            return obj

    @property
    def channels(self):
        blocks = self._metadata['blocks']

        # Get grid info from block 0x0066
        grid_info = blocks.get(0x0066, blocks.get('grid_info', {}))
        nb_grid_pts_x = grid_info.get('nb_grid_pts_x')
        nb_grid_pts_y = grid_info.get('nb_grid_pts_y')
        bytes_per_pixel = grid_info.get('bytes_per_pixel')

        # Get physical params from block 0x0067
        physical_params = blocks.get(0x0067, blocks.get('physical_params', {}))
        physical_size_x = physical_params.get('physical_size_x')
        physical_size_y = physical_params.get('physical_size_y')

        # Get height scale from block 0x006c
        height_scale = blocks.get(0x006c, blocks.get('height_scale', {}))
        height_scale_factor = height_scale.get('height_scale_factor')

        # Get data block info (block 0x000b - not defined, stored as raw)
        data_block = blocks.get(0x000b, {})
        if isinstance(data_block, dict):
            data_info = {
                'block_size': data_block.get('_size', 0),
                'dtype': self._bytes_per_pixel_to_dtype.get(bytes_per_pixel, np.dtype('<H')),
            }
        else:
            data_info = {}

        # Filter out raw bytes from metadata for JSON serialization
        filtered_metadata = self._filter_raw_bytes(self._metadata)

        return [
            ChannelInfo(
                self,
                0,
                name='default',
                dim=2,
                nb_grid_pts=(nb_grid_pts_x, nb_grid_pts_y),
                physical_sizes=(physical_size_x, physical_size_y),
                unit='m',
                height_scale_factor=height_scale_factor,
                periodic=False,
                uniform=True,
                info={'raw_metadata': filtered_metadata},
                tags={
                    'dtype': data_info.get('dtype', np.dtype('<H')),
                    'block_size': data_info.get('block_size', 0),
                }
            )
        ]

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        """Read topography data from file.

        Note: This method needs custom implementation because the data block
        offset must be determined by scanning through blocks.
        """
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        if channel_index is None:
            channel_index = self._default_channel_index

        channel = self.channels[channel_index]

        # Check for fixed metadata
        from ..Exceptions import MetadataAlreadyFixedByFile
        if physical_sizes is not None and channel.physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')
        if height_scale_factor is not None and channel.height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')
        if unit is not None and channel.unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        # Use channel values if not overridden
        if physical_sizes is None:
            physical_sizes = channel.physical_sizes
        if height_scale_factor is None:
            height_scale_factor = channel.height_scale_factor
        if unit is None:
            unit = channel.unit

        # Read data by scanning through the file to find block 0x000b
        with OpenFromAny(self.file_path, 'rb') as f:
            # Skip header: magic (12) + version (4) + nb_blocks (2)
            f.read(12 + 4 + 2)
            version = self._metadata['header']['version']
            nb_blocks = self._metadata['header']['nb_blocks']

            # Determine size format
            if version == '1.00':
                size_bytes = 4
                size_fmt = '<I'
            else:
                size_bytes = 8
                size_fmt = '<Q'

            # Scan through blocks to find 0x000b
            from struct import unpack
            for _ in range(nb_blocks):
                tag_data = f.read(2)
                if len(tag_data) < 2:
                    break
                tag, = unpack('<H', tag_data)
                size_data = f.read(size_bytes)
                if len(size_data) < size_bytes:
                    break
                size, = unpack(size_fmt, size_data)

                if tag == 0x000b:
                    # Found the data block
                    nb_grid_pts_x, nb_grid_pts_y = channel.nb_grid_pts
                    dtype = channel.tags['dtype']
                    height_data = np.frombuffer(
                        f.read(np.prod(channel.nb_grid_pts) * dtype.itemsize),
                        dtype=dtype
                    ).reshape((nb_grid_pts_y, nb_grid_pts_x)).T
                    height_data = np.ma.masked_array(
                        height_data,
                        mask=height_data == self._UNDEFINED_DATA
                    )
                    break
                else:
                    f.seek(size, 1)  # Skip this block
            else:
                raise CorruptFile('Could not find data block 0x000b')

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(
            height_data,
            physical_sizes,
            unit=unit,
            periodic=False if periodic is None else periodic,
            info=_info
        )
        return topo.scale(height_scale_factor)
