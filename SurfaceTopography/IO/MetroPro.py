#
# Copyright 2022-2023 Lars Pastewka
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
# Reference information and implementations:
# https://www.seas.upenn.edu/~nanosop/documents/MetroProReferenceGuide0347_M.pdf
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/metropro.c
#

import datetime
import os

import numpy as np

from ..Exceptions import (CorruptFile, FileFormatMismatch,
                          MetadataAlreadyFixedByFile)
from ..UniformLineScanAndTopography import Topography
from .binary import decode
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase


class MetroProReader(ReaderBase):
    _format = 'metropro'
    _mime_types = ['application/x-zygo-spm']
    _file_extensions = ['dat']

    _name = 'Zygo Metropro DAT'
    _description = '''
This reader imports Zygo MetroPro data files.
'''

    _MAGIC1 = b'\x88\x1b\x03\x6f'
    _MAGIC2 = b'\x88\x1b\x03\x70'
    _MAGIC3 = b'\x88\x1b\x03\x71'

    _HEADER_SIZE12 = 834
    _HEADER_SIZE3 = 4096

    _HEADER_FORMATS = {
        _MAGIC1: _HEADER_SIZE12,
        _MAGIC2: _HEADER_SIZE12,
        _MAGIC3: _HEADER_SIZE3
    }

    _MAX_PHASE = {
        0: 4096,
        1: 32768,
        2: 131072
    }

    _header_structure1 = [
        ('header_format', '>H'),
        ('header_size', '>I'),
        ('swinfo_type', '>h'),
        ('swinfo_date', '30s'),
        ('swinfo_version_major', '>h'),
        ('swinfo_version_minor', '>h'),
        ('swinfo_version_bug', '>h'),
        ('ac_org_x', '>h'),
        ('ac_org_y', '>h'),
        ('ac_org_width', '>H'),
        ('ac_org_height', '>H'),
        ('ac_n_buckets', '>H'),
        ('ac_range', '>h'),
        ('ac_n_bytes', '>I'),
        ('cn_org_x', '>h'),
        ('cn_org_y', '>h'),
        ('cn_org_width', '>H'),
        ('cn_org_height', '>H'),
        ('cn_n_bytes', '>I'),
        ('time_stamp', '>i'),
        ('comment', '82s'),
        ('source', '>h'),
        ('intf_scale_factor', '>f'),
        ('wavelength_in', '>f'),
        ('num_aperture', '>f'),
        ('obliquity_factor', '>f'),
        ('magnification', '>f'),
        ('lateral_res', '>f'),
        ('acq_type', '>h'),
        ('intens_avg_cnt', '>h'),
        ('ramp_cal', '>h'),
        ('sfac_limit', '>h'),
        ('ramp_gain', '>h'),
        ('part_thickness', '>f'),
        ('sw_llc', '>h'),
        ('target_range', '>f'),
        ('rad_crv_veasure_eeq', '<h'),
        ('min_mod', '>i'),
        ('min_mod_count', '>i'),
        ('phase_res', '>h'),
        ('min_area', '>i'),
        ('discon_action', '>h'),
        ('discon_filter', '>f'),
        ('connect_order', '>h'),
        ('sign', '>h'),
        ('camera_width', '>h'),
        ('camera_height', '>h'),
        ('sys_type', '>h'),
        ('sys_board', '>h'),
        ('sys_serial', '>h'),
        ('inst_id', '>h'),
        ('obj_name', '12s'),
        ('part_name', '40s'),
        ('codev_type', '>h'),
        ('phase_avg_cnt', '>h'),
        ('sub_sys_err', '>h'),
        (None, '16b'),
        ('part_ser_num', '40s'),
        ('refractive_index', '>f'),
        ('rem_tilt_bias', '>h'),
        ('rem_fringes', '>h'),
        ('max_area', '>i'),
        ('setup_type', '>h'),
        ('wrapped', '>h'),
        ('pre_connect_filter', '>f'),
        ('wavelength_in_2', '>f'),
        ('wavelength_fold', '>h'),
        ('wavelength_in_1', '>f'),
        ('wavelength_in_3', '>f'),
        ('wavelength_in_4', '>f'),
        ('wavelen_select', '8s'),
        ('fda_res', '>h'),
        ('scan_descr', '20s'),
        ('n_fiducials_a', '>h'),
        ('fiducials_a', '14f'),
        ('pixel_width', '>f'),
        ('pixel_height', '>f'),
        ('exit_pupil_diam', '>f'),
        ('light_level_pct', '>f'),
        ('coords_state', '<i'),
        ('coords_x_pos', '<f'),
        ('coords_y_pos', '<f'),
        ('coords_z_pos', '<f'),
        ('coords_x_rot', '<f'),
        ('coords_y_rot', '<f'),
        ('coords_z_rot', '<f'),
        ('coherence_mode', '<h'),
        ('surface_filter', '<h'),
        ('sys_err_file_name', '28s'),
        ('zoom_descr', '8s'),
        ('alpha_part', '<f'),
        ('beta_part', '<f'),
        ('dist_part', '<f'),
        ('cam_split_loc_x', '<h'),
        ('cam_split_loc_y', '<h'),
        ('cam_split_trans_x', '<h'),
        ('cam_split_trans_y', '<h'),
        ('material_a', '24s'),
        ('material_b', '24s'),
        ('cam_split_unused', '<h'),
        (None, '2b'),
        ('dmi_ctr_x', '<f'),
        ('dmi_ctr_y', '<f'),
        ('sph_dist_corr', '<h'),
        (None, '2b'),
        ('sph_dist_part_na', '<f'),
        ('sph_dist_part_radius', '<f'),
        ('sph_dist_cal_na', '<f'),
        ('sph_dist_cal_radius', '<f'),
        ('surface_type', '<h'),
        ('ac_surface_type', '<h'),
        ('z_position', '<f'),
        ('power_multiplier', '<f'),
        ('focus_multiplier', '<f'),
        ('rad_crv_vocus_sal_lactor', '<f'),
        ('rad_crv_vower_ral_lactor', '<f'),
        ('ftp_left_pos', '<f'),
        ('ftp_right_pos', '<f'),
        ('ftp_pitch_pos', '<f'),
        ('ftp_roll_pos', '<f'),
        ('min_mod_pct', '<f'),
        ('max_inten', '<i'),
        ('ring_of_fire', '<h'),
        (None, '1b'),
        ('rc_orientation', 'B'),
        ('rc_distance', '<f'),
        ('rc_angle', '<f'),
        ('rc_diameter', '<f'),
        ('rem_fringes_mode', '>h'),
        (None, '1b'),
        ('ftpsi_phase_res', 'B'),
        ('frames_acquired', '<h'),
        ('cavity_type', '<h'),
        ('cam_frame_rate', '<f'),
        ('tune_range', '<f'),
        ('cal_pix_loc_x', '<h'),
        ('cal_pix_loc_y', '<h'),
        ('n_tst_cal_pts', '<h'),
        ('n_ref_cal_pts', '<h'),
        ('tst_cal_pts', '<4f'),
        ('ref_cal_pts', '<4f'),
        ('tst_cal_pix_opd', '<f'),
        ('ref_cal_pix_opd', '<f'),
        ('sys_serial2', '<i'),
        ('flash_phase_dc_mask', '<f'),
        ('flash_phase_alias_mask', '<f'),
        ('flash_phase_filter', '<f'),
        ('scan_direction', 'B'),
        (None, '1b'),
        ('pre_fda_filter', '>h'),
        (None, '4b'),
        ('ftpsi_res_factor', '<i'),
        (None, '8b'),
        (None, '4b')
    ]

    _header_structure3 = [
        ('films_mode', '>h'),
        ('films_reflectivity_ratio', '>h'),
        ('films_obliquity_correction', '>f'),
        ('films_refraction_index', '>f'),
        ('films_min_mod', '>f'),
        ('films_min_thickness', '>f'),
        ('films_max_thickness', '>f'),
        ('films_min_refl_ratio', '>f'),
        ('films_max_refl_ratio', '>f'),
        ('films_sys_char_file_name', '28s'),
        ('films_dfmt', '>h'),
        ('films_merit_mode', '>h'),
        ('films_h2g', '>h'),
        ('anti_vibration_cal_file_name', '28s'),
        (None, '2b'),
        ('films_fringe_remove_perc', '>f'),
        ('asphere_job_file_name', '28s'),
        ('asphere_test_plan_name', '28s'),
        (None, '4b'),
        ('asphere_nzones', '>f'),
        ('asphere_rv', '>f'),
        ('asphere_voffset', '>f'),
        ('asphere_att4', '>f'),
        ('asphere_r0', '>f'),
        ('asphere_att6', '>f'),
        ('asphere_r0_optimization', '>f'),
        ('asphere_att8', '>f'),
        ('asphere_aperture_pct', '>f'),
        ('asphere_optimized_r0', '>f'),
        ('iff_state', '<i'),
        ('iff_idr_filename', '42s'),
        ('iff_ise_filename', '42s'),
        (None, '2b'),
        ('asphere_eqn_r0', '>f'),
        ('asphere_eqn_k', '>f'),
        ('asphere_eqn_coef', '>21f'),
        ('awm_enable', '<i'),
        ('awm_vacuum_wavelength_nm', '<f'),
        ('awm_air_wavelength_nm', '<f'),
        ('awm_air_temperature_degc', '<f'),
        ('awm_air_pressure_mmhg', '<f'),
        ('awm_air_rel_humidity_pct', '<f'),
        ('awm_air_quality', '<f'),
        ('awm_input_power_mw', '<f'),
        ('asphere_optimizations', '>i'),
        ('asphere_optimization_mode', '>i'),
        ('asphere_optimized_k', '>f'),
        (None, '2b'),
        ('n_fiducials_b', '>h'),
        ('fiducials_b', '>14f'),
        (None, '2b'),
        ('n_fiducials_c', '>h'),
        ('fiducials_c', '>14f'),
        (None, '2b'),
        ('n_fiducials_d', '>h'),
        ('fiducials_d', '>14f'),
        ('gpi_enc_zoom_mag', '<f'),
        ('asphere_max_distortion', '>f'),
        ('asphere_distortion_uncert', '>f'),
        ('field_stop_name', '12s'),
        ('apert_stop_name', '12s'),
        ('illum_filt_name', '12s'),
        (None, '2608b')
    ]

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Detect file magic
            magic = f.read(4)
            if magic == self._MAGIC1:
                self._file_version = 1
            elif magic == self._MAGIC2:
                self._file_version = 2
            elif magic == self._MAGIC3:
                self._file_version = 3
            else:
                raise FileFormatMismatch('File magic does not match. This is not a Zygo MetroPro file.')

            # Decode header
            self._header, size = decode(f, self._header_structure1, return_size=True)
            # Idiot check, should not fail
            assert size == self._HEADER_SIZE12

            if self._file_version in [1, 2]:
                if self._header['header_size'] != self._HEADER_SIZE12:
                    raise CorruptFile('Reported header size does not match expected header size.')
            else:
                if self._header['header_size'] != self._HEADER_SIZE3:
                    raise CorruptFile('Reported header size does not match expected header size.')
                header, size3 = decode(f, self._header_structure3, return_size=True)
                assert size + size3 == self._HEADER_SIZE3
                self._header.update(header)

            # Check that file size is large enough to hold data
            ac_nb_pixels = self._header['ac_org_height'] * self._header['ac_org_width']
            nx, ny = self._header['cn_org_height'], self._header['cn_org_width']
            cn_nb_pixels = nx * ny
            self._data_offset = \
                self._header['header_size'] + \
                2 * self._header['ac_n_buckets'] * ac_nb_pixels
            expected_file_size = self._data_offset + 4 * cn_nb_pixels
            actual_file_size = f.seek(0, os.SEEK_END)
            if actual_file_size < expected_file_size:
                raise CorruptFile('File is too small to hold reported data buffers.')

        # All good, now initialize some convenience variables
        self._nb_grid_pts = (nx, ny)
        self._physical_sizes = (self._header['lateral_res'] * nx, self._header['lateral_res'] * ny)

        # `phase_res` should only have values 0, 1 or 2. If this fails, the file is likely corrupt.
        max_phase = self._MAX_PHASE[self._header['phase_res']]

        self._height_scale_factor = \
            self._header['intf_scale_factor'] * \
            self._header['obliquity_factor'] * \
            self._header['wavelength_in'] / max_phase
        if self._header['sign']:
            self._height_scale_factor *= -1
        self._unit = 'm'
        self._info = {
            'acquisition_time': str(datetime.datetime.fromtimestamp(self._header['time_stamp'])),
            'raw_metadata': self._header
        }

    def read_height_data(self, f):
        f.seek(self._data_offset)

        dtype = np.dtype('>i')

        nx, ny = self._nb_grid_pts
        buffer = f.read(nx * ny * dtype.itemsize)
        data = np.frombuffer(buffer, dtype=dtype).reshape((nx, ny))
        mask = data >= 2147483640
        if mask.sum() > 0:
            return np.ma.masked_array(data, mask=mask)
        else:
            return data

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=2,
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            height_scale_factor=self._height_scale_factor,
                            uniform=True,
                            unit=self._unit,
                            info=self._info)]

    def topography(self, channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=None, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError('This reader does not support MPI parallelization.')

        if channel_index is None:
            channel_index = self._default_channel_index

        if channel_index != self._default_channel_index:
            raise RuntimeError(f'There is only a single channel. Channel index must be {self._default_channel_index}.')

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        with OpenFromAny(self.file_path, 'rb') as f:
            height_data = self.read_height_data(f)

        _info = self._info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          self._physical_sizes,
                          unit=self._unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(self._height_scale_factor)
