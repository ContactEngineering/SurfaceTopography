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

#
# Reference information and implementations:
# https://www.seas.upenn.edu/~nanosop/documents/MetroProReferenceGuide0347_M.pdf
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/metropro.c
#

import datetime

import numpy as np

from .binary import decode
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor


class MetroProReader(ReaderBase):
    _format = 'metropro'
    _name = 'Zygo Metropro'
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
        ('', '16b'),
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
        ('fiducials_a',  '14f'),
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
        ('', '2b'),
        ('dmi_ctr_x', '<f'),
        ('dmi_ctr_y', '<f'),
        ('sph_dist_corr', '<h'),
        ('', '2b'),
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
        ('', '1b'),
        ('rc_orientation', 'B'),
        ('rc_distance', '<f'),
        ('rc_angle', '<f'),
        ('rc_diameter', '<f'),
        ('rem_fringes_mode', '>h'),
        ('', '1b'),
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
        ('', '1b'),
        ('pre_fda_filter', '>h'),
        ('', '4b'),
        ('ftpsi_res_factor', '<i'),
        ('', '8b'),
        ('', '4b')
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
                raise FileFormatMismatch('File magic does not match. This is not a Digital Surf file.')

            self._header, size = decode(f, self._header_structure1, return_size=True)
            # Idiot check, should not fail
            assert size == self._HEADER_SIZE12

            if self._file_version in [1, 2]:
                if self._header['header_size'] != self._HEADER_SIZE12:
                    raise CorruptFile('Reported header size does not match expected header size.')
            else:
                if self._header['header_size'] != self._HEADER_SIZE3:
                    raise CorruptFile('Reported header size does not match expected header size.')
                # TODO: Read rest of header

            # Convert time stamp to datetime object
            self._header['time_stamp'] = datetime.datetime.fromtimestamp(self._header['time_stamp'])

        nx = self._header['nb_grid_pts_x']
        ny = self._header['nb_grid_pts_y']
        if nx * ny != self._header['nb_points']:
            raise CorruptFile(
                'The file reported a grid of {} x {} data points and a total number of {} data points, which is '
                'inconsistent'.format(self._nb_grid_pts[0], self._nb_grid_pts[1], self._header['nb_points']))

        if self._header['itemsize'] not in [16, 32]:
            raise CorruptFile('The file reported an item size of {} bits, which I cannot read.'
                              .format(self._header['itemsize']))

        # Check that units and delta units are the same. Not sure if they differ and why there are two different sets
        # of units provided.
        unit_x = self._header['unit_x']
        unit_y = self._header['unit_y']
        self._unit = self._header['data_unit']  # We use the data unit as the primary unit for topography objects

        if self._header['unit_delta_x'] != unit_x or self._header['unit_delta_y'] != unit_y or \
                self._header['delta_data_unit'] != self._unit:
            raise CorruptFile('Units and delta units differ. Not sure how to handle this.')

        # Get the conversion factors for converting x,y to the main system of units
        fac_x = get_unit_conversion_factor(unit_x, self._unit)
        fac_y = get_unit_conversion_factor(unit_y, self._unit)

        # All good, now initialize some convenience variables

        self._nb_grid_pts = (nx, ny)
        self._physical_sizes = \
            (fac_x * self._header['grid_spacing_x'] * nx, fac_y * self._header['grid_spacing_y'] * ny)

        self._info = {
            'instrument': {'name': self._header['instrument_name']},
            'raw_metadata': self._header
        }

        try:
            self._info['acquisition_time'] = \
                str(datetime.datetime(self._header['year'], self._header['month'], self._header['day'],
                                      self._header['hour'], self._header['minute'], self._header['second']))
        except ValueError:
            # This can fail if the date is not valid, e.g. if there are just zeros
            pass

    def read_height_data(self, f):
        if self._header['itemsize'] == 16:
            dtype = np.dtype('<i2')
        elif self._header['itemsize'] == 32:
            dtype = np.dtype('<i4')
        else:
            raise RuntimeError('Unknown itemsize')  # Should not happen because we check this in the constructor

        f.seek(512)

        nx, ny = self._nb_grid_pts
        buffer = f.read(nx * ny * np.dtype(dtype).itemsize)
        return np.frombuffer(buffer, dtype=dtype).reshape(self._nb_grid_pts)

    @property
    def channels(self):
        return [ChannelInfo(self,
                            0,  # channel index
                            name='Default',
                            dim=2,
                            nb_grid_pts=self._nb_grid_pts,
                            physical_sizes=self._physical_sizes,
                            height_scale_factor=float(self._header['height_scale_factor']),
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
        return topo.scale(float(self._header['height_scale_factor']))
