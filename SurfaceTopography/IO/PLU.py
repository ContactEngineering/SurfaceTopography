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

#
# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/sensofar.c
#

import dateutil
from struct import unpack

import numpy as np

from .binary import decode
from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import MetadataAlreadyFixedByFile
from ..UniformLineScanAndTopography import Topography


class PLUReader(ReaderBase):
    _format = 'plu'
    _mime_types = ['application/x-sensofar-spm']
    _file_extensions = ['plu', 'apx']

    _name = 'Sensorfar SPM'
    _description = '''
This reader imports Sensofar's SPM file format.
'''

    _DATE_SIZE = 128
    _COMMENT_SIZE = 256
    _HEADER_SIZE = 500

    _AREA_COORDINATES = 6

    _UNDEFINED_DATA = 1000001

    # Measurement types
    _TYPE_PROFILE = 1
    _TYPE_TOPOGRAPHY = 3

    _fov_scan_settings_structure = [
        ('xres_area', 'I'),
        ('yres_area', 'I'),
        ('xres', 'I'),
        ('yres', 'I'),
        ('na', 'I'),
        ('incr_z', 'd'),
        ('range', 'f'),
        ('n_planes', 'I'),
        ('tpc_umbral_F', 'I')
    ]

    _point_scan_settings_structure = [
        ('tracking_range', 'f'),
        ('tracking_speed', 'f'),
        ('tracking_direction', 'I'),
        ('tracking_threshold', 'f'),
        ('tracking_min_angle', 'f'),
        ('confocal_scan_type', 'I'),
        ('confocal_scan_range', 'f'),
        ('confocal_speed_factor', 'f'),
        ('confocal_threshold', 'f'),
        ('reserved', '4B')
    ]

    _measurement_configuration_structure1 = [
        ('type', 'I'),
        ('algorithm', 'I'),
        ('method', 'I'),
        ('objective', 'I'),
        ('area_type', 'I')
    ]

    _measurement_configuration_structure2 = [
        ('restore', 'B'),
        ('num_layers', 'B'),
        ('version', 'B'),
        ('config_hardware', 'B'),
        ('num_images', 'B'),
        ('reserved', '3B'),
        ('factor_delmacio', 'I')
    ]

    _calibration_structure = [
        ('nb_grid_pts_y', 'I'),
        ('nb_grid_pts_x', 'I'),
        ('N_tall', 'I'),
        ('dy_multip', 'f'),
        ('micrometers_per_pixel_x', 'f'),
        ('micrometers_per_pixel_y', 'f'),
        ('offset_x', 'f'),
        ('offset_y', 'f'),
        ('micrometers_per_pixel_tall', 'f'),
        ('offset_z', 'f')
    ]

    _objective_names = [
        "Unknown",
        "Nikon CFI Fluor Plan EPI SLWD 20x",
        "Nikon CFI Fluor Plan EPI SLWD 50x",
        "Nikon CFI Fluor Plan EPI SLWD 100x",
        "Nikon CFI Fluor Plan EPI 20x",
        "Nikon CFI Fluor Plan EPI 50x",
        "Nikon CFI Fluor Plan EPI 10x",
        "Nikon CFI Fluor Plan EPI 100x",
        "Nikon CFI Fluor Plan EPI ELWD 10x",
        "Nikon CFI Fluor Plan EPI ELWD 20x",
        "Nikon CFI Fluor Plan EPI ELWD 50x",
        "Nikon CFI Fluor Plan EPI ELWD 100x",
        "Nikon CFI Plan Interferential 2.5X",
        "Nikon CFI Plan Interferential 5X T",
        "Nikon CFI Plan Interferential 10X",
        "Nikon CFI Plan Interferential 20X",
        "Nikon CFI Plan Interferential 50X",
        "Nikon CFI Fluor Plan EPI 5X",
        "Nikon CFI Fluor Plan EPI 150X",
        "Nikon CFI Fluor Plan Apo EPI 50X",
        "Nikon CFI Fluor Plan EPI 1.5X",
        "Nikon CFI Fluor Plan EPI 2.5X",
        "Nikon CFI Fluor Plan Apo EPI 100X",
        "Nikon CFI Fluor Plan EPI 200X",
        "Nikon CFI Plan Water Immersion 10X",
        "Nikon CFI Plan Water Immersion 20X",
        "Nikon CFI Plan Water Immersion 150X",
        "Nikon CFI Plan EPI CR ELWD 10X",
        "Nikon CFI Plan EPI CR 20X",
        "Nikon CFI Plan EPI CR 50X",
        "Nikon CFI Plan EPI CR 100X A",
        "Nikon CFI Plan EPI CR 100X B",
        "Leica HCX FL Plan 2.5X",
        "Leica HC PL Fluotar EPI 5X",
        "Leica HC PL Fluotar EPI 10X",
        "Leica HC PL Fluotar EPI 20X",
        "Leica HC PL Fluotar EPI 50X",
        "Leica HC PL Fluotar EPI 50X HNA",
        "Leica HC PL Fluotar EPI 100X",
        "Leica HC PL Fluotar EPI 50X",
        "Leica N Plan EPI LWD 10X",
        "Leica N Plan EPI LWD 20X",
        "Leica HCX PL Fluotar LWD 50X",
        "Leica HCX PL Fluotar LWD 100X",
        "Leica HC PL Fluotar – Interferential Michelson MR 5X",
        "Leica HC PL Fluotar – Interferential Mirau MR 10X",
        "Leica N PLAN H - Interferential Mirau MR 20X",
        "Leica N PLAN H -Interferential Mirau MR 50X",
        "Nikon Interferential Linnik EPI 20X",
        "Nikon CFI Plan Interferential 100X DI",
        "Leica HCX PL FLUOTAR 1.25X",
        "Leica N PLAN EPI 20X",
        "Leica N PLAN EPI 40X",
        "Leica N PLAN L 50X",
        "Leica PL APO 100X",
        "Leica HCX APO L U-V-I 20X",
        "Leica HCX APO L U-V-I 40X",
        "Leica HCX APO L U-V-I 63X",
        "Leica HCX PL FLUOTAR 20X",
        "Leica N PLAN L 40X",
        "Leica Interferential Mirau SR 5X",
        "Leica Interferential Mirau SR 10X",
        "Leica Interferential Mirau SR 20X",
        "Leica Interferential Mirau SR 50X",
        "Leica Interferential Mirau SR 100X",
        "Leica HC PL Fluotar EPI 50X 0.8",
        "Leica HC PL Fluotar EPI 100X 0.9",
        "Nikon CFI T Plan EPI 1X",
        "Nikon CFI T Plan EPI 2.5X",
        "Nikon CFI TU Plan Fluor EPI 5X",
        "Nikon CFI TU Plan Fluor EPI 10X",
        "Nikon CFI TU Plan Fluor EPI 20X",
        "Nikon CFI LU Plan Fluor EPI 50X",
        "Nikon CFI TU Plan Fluor EPI 100X",
        "Nikon CFI EPI Plan Apo 150X",
        "Nikon CFI T Plan EPI ELWD 20X (AV 3.5)",
        "Nikon CFI T Plan EPI ELWD 50X (AV 3.5)",
        "Nikon CFI T Plan EPI ELWD 100X (AV 3.5)",
        "Nikon CFI T Plan EPI SLWD 10X (AV 3.5)",
        "Nikon CFI T Plan EPI SLWD 20X (AV 3.5)",
        "Nikon CFI T Plan EPI SLWD 50X (AV 3.5)",
        "Nikon CFI T Plan EPI SLWD 100X (AV 3.5)",
        "Nikon CFI Fluor Water Immersion 63X",
        "Nikon CFI TU Plan Fluor EPI 50X",
        "Nikon CFI TU Plan Apo EPI 100X",
        "Nikon CFI TU Plan Apo EPI 150X"
    ]

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self.file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            # Just read header, there is no file magic
            self._acquisition_date = dateutil.parser.parse(f.read(self._DATE_SIZE).decode('ascii'))
            self._acquisition_time, = unpack('<I', f.read(4))
            self._comment = f.read(self._COMMENT_SIZE).decode('ascii')

            # Read file metadata
            self._calibration = decode(f, self._calibration_structure, '<')

            self._measurement_configuration = decode(f, self._measurement_configuration_structure1, '<')
            if self._measurement_configuration['area_type'] == self._AREA_COORDINATES:
                self._scan_settings = decode(f, self._point_scan_settings_structure, '<')
            else:
                self._scan_settings = decode(f, self._fov_scan_settings_structure, '<')
            self._measurement_configuration.update(decode(f, self._measurement_configuration_structure2, '<'))

            self._measurement_configuration['objective'] = \
                self._objective_names[self._measurement_configuration['objective']]

            # Create global metadata dictionary
            self._info = {
                'raw_metadata': {
                    'calibration': self._calibration,
                    'measurement_configuration': self._measurement_configuration,
                    'scan_settings': self._scan_settings
                }
            }

            # Read and fill channel information
            self._channels = []

            # Topography
            measurement_type = self._measurement_configuration['type']
            if measurement_type == self._TYPE_TOPOGRAPHY:
                index = 0
                for i in range(self._measurement_configuration['num_layers']):
                    # Get number of data points
                    nb_grid_pts_y, nb_grid_pts_x = unpack('<II', f.read(8))

                    # Compute physical sizes (units are um)
                    physical_size_x = nb_grid_pts_x * self._calibration['micrometers_per_pixel_x']
                    physical_size_y = nb_grid_pts_y * self._calibration['micrometers_per_pixel_y']

                    # Construct channel info
                    self._channels += [
                        ChannelInfo(
                            self,
                            index,
                            name='default',
                            dim=2,
                            nb_grid_pts=(nb_grid_pts_x, nb_grid_pts_y),
                            physical_sizes=(physical_size_x, physical_size_y),
                            unit='µm',
                            height_scale_factor=1,  # All units µm
                            periodic=False,
                            uniform=True,
                            info=self._info,
                            tags={'offset': f.tell()}  # This is where the data block starts
                        )
                    ]
                    index += 1

                    # Skip data
                    f.seek(4 * (nb_grid_pts_x * nb_grid_pts_y + 2))  # The last two floats are min/max values
            else:
                raise NotImplementedError(f'Reading measurements of type {measurement_type} is not implemented.')

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

        if channel_index != self._default_channel_index:
            raise RuntimeError(f'There is only a single channel. Channel index must be {self._default_channel_index}.')

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if unit is not None:
            raise MetadataAlreadyFixedByFile('unit')

        channel = self._channels[channel_index]
        with OpenFromAny(self.file_path, 'rb') as f:
            nb_grid_pts_x, nb_grid_pts_y = channel.nb_grid_pts
            f.seek(channel.tags['offset'])
            height_data = np.fromfile(f, np.float32, count=np.prod(channel.nb_grid_pts)) \
                .reshape((nb_grid_pts_y, nb_grid_pts_x)).T
            height_data = np.ma.masked_array(height_data, mask=height_data == self._UNDEFINED_DATA)

        _info = self._info.copy()
        _info.update(info)

        topo = Topography(height_data,
                          channel.physical_sizes,
                          unit=channel.unit,
                          periodic=False if periodic is None else periodic,
                          info=_info)
        return topo.scale(1)  # This fixes the height scale factor in topobank
