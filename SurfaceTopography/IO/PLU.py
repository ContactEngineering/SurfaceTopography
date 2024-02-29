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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/sensofar.c
#

import dateutil

import numpy as np

from .binary import BinaryArray, BinaryStructure, Convert, Validate
from .Reader import ChannelInfo, DeclarativeReaderBase, CompoundLayout, If, For
from ..Exceptions import UnsupportedFormatFeature

# Measurement types
_TYPE_PROFILE = 1
_TYPE_TOPOGRAPHY = 3

_AREA_COORDINATES = 6

_UNDEFINED_DATA = 1000001

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


class PLUReader(DeclarativeReaderBase):
    _format = 'plu'
    _mime_types = ['application/x-sensofar-spm']
    _file_extensions = ['plu', 'apx']

    _name = 'Sensorfar SPM'
    _description = '''
This reader imports Sensofar's SPM file format.
'''

    _file_layout = CompoundLayout([
        BinaryStructure([
            ('data', '128s', Convert(lambda x: str(dateutil.parser.parse(x)))),
            ('time', 'I'),
            ('comment', '256s')
        ], name='header'),
        BinaryStructure([
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
        ], name='calibration'),
        BinaryStructure([
            # We only support topographies at present
            ('type', 'I', Validate(_TYPE_TOPOGRAPHY, UnsupportedFormatFeature)),
            ('algorithm', 'I'),
            ('method', 'I'),
            ('objective', 'I', Convert(lambda x: _objective_names[x])),
            ('area_type', 'I')
        ], name='measurement_configuration1'),
        If(
            lambda data: data.measurement_configuration1.area_type == _AREA_COORDINATES,
            BinaryStructure([
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
            ], name='scan_settings'),
            BinaryStructure([
                ('xres_area', 'I'),
                ('yres_area', 'I'),
                ('xres', 'I'),
                ('yres', 'I'),
                ('na', 'I'),
                ('incr_z', 'd'),
                ('range', 'f'),
                ('n_planes', 'I'),
                ('tpc_umbral_F', 'I')
            ], name='scan_settings')
        ),
        BinaryStructure([
            ('restore', 'B'),
            ('num_layers', 'B'),
            ('version', 'B'),
            ('config_hardware', 'B'),
            ('num_images', 'B'),
            ('reserved', '3B'),
            ('factor_delmacio', 'I')
        ], name='measurement_configuration2'),
        For(
            lambda data: data.measurement_configuration2.num_layers,
            CompoundLayout([
                BinaryStructure([
                    ('y', 'I'),
                    ('x', 'I')
                ], name='nb_grid_pts'),
                BinaryArray(
                    'data',
                    lambda context: (context.nb_grid_pts.y, context.nb_grid_pts.x),
                    lambda context: np.dtype(np.float32),
                    conversion_fun=lambda arr: arr.T,
                    mask_fun=lambda arr, data: arr == _UNDEFINED_DATA
                ),
                BinaryStructure([
                    ('min', 'f'),
                    ('max', 'f'),
                ], name='min_max')
            ]),
            name='layers'
        )
    ])

    @property
    def channels(self):
        return [
            ChannelInfo(
                self,
                index,
                name=f'layer{index}',
                dim=2,
                nb_grid_pts=(layer.nb_grid_pts.x,
                             layer.nb_grid_pts.y),
                physical_sizes=(layer.nb_grid_pts.x * self._metadata.calibration.micrometers_per_pixel_x,
                                layer.nb_grid_pts.y * self._metadata.calibration.micrometers_per_pixel_y),
                unit='µm',
                height_scale_factor=1,  # All units µm
                periodic=False,
                uniform=True,
                tags={'reader': layer.data}
            )
            for index, layer in enumerate(self._metadata.layers)
        ]
