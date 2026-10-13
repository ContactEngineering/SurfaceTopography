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

from ..Exceptions import UnsupportedFormatFeature
from .binary import BinaryArray, BinaryStructure, Validate
from .expr import C, Cond, F, Lit, Tup, V
from .Reader import CompoundLayout, DeclarativeReaderBase, For, If, Skip

# Measurement types
_TYPE_PROFILE = 1
_TYPE_TOPOGRAPHY = 3

_AREA_COORDINATES = 6

_UNDEFINED_DATA = 1000001

# Format versions are stored as a single byte. Version 2000 is zero, later
# versions count *down* from 0xff (2006) to 0xfa (2013).
_VERSION_2000 = 0x00
_VERSION_2012 = 0xFB

# Instrument (hardware configuration) names, keyed by the decimal string of
# the hardware configuration ID. The ID is only meaningful for format
# versions 2012 and newer; older files carry unreliable values here.
_hardware_names = {
    "0": "PLu",
    "1": "PLu 2300 XGA",
    "2": "PLu 2300 XGA T5",
    "3": "PLu 2300 SXGA",
    "4": "PLu 3300",
    "5": "DCM 3D",
    "6": "PLu neox",
    "7": "DCM 3D rev. 2",
    "8": "PLu Apex (prototype)",
    "9": "S neox",
    "10": "DCM8",
    "11": "PLu Apex",
}

# Objectives, keyed by the (decimal string of the) objective ID stored in
# the file. Note that IDs 65 to 71 are not assigned.
_objective_names = {
    "0": "Unknown",
    "1": "Nikon CFI Fluor Plan EPI SLWD 20x",
    "2": "Nikon CFI Fluor Plan EPI SLWD 50x",
    "3": "Nikon CFI Fluor Plan EPI SLWD 100x",
    "4": "Nikon CFI Fluor Plan EPI 20x",
    "5": "Nikon CFI Fluor Plan EPI 50x",
    "6": "Nikon CFI Fluor Plan EPI 10x",
    "7": "Nikon CFI Fluor Plan EPI 100x",
    "8": "Nikon CFI Fluor Plan EPI ELWD 10x",
    "9": "Nikon CFI Fluor Plan EPI ELWD 20x",
    "10": "Nikon CFI Fluor Plan EPI ELWD 50x",
    "11": "Nikon CFI Fluor Plan EPI ELWD 100x",
    "12": "Nikon CFI Plan Interferential 2.5X",
    "13": "Nikon CFI Plan Interferential 5X T",
    "14": "Nikon CFI Plan Interferential 10X",
    "15": "Nikon CFI Plan Interferential 20X",
    "16": "Nikon CFI Plan Interferential 50X",
    "17": "Nikon CFI Fluor Plan EPI 5X",
    "18": "Nikon CFI Fluor Plan EPI 150X",
    "19": "Nikon CFI Fluor Plan Apo EPI 50X",
    "20": "Nikon CFI Fluor Plan EPI 1.5X",
    "21": "Nikon CFI Fluor Plan EPI 2.5X",
    "22": "Nikon CFI Fluor Plan Apo EPI 100X",
    "23": "Nikon CFI Fluor Plan EPI 200X",
    "24": "Nikon CFI Plan Water Immersion 10X",
    "25": "Nikon CFI Plan Water Immersion 20X",
    "26": "Nikon CFI Plan Water Immersion 150X",
    "27": "Nikon CFI Plan EPI CR ELWD 10X",
    "28": "Nikon CFI Plan EPI CR 20X",
    "29": "Nikon CFI Plan EPI CR 50X",
    "30": "Nikon CFI Plan EPI CR 100X A",
    "31": "Nikon CFI Plan EPI CR 100X B",
    "32": "Leica HCX FL Plan 2.5X",
    "33": "Leica HC PL Fluotar EPI 5X",
    "34": "Leica HC PL Fluotar EPI 10X",
    "35": "Leica HC PL Fluotar EPI 20X",
    "36": "Leica HC PL Fluotar EPI 50X",
    "37": "Leica HC PL Fluotar EPI 50X HNA",
    "38": "Leica HC PL Fluotar EPI 100X",
    "39": "Leica HC PL Fluotar EPI 50X",
    "40": "Leica N Plan EPI LWD 10X",
    "41": "Leica N Plan EPI LWD 20X",
    "42": "Leica HCX PL Fluotar LWD 50X",
    "43": "Leica HCX PL Fluotar LWD 100X",
    "44": "Leica HC PL Fluotar – Interferential Michelson MR 5X",
    "45": "Leica HC PL Fluotar – Interferential Mirau MR 10X",
    "46": "Leica N PLAN H - Interferential Mirau MR 20X",
    "47": "Leica N PLAN H -Interferential Mirau MR 50X",
    "48": "Nikon Interferential Linnik EPI 20X",
    "49": "Nikon CFI Plan Interferential 100X DI",
    "50": "Leica HCX PL FLUOTAR 1.25X",
    "51": "Leica N PLAN EPI 20X",
    "52": "Leica N PLAN EPI 40X",
    "53": "Leica N PLAN L 50X",
    "54": "Leica PL APO 100X",
    "55": "Leica HCX APO L U-V-I 20X",
    "56": "Leica HCX APO L U-V-I 40X",
    "57": "Leica HCX APO L U-V-I 63X",
    "58": "Leica HCX PL FLUOTAR 20X",
    "59": "Leica N PLAN L 40X",
    "60": "Leica Interferential Mirau SR 5X",
    "61": "Leica Interferential Mirau SR 10X",
    "62": "Leica Interferential Mirau SR 20X",
    "63": "Leica Interferential Mirau SR 50X",
    "64": "Leica Interferential Mirau SR 100X",
    "72": "Leica HC PL Fluotar EPI 50X 0.8",
    "73": "Leica HC PL Fluotar EPI 100X 0.9",
    "74": "Nikon CFI T Plan EPI 1X",
    "75": "Nikon CFI T Plan EPI 2.5X",
    "76": "Nikon CFI TU Plan Fluor EPI 5X",
    "77": "Nikon CFI TU Plan Fluor EPI 10X",
    "78": "Nikon CFI TU Plan Fluor EPI 20X",
    "79": "Nikon CFI LU Plan Fluor EPI 50X",
    "80": "Nikon CFI TU Plan Fluor EPI 100X",
    "81": "Nikon CFI EPI Plan Apo 150X",
    "82": "Nikon CFI T Plan EPI ELWD 20X (AV 3.5)",
    "83": "Nikon CFI T Plan EPI ELWD 50X (AV 3.5)",
    "84": "Nikon CFI T Plan EPI ELWD 100X (AV 3.5)",
    "85": "Nikon CFI T Plan EPI SLWD 10X (AV 3.5)",
    "86": "Nikon CFI T Plan EPI SLWD 20X (AV 3.5)",
    "87": "Nikon CFI T Plan EPI SLWD 50X (AV 3.5)",
    "88": "Nikon CFI T Plan EPI SLWD 100X (AV 3.5)",
    "89": "Nikon CFI Fluor Water Immersion 63X",
    "90": "Nikon CFI TU Plan Fluor EPI 50X",
    "91": "Nikon CFI TU Plan Apo EPI 100X",
    "92": "Nikon CFI TU Plan Apo EPI 150X",
}

_version = C.measurement_configuration2.version
_hardware_name = Cond(
    (_version != _VERSION_2000) & (_version <= _VERSION_2012),
    F.get(
        Lit(_hardware_names),
        F.str(C.measurement_configuration2.config_hardware),
        None,
    ),
    None,
)


class PLUReader(DeclarativeReaderBase):
    _format = "plu"
    _mime_types = ["application/x-sensofar-spm"]
    _file_extensions = ["plu", "apx"]

    _name = "Sensofar PLU"
    _description = """
PLU (and APX) files of Sensofar 3D optical profilometers (confocal,
interferometric and focus-variation instruments).
"""

    _file_layout = CompoundLayout(
        [
            BinaryStructure(
                [
                    (
                        "data",
                        "128s",
                        F.parse_datetime(V),
                    ),
                    ("time", "I"),
                    ("comment", "256s"),
                ],
                name="header",
            ),
            BinaryStructure(
                [
                    ("nb_grid_pts_y", "I"),
                    ("nb_grid_pts_x", "I"),
                    ("N_tall", "I"),
                    ("dy_multip", "f"),
                    ("micrometers_per_pixel_x", "f"),
                    ("micrometers_per_pixel_y", "f"),
                    ("offset_x", "f"),
                    ("offset_y", "f"),
                    ("micrometers_per_pixel_tall", "f"),
                    ("offset_z", "f"),
                ],
                name="calibration",
            ),
            BinaryStructure(
                [
                    # We only support topographies at present
                    ("type", "I", Validate(_TYPE_TOPOGRAPHY, UnsupportedFormatFeature)),
                    ("algorithm", "I"),
                    ("method", "I"),
                    (
                        "objective",
                        "I",
                        F.get(Lit(_objective_names), F.str(V), None),
                    ),
                    ("area_type", "I"),
                ],
                name="measurement_configuration1",
            ),
            If(
                C.measurement_configuration1.area_type == _AREA_COORDINATES,
                BinaryStructure(
                    [
                        ("tracking_range", "f"),
                        ("tracking_speed", "f"),
                        ("tracking_direction", "I"),
                        ("tracking_threshold", "f"),
                        ("tracking_min_angle", "f"),
                        ("confocal_scan_type", "I"),
                        ("confocal_scan_range", "f"),
                        ("confocal_speed_factor", "f"),
                        ("confocal_threshold", "f"),
                        ("reserved", "4B"),
                    ],
                    name="scan_settings",
                ),
                BinaryStructure(
                    [
                        ("xres_area", "I"),
                        ("yres_area", "I"),
                        ("xres", "I"),
                        ("yres", "I"),
                        ("na", "I"),
                        ("incr_z", "d"),
                        ("range", "f"),
                        ("n_planes", "I"),
                        ("tpc_umbral_F", "I"),
                    ],
                    name="scan_settings",
                ),
            ),
            BinaryStructure(
                [
                    ("restore", "B"),
                    ("num_layers", "B"),
                    ("version", "B"),
                    ("config_hardware", "B"),
                    ("num_images", "B"),
                    ("reserved", "3B"),
                    ("factor_delmacio", "I"),
                ],
                name="measurement_configuration2",
            ),
            For(
                C.measurement_configuration2.num_layers,
                CompoundLayout(
                    [
                        BinaryStructure([("y", "I"), ("x", "I")], name="nb_grid_pts"),
                        BinaryArray(
                            "data",
                            Tup(C.nb_grid_pts.y, C.nb_grid_pts.x),
                            F.dtype("<f4"),
                            conversion_fun=F.transpose(V),
                            mask_fun=V == _UNDEFINED_DATA,
                        ),
                        BinaryStructure(
                            [
                                ("min", "f"),
                                ("max", "f"),
                            ],
                            name="min_max",
                        ),
                        # Each topography layer is followed by
                        # `num_images` 24-bit RGB images (e.g. the
                        # confocal or colour image) of the same size
                        Skip(
                            3
                            * C.nb_grid_pts.x
                            * C.nb_grid_pts.y
                            * C.__parent__.measurement_configuration2.num_images,
                            comment="RGB images",
                        ),
                    ]
                ),
                name="layers",
            ),
        ]
    )

    _channel_bindings = [
        {
            # One channel per measurement layer
            "foreach": C.layers,
            "name": "layer" + F.str(C.item_index),
            "dim": 2,
            "nb_grid_pts": Tup(C.item.nb_grid_pts.x, C.item.nb_grid_pts.y),
            "physical_sizes": Tup(
                C.item.nb_grid_pts.x * C.calibration.micrometers_per_pixel_x,
                C.item.nb_grid_pts.y * C.calibration.micrometers_per_pixel_y,
            ),
            "unit": "µm",
            "height_scale_factor": 1,  # All units µm
            "periodic": False,
            "uniform": True,
            "info": {
                "acquisition_time": C.header.data,
                "instrument": {
                    "vendor": "Sensofar",
                    "name": _hardware_name,
                },
                "raw_metadata": {
                    "comment": C.header.comment,
                    "calibration": C.calibration,
                    "measurement_configuration": F.merge(
                        C.measurement_configuration1,
                        C.measurement_configuration2,
                    ),
                    "scan_settings": C.scan_settings,
                },
            },
            "data": C.item.data,
        }
    ]
