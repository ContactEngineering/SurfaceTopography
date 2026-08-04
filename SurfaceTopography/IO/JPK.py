#
# Copyright 2023-2026 Lars Pastewka
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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/jpkscan.c
#

from .binary import TIFFContainer
from .expr import C, Cond, DictExpr, F, Tup, V
from .Reader import DeclarativeReaderBase, MagicMatch

# Global metadata tags; these appear on the first (thumbnail) page
_GLOBAL_TAG_NAMES = {
    0x8000: "FileFormatVersion",
    0x8001: "ProgramVersion",
    0x8002: "SavedByProgram",
    0x8003: "StartDate",
    0x8004: "Name",
    0x8005: "Comment",
    0x8006: "EndDate",
    0x8007: "Sample",
    0x8008: "UniqueID",
    0x8009: "AccountName",
    0x8010: "CantileverComment",
    0x8011: "CantileverSpringConst",
    0x8012: "CantileverCalibrated",
    0x8013: "CantileverShape",
    0x8014: "CantileverRadius",
    0x8015: "ApproachID",
    0x8030: "FeedbackMode",
    0x8031: "FeedbackPGain",
    0x8032: "FeedbackIGain",
    0x8033: "FeedbackSetpoint",
    0x8034: "FeedbackVar1",
    0x8035: "FeedbackVar2",
    0x8036: "FeedbackVar3",
    0x8037: "FeedbackVar4",
    0x8038: "FeedbackVar5",
    0x8039: "ApproachIGain",
    0x801A: "ApproachPGain",
    0x801B: "TipsaverSetpoint",
    0x801C: "TipsaverActive",
    0x801D: "TipsaverLowerLimit",
    0x8040: "GridX0",
    0x8041: "GridY0",
    0x8042: "GridULength",
    0x8043: "GridVLength",
    0x8044: "GridTheta",
    0x8045: "GridReflect",
    0x8046: "ScanWidth",
    0x8047: "ScanLength",
    0x8048: "Lineend",
    0x8049: "ScanrateFrequency",
    0x804A: "ScanrateDutycycle",
    0x804B: "Motion",
    0x804C: "ScanlineStart",
    0x804D: "ScanlineSize",
}

# Per-channel metadata tags
_CHANNEL_TAG_NAMES = {
    0x8050: "Channel",
    0x8051: "ChannelRetrace",
    0x8052: "ChannelFancyName",
    0x8060: "CalibrationAge",
    0x8061: "CalibrationOperator",
    0x8080: "NrOfSlots",
    0x8081: "DefaultSlot",
}

# Each channel carries a number of data-interpretation "slots" as
# repeating tag blocks of stride 0x30, starting at tag 0x8090
_SLOT_TAG_NAMES = {
    0x00: "SlotName",
    0x01: "SlotType",
    0x02: "SlotParent",
    0x10: "CalibrationName",
    0x11: "EncoderName",
    0x12: "EncoderUnit",
    0x13: "ScalingType",
    # Multiplier and offset apply if ScalingType == 'LinearScaling'
    0x14: "ScalingMultiply",
    0x15: "ScalingOffset",
    0x16: "ScalingVar3",
    0x17: "ScalingVar4",
    0x18: "ScalingVar5",
}

# Global metadata lives on the first page
_global = C.pages[0].tags

# The default slot of the current channel (page) determines the physical
# interpretation of its data
_slots_by_name = F.index_by(C.item.slots, "SlotName")
_default_slot = F.get(
    _slots_by_name, F.get(C.item.tags, "DefaultSlot", ""), {}
)
_unit = F.get(_default_slot, "EncoderUnit", "")

# Channel name, with a retrace marker appended for retrace channels
_fancy_name = C.item.tags.ChannelFancyName
_channel_name = Cond(
    C.item.tags.ChannelRetrace != 0,
    Cond(
        _fancy_name[-1:] == ")",
        _fancy_name[:-1] + ", retrace)",
        _fancy_name + " (retrace)",
    ),
    _fancy_name,
)

# Conversion from meters to the reported unit
_lateral_conversion = F.unit_conversion_factor("m", _unit)


class JPKReader(DeclarativeReaderBase):
    _format = "jpk"
    _mime_types = ["application/x-jpk-image-scan"]
    _file_extensions = ["jpk"]

    _name = "JPK image scan"
    _description = """
TIFF-based file format of JPK instruments (now Bruker)
"""

    # TIFF magic bytes (little- and big-endian)
    _MAGIC_TIFF = (b"II*\x00", b"MM\x00*")

    # A TIFF magic can only tell that the file *may* be a JPK file;
    # detection needs to try parsing
    _magic = None

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < 4:
            return MagicMatch.MAYBE  # Buffer too short to determine
        if buffer.startswith(cls._MAGIC_TIFF):
            # TIFF file, could be a JPK file
            return MagicMatch.MAYBE
        return MagicMatch.NO

    _file_layout = TIFFContainer(
        tag_names={**_GLOBAL_TAG_NAMES, **_CHANNEL_TAG_NAMES},
        tag_groups={
            "slots": {"first": 0x8090, "stride": 0x30, "names": _SLOT_TAG_NAMES}
        },
        # The data is stored line by line; transpose to (nx, ny) order
        image_conversion=F.transpose(V),
    )

    _channel_bindings = [
        {
            # One channel per TIFF page; we only report channels whose
            # data carries units of length (these are probably height
            # information). The first page is a thumbnail without a
            # channel name.
            "foreach": C.pages,
            "where": F.is_length_unit(_unit)
            & (F.get(C.item.tags, "ChannelFancyName", None) != None),  # noqa: E711
            "checks": [
                {
                    "condition": F.get(_default_slot, "ScalingType", None)
                    != None,  # noqa: E711
                    "error": "corrupt_file",
                    "message": "Default slot does not contain 'ScalingType' "
                    "entry.",
                },
                {
                    "condition": F.get(_default_slot, "ScalingType", None)
                    == "LinearScaling",
                    "error": "unsupported_feature",
                    "message": "Cannot read file that contains data with a "
                    "scaling type other than 'LinearScaling'.",
                },
                {
                    "condition": F.get(_default_slot, "ScalingMultiply", None)
                    != None,  # noqa: E711
                    "error": "corrupt_file",
                    "message": "Default slot does not contain "
                    "'ScalingMultiply' entry.",
                },
            ],
            "name": _channel_name,
            "dim": 2,
            "nb_grid_pts": Tup(
                C.item.tags.ImageWidth, C.item.tags.ImageLength
            ),
            "physical_sizes": Tup(
                _lateral_conversion * _global.GridULength,
                _lateral_conversion * _global.GridVLength,
            ),
            "height_scale_factor": _default_slot.ScalingMultiply,
            "uniform": True,
            "unit": _unit,
            "info": {
                "acquisition_time": F.parse_datetime(_global.StartDate),
                "instrument": {
                    "vendor": "JPK Instruments",
                    "software": Cond(
                        F.get(_global, "ProgramVersion", None)
                        != None,  # noqa: E711
                        F.str(_global.ProgramVersion),
                        None,
                    ),
                },
                "raw_metadata": F.merge(
                    _global,
                    F.merge(
                        C.item.tags,
                        DictExpr({"Slots": _slots_by_name}),
                    ),
                ),
            },
            "data": C.item.image,
        }
    ]
