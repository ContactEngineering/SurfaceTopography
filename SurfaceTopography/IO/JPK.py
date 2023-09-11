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

# Reference information and implementations:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/jpkscan.c

import enum

import dateutil
from tiffile import TiffFile, TiffFileError

from .common import OpenFromAny
from .Reader import ReaderBase, ChannelInfo
from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile, UnsupportedFormatFeature
from ..UniformLineScanAndTopography import Topography
from ..Support.UnitConversion import get_unit_conversion_factor, is_length_unit


class JPKReader(ReaderBase):
    _format = 'jpk'
    _mime_types = ['application/x-jpk-image-scan']
    _file_extensions = ['jpk']

    _name = 'JPK image scan'
    _description = '''
TIFF-based file format of JPK instruments (now Bruker)
'''

    _global_tag_names = {
        0x8000: 'FileFormatVersion',
        0x8001: 'ProgramVersion',
        0x8002: 'SavedByProgram',
        0x8003: 'StartDate',
        0x8004: 'Name',
        0x8005: 'Comment',
        0x8006: 'EndDate',
        0x8007: 'Sample',
        0x8008: 'UniqueID',
        0x8009: 'AccountName',
        0x8010: 'CantileverComment',
        0x8011: 'CantileverSpringConst',
        0x8012: 'CantileverCalibrated',
        0x8013: 'CantileverShape',
        0x8014: 'CantileverRadius',
        0x8015: 'ApproachID',
        0x8030: 'FeedbackMode',
        0x8031: 'FeedbackPGain',
        0x8032: 'FeedbackIGain',
        0x8033: 'FeedbackSetpoint',
        0x8034: 'FeedbackVar1',
        0x8035: 'FeedbackVar2',
        0x8036: 'FeedbackVar3',
        0x8037: 'FeedbackVar4',
        0x8038: 'FeedbackVar5',
        0x8039: 'ApproachIGain',
        0x801A: 'ApproachPGain',
        0x801B: 'TipsaverSetpoint',
        0x801C: 'TipsaverActive',
        0x801D: 'TipsaverLowerLimit',
        0x8040: 'GridX0',
        0x8041: 'GridY0',
        0x8042: 'GridULength',
        0x8043: 'GridVLength',
        0x8044: 'GridTheta',
        0x8045: 'GridReflect',
        0x8046: 'ScanWidth',
        0x8047: 'ScanLength',
        0x8048: 'Lineend',
        0x8049: 'ScanrateFrequency',
        0x804A: 'ScanrateDutycycle',
        0x804B: 'Motion',
        0x804C: 'ScanlineStart',
        0x804D: 'ScanlineSize',
        # 0x8050: 'ForceSettingsName',
        # 0x8051: 'K_Length',
        # 0x8052: 'ForceMap_Feedback_Mode',
        # 0x8053: 'Z_Start',
        # 0x8054: 'Z_End',
        # 0x8055: 'Setpoint',
        # 0x8056: 'PauseAtEnd',
        # 0x8057: 'PauseAtStart',
        # 0x8058: 'PauseOnTipsaver',
        # 0x8059: 'TraceScanTime',
        # 0x805A: 'RetraceScanTime',
        # 0x805B: 'Z_Start_Pause_Option',
        # 0x805C: 'Z_End_Pause_Option',
        # 0x805D: 'Tipsaver_Pause_Option',
        # 0x8060: 'Scanner',
        # 0x8061: 'FitAlgorithmName',
        # 0x8062: 'LastIndex',
        # 0x8063: 'BackAndForth',

    }

    _channel_tag_names = {
        0x8050: 'Channel',
        0x8051: 'ChannelRetrace',
        0x8052: 'ChannelFancyName',
        0x8060: 'CalibrationAge',
        0x8061: 'CalibrationOperator',
        0x8080: 'NrOfSlots',
        0x8081: 'DefaultSlot',
    }

    _slot_tag_names = {
        0x8090: 'SlotName',
        0x8091: 'SlotType',
        0x8092: 'SlotParent',
        0x80A0: 'CalibrationName',
        0x80A1: 'EncoderName',
        0x80A2: 'EncoderUnit',
        0x80A3: 'ScalingType',
        0x80A4: 'ScalingMultiply',  # This is only the multiplier if ScalingType == 'LinearScaling'
        0x80A5: 'ScalingOffset',  # This is only the offset if ScalingType == 'LinearScaling'
        0x80A6: 'ScalingVar3',
        0x80A7: 'ScalingVar4',
        0x80A8: 'ScalingVar5',
    }

    _tag_readers = {}

    @classmethod
    def _tag_value(cls, key, value):
        # Try dedicated tag readers first
        try:
            return cls._tag_readers[key](value)
        except KeyError:
            pass

        # Generic treatment if failed
        if isinstance(value, enum.Enum):
            return [value.name, value.value]
        else:
            if isinstance(value, tuple):
                value = list(value)
            return value

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        self._file_path = file_path

        # All units appear to be picometers, also some files report 'INCH' as resolution unit
        self._unit = "µm"

        with OpenFromAny(self._file_path, "rb") as f:
            try:
                with TiffFile(f) as t:
                    # Go through all pages and see what is in there
                    global_metadata = {}
                    channels = []
                    for i, p in enumerate(t.pages):
                        # We distinguish between global metadata, channel metadata and slots
                        channel_metadata = {}
                        slots = []
                        uncategorized_metadata = {}
                        for key, value in p.tags.items():
                            v = self._tag_value(key, value.value)
                            if value.name == str(key):
                                try:
                                    # Try to get name from global dictionary
                                    key = self._global_tag_names[key]
                                    global_metadata.update({key: v})
                                except KeyError:
                                    try:
                                        # Try to get from channel dictionary
                                        key = self._channel_tag_names[key]
                                        channel_metadata.update({key: v})
                                    except KeyError:
                                        try:
                                            # Try to get key from slot dictionary
                                            slot_index = (key - 0x8090) // 0x30
                                            if slot_index < 0:
                                                raise KeyError
                                            key = self._slot_tag_names[key - slot_index * 0x30]
                                            if slot_index == len(slots):
                                                slots += [{key: v}]
                                            elif slot_index == len(slots) - 1:
                                                slots[slot_index][key] = v
                                            else:
                                                raise CorruptFile('Metadata is missing a slot')
                                        except KeyError:
                                            # We don't know what this is
                                            uncategorized_metadata.update({key: v})
                            else:
                                channel_metadata.update({value.name: v})

                        # Reorganize slots into dictionary
                        slot_dict = {}
                        for slot in slots:
                            slot_dict[slot['SlotName']] = slot

                        # Store slots to channel metadata
                        channel_metadata['Slots'] = slot_dict
                        channels += [channel_metadata]

            except TiffFileError:
                raise FileFormatMismatch("This is not a TIFF file, so it cannot be a JPK file.")

        # Acquisition date
        acquisition_date = dateutil.parser.parse(global_metadata['StartDate'])  # There is also an EndData

        # We now detect channels with height information
        self._channels = []
        for page_index, channel in enumerate(channels):
            slots = channel['Slots']
            default_slot_name = channel['DefaultSlot']
            default_slot = slots[default_slot_name]
            default_slot_unit = default_slot['EncoderUnit']
            if is_length_unit(default_slot_unit):
                # This has units of length, so it is probably height information
                name = channel['ChannelFancyName']
                if channel['ChannelRetrace']:
                    if name.endswith(')'):
                        name = name[:-1] + ', retrace)'
                    else:
                        name = name + ' (retrace)'

                slot = default_slot
                try:
                    scaling_type = slot['ScalingType']
                except KeyError as exc:
                    raise CorruptFile("Default slot does not contain 'ScalingType' entry.") from exc
                if scaling_type == 'LinearScaling':
                    try:
                        height_scale_factor = slot['ScalingMultiply']
                    except KeyError as exc:
                        raise CorruptFile("Default slot does not contain 'ScalingMultiply' entry.") from exc
                else:
                    raise UnsupportedFormatFeature("Cannot read file that contains data with scaling type "
                                                   f"'{scaling_type}'.")

                # Conversion from meters to reported unit
                fac = get_unit_conversion_factor('m', default_slot_unit)

                # Construct raw metadata dictionary
                raw_metadata = global_metadata.copy()
                raw_metadata.update(channel_metadata)

                # Construct channel info
                self._channels += [ChannelInfo(self,
                                               len(self._channels),  # channel index
                                               name=name,
                                               dim=2,
                                               nb_grid_pts=(channel['ImageWidth'],
                                                            channel['ImageLength']),
                                               physical_sizes=(fac * global_metadata['GridULength'],
                                                               fac * global_metadata['GridVLength']),
                                               height_scale_factor=height_scale_factor,
                                               uniform=True,
                                               unit=default_slot_unit,
                                               info={'acquisition_date': str(acquisition_date),
                                                     'raw_metadata': raw_metadata},
                                               tags={'page_index': page_index})]

    @property
    def channels(self):
        return self._channels

    def topography(
            self,
            channel_index=None,
            physical_sizes=None,
            height_scale_factor=None,
            unit=None,
            info={},
            periodic=None,
            subdomain_locations=None,
            nb_subdomain_grid_pts=None,
    ):
        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        if channel_index is None:
            channel_index = self._default_channel_index

        try:
            channel = self.channels[channel_index]
        except KeyError as exc:
            raise RuntimeError(f"Channel index must be in range 0 to {len(self._channels) - 1}.") from exc

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")

        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")

        with OpenFromAny(self._file_path, "rb") as f:
            with TiffFile(f) as t:
                height_data = t.pages[channel.tags['page_index']].asarray().T
                assert height_data.shape == channel.nb_grid_pts

        _info = channel.info.copy()
        _info.update(info)

        topo = Topography(
            height_data,
            channel.physical_sizes,
            unit=channel.unit,
            periodic=False if periodic is None else periodic,
            info=_info,
        )
        return topo.scale(channel.height_scale_factor)
