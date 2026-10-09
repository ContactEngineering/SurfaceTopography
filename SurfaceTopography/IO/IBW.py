#
# Copyright 2019-2021, 2023 Lars Pastewka
#           2019-2021 Michael Röttger
#           2019 Kai Haase
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
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/igorfile.c
#

import datetime
import re

import numpy as np
from igor2.binarywave import load as loadibw

from ..Exceptions import MetadataAlreadyFixedByFile, UnsupportedFormatFeature
from ..Support.UnitConversion import (
    get_unit_conversion_factor,
    is_length_unit,
    mangle_length_unit_utf8,
)
from ..UniformLineScanAndTopography import Topography, UniformLineScan
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase

# Asylum Research writes the same `dataUnits` into the wave header for all
# channels of an image, even for channels that are not lengths (e.g. phase
# in degrees). The physical unit of a channel follows from the prefix of
# its base name instead, unless the note has an explicit `<base name>Unit`
# entry. Channels whose prefix is not listed here are voltages.
_ASYLUM_UNIT_BY_PREFIX = [
    ("Height", "m"),
    ("ZSensor", "m"),
    ("Deflection", "m"),
    ("Amplitude", "m"),
    ("Phase", "deg"),
    ("Current", "A"),
    ("Frequency", "Hz"),
    ("Capacitance", "F"),
    ("Potential", "V"),
    ("Count", None),
    ("QFactor", None),
]
_ASYLUM_DEFAULT_UNIT = "V"


def _normalize_unit(unit):
    """
    Normalize a unit string (e.g. 'um' to 'µm'). Igor and Asylum Research
    use SI symbols, hence 'A' is the ampere and not the ångström.
    """
    return mangle_length_unit_utf8(unit, ascii_angstrom=False)


def _is_length_unit(unit):
    """Whether `unit` is a length unit; 'A' is the ampere (see above)."""
    return is_length_unit(unit, ascii_angstrom=False)


def _decode_unit(raw):
    """Decode a zero-padded unit entry of the wave header."""
    return _normalize_unit(raw.tobytes().partition(b"\0")[0].decode("latin-1"))


def _asylum_base_name(title):
    """
    Base name of an Asylum Research channel: The channel name without a
    trailing modulation suffix ('Mod' plus digits) and without the scan
    direction suffix ('Trace' or 'Retrace').
    """
    name = re.sub(r"Mod[0-9]*$", "", title)
    for suffix in ("Trace", "Retrace"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _asylum_channel_unit(title, note):
    """Physical unit of the data of an Asylum Research channel."""
    base_name = _asylum_base_name(title)
    unit = note.get(f"{base_name}Unit")
    if unit is not None:
        return _normalize_unit(unit)
    # DAC and Nap (second pass) channels report the same quantity as the
    # channel without this prefix
    for prefix in ("DAC", "Nap"):
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
            break
    for prefix, unit in _ASYLUM_UNIT_BY_PREFIX:
        if base_name.startswith(prefix):
            return unit
    return _ASYLUM_DEFAULT_UNIT


def _parse_note(note):
    """Parse the 'key: value' lines of the wave note."""
    metadata = {}
    for line in note.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def _acquisition_time(note):
    """Acquisition time from the (Asylum Research) note, if present."""
    date, time = note.get("Date"), note.get("Time")
    if not date or not time:
        return None
    for fmt in ("%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(f"{date} {time}", fmt)
        except ValueError:
            pass
    return None


class IBWReader(ReaderBase):
    _format = 'ibw'
    _mime_types = ['application/x-igor-binary-wave']
    _file_extensions = ['ibw']

    _name = 'Igor binary wave'
    _description = '''
Igor binary wave is a container format of the
[Igor Pro](https://www.wavemetrics.com/products/igorpro/programming)
language. This format is used by AFMs from Asylum Research (now Oxford
Instruments) to store topography information. This format contains information
on the physical size of the topography map as well as its units.
'''

    # Reads in the positions of all the data and metadata
    def __init__(self, file_path):
        # Note: igor2 has no partial-read support, so the wave data is
        # loaded here as well - but only the metadata is kept on the
        # reader; the data itself is dropped when this method returns and
        # loaded again in `topography()`. Readers are long-lived metadata
        # handles and must not pin the full data in memory.
        self._file_path = file_path
        with OpenFromAny(file_path, 'rb') as f:
            file = loadibw(f)

        if file['version'] != 5:
            raise UnsupportedFormatFeature('Only IBW version 5 is supported!')

        data = file['wave']
        wave_header = data['wave_header']

        if np.iscomplexobj(data['wData']):
            raise UnsupportedFormatFeature(
                'Igor waves with complex data are not supported.')

        #
        # Determine the shape of the data. A wave contains either a stack
        # of images (shape rows x columns x channels), a single image
        # (square two-dimensional wave) or curves (shape points x channels).
        #
        n_dim = [int(n) for n in wave_header['nDim']]
        if n_dim[3] != 0:
            raise UnsupportedFormatFeature(
                'Igor waves with four dimensions are not supported.')
        elif n_dim[2] != 0:
            self._dim, num_channels, label_dim = 2, n_dim[2], 2
        elif n_dim[1] != 0 and n_dim[1] == n_dim[0]:
            self._dim, num_channels, label_dim = 2, 1, 2
        elif n_dim[1] != 0:
            self._dim, num_channels, label_dim = 1, n_dim[1], 1
        else:
            self._dim, num_channels, label_dim = 1, 1, 1

        # The first label of a dimension names the dimension itself, the
        # following ones name its elements, i.e. the channels
        labels = data['labels']
        labels = labels[label_dim] if len(labels) > label_dim else []
        self._channel_names = [label.decode('latin-1') for label in labels[1:]]
        self._default_channel = 0

        # ensure that there are not too many channel names
        self._channel_names = self._channel_names[:num_channels]

        # add channel names for all channels without name
        no_name_idx = 1
        while len(self._channel_names) < num_channels:
            self._channel_names.append("no name ({})".format(no_name_idx))
            no_name_idx += 1

        #
        # Decode units
        #
        data_unit = _decode_unit(wave_header['dataUnits'])
        lateral_unit = _decode_unit(wave_header['dimUnits'][0])

        #
        # Decode sizes
        #
        sfA = wave_header['sfA']
        if self._dim == 2:
            nb_grid_pts = (n_dim[0], n_dim[1])
            self._physical_sizes = (n_dim[0] * sfA[0], n_dim[1] * sfA[1])
        else:
            nb_grid_pts = (n_dim[0],)
            self._physical_sizes = (n_dim[0] * sfA[0],)
        # Comment in C header file on these fields: Index value for element e
        # of dimension d = sfA[d]*e + sfB[d]. sfB is left out here, because we
        # are interested in the width and height, not the absolute offsets.

        #
        # The note contains the metadata of Asylum Research instruments
        #
        try:
            note = _parse_note(data["note"].decode("latin-1"))
        except (KeyError, UnicodeDecodeError):
            note = {}

        #
        # Build instrument information
        #
        instrument_info = {"vendor": "Asylum Research"}
        if "MicroscopeModel" in note:
            instrument_info["name"] = note["MicroscopeModel"]
        if "IgorFileVersion" in note:
            instrument_info["software"] = f"Igor Pro {note['IgorFileVersion']}"
        if "Version" in note:
            # This seems to be the software version of the Asylum software
            if "software" in instrument_info:
                instrument_info["software"] += f" (Asylum {note['Version']})"
            else:
                instrument_info["software"] = f"Asylum {note['Version']}"

        info = {"instrument": instrument_info}
        acquisition_time = _acquisition_time(note)
        if acquisition_time is not None:
            info["acquisition_time"] = acquisition_time
        if note:
            info["raw_metadata"] = note

        #
        # Build channel information
        #
        self._channels = []
        for i, channel_name in enumerate(self._channel_names):
            # Unit of the data values. Files with a note (Asylum Research)
            # store a single unit in the wave header for all channels; the
            # actual unit is derived from the channel name.
            if note and i < len(labels) - 1:
                channel_data_unit = _asylum_channel_unit(channel_name, note)
            else:
                channel_data_unit = data_unit

            # The lateral unit is the unit of the topography; heights that
            # are lengths are converted to this unit
            unit = lateral_unit
            height_scale_factor = 1
            if _is_length_unit(channel_data_unit):
                if unit is None:
                    unit = channel_data_unit
                elif _is_length_unit(unit):
                    height_scale_factor = get_unit_conversion_factor(
                        channel_data_unit, unit)

            self._channels += [
                ChannelInfo(
                    self,
                    i,
                    name=channel_name,
                    dim=self._dim,
                    nb_grid_pts=nb_grid_pts,
                    physical_sizes=self._physical_sizes,
                    unit=unit,
                    data_unit=channel_data_unit,
                    height_scale_factor=height_scale_factor,
                    uniform=True,
                    info=info,
                )
            ]

    @property
    def channels(self):
        return self._channels

    def topography(self, channel_index=None, channel_id=None,
                   height_channel_index=None, physical_sizes=None,
                   height_scale_factor=None, unit=None, info={},
                   periodic=False, subdomain_locations=None,
                   nb_subdomain_grid_pts=None):
        channel, channel_index = self._resolve_channel(
            channel_index, channel_id, height_channel_index
        )

        if unit is not None and channel.unit is not None:
            raise MetadataAlreadyFixedByFile('unit')
        if channel.unit is not None:
            unit = channel.unit

        if height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile('height_scale_factor')

        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile('physical_sizes')

        if subdomain_locations is not None or \
                nb_subdomain_grid_pts is not None:
            raise RuntimeError(
                'This reader does not support MPI parallelization.')

        # Load the data on demand; the reader itself only holds metadata
        with OpenFromAny(self._file_path, 'rb') as f:
            wave_data = loadibw(f)['wave']['wData']

        _info = channel.info.copy()
        _info.update(info)
        _info.update({'instrument': channel.info['instrument']})

        if self._dim == 2:
            if wave_data.ndim == 3:
                wave_data = wave_data[:, :, channel_index]
            # The orientation matches Gwyddion's when plotted with
            # imshow(t.heights().T)
            topo = Topography(np.fliplr(wave_data).astype(float),
                              self._physical_sizes, unit=unit, info=_info,
                              periodic=periodic)
        else:
            if wave_data.ndim == 2:
                wave_data = wave_data[:, channel_index]
            topo = UniformLineScan(wave_data.astype(float),
                                   self._physical_sizes[0], unit=unit,
                                   info=_info, periodic=periodic)

        return topo.scale(channel.height_scale_factor)
