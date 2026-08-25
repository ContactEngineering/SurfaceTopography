#
# Copyright 2026 Lars Pastewka
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

import datetime
import re
import struct
import zipfile

import numpy as np

from ..Exceptions import CorruptFile, FileFormatMismatch, MetadataAlreadyFixedByFile
from ..NonuniformLineScan import NonuniformLineScan
from ..UniformLineScanAndTopography import UniformLineScan
from .common import OpenFromAny
from .Reader import ChannelInfo, MagicMatch, ReaderBase

# Each MarWin '[p3dlin]' profile blob consists of a fixed-size header, a
# table of length-prefixed metadata strings, two marker-delimited sections
# ('[sellsg]' and '[Trajec]') and four packed data arrays.
_PRF_MAGIC = b"[p3dlin]"

# Byte offset of the point count (int32) within the profile header
_NB_POINTS_OFFSET = 0x14

# Byte offset of the metadata string table (int32 entry count followed by
# int32-length-prefixed latin-1 strings)
_STRING_TABLE_OFFSET = 0xC4

# The data arrays follow the '[Trajec]' marker at this distance
_TRAJEC_DATA_OFFSET = 8 + 41

# Data arrays per point: x, y, z as float64 plus a float32 validity flag
_BYTES_PER_POINT = 28

# Zip members holding profiles, with the channel names they map to. The
# measured profile comes first and is the default channel.
_PROFILE_MEMBERS = (
    ("Kontur1.prf", "measured"),
    ("qe_contour_reference_profil_001.tmp", "reference"),
)

# `>DateTime=` values look like
# '2026-03-20  02:44:41  (Mitteleuropäische Zeit)'
_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})")


def _parse_prf(blob, member_name):
    """
    Parse a MarWin '[p3dlin]' profile blob into metadata and data arrays.

    Returns a tuple (meta, x, z, flags) with the `>Key=Value` metadata
    entries as a dictionary, positions and heights in mm as float64 arrays
    and the per-point validity flags as a float32 array (nonzero marks an
    invalid point).
    """
    if blob[: len(_PRF_MAGIC)] != _PRF_MAGIC:
        raise CorruptFile(
            f"Archive member `{member_name}` is not a MarWin '[p3dlin]' "
            f"profile."
        )
    if len(blob) < _STRING_TABLE_OFFSET + 4:
        raise CorruptFile(
            f"Archive member `{member_name}` is truncated within the "
            f"profile header."
        )
    (nb_points,) = struct.unpack_from("<i", blob, _NB_POINTS_OFFSET)
    if nb_points <= 0:
        raise CorruptFile(
            f"Archive member `{member_name}` reports a non-positive number "
            f"of points ({nb_points})."
        )

    # Metadata string table: entry count, then int32-length-prefixed
    # latin-1 strings; entries of the form '>Key=Value' carry metadata
    offset = _STRING_TABLE_OFFSET
    (nb_entries,) = struct.unpack_from("<i", blob, offset)
    offset += 4
    if nb_entries < 0:
        raise CorruptFile(
            f"Archive member `{member_name}` reports a negative metadata "
            f"entry count."
        )
    meta = {}
    for _ in range(nb_entries):
        if offset + 4 > len(blob):
            raise CorruptFile(
                f"Archive member `{member_name}` is truncated within the "
                f"metadata string table."
            )
        (strlen,) = struct.unpack_from("<i", blob, offset)
        offset += 4
        if strlen < 0 or offset + strlen > len(blob):
            raise CorruptFile(
                f"Archive member `{member_name}` contains a metadata string "
                f"that exceeds the file size."
            )
        entry = blob[offset:offset + strlen].decode("latin-1", "replace")
        offset += strlen
        if entry.startswith(">") and "=" in entry:
            key, _, value = entry[1:].partition("=")
            meta[key] = value

    # The '[sellsg]' and '[Trajec]' sections follow the string table at
    # version-dependent distances; locate them by their markers
    sellsg = blob.find(b"[sellsg]", offset)
    if sellsg < 0:
        raise CorruptFile(
            f"Archive member `{member_name}` is missing the '[sellsg]' "
            f"section."
        )
    trajec = blob.find(b"[Trajec]", sellsg)
    if trajec < 0:
        raise CorruptFile(
            f"Archive member `{member_name}` is missing the '[Trajec]' "
            f"section."
        )
    data_start = trajec + _TRAJEC_DATA_OFFSET
    if data_start + nb_points * _BYTES_PER_POINT != len(blob):
        raise CorruptFile(
            f"Archive member `{member_name}` has a size that disagrees "
            f"with its header ({nb_points} points)."
        )

    # x, y, z as float64 followed by float32 validity flags; the y
    # positions are constant for a line scan and are ignored
    x = np.frombuffer(blob, "<f8", nb_points, data_start)
    z = np.frombuffer(blob, "<f8", nb_points, data_start + nb_points * 16)
    flags = np.frombuffer(blob, "<f4", nb_points, data_start + nb_points * 24)
    return meta, x, z, flags


def _acquisition_time(meta):
    """Parse the `>DateTime=` metadata entry, if present and well-formed."""
    match = _DATETIME_RE.search(meta.get("DateTime", ""))
    if match is None:
        return None
    return datetime.datetime.strptime(
        f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M:%S"
    )


def _instrument(meta):
    """Instrument metadata from the profile's `>Key=Value` entries."""
    instrument = {"vendor": "Mahr"}
    machine_name = meta.get("MachineName")
    if machine_name:
        instrument["name"] = machine_name
    software = " ".join(
        s for s in (meta.get("ProductName"), meta.get("ProductVersion")) if s
    )
    if software:
        instrument["software"] = software
    try:
        instrument["parameters"] = {
            "tip_radius": {"value": float(meta["ProbeRadius"]), "unit": "mm"}
        }
    except (KeyError, ValueError):
        pass
    return instrument


class QEPReader(ReaderBase):
    """
    Reader for contour/profile scans of Mahr instruments driven by the
    MarWin software (e.g. the MarSurf and MarForm series).
    """

    _format = "qep"
    _mime_types = ["application/zip"]
    _file_extensions = ["qep"]

    _name = "Mahr MarWin QEP"
    _description = """
QEP profile archives written by the MarWin software of Mahr contour and
surface measuring instruments. A QEP file is a ZIP archive holding a
measured profile ('Kontur1.prf') and optionally a reference profile, each
stored as a binary MarWin '[p3dlin]' line scan with positions and heights
in mm. Points flagged invalid by the instrument are reported as undefined
(uniform profiles) or omitted (nonuniform profiles).
"""

    @classmethod
    def can_read(cls, buffer: bytes) -> MagicMatch:
        if len(buffer) < 4:
            return MagicMatch.MAYBE
        # A QEP is a ZIP archive; the magic can only tell that the file
        # *may* be one
        if buffer.startswith(b"PK\x03\x04"):
            return MagicMatch.MAYBE
        return MagicMatch.NO

    def __init__(self, fobj):
        """
        Open a QEP profile archive.

        Parameters
        ----------
        fobj : str or file object
            File name or stream.
        """
        self.file_path = fobj
        profiles = []
        with OpenFromAny(fobj, "rb") as f:
            try:
                with zipfile.ZipFile(f) as zip_file:
                    member_names = set(zip_file.namelist())
                    for member_name, channel_name in _PROFILE_MEMBERS:
                        if member_name not in member_names:
                            continue
                        meta, x, z, flags = _parse_prf(
                            zip_file.read(member_name), member_name
                        )
                        profiles.append(
                            self._process_profile(channel_name, meta, x, z, flags)
                        )
            except zipfile.BadZipFile as exc:
                raise FileFormatMismatch(
                    "This is not a ZIP archive."
                ) from exc
        if not profiles:
            raise FileFormatMismatch(
                "ZIP archive contains no MarWin profile members."
            )

        self._channels = [
            ChannelInfo(
                self,
                index,
                name=profile["name"],
                dim=1,
                nb_grid_pts=len(profile["heights"]),
                physical_sizes=profile["physical_sizes"],
                uniform=profile["uniform"],
                undefined_data=profile["has_undefined_data"],
                unit="mm",
                info=profile["info"],
                tags={"profile": profile},
            )
            for index, profile in enumerate(profiles)
        ]

    @staticmethod
    def _process_profile(channel_name, meta, x, z, flags):
        """
        Turn parsed profile arrays into the channel record: decide between
        the uniform and nonuniform representations and apply the
        invalid-point policy (mask on uniform grids, omit otherwise).
        """
        if len(x) < 2:
            raise CorruptFile(
                f"The {channel_name} profile has fewer than two points."
            )
        invalid = flags != 0

        steps = np.diff(x)
        uniform = bool(np.all(np.isclose(steps, steps[0])))
        if uniform:
            # Keep the grid intact; invalid points become undefined data
            nb_points = len(x)
            span = x[-1] - x[0]
            if span <= 0:
                raise CorruptFile(
                    f"The {channel_name} profile has a non-positive extent."
                )
            if invalid.any():
                heights = np.ma.masked_array(z, mask=invalid)
            else:
                heights = np.asarray(z)
            positions = None
            # A uniform line scan of length s has grid points at
            # [0, s/n, ..., (n-1) s/n]; the physical size must therefore
            # capture n*dx, not the span (n-1)*dx
            physical_sizes = nb_points * span / (nb_points - 1)
            has_undefined_data = bool(invalid.any())
        else:
            # Nonuniform positions are explicit; omit invalid points
            valid = np.logical_not(invalid)
            if np.count_nonzero(valid) < 2:
                raise CorruptFile(
                    f"The {channel_name} profile has fewer than two valid "
                    f"points."
                )
            positions = x[valid] - x[valid][0]
            heights = np.asarray(z[valid])
            physical_sizes = positions[-1]
            if physical_sizes <= 0:
                raise CorruptFile(
                    f"The {channel_name} profile has a non-positive extent."
                )
            has_undefined_data = False

        info = {
            "channel_name": channel_name,
            "acquisition_time": _acquisition_time(meta),
            "instrument": _instrument(meta),
            "raw_metadata": meta,
        }
        info = {key: value for key, value in info.items() if value is not None}

        return {
            "name": channel_name,
            "positions": positions,
            "heights": heights,
            "uniform": uniform,
            "physical_sizes": physical_sizes,
            "has_undefined_data": has_undefined_data,
            "info": info,
        }

    @property
    def channels(self):
        return self._channels

    def topography(
        self,
        channel_index=None,
        channel_id=None,
        height_channel_index=None,
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

        channel, channel_index = self._resolve_channel(
            channel_index, channel_id, height_channel_index
        )
        profile = channel.tags["profile"]

        # Positions and heights are calibrated in mm by the file
        if unit is not None:
            raise MetadataAlreadyFixedByFile("unit")
        unit = channel.unit
        physical_sizes = self._check_physical_sizes(
            physical_sizes, channel.physical_sizes
        )

        _info = channel.info.copy()
        _info.update(info)

        if profile["uniform"]:
            topography = UniformLineScan(
                profile["heights"],
                physical_sizes,
                periodic=False if periodic is None else periodic,
                unit=unit,
                info=_info,
            )
        else:
            if periodic is not None and periodic:
                raise ValueError(
                    "The profile is nonuniform; nonuniform line scans "
                    "cannot be periodic."
                )
            topography = NonuniformLineScan(
                profile["positions"], profile["heights"], unit=unit, info=_info
            )

        if height_scale_factor is not None:
            topography = topography.scale(height_scale_factor)
        return topography

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
