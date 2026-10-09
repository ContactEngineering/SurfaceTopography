#
# Copyright 2019-2023 Lars Pastewka
#           2020-2021 Michael Röttger
#           2019 Antoine Sanner
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
# The DI file format is described in detail here:
# http://www.physics.arizona.edu/~smanne/DI/software/fileformats.html
#


import dateutil.parser
import numpy as np

from ..Exceptions import (
    CorruptFile,
    MetadataAlreadyFixedByFile,
    UnsupportedFormatFeature,
)
from ..Support.UnitConversion import (
    get_unit_conversion_factor,
    is_length_unit,
    mangle_length_unit_utf8,
)
from ..UniformLineScanAndTopography import Topography
from .common import OpenFromAny
from .Reader import ChannelInfo, ReaderBase

###

# Section names. Besides the common "Ciao" names written by the
# Nanoscope software, older software versions (and the "Nanoscope E"
# software) use different names for the same sections; see also the
# `nanoscope.c` module of Gwyddion.
_FILE_LIST_SECTIONS = ("file list", "ec file list")
_SCANNER_SECTIONS = ("scanner list", "microscope list")
_SCAN_SECTIONS = ("ciao scan list", "afm list", "stm list", "nc afm list")
_IMAGE_SECTIONS = (
    "ciao image list",
    "afm image list",
    "stm image list",
    "ncafm image list",
    "image list",
)

# Starting with version 9.2 of the Nanoscope software, the raw data is
# always stored as 32-bit integers. The "Bytes/pixel" entry then only
# determines the scaling of the integers.
_VERSION_32BIT = 0x09200000


def _parse_value(value):
    """
    Parse a "@"-type header value, e.g.
    `V [Sens. Zsens] (0.006713867 V/LSB) 2.910461 V`.

    Returns
    -------
    value_type : str
        Type of the value: 'V' (value), 'C' (scale) or 'S' (select)
    soft_scale : str or None
        Name of the soft scale (the entry in square brackets)
    hard_scale : str or None
        Content of the parentheses (the per-LSB scale)
    hard_value : str
        The remaining (hard) value, including its unit
    """
    value = value.strip()
    if not value:
        raise CorruptFile("Empty value in Nanoscope header.")
    value_type, rest = value[0], value[1:].strip()
    soft_scale = None
    if rest.startswith("["):
        end = rest.find("]")
        if end < 0:
            raise CorruptFile(f"Cannot parse soft scale of Nanoscope header value '{value}'.")
        soft_scale = rest[1:end].strip() or None
        rest = rest[end + 1:].strip()
    hard_scale = None
    if rest.startswith("("):
        # Units can themselves contain parentheses, e.g. log(Arb)
        level = 0
        for i, c in enumerate(rest):
            if c == "(":
                level += 1
            elif c == ")":
                level -= 1
                if level == 0:
                    break
        if level != 0:
            raise CorruptFile(f"Cannot parse hard scale of Nanoscope header value '{value}'.")
        hard_scale = rest[1:i].strip()
        rest = rest[i + 1:].strip()
    return value_type, soft_scale, hard_scale, rest


def _split_number_and_unit(value):
    """Split e.g. '2.910461 V' into (2.910461, 'V')."""
    s = value.split(None, 1)
    number = float(s[0])
    unit = s[1].strip() if len(s) > 1 else ""
    return number, unit


def _parse_size(value):
    """Parse a size with unit, e.g. '1000 nm' or '10 ~m'."""
    number, unit = _split_number_and_unit(value)
    return number, mangle_length_unit_utf8(unit)


def _has_nonsquare_aspect(section):
    aspect = section.get("aspect ratio")
    if aspect is None or aspect.strip() == "1:1":
        return False
    try:
        ratio = float(aspect.split(":")[0])
    except ValueError:
        return False
    return ratio > 0 and ratio != 1


def _image_data_name(section):
    """Name of the channel stored in an image section."""
    for key in ("@2:image data", "@3:image data", "@4:image data"):
        if key in section:
            _, soft_scale, _, hard_value = _parse_value(section[key])
            if soft_scale is not None:
                return soft_scale
            return hard_value.strip('"')
    return section.get("image data")


class DIReader(ReaderBase):
    _format = "di"
    _mime_types = ["application/x-nanoscope-iii-spm"]
    _file_extensions = ["spm", "001", "002", "003", "004", "005"]

    _name = "Bruker/Veeco/DI Nanoscope"
    _description = """
Digital Instruments Nanoscope (also Veeco Nanoscope and Bruker Dimension)
files typically have a three-digit number as the file extension (.001, .002, .003, ...).
Newer versions of this file format have the extension .spm. This format contains
information on the physical size of the topography map as well as its units.
The reader supports V4.3 and later version of the format.
"""

    def __init__(self, file_path):
        """
        Load Digital Instrument's Nanoscope files.

        Arguments
        ---------
        file_path : filename or file object
             File or data stream to open.
        """
        self._file_path = file_path
        with OpenFromAny(self._file_path, "rb") as fobj:
            # Get file size
            pos = fobj.tell()
            fobj.seek(0, 2)
            file_size = fobj.tell()
            fobj.seek(pos)

            parameters = []
            section_name = None
            section_dict = {}

            L = fobj.readline().decode("latin-1").strip()
            if L.startswith("?*"):
                raise UnsupportedFormatFeature(
                    "This is a Nanoscope file with data stored as text. Only files with binary data are supported."
                )
            while L and L.lower() != r"\*file list end":
                if L.startswith("\\*"):
                    if section_name is not None:
                        parameters += [(section_name, section_dict)]
                    new_section_name = L[2:].lower()
                    if section_name is None:
                        if new_section_name not in _FILE_LIST_SECTIONS:
                            raise IOError(
                                "Header must start with the " "'File list' section."
                            )
                    section_name = new_section_name
                    section_dict = {}
                elif L.startswith("\\"):
                    if section_name is None:
                        raise IOError("Encountered key before section " "header.")
                    s = L[1:].split(": ", 1)
                    try:
                        key, value = s
                    except ValueError:
                        (key,) = s
                        value = ""
                    section_dict[key.lower()] = value.strip()
                else:
                    raise IOError("Header line does not start with a slash.")
                L = fobj.readline().decode("latin-1").strip()
            if section_name is None:
                raise IOError("No sections found in header.")
            parameters += [(section_name, section_dict)]

            self._channels = []
            self._offsets = []

            # Collect the global sections first; they are needed to
            # interpret the image sections
            file_list = {}
            scanner_list = {}
            scan_list = {}
            equipment = {}
            operating_mode = None
            for n, p in parameters:
                if n in _FILE_LIST_SECTIONS:
                    file_list.update(p)
                elif n in _SCANNER_SECTIONS:
                    scanner_list.update(p)
                elif n in _SCAN_SECTIONS:
                    scan_list.update(p)
                elif n == "equipment list":
                    equipment.update(p)
                if "operating mode" in p:
                    operating_mode = p["operating mode"].strip()

            try:
                version = int(file_list.get("version", "0"), 16)
            except ValueError:
                version = 0
            start_context = file_list.get("start context", "").strip()

            # Files with single force curves or (Deep Trench) sets of
            # unevenly spaced profiles have image sections, but these do
            # not contain topography maps
            if operating_mode == "Force" or start_context == "FOL":
                raise UnsupportedFormatFeature(
                    "This Nanoscope file contains force curves, which are not supported."
                )
            if start_context.endswith("VAR") and operating_mode != "Force Volume":
                raise UnsupportedFormatFeature(
                    "This Nanoscope file contains a set of unevenly spaced profiles, which is not supported."
                )

            # Global resolution, which is used when the resolution of the
            # image sections does not match the size of the data block
            try:
                global_nx = int(scan_list["samps/line"])
                global_ny = int(scan_list["lines"])
            except (KeyError, ValueError):
                global_nx = global_ny = None

            # Newer files store the (correct) slow-axis size separately;
            # the second number in "Scan size" can then be wrong
            slow_axis_sizes = None
            if "slow axis size" in scan_list and "scan size" in scan_list:
                fast_size, fast_unit = _parse_size(scan_list["scan size"])
                slow_size, slow_unit = _parse_size(scan_list["slow axis size"])
                fac = get_unit_conversion_factor(slow_unit, fast_unit)
                if fac is not None:
                    slow_axis_sizes = (fast_size, slow_size * fac, fast_unit)

            instrument = {"vendor": "Bruker"}
            if "microscope" in equipment:
                instrument["name"] = equipment["microscope"]
            elif "description" in equipment:
                instrument["name"] = equipment["description"]
            # DI files only carry the serial number of the scanner,
            # not that of the controller
            if "serial number" in scanner_list or "serial number" in scan_list:
                instrument["scanner_serial"] = scanner_list.get(
                    "serial number", scan_list.get("serial number")
                )
            if "version" in file_list:
                instrument["software"] = file_list["version"]

            for n, p in parameters:
                if n not in _IMAGE_SECTIONS:
                    continue

                info = {}
                if "date" in file_list:
                    info["acquisition_time"] = dateutil.parser.parse(file_list["date"])
                info["instrument"] = instrument.copy()
                info["raw_metadata"] = p

                image_data_key = _image_data_name(p)

                nx = int(p["samps/line"])
                ny = int(p["number of lines"])

                # Scan size; old files report a single number for square
                # scans
                s = p["scan size"].split()
                sx = float(s[0])
                try:
                    sy = float(s[1])
                    xy_unit = " ".join(s[2:])
                except (IndexError, ValueError):
                    sy = sx
                    xy_unit = " ".join(s[1:])
                xy_unit = mangle_length_unit_utf8(xy_unit)
                if slow_axis_sizes is not None:
                    fast_size, slow_size, fast_unit = slow_axis_sizes
                    fac = get_unit_conversion_factor(fast_unit, xy_unit)
                    if fac is None:
                        fac = 1
                        xy_unit = fast_unit
                    sx, sy = fast_size * fac, slow_size * fac

                offset = int(p["data offset"])
                self._offsets.append(offset)

                length = int(p["data length"])
                # Bytes/pixel determines the scaling of the integers,
                # starting with version 9.2 the raw data is 32-bit
                # irrespective of this value
                bytes_per_pixel = int(p.get("bytes/pixel", "2"))
                if bytes_per_pixel not in (2, 4):
                    raise IOError(
                        f"Don't know how to handle {bytes_per_pixel} bytes per pixel data."
                    )
                elsize = bytes_per_pixel
                if version >= _VERSION_32BIT and length >= 4 * nx * ny:
                    elsize = 4

                # Some files report wrong resolutions in the image
                # sections; the global resolution is then correct
                use_global = False
                if global_nx is not None and length != nx * ny * elsize:
                    if length == global_nx * global_ny * elsize:
                        use_global = True
                    elif nx * ny * elsize > length >= global_nx * global_ny * elsize:
                        use_global = True
                if use_global:
                    if slow_axis_sizes is None:
                        sx *= global_nx / nx
                        sy *= global_ny / ny
                    nx, ny = global_nx, global_ny
                elif (
                    slow_axis_sizes is not None
                    and _has_nonsquare_aspect(p)
                    and global_ny is not None
                    and ny < global_ny
                ):
                    # Scan was stopped early ("Capture Now"), the slow
                    # axis size refers to the full scan
                    sy *= ny / global_ny

                if nx * ny * elsize > length:
                    raise IOError(
                        f"File reports a data block of length {length}, but computing the size of the "
                        f"data block from {nx} x {ny} grid points and the per-pixel storage of {elsize} "
                        f"bytes yields a larger value of {nx * ny * elsize}."
                    )

                height_unit = None
                height_scale_factor = None
                z_scale = p.get("@4:z scale", p.get("@2:z scale"))
                if z_scale is None:
                    raise UnsupportedFormatFeature(
                        "Nanoscope files without '@2:Z scale' entries (version 4.2 and earlier) are not supported."
                    )
                _, quantity, _, hard_value = _parse_value(z_scale)
                hard_value, hard_unit = _split_number_and_unit(hard_value)
                hard_scale = hard_value / 256**bytes_per_pixel

                if quantity is not None:
                    key = "@" + quantity.lower()
                    soft = scanner_list.get(key, scan_list.get(key))
                    if soft is not None:
                        soft_type, _, _, soft_value = _parse_value(soft)
                        if soft_type != "V":
                            raise CorruptFile("Malformed Nanoscope DI file.")
                        soft_scale, soft_unit = _split_number_and_unit(soft_value)
                        if "/" in soft_unit:
                            # Check units
                            height_unit, soft_unit = soft_unit.split("/", 1)
                            hard_to_soft = get_unit_conversion_factor(hard_unit, soft_unit)
                            if hard_to_soft is None:
                                raise RuntimeError(
                                    "Units for hard (={}) and soft (={}) "
                                    "scale differ for '{}'. Don't know how "
                                    "to handle this.".format(
                                        hard_unit, soft_unit, image_data_key
                                    )
                                )
                            if is_length_unit(height_unit):
                                height_scale_factor = hard_scale * hard_to_soft * soft_scale

                # We only report channels with height information
                if height_scale_factor is not None:
                    height_unit = mangle_length_unit_utf8(height_unit)
                    if xy_unit != height_unit:
                        fac = get_unit_conversion_factor(xy_unit, height_unit)
                        sx *= fac
                        sy *= fac
                    unit = height_unit
                else:
                    unit = (xy_unit, height_unit)

                self._channels += [
                    ChannelInfo(
                        self,
                        len(self._channels),
                        name=image_data_key,
                        dim=2,
                        nb_grid_pts=(nx, ny),
                        physical_sizes=(sx, sy),
                        height_scale_factor=height_scale_factor,
                        periodic=False,
                        uniform=True,
                        unit=unit,
                        info=info,
                        tags={"elsize": elsize},
                    )
                ]

                # We seek to the end of the data buffer, this should not raise an exception
                if offset + nx * ny * elsize > file_size:
                    raise CorruptFile(
                        "File is not large enough to contain all data buffers."
                    )

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
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        channel, channel_index = self._resolve_channel(
            channel_index, channel_id, height_channel_index
        )

        if subdomain_locations is not None or nb_subdomain_grid_pts is not None:
            raise RuntimeError("This reader does not support MPI parallelization.")

        with OpenFromAny(self._file_path, "rb") as fobj:

            if unit is not None:
                raise MetadataAlreadyFixedByFile("unit")

            sx, sy = self._check_physical_sizes(physical_sizes, channel.physical_sizes)

            nx, ny = channel.nb_grid_pts

            offset = self._offsets[channel_index]
            elsize = channel.tags["elsize"]
            if elsize == 2:
                dtype = np.dtype("<i2")
            elif elsize == 4:
                dtype = np.dtype("<i4")
            else:
                raise IOError(
                    f"Don't know how to handle {elsize} bytes per pixel data."
                )

            assert elsize == dtype.itemsize

            ###################################

            fobj.seek(offset)
            rawdata = fobj.read(nx * ny * dtype.itemsize)
            # The data is stored line by line, i.e. the buffer has C-order
            # shape (ny, nx). Transposing yields the (nx, ny) array with the
            # x index first that `Topography` expects.
            unscaleddata = (
                np.frombuffer(rawdata, count=nx * ny, dtype=dtype)
                .reshape(ny, nx)
                .T
            )

        # internal information from file
        _info = channel.info.copy()
        _info.update(info)

        # it is not allowed to provide extra `physical_sizes` here:
        if physical_sizes is not None:
            raise MetadataAlreadyFixedByFile("physical_sizes")

        # the orientation of the heights is modified in order to match
        # the image of gwyddion when plotted with imshow(t.heights().T)
        # or pcolormesh(t.heights().T) for origin in lower left and
        # with inverted y axis (cartesian coordinate system)

        surface = Topography(
            np.fliplr(unscaleddata),
            physical_sizes=(sx, sy),
            unit=channel.unit,
            info=_info,
            periodic=periodic,
        )
        if height_scale_factor is None:
            height_scale_factor = channel.height_scale_factor
        elif channel.height_scale_factor is not None:
            raise MetadataAlreadyFixedByFile("height_scale_factor")
        if height_scale_factor is not None:
            surface = surface.scale(height_scale_factor)

        return surface

    channels.__doc__ = ReaderBase.channels.__doc__
    topography.__doc__ = ReaderBase.topography.__doc__
