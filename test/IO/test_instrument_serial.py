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

"""
Serial numbers of the instrument (`info["instrument"]["serial"]`) and of its
scanner (`info["instrument"]["scanner_serial"]`)
"""

import os

import pytest
from NuMPI import MPI

from SurfaceTopography.IO import open_topography
from SurfaceTopography.IO.GWY import _instrument_from_meta
from SurfaceTopography.Metadata import InfoModel, InstrumentModel

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


@pytest.mark.parametrize(
    "filename,serial,scanner_serial",
    [
        ("datx-1.datx", "87266", None),
        ("datx-2.datx", "78137", None),
        # DI files only carry the serial number of the scanner
        ("di-1.di", None, "Hybrid 306"),
        ("di-2.di", None, "1704GN"),
        ("di-5.di", None, "1A03F9"),
        ("frt-1.frt", "SN_1036", None),
        # Converted by Gwyddion from a MetroPro file of the instrument that
        # also wrote datx-2.datx
        ("gwy-2.gwy", "78137", None),
        ("gwy-1.gwy", None, None),
        # sys_serial and sys_serial2 both hold 59407
        ("metropro-1.dat", "59407", None),
        # Controller and scan head
        ("nid-1.nid", "091-21-061", "101-20-000"),
        # Dulcinea controller
        ("stp-1.stp", "24", None),
        ("top-1.top", "24", None),
        ("zon-1.zon", "#3C810114", None),
        # Placeholders for unknown serial numbers
        ("x3p-1.x3p", None, None),
        ("x3p-3.x3p", None, None),
        # The serial number in MNT files is that of the Mountains software
        ("mnt-1.mnt", None, None),
        # AL3D files carry no serial number
        ("al3d-1.al3d", None, None),
    ],
)
def test_instrument_serial(file_format_examples, filename, serial, scanner_serial):
    reader = open_topography(os.path.join(file_format_examples, filename))
    for info in [reader.default_channel.info, reader.topography().info]:
        instrument = info.get("instrument", {})
        assert instrument.get("serial") == serial
        assert instrument.get("scanner_serial") == scanner_serial


def test_metropro_serial_fields(file_format_examples):
    reader = open_topography(os.path.join(file_format_examples, "metropro-1.dat"))
    raw_metadata = reader.default_channel.info["raw_metadata"]
    # The 16-bit field is unsigned, the 32-bit field is little endian
    assert raw_metadata["sys_serial"] == 59407
    assert raw_metadata["sys_serial2"] == 59407


def test_nid_instrument(file_format_examples):
    reader = open_topography(os.path.join(file_format_examples, "nid-1.nid"))
    instrument = reader.topography().info["instrument"]
    assert instrument["vendor"] == "Nanosurf"
    assert instrument["name"] == "DriveAFM"


@pytest.mark.parametrize(
    "meta,instrument",
    [
        (None, None),
        ({"GwyContainer": {}}, None),
        (
            {
                "GwyContainer": {
                    "Instrument serial number": "12601",
                    "Instrument serial number 2": "78137",
                }
            },
            {"serial": "78137"},
        ),
        # Old MetroPro files only have the 16-bit field, which Gwyddion
        # stores as a signed number
        (
            {
                "GwyContainer": {
                    "Instrument serial number": "-6129",
                    "Instrument serial number 2": "0",
                }
            },
            {"serial": "59407"},
        ),
    ],
)
def test_gwy_instrument_from_meta(meta, instrument):
    assert _instrument_from_meta(meta) == instrument


@pytest.mark.parametrize(
    "value,serial",
    [
        ("SN_1036", "SN_1036"),
        ("  1A03F9\x00\x00", "1A03F9"),
        (b"12345", "12345"),
        (59407, "59407"),
        ("not available", None),
        ("Unknown", None),
        ("N/A", None),
        ("0", None),
        ("", None),
        (None, None),
    ],
)
@pytest.mark.parametrize("field", ["serial", "scanner_serial"])
def test_serial_normalization(field, value, serial):
    assert getattr(InstrumentModel(**{field: value}), field) == serial
    info = InfoModel(instrument={field: value})
    assert info.model_dump(exclude_none=True)["instrument"].get(field) == serial
