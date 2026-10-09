#
# Copyright 2020-2024 Lars Pastewka
#           2021 Michael Röttger
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

import json
import os
import tempfile
import zipfile
from datetime import datetime

import numpy as np
import pytest
import yaml
from NuMPI import MPI
from numpy.testing import assert_allclose

import SurfaceTopography
from SurfaceTopography import (
    open_topography,
    read_container,
    read_published_container,
    read_topography,
)
from SurfaceTopography.Container.IO import CEReader
from SurfaceTopography.Container.SurfaceContainer import InMemorySurfaceContainer

from .test_io import binary_example_file_list, text_example_file_list

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest",
)


def test_read_just_uniform(file_format_examples):
    for (c,) in [
        read_container(f"{file_format_examples}/container-1.zip"),
        # read_published_container('https://contact.engineering/go/867nv/')  # Same a container-1.zip
        # TODO maybe this makes the web app stall when running MPI tests von Travis, further investigation needed
    ]:
        assert len(c) == 3

        assert not c[0].is_periodic
        assert not c[1].is_periodic
        assert not c[2].is_periodic

        assert c[0].nb_grid_pts == (500, 500)
        assert c[1].nb_grid_pts == (500, 500)
        assert c[2].nb_grid_pts == (500, 500)

        assert c[0].physical_sizes == (100, 100)
        assert c[1].physical_sizes == (10, 10)
        assert c[2].physical_sizes == (1, 1)

        assert c[0].unit == "µm"
        assert c[1].unit == "µm"
        assert c[2].unit == "µm"

        assert "unit" not in c[0].info
        assert "unit" not in c[1].info
        assert "unit" not in c[2].info


def test_read_mixed(file_format_examples):
    (c,) = read_container(f"{file_format_examples}/container-2.zip")

    assert len(c) == 3


def test_write(file_format_examples):
    t1 = read_topography(f"{file_format_examples}/di-1.di")
    t2 = read_topography(f"{file_format_examples}/opd-1.opd")
    t3 = read_topography(f"{file_format_examples}/matrix-2.txt")

    c = InMemorySurfaceContainer([t1, t2, t3])

    with tempfile.TemporaryFile() as fobj:
        c.to_zip(fobj)

        (c2,) = read_container(fobj)

        assert len(c2) == 3

        assert not c2[0].is_periodic
        assert not c2[1].is_periodic
        assert not c2[2].is_periodic

        assert c2[0].nb_grid_pts == t1.nb_grid_pts
        assert c2[1].nb_grid_pts == t2.nb_grid_pts
        assert c2[2].nb_grid_pts == t3.nb_grid_pts

        assert_allclose(c2[0].physical_sizes, t1.physical_sizes)
        assert_allclose(c2[1].physical_sizes, t2.physical_sizes)
        assert_allclose(c2[2].physical_sizes, t3.physical_sizes)

        assert c2[0].unit == t1.unit
        assert c2[1].unit == t2.unit
        assert c2[2].unit == t3.unit

        assert "unit" not in c2[0].info
        assert "unit" not in c2[1].info
        assert "unit" not in c2[2].info

        assert "unit" not in t1.info
        assert "unit" not in t2.info
        assert "unit" not in t3.info


def test_periodic():
    (container,) = read_published_container("https://contact.engineering/go/v9qwe/")

    pristine = container[0]
    convoluted = container[1]
    assert pristine.is_periodic
    assert convoluted.is_periodic


@pytest.mark.parametrize(
    "filenames", [binary_example_file_list, text_example_file_list]
)
def test_read_files_from_container(filenames):
    """BCRF and GWY file use np.fromfile to read data, which has issues when reading within a ZIP file"""
    with tempfile.TemporaryDirectory() as d:
        containerfn = f"{d}/container.zip"

        # Write container with raw data files
        with zipfile.ZipFile(containerfn, "w") as z:
            topographies = []
            for filepath in filenames:
                _, fn = os.path.split(filepath)
                z.write(filepath, fn)
                topography = {"datafile": {"original": fn}}

                reader = open_topography(filepath)
                if reader.default_channel.physical_sizes is None:
                    topography["size"] = (1.0,) * reader.default_channel.dim

                topographies += [topography]

            metadata = {
                "versions": {"SurfaceTopography": SurfaceTopography.__version__},
                "surfaces": [{"topographies": topographies}],
                "creation_time": datetime.now(),
            }
            z.writestr("meta.yml", yaml.dump(metadata))

        r = CEReader(containerfn)
        c = r.container()
        for t in c:
            # This loop actually reads the files
            # The test is that file reading progresses without issues
            pass


def test_ce_container():
    (surface,) = read_published_container("https://doi.org/10.57703/ce-mg4cy")
    rms_heights = [t.rms_height_from_profile() for t in surface]
    # This is a regression test. The values are not checked for correctness.
    np.testing.assert_allclose(
        rms_heights,
        [
            0.003949841631562571,
            0.004032999294137858,
            0.013820472252164628,
            0.0217028756922634,
            0.004851062064249796,
            0.02123958101044809,
            0.030065647090746887,
            0.009013796742461624,
            0.019992889576544735,
            0.03571244942555436,
            0.019188774584700925,
            0.03026452140452409,
            0.04009296334584872,
            0.03591547319579487,
            0.026463673998312873,
            0.18154731113775002,
            1.123743993604103,
            0.078929223879882,
        ],
    )


def test_read_multiple_surfaces(file_format_examples):
    c = read_container(f"{file_format_examples}/container-4.zip")
    assert len(c) == 2
    s1, s2 = c
    assert len(s1) == 1
    assert len(s2) == 1
    assert not np.allclose(s1[0].heights(), s2[0].heights())


#
# `index.json` metadata and member enumeration
#


def _make_index_json_container(fn, file_format_examples, with_legacy_yaml=False):
    """Build a container with `index.json` metadata (the canonical layout
    written by TopoBank) around two example data files. Returns the paths of
    the bundled data files."""
    datafiles = ["di-1.di", "opd-1.opd"]
    topographies = []
    for i, datafile in enumerate(datafiles):
        t_ref = read_topography(os.path.join(file_format_examples, datafile))
        # TopoBank exports always carry an explicit detrend mode; sizes and
        # unit mirror the data file, as on a real export.
        topographies.append(
            {
                "name": f"Measurement {i}",
                "datafile": {"original": datafile},
                "size": [float(s) for s in t_ref.physical_sizes],
                "unit": t_ref.unit,
                "detrend_mode": "center",
            }
        )
    index = {
        "versions": {},
        "surfaces": [{"name": "My surface", "topographies": topographies}],
    }
    with zipfile.ZipFile(fn, "w") as z:
        for datafile in datafiles:
            z.write(os.path.join(file_format_examples, datafile), datafile)
        z.writestr("index.json", json.dumps(index))
        if with_legacy_yaml:
            # Conflicting legacy metadata; `index.json` must win
            z.writestr(
                "meta.yml",
                yaml.dump({"surfaces": [{"name": "WRONG", "topographies": []}]}),
            )
    return datafiles


def test_read_index_json_container(file_format_examples, tmp_path):
    """CEReader must read containers with `index.json` metadata -- the only
    kind of metadata that current TopoBank exports carry."""
    fn = str(tmp_path / "container.zip")
    _make_index_json_container(fn, file_format_examples)

    r = CEReader(fn)
    assert r.nb_containers == 1
    c = r.container()
    assert len(c) == 2

    # The pipeline (here: center detrending) is reconstructed from the
    # metadata, exactly as for legacy `meta.yml` containers
    t1_ref = read_topography(f"{file_format_examples}/di-1.di").detrend("center")
    t2_ref = read_topography(f"{file_format_examples}/opd-1.opd").detrend("center")
    assert_allclose(c[0].heights(), t1_ref.heights())
    assert_allclose(c[1].heights(), t2_ref.heights())
    assert c[0].unit == t1_ref.unit
    assert c[1].unit == t2_ref.unit


def test_detect_format_index_json_container(file_format_examples, tmp_path):
    from SurfaceTopography.Container.IO import detect_format

    fn = str(tmp_path / "container.zip")
    _make_index_json_container(fn, file_format_examples)
    assert detect_format(fn) == "ce"


def test_index_json_takes_precedence_over_legacy_yaml(
    file_format_examples, tmp_path
):
    fn = str(tmp_path / "container.zip")
    _make_index_json_container(fn, file_format_examples, with_legacy_yaml=True)

    r = CEReader(fn)
    assert len(r.container()) == 2
    assert r.container().info["name"] == "My surface"


def test_invalid_index_json_raises(file_format_examples, tmp_path):
    from SurfaceTopography.Exceptions import CorruptFile

    fn = str(tmp_path / "container.zip")
    with zipfile.ZipFile(fn, "w") as z:
        z.write(os.path.join(file_format_examples, "di-1.di"), "di-1.di")
        # `surfaces` must be a list
        z.writestr("index.json", json.dumps({"surfaces": {"oops": 1}}))
    with pytest.raises(CorruptFile):
        CEReader(fn)


def test_missing_metadata_raises_format_mismatch(file_format_examples, tmp_path):
    from SurfaceTopography.Container.IO import detect_format
    from SurfaceTopography.Exceptions import (
        CannotDetectFileFormat,
        FileFormatMismatch,
    )

    fn = str(tmp_path / "batch.zip")
    with zipfile.ZipFile(fn, "w") as z:
        z.write(os.path.join(file_format_examples, "di-1.di"), "di-1.di")
    with pytest.raises(FileFormatMismatch):
        CEReader(fn)
    # A bare batch of data files is not a container of any known format
    with pytest.raises(CannotDetectFileFormat):
        detect_format(fn)


def test_corrupt_datafile_surfaces_on_access_not_on_open(
    file_format_examples, tmp_path
):
    """Opening a container parses only the metadata; a corrupt data file must
    not prevent opening (or format detection), only reading the affected
    topography."""
    fn = str(tmp_path / "container.zip")
    index = {
        "surfaces": [
            {
                "name": "My surface",
                "topographies": [
                    {
                        "name": "Broken measurement",
                        "datafile": {"original": "broken.dat"},
                        "size": [1.0, 1.0],
                        "unit": "µm",
                    }
                ],
            }
        ]
    }
    with zipfile.ZipFile(fn, "w") as z:
        z.writestr("broken.dat", b"\x00\x01\x02 this is not a topography")
        z.writestr("index.json", json.dumps(index))

    r = CEReader(fn)  # Must not raise
    c = r.container()
    assert len(c) == 1
    with pytest.raises(Exception):
        c[0]


def test_ce_members_index_json(file_format_examples, tmp_path):
    from SurfaceTopography.Container.IO.Schema import TopographyMeta

    fn = str(tmp_path / "container.zip")
    datafiles = _make_index_json_container(fn, file_format_examples)

    r = CEReader(fn)
    members = r.members()
    assert [m.name for m in members] == ["Measurement 0", "Measurement 1"]
    assert all(m.visible for m in members)

    # Members of an `index.json` container carry validated metadata
    for member in members:
        assert isinstance(member.metadata, TopographyMeta)
    assert members[0].metadata.datafile.original == "di-1.di"
    assert members[0].metadata.unit is not None

    # The member streams must return the verbatim bytes of the bundled files
    for member, datafile in zip(members, datafiles):
        with open(os.path.join(file_format_examples, datafile), "rb") as f:
            reference_bytes = f.read()
        with member.open() as f:
            assert f.read() == reference_bytes


def test_ce_members_legacy_yaml_without_names(file_format_examples, tmp_path):
    """Legacy containers may lack measurement names entirely; members fall
    back to the data file name and carry no validated metadata."""
    fn = str(tmp_path / "container.zip")
    with zipfile.ZipFile(fn, "w") as z:
        z.write(os.path.join(file_format_examples, "di-1.di"), "data/di-1.di")
        z.writestr(
            "meta.yml",
            yaml.dump(
                {
                    "surfaces": [
                        {
                            "topographies": [
                                {"datafile": {"original": "data/di-1.di"}}
                            ]
                        }
                    ]
                }
            ),
        )

    r = CEReader(fn)
    (member,) = r.members()
    assert member.name == "di-1.di"
    assert member.metadata is None
    with open(os.path.join(file_format_examples, "di-1.di"), "rb") as f:
        with member.open() as g:
            assert g.read() == f.read()


def test_write_creates_valid_index_json(file_format_examples):
    """Containers written by `write_containers` must carry schema-valid
    `index.json` metadata alongside the legacy `meta.yml`."""
    from SurfaceTopography.Container.IO.Schema import ContainerMeta

    t1 = read_topography(f"{file_format_examples}/di-1.di")
    t2 = read_topography(f"{file_format_examples}/opd-1.opd")
    c = InMemorySurfaceContainer([t1, t2])

    with tempfile.TemporaryFile() as fobj:
        c.to_zip(fobj)

        fobj.seek(0)
        with zipfile.ZipFile(fobj) as z:
            names = set(z.namelist())
            assert "index.json" in names
            assert "meta.yml" in names
            meta = ContainerMeta.model_validate_json(z.read("index.json"))
            meta_yaml = ContainerMeta.model_validate(
                yaml.safe_load(z.read("meta.yml"))
            )
        # Both metadata files carry identical content
        assert meta == meta_yaml
        assert len(meta.surfaces) == 1
        assert len(meta.surfaces[0].topographies) == 2
        assert meta.surfaces[0].topographies[0].datafile.squeezed_netcdf is not None

        # ... and the container reads back, through the `index.json` path
        (c2,) = read_container(fobj)
        assert len(c2) == 2
        assert_allclose(c2[0].heights(), t1.heights())
        assert_allclose(c2[1].heights(), t2.heights())
        assert c2[0].unit == t1.unit
        assert c2[1].unit == t2.unit
