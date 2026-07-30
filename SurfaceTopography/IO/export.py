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
Exporter for format description documents and conformance goldens, per the
format description contract (docs/format_description_contract.rst).

SurfaceTopography is the reference implementation; the artifacts written
here are vendored into consumers (e.g. SDSAlgorithms/libsdsio) and pinned
by their CI.

Run as a script::

    python -m SurfaceTopography.IO.export --descriptions DIR
    python -m SurfaceTopography.IO.export --goldens DIR --corpus CORPUS_DIR
"""

import argparse
import json
import os

import numpy as np

from .description import SCHEMA_VERSION, count_opaque, reader_description

# Readers exported to format description documents, and the corpus
# fixtures backing their conformance goldens. Only fully declarative
# readers (zero opaque hooks) can be exported.
_EXPORTED_READERS = {
    "zmg": ["zmg-1.zmg", "zmg-2.zmg"],
    "tmd": ["tmd-1.tmd"],
    "al3d": ["al3d-1.al3d"],
    "sur": ["sur-1.sur", "sur-2.sur", "sur-3.sur", "sur-4.sur", "sur-5.sur"],
    "metropro": ["metropro-1.dat"],
    "plu": ["plu-1.plu"],
    "zon": ["zon-1.zon"],
}


def exported_reader_classes():
    """The reader classes exported to description documents, by format id."""
    from . import AL3DReader, MetroProReader, PLUReader, SURReader, TMDReader, ZMGReader, ZONReader

    classes = [
        ZMGReader,
        TMDReader,
        AL3DReader,
        SURReader,
        MetroProReader,
        PLUReader,
        ZONReader,
    ]
    return {cls._format: cls for cls in classes}


def _dump(document, file_path):
    with open(file_path, "w") as f:
        json.dump(document, f, indent=2, sort_keys=True)
        f.write("\n")


def export_descriptions(output_dir):
    """
    Write one description document per exported reader into `output_dir`.
    Refuses to export incomplete (opaque-hook-containing) documents.
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for format_id, reader_class in exported_reader_classes().items():
        document = reader_description(reader_class)
        nb_opaque = count_opaque(document)
        if nb_opaque > 0:
            raise RuntimeError(
                f"The description of `{format_id}` contains {nb_opaque} "
                f"non-serializable hooks and cannot be exported."
            )
        file_path = os.path.join(output_dir, f"{format_id}.json")
        _dump(document, file_path)
        written.append(file_path)
    return written


def _probe_indices(nx, ny):
    """Corners, center, and an interior point."""
    return [
        (0, 0),
        (nx - 1, 0),
        (0, ny - 1),
        (nx - 1, ny - 1),
        (nx // 2, ny // 2),
        (nx // 3, (2 * ny) // 3),
    ]


def _golden_channel(reader, index):
    channel = reader.channels[index]
    topography = reader.topography(channel_index=index)
    heights = np.ma.masked_invalid(
        np.ma.filled(np.ma.asarray(topography.heights(), dtype=float), np.nan)
    )
    nx, ny = heights.shape
    golden = {
        "index": index,
        "name": channel.name,
        "nb_grid_pts": [nx, ny],
        "physical_sizes": [float(s) for s in topography.physical_sizes],
        "unit": topography.unit,
        "nb_undefined": int(np.ma.count_masked(heights)),
        "probe_pixels": [
            {
                "pos": [i, j],
                "value": None if heights[i, j] is np.ma.masked else float(heights[i, j]),
            }
            for i, j in _probe_indices(nx, ny)
        ],
        "masked_mean": float(heights.mean()),
        "masked_rms": float(heights.std()),
    }
    return golden


def golden_document(reader_class, file_path, fixture_name):
    """Build the conformance golden document for one corpus fixture."""
    reader = reader_class(file_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "format": reader_class._format,
        "fixture": fixture_name,
        "channels": [
            _golden_channel(reader, index)
            for index in range(len(reader.channels))
        ],
    }


def export_goldens(output_dir, corpus_dir):
    """
    Write one conformance golden document per corpus fixture of every
    exported reader into `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = exported_reader_classes()
    written = []
    for format_id, fixtures in _EXPORTED_READERS.items():
        for fixture in fixtures:
            document = golden_document(
                classes[format_id], os.path.join(corpus_dir, fixture), fixture
            )
            file_path = os.path.join(
                output_dir, f"{os.path.splitext(fixture)[0]}.json"
            )
            _dump(document, file_path)
            written.append(file_path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export format descriptions and conformance goldens."
    )
    parser.add_argument(
        "--descriptions", metavar="DIR", help="Write description documents here"
    )
    parser.add_argument(
        "--goldens", metavar="DIR", help="Write conformance goldens here"
    )
    parser.add_argument(
        "--corpus",
        metavar="DIR",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "test", "file_format_examples"
        ),
        help="Corpus directory with example files (for --goldens)",
    )
    args = parser.parse_args(argv)
    if not args.descriptions and not args.goldens:
        parser.error("Nothing to do; pass --descriptions and/or --goldens.")
    if args.descriptions:
        for file_path in export_descriptions(args.descriptions):
            print(f"Wrote {file_path}")
    if args.goldens:
        for file_path in export_goldens(args.goldens, args.corpus):
            print(f"Wrote {file_path}")


if __name__ == "__main__":
    main()
