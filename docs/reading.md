# Reading and writing files

```python
from SurfaceTopography import open_topography, read_topography
```

A file is opened with `open_topography`, which returns a *reader*. The reader
lists the data *channels* in the file with their metadata, without reading the
height data. The height data is read when a topography is requested from the
reader. `read_topography` does both steps at once. The topography classes
that are returned are described in [topographies](topographies.md); the list of
formats is in [supported file formats](formats.md).

## Opening a file

```python
open_topography(fobj, format=None, communicator=None)
```

| Argument | Meaning |
|-|-|
| `fobj` | path, binary file-like object, or callable that returns a file-like object |
| `format` | format identifier from the [format table](formats.md#topography-files); `None` detects the format |
| `communicator` | MPI communicator for [parallel reading](#mpi-parallel-reading) |

The reader is a context manager that closes the file on exit:

```python
with open_topography("di-1.di") as reader:
    for channel in reader.channels:
        print(channel.index, channel.channel_id, channel.nb_grid_pts,
              channel.physical_sizes, channel.unit)
    topography = reader.topography()  # default channel
```

The example files used on this page are in `test/file_format_examples` of the
source repository. Opening a file is fast: readers parse the headers and
detect corrupt files at this point, but read the height data only in
`topography`.

### File-like objects

Instead of a path, `fobj` can be a file-like object opened in binary mode,
e.g. a file in memory or a decompressed stream:

```python
import gzip

with gzip.open("matrix-4.txt.gz") as f:
    topography = read_topography(f)
```

`fobj` can also be a callable without arguments that returns a new file-like
object each time it is called. The readers use this to reopen the file, e.g.
for members of [container files](containers.md).

### Format detection

Without `format`, the format is detected from the file contents; the file
name and extension are not used. The readers are tried in the order of the
list `SurfaceTopography.IO.readers`. A reader is skipped if the first 512 bytes
of the file rule out its format (its `can_read` class method compares
signatures, "magic bytes"); otherwise it attempts to open the file. The first
reader that opens the file without error is returned. If no reader succeeds,
`CannotDetectFileFormat` is raised; its message lists the error of every
reader that was tried. The order is listed in
[supported file formats](formats.md#order-of-detection).

```python
from SurfaceTopography.IO import detect_format

detect_format("opd-1.opd")                      # 'opd'
reader = open_topography("opd-1.opd", format="opd")
```

Specifying `format` skips detection, which is faster and avoids ambiguities
between formats that share a container (ZIP, TIFF, HDF5, plain text).

## Channels

A file contains one or more channels, e.g. height, phase and amplitude images
of an AFM scan, or forward and backward scans. `reader.channels` is a list of
`ChannelInfo` objects; `reader.default_channel` is the channel loaded when
none is selected, typically the first channel with height information.

| Property | Meaning |
|-|-|
| `index` | position in `reader.channels` |
| `name` | channel name stored in the file (`"Default"` or `"channel <index>"` if there is none) |
| `channel_id` | identifier that is unique within the file, see [below](#channel-identifiers) |
| `dim` | 1 for line scans, 2 for topography maps |
| `nb_grid_pts` | number of grid points, $(N_x,)$ or $(N_x, N_y)$ |
| `physical_sizes` | $(L,)$ or $(L_x, L_y)$, `None` if not stored in the file |
| `pixel_size` | $(\Delta x,)$ or $(\Delta x, \Delta y)$, `None` without physical sizes |
| `area_per_pt` | $\Delta x\,\Delta y$ |
| `unit` (alias `lateral_unit`) | length unit, `None` if not stored in the file |
| `data_unit` | unit of the data values; for height channels the length unit |
| `height_scale_factor` | factor applied to the raw values, `None` if not stored in the file |
| `is_periodic`, `is_uniform`, `has_undefined_data` | as stored in the file (may be `None` if unknown) |
| `data_kind` | `DataKind` of the values (`HEIGHT`, `PHASE`, `AMPLITUDE`, `VOLTAGE`, ...) |
| `is_height_channel` | `data_kind == DataKind.HEIGHT` |
| `height_index` | position among the height channels, `None` for other channels |
| `info` | metadata found in the file, see [topographies](topographies.md) |

`reader.height_channels` lists only the height channels.

A unit given as a tuple, e.g. `('nm', None)`, means that the lateral unit is
known but the unit of the data values is not (e.g. phase images in Nanoscope
files).

### Selecting a channel

The `topography` method of the reader loads a channel by its position in
`reader.channels`:

```python
reader = open_topography("di-1.di")
phase = reader.topography(channel_index=2)
phase = reader.channels[2].topography()  # equivalent
```

Some readers (those built on the declarative reader engine, see
[declarative readers](declarative-readers.md), and the DI, IBW and QEP
readers) additionally select a channel by its identifier or by its position
among the height channels. At most one of the three selection
arguments may be given:

```python
phase = reader.topography(channel_id="Phase")
height = reader.topography(height_channel_index=1)
```

`ChannelInfo.topography` accepts the same metadata arguments as the reader
(see [below](#metadata-from-the-file-and-from-arguments)) and works with every
reader.

### Channel identifiers

The `channel_id` of a channel is its name. If several channels in the file
have the same name, they are numbered in the order of the file:
`"Height#1"`, `"Height#2"`. The identifier depends only on the channel names,
not on the number or order of channels with other names. Use it to store a
reference to a channel, e.g. in a database:

```python
channel_id = reader.channels[1].channel_id     # 'Height#2'

# Later
channel = next(ch for ch in reader.channels if ch.channel_id == channel_id)
topography = channel.topography()
```

`reader.height_index_to_channel_id(height_index)` returns the identifier of the
height channel at a position among the height channels;
`reader.channel_id_to_height_index(channel_id)` does the reverse and raises
`ValueError` for channels that do not contain heights. Use them when a channel
reference is given as a position among the height channels, e.g. the
`height_channel_index` argument, and needs to be stored as an identifier or
vice versa:

```python
reader.height_index_to_channel_id(1)            # 'Height#2'
reader.channel_id_to_height_index("Height#2")   # 1
```

## Reading a topography

```python
reader.topography(channel_index=None, physical_sizes=None,
                  height_scale_factor=None, unit=None, info={},
                  periodic=False, subdomain_locations=None,
                  nb_subdomain_grid_pts=None)

read_topography(fn, format=None, communicator=None, **kwargs)
```

`read_topography` opens the file, calls `reader.topography(**kwargs)` and
closes the file. Depending on the file contents, the result is a `Topography`,
a `UniformLineScan` or a `NonuniformLineScan` (see
[topographies](topographies.md)).

### Metadata from the file and from arguments

| Argument | Meaning |
|-|-|
| `physical_sizes` | $(L_x, L_y)$ or $L$ |
| `unit` | length unit of lateral sizes and heights, e.g. `"µm"` |
| `height_scale_factor` | factor by which the stored values are multiplied |
| `info` | dictionary merged into the metadata read from the file |
| `periodic` | whether the topography is one period of a periodic surface |

Metadata that is stored in the file is fixed: passing `physical_sizes`, `unit`
or `height_scale_factor` for a channel that has it raises
`MetadataAlreadyFixedByFile`. Metadata that is not stored in the file must be
passed; a missing physical size raises `ValueError`. Inspect the
`ChannelInfo` (`physical_sizes`, `unit`, `height_scale_factor` are `None` when
missing) to decide which arguments to pass.

```python
topography = read_topography("example-2d.npy", physical_sizes=(10.0, 10.0),
                             unit="µm", height_scale_factor=1e-3,
                             periodic=True, info={"sample": "A"})
```

Periodicity affects the windowing in spectral analysis (see
[spectral analysis](spectral.md)). The default of `periodic` depends on the
reader: readers of formats that store periodicity (e.g. NetCDF) have the
default `None` and then take it from the file; the others default to
`False`. `ChannelInfo.topography` always passes `periodic=False` unless
`periodic` is given.

## MPI-parallel reading

The NetCDF and NPY readers read large maps in parallel: every MPI process
reads only its subdomain. Pass the communicator to `open_topography` and the
subdomain of the present process to `topography`:

```python
from NuMPI import MPI
from NuMPI.FFT import make_fft_engine
from SurfaceTopography import open_topography

comm = MPI.COMM_WORLD
reader = open_topography("example-2d.npy", communicator=comm)
nb_grid_pts = reader.channels[0].nb_grid_pts

engine = make_fft_engine(nb_grid_pts, communicator=comm)
topography = reader.topography(
    physical_sizes=(1.0, 1.0),
    subdomain_locations=engine.subdomain_locations,
    nb_subdomain_grid_pts=engine.nb_subdomain_grid_pts,
)
```

- `subdomain_locations` is the index of the first grid point of the subdomain,
  `nb_subdomain_grid_pts` its number of grid points. The decomposition is
  usually that of the FFT engine (see [topographies](topographies.md#mpi)).
- Parallel reading works for two-dimensional maps only.
- For a communicator with more than one process, `open_topography` skips
  readers without parallel support during format detection, and raises
  `ValueError` if such a format is given explicitly.
- Parallel NetCDF I/O requires the `netCDF4` package with parallel support.

## Writing files

Uniform topographies and line scans have methods that write them to a file:

| Method | Format | Notes |
|-|-|-|
| `to_netcdf(fobj, format="NETCDF3_64BIT_OFFSET")` | NetCDF | heights, mask of undefined data, physical sizes, unit, periodicity and `info`; also nonuniform line scans; parallel for domain-decomposed topographies |
| `to_x3p(fobj, dtype="D", manufacturer="SurfaceTopography", model="Python Library", version=None)` | X3P | two-dimensional maps; converted to m; undefined data marked invalid |
| `to_gwy(fobj, name="Topography")` | Gwyddion | two-dimensional maps; converted to m; undefined data as mask |
| `to_matrix(fname)` | plain text | heights only |
| `to_dzi(name, root_directory=".", tile_size=256, overlap=1, format="jpg", meta_format="xml", cmap=None, **kwargs)` | Deep Zoom Image | image pyramid for web viewers |

NPY files are written with `SurfaceTopography.IO.NPY.save_npy(fn, topography)`,
which writes in parallel for domain-decomposed topographies.

```python
import numpy as np
from SurfaceTopography import Topography
from SurfaceTopography.IO.NPY import save_npy

topography = Topography(np.random.rand(64, 32), (2.0, 1.0), unit="µm")

topography.to_netcdf("topography.nc")
topography.to_x3p("topography.x3p")
topography.to_gwy("topography.gwy")
topography.to_matrix("topography.txt")
save_npy("topography.npy", topography)
files = topography.to_dzi("topography", root_directory="dzi")
```

NetCDF is the format that preserves all metadata of a topography; reading the
file returns the topography as it was written.

**X3P.** `dtype` selects the storage of the heights: `"D"` (64-bit float),
`"F"` (32-bit float), `"L"` (32-bit integer) or `"I"` (16-bit integer). The
integer types store the heights $h_i$ as integers $n_i$ with an increment
$\delta$ and an offset $h_0$, $h_i = h_0 + n_i\,\delta$, and therefore round
the heights to $2^{32}$ or $2^{16}$ levels between the minimum and maximum
height. Lateral sizes and heights are converted to meters, the unit of the
X3P format.

**Text matrix.** `to_matrix` writes the heights with `numpy.savetxt`: row $i$
of the file contains $h_{i0}, h_{i1}, \dots$, i.e. rows run along the first
axis $x$. The [plain text reader](formats.md#topography-files) interprets
rows as $y$, so reading the file back yields the transposed map. The file
contains no physical sizes and no unit.

**Deep Zoom Image.** `to_dzi` writes the metadata file `name.xml` (or
`name.json` with `meta_format="json"`) and the directory `name_files` with
image tiles of `tile_size` pixels at every zoom level, for zoomable viewers
such as [OpenSeadragon](https://openseadragon.github.io/). The heights are
mapped to colors with the colormap `cmap`. With `format="npy"` or
`format="nc"`, the tiles contain the raw heights instead of images. Further
keyword arguments are passed to Pillow's `save`. The method returns the list
of files written.
