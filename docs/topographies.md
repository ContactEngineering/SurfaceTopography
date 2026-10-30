# Topographies

```python
from SurfaceTopography import Topography, UniformLineScan, NonuniformLineScan
```

A *topography* is a container of heights $h$ on a set of lateral positions.
Three classes hold data; everything else in SurfaceTopography either creates
them ([reading](reading.md), [generation](generation.md)), analyzes them, or
transforms them into new topographies ([pipelines](pipelines.md)).

| Class | Grid | Heights | Interpolation between points |
|-|-|-|-|
| `Topography` | uniform 2D, $N_x\times N_y$ | $h_{ij}$ | none |
| `UniformLineScan` | uniform 1D, $N$ | $h_i$ | none |
| `NonuniformLineScan` | arbitrary positions $x_0,\dots,x_{N-1}$ | $h_i$ | linear |

Uniform data is treated as samples on grid points: analyses work on the
discrete values (sums, FFTs, finite differences). Nonuniform line scans are
interpreted as the piecewise linear function through the data points, and
their analyses are computed for this function.

## Constructing topographies from arrays

```python
Topography(heights, physical_sizes, periodic=False, nb_grid_pts=None,
           subdomain_locations=None, nb_subdomain_grid_pts=None, unit=None,
           info={}, decomposition="serial", communicator=MPI.COMM_SELF)
UniformLineScan(heights, physical_sizes, periodic=False, unit=None, info={},
                communicator=MPI.COMM_SELF)
NonuniformLineScan(x, y, unit=None, info={})
```

`Topography` requires a two-dimensional numpy array (a nested list is not
converted); `UniformLineScan` accepts any one-dimensional array-like. For
`NonuniformLineScan`, `x` are the positions and `y` the heights. Heights are
stored as `float64`.

```python
import numpy as np
from SurfaceTopography import Topography, UniformLineScan, NonuniformLineScan

h = np.random.default_rng(0).normal(size=(4, 3))
t = Topography(h, physical_sizes=(2.0, 3.0), unit="µm")
print(t.nb_grid_pts, t.pixel_size, t.area_per_pt)  # (4, 3) [0.5 1. ] 0.5

l = UniformLineScan([1.0, 2.0, 3.0, 4.0], physical_sizes=4.0, unit="nm")
n = NonuniformLineScan([0.0, 1.0, 3.0], [1.0, 2.0, 3.0], unit="nm")
print(n.physical_sizes, n.x_range)  # (3.0,) (0.0, 3.0)
```

### Common interface

| Attribute / method | Meaning |
|-|-|
| `dim` | 1 for line scans, 2 for topography maps |
| `is_uniform` | `True` for `Topography` and `UniformLineScan` |
| `nb_grid_pts` | $(N_x, N_y)$ or $(N,)$ |
| `physical_sizes` | $(L_x, L_y)$ or $(L,)$; nonuniform: $(x_{N-1}-x_0,)$ |
| `positions()` | $x$ (line scans) or $(x, y)$ (maps) |
| `heights()` | $h$ as a numpy (masked) array |
| `positions_and_heights()` | `(x, h)` or `(x, y, h)` |
| `unit` | length unit, a string or `None` |
| `is_periodic` | periodic boundary conditions |
| `has_undefined_data` | some heights are masked (uniform data only) |
| `is_reentrant` | positions not strictly increasing (nonuniform only) |
| `info` | metadata dictionary |
| `pipeline()` | list of objects from the data source to this object |
| `squeeze()` | evaluate the pipeline into a data-owning object |
| `communicator` | MPI communicator |

Uniform topographies additionally have `pixel_size`, `area_per_pt`,
`nb_subdomain_grid_pts`, `subdomain_locations`, `subdomain_slices` and
`is_domain_decomposed` ([see below](#mpi-parallel-topographies));
`NonuniformLineScan` has `x_range` $=(x_0, x_{N-1})$.

`t[i]` indexes the heights (`t.heights()[i]`); for nonuniform line scans it
returns the tuple `(x[i], h[i])`. `t1 == t2` is true if unit, info,
periodicity and number of grid points are equal and positions and heights
agree within `np.allclose`. Topographies can be pickled.

## Grid conventions

The first array axis is the $x$-direction and the second the $y$-direction:
`heights()[i, j]` is $h_{ij} = h(x_i, y_j)$. Uniform grid points are at the
lower-left corner of each pixel,

$$
x_i = i\,\Delta x,\quad y_j = j\,\Delta y,\qquad
\Delta x = \frac{L_x}{N_x},\quad \Delta y = \frac{L_y}{N_y},
$$

with $i = 0,\dots,N_x-1$ and $j=0,\dots,N_y-1$, so the last grid point is
at $L_x - \Delta x$. `pixel_size` is $(\Delta x, \Delta y)$ and
`area_per_pt` is $\Delta x\,\Delta y$ ($\Delta x$ for line scans), the
weight of one grid point in an integral over the surface,
$\int h\,\mathrm d^2r \approx \Delta x\,\Delta y\sum_{ij} h_{ij}$.

`positions()` returns arrays of the same shape as the heights
(`np.meshgrid(..., indexing="ij")`); `positions(meshgrid=False)` returns the
one-dimensional arrays $x_i$ and $y_j$.

```python
x, y = t.positions()
print(x[:, 0], y[0])  # [0.  0.5 1.  1.5] [0. 1. 2.]
```

## Data orientation for plotting

Because $x$ is the first axis, `heights()` is the transpose of an image in
matplotlib's row-major convention. To plot with the origin in the lower left
(Cartesian axes):

```python
import matplotlib.pyplot as plt

plt.pcolormesh(t.heights().T)               # axes labeled by grid indices
plt.pcolormesh(*t.positions_and_heights())  # axes labeled by positions
```

`imshow` places the first row at the top, so

```python
plt.imshow(t.heights().T, extent=(0, t.physical_sizes[0], t.physical_sizes[1], 0))
```

shows the origin in the upper left with $y$ pointing down. Readers store
data such that this view matches the image shown by the instrument software
and by Gwyddion (see `test/compare-orientation-with-gwyddion.py`).
`t.plot()` (requires matplotlib) produces this view, with axes in a
suitable length unit.

## Units

`unit` is a length unit string, e.g. `"m"`, `"mm"`, `"µm"` (also `"um"`),
`"nm"`, `"Å"` (also `"A"`), `"pm"`. The same unit applies to positions and
heights. `None` means that the unit is unknown; analysis functions then work
in the units of the numbers, but [unit conversion](pipelines.md) with
`to_unit` fails when the converted data is accessed. `to_unit("nm")` rescales positions, physical sizes
and heights by the conversion factor; `scale(factor)` multiplies heights by a
factor without changing the unit.

For data channels that are not heights (e.g. phase or amplitude images in
Nanoscope files), readers return a tuple `(lateral_unit, data_unit)`, e.g.
`('nm', None)`: the positions are in nm, and the unit of the data values is
unknown. Such topographies can be analyzed, but `to_unit` fails with
`ValueError`. See [reading](reading.md) for channel units.

## Periodicity

`periodic=True` declares that the topography repeats with period $L_x$
(and $L_y$), i.e. $h_{N_x,j} \equiv h_{0,j}$. Periodicity changes how
analyses treat the boundary: derivatives use differences across the boundary
instead of dropping the last point, Fourier transforms need no window, and
some operations (e.g. `translate`) are only defined for periodic data.
Nonuniform line scans are never periodic. Measured data is normally
nonperiodic; [synthetic topographies](generation.md) are periodic.

## Undefined data

Missing measurement points are represented by numpy masked arrays. The
constructors of `Topography` and `UniformLineScan` convert non-finite values
(NaN, $\pm\infty$) into masked entries; masked arrays can also be passed
directly:

```python
l = UniformLineScan([1.0, 2.0, np.nan, 4.0], 4.0)
print(l.has_undefined_data, l.heights())  # True [1.0 2.0 -- 4.0]
```

Functions that cannot handle undefined data raise `UndefinedDataError`, e.g.
the power spectral densities; the autocorrelation functions average over the
defined points. The
pipeline functions `fill_undefined_data` and
`interpolate_undefined_data` remove the mask ([imputation](pipelines.md)).
`NonuniformLineScan` cannot hold undefined data; its constructor rejects
masked arrays.

## Metadata: the `info` dictionary

`info` holds metadata that SurfaceTopography does not use for computation:
data read from the file, instrument information or arbitrary entries added
by the user. It is validated by the pydantic model
`SurfaceTopography.Metadata.InfoModel`; known keys are converted to their
types, unknown keys are kept as they are, and entries that are `None` are
omitted from `t.info`.

| Key | Type | Meaning |
|-|-|-|
| `acquisition_time` | `datetime` | date and time of the measurement |
| `instrument` | `InstrumentModel` | `name`, `vendor`, `serial` (instrument/controller), `scanner_serial`, `software`, `parameters` |
| `instrument.parameters` | `InstrumentParametersModel` | `name`, `resolution` and `tip_radius`, each `{"value": float, "unit": str}` |
| `raw_metadata` | `dict` or `list` | reader-specific metadata as found in the file |
| `channel_name` | `str` | name of the data channel |
| `datafile` | `dict` | information on the data file, attached by container readers |

Serial numbers that are placeholders in the file (empty, `"0"`, `"n/a"`,
`"unknown"`, ...) are stored as `None`.

```python
t = Topography(h, (2.0, 3.0), info={"instrument": {"name": "AFM", "serial": "n/a"},
                                    "sample": "A1"})
print(t.info)  # {'instrument': {'name': 'AFM'}, 'sample': 'A1'}
```

`info` returns a new dictionary on every access; modifying it does not
change the topography. Pipeline functions accept `info={...}` with entries
that are merged into the `info` of their parent.

## Analysis and pipeline functions

Operations on topographies are methods, and they come in two kinds:

- **Analysis functions** return values, e.g. `t.Sq()` or
  `t.power_spectrum_from_profile()`.
- **Pipeline functions** return a new topography derived from `t`, e.g.
  `t.detrend()` or `t.to_unit("nm")`.

| Topic | Functions | Page |
|-|-|-|
| scalar statistics | `mean`, `min`, `max`, `median`, `Sq`/`rms_height_from_area`, `Rq`/`rms_height_from_profile`, `rms_gradient`, `Rdq`/`rms_slope_from_profile`, `Rddq`/`rms_curvature_from_profile`, `rms_curvature_from_area`, `rms_laplacian`, `mad_height`, `moment`, `bearing_area`, `bandwidth` | [statistics](statistics.md) |
| scale-dependent analysis | `derivative`, `scale_dependent_statistical_property`, `scale_dependent_slope_from_*`, `scale_dependent_curvature_from_*`, `variable_bandwidth_from_*`, `checkerboard_detrend_*`, `short_reliability_cutoff`, `scanning_probe_reliability_cutoff`, `scan_with_rigid_sphere` | [scale-dependent](scale-dependent.md) |
| spectral analysis | `power_spectrum_from_profile`, `power_spectrum_from_area`, `autocorrelation_from_*`, `moment_power_spectrum`, `integrate_psd*`, `variance_half_derivative*`, `elastic_energy`, `fftfreq`, `wavevectors_norm2`, `fourier_derivative`, `make_fft` | [spectral](spectral.md) |
| detrending | `detrend`, `polyfit`, `mad_polyfit`, `scan_line_align` | [pipelines](pipelines.md) |
| filtering | `window`, `filter`, `shortcut`, `longcut`, `downsample` | [pipelines](pipelines.md) |
| interpolation | `interpolate_linear`, `interpolate_bicubic`, `interpolate_fourier`, `interpolate_cubic`, `mirror_stitch`, `to_uniform`, `to_nonuniform` | [pipelines](pipelines.md) |
| imputation | `fill_undefined_data`, `interpolate_undefined_data`, `interpolate_undefined_data_with_harmonic_function` | [pipelines](pipelines.md) |
| arithmetic, units | `add`, `subtract`, `superpose`, `scale`, `to_unit`, `transpose`, `translate` | [pipelines](pipelines.md) |
| output | `to_matrix`, `to_netcdf`, `to_x3p`, `to_gwy`, `to_dzi` | [reading and writing](reading.md) |
| plotting | `plot` | [above](#data-orientation-for-plotting) |
| parallelization | `domain_decompose` | [below](#mpi-parallel-topographies) |

Not every function exists for every class; `dir(t)` lists the functions
available for `t`, and functions specific to maps (`*_from_area`) raise an
error when called on line scans. Docstrings are available through `help(t.detrend)`.

### Registration

Functions are kept in a registry on the interface classes
`UniformTopographyInterface` (for `Topography` and `UniformLineScan`) and
`NonuniformLineScanInterface` (in `SurfaceTopography.HeightContainer`), not
on the topography classes themselves. The registry is determined by what
kind of data an object represents, not by how it was obtained: a uniform
topography created by `to_uniform` from a nonuniform line scan has the
uniform functions. Attribute access on a topography that does not find a
regular attribute looks up the registry (`__getattr__`) and calls the
function with the topography as first argument; `t.apply("detrend", ...)` is
equivalent to `t.detrend(...)`.

```python
from SurfaceTopography.HeightContainer import UniformTopographyInterface

def peak_to_valley(topography):
    h = topography.heights()
    return h.max() - h.min()

UniformTopographyInterface.register_function("peak_to_valley", peak_to_valley)
print(t.peak_to_valley())
```

`register_function(name, function, deprecated=False)` wraps the function so
that it accepts the `dois` argument ([bibliography](#bibliography)) and, for
`deprecated=True`, emits a `DeprecationWarning` on every call. Functions are
visible on the class they are registered on and its subclasses. The package
registers its functions when `SurfaceTopography` is imported.

### Pipelines

A pipeline function returns an object derived from `DecoratedTopography`
that stores a reference to its parent and its arguments, but no heights.
`heights()` recomputes the transformation from the parent every time it is
called; chains like `t.detrend().scale(2.0).window()` therefore evaluate the
whole chain on each access. The chain documents how the data was obtained:
`pipeline()` returns the list of objects from the data source to the last
step. Geometry, unit, periodicity and `info` are passed through from the
parent unless the step changes them.

`squeeze()` evaluates the pipeline once and returns a `Topography`,
`UniformLineScan` or `NonuniformLineScan` that owns the result. Squeeze
before repeated analyses of an expensive pipeline; squeezing a data-owning
object returns the object itself.

```python
d = t.detrend("center")
print([type(p).__name__ for p in d.pipeline()])  # ['Topography', 'DetrendedUniformTopography']
s = d.squeeze()
print(type(s).__name__)  # Topography
```

Custom pipeline steps can be written as classes derived from
`DecoratedUniformTopography` (or `DecoratedNonuniformTopography`) that
override `heights()`, or with the decorator
`SurfaceTopography.Pipeline.pipeline_function`, which generates such a class
from a function that computes the new heights:

```python
from SurfaceTopography.Pipeline import pipeline_function
from SurfaceTopography.UniformLineScanAndTopography import DecoratedUniformTopography

@pipeline_function(DecoratedUniformTopography)
def clip(topography, hmax):
    return np.minimum(topography.heights(), hmax)

UniformTopographyInterface.register_function("clip", clip)
c = t.clip(hmax=0.5)
print(c.hmax, c.heights().max() <= 0.5)  # 0.5 True
```

Arguments passed by keyword are available as attributes (`c.hmax`); the
keyword arguments `unit` and `info` update the metadata of the step.

## Bibliography

Analysis and pipeline functions record the publications that describe the
methods they use. Pass an empty set as `dois`; after the call it contains the
DOIs of the called function, of all functions it called in turn (including
the steps of the pipeline it was called on) and of the
[contact.engineering paper](https://doi.org/10.1088/2051-672X/ac860a), which
is always included:

```python
dois = set()
t.power_spectrum_from_area(dois=dois)
print(dois)  # {'10.1088/2051-672X/aa51f8', '10.1088/2051-672X/ac860a'}
```

Functions declare their references with the decorator
`SurfaceTopography.Support.doi`:

```python
from SurfaceTopography.Support import doi

@doi("10.1088/2051-672X/aa51f8")
def my_analysis(topography):
    ...
```

The set is filled while the outermost call that received `dois` runs; the
bookkeeping uses module-level state and is not thread-safe.

## Exceptions

Defined in `SurfaceTopography.Exceptions`:

| Exception | Raised when |
|-|-|
| `CannotPerformAnalysisError` | an analysis cannot be performed (base class of the next three) |
| `NoReliableDataError` | no data is left within the reliable bandwidth, see [reliability cutoff](scale-dependent.md) |
| `ReentrantDataError` | the analysis does not support reentrant line scans |
| `UndefinedDataError` | the analysis does not support undefined (masked) data |
| `UnknownFileFormat` | an unknown format identifier was passed to a reader function |
| `ReadFileError` | base class of the reader errors below |
| `CannotDetectFileFormat` | no reader can open the file |
| `FileFormatMismatch` | the reader cannot interpret the file at all |
| `UnsupportedFormatFeature` | the format is correct but uses a feature that is not implemented |
| `CorruptFile` | the format matches but the content is inconsistent |
| `UnsupportedSchema` | a [format description](format-description-contract.md) cannot be executed |
| `MetadataAlreadyFixedByFile` | a metadata argument (e.g. `physical_sizes`) is passed although the file defines it |

Averaging analyses of [containers](containers.md) skip topographies that
raise `NoReliableDataError` or `UndefinedDataError`.

<a id="mpi"></a>

## MPI-parallel topographies

Topography maps can be distributed over the processes of an MPI
communicator, such that each process stores only a rectangular subdomain of
the grid. This is used to analyze and process maps that do not fit into the
memory of a single node, and by codes built on SurfaceTopography (e.g.
contact mechanics with domain-decomposed FFTs).

MPI support is optional. The serial package depends on
[NuMPI](https://github.com/ComputationalMechanics/NuMPI), which provides a
stub of the `mpi4py` interface (`NuMPI.MPI`) for a single process and serial
FFTs. The extra `mpi` (`pip install SurfaceTopography[mpi]`, see
[installation](installation.md)) adds `mpi4py`, muGrid (parallel FFTs) and
netCDF4 (parallel I/O), which must be compiled against the same MPI library.

### Decomposition

Each process holds the heights of the subdomain

$$
i = s_x,\dots,s_x+n_x-1,\qquad j = s_y,\dots,s_y+n_y-1
$$

of the global $N_x\times N_y$ grid, where $(s_x, s_y)$ is
`subdomain_locations` and $(n_x, n_y)$ is `nb_subdomain_grid_pts`.
`nb_grid_pts` and `physical_sizes` are global, `heights()` and `positions()`
are local, and `subdomain_slices` cuts the subdomain out of a global array.
`is_domain_decomposed` is true if the subdomain is smaller than the grid.

SurfaceTopography does not choose the decomposition; it adopts the one of
the FFT engine, because FFT-based analyses require that the data layout
matches the engine. NuMPI's
[`make_fft_engine`](https://github.com/ComputationalMechanics/NuMPI) returns
a muGrid FFT engine for communicators with several processes (a serial numpy
engine for one process); its `subdomain_locations` and
`nb_subdomain_grid_pts` define the decomposition. muGrid splits 2D grids
along the last ($y$) axis.

```python
import numpy as np
from NuMPI import MPI
from NuMPI.FFT import make_fft_engine
from SurfaceTopography.Generation import fourier_synthesis

comm = MPI.COMM_WORLD
np.random.seed(0)  # every process generates the same global topography
t = fourier_synthesis((64, 64), (1.0, 1.0), hurst=0.8, rms_slope=0.1)

engine = make_fft_engine(t.nb_grid_pts, communicator=comm)
local = t.domain_decompose(engine.subdomain_locations,
                           engine.nb_subdomain_grid_pts, communicator=comm)
print(local.nb_subdomain_grid_pts, local.Sq())
```

`domain_decompose(subdomain_locations, nb_subdomain_grid_pts,
communicator=None)` turns a `Topography` that holds the global data on every
process into a decomposed one. Data that is only available locally is passed
to the constructor with `decomposition="subdomain"`; global data with
`decomposition="domain"` ([constructor signature](#constructing-topographies-from-arrays)):

```python
Topography(local_heights, physical_sizes, decomposition="subdomain",
           nb_grid_pts=nb_grid_pts, subdomain_locations=engine.subdomain_locations,
           communicator=comm)
```

For a communicator with a single process, the decomposition arguments are
ignored (or checked for consistency) and the topography is serial. The
default communicator is `MPI.COMM_SELF`, so that topographies created
without a communicator, e.g. by readers that do not support parallel I/O,
are serial and complete on every process of an MPI program. NetCDF files can
be read and written in parallel ([reading](reading.md)).

### What runs in parallel

Reductions (`mean`, `min`, `max`, `Sq`,
`has_undefined_data`, `==`) are computed over all processes with NuMPI's
`Reduction`. FFT-based functions use the engine returned by `make_fft`, which
checks that the engine's decomposition equals the topography's. Functions
that need data across subdomain boundaries without a parallel algorithm,
e.g. `Rq`, `rms_curvature_from_area` and
`rms_laplacian`, raise `NotImplementedError`.

Pipeline steps on a decomposed topography operate on the local data and keep
its communicator, but report `is_domain_decomposed = False`, and `squeeze()`
returns a serial topography of the local subdomain only.
<!-- TODO: rationale for decorated topographies reporting is_domain_decomposed = False? -->

## The `show-topography` command

Installing the package provides a command-line tool that reads a file with
[`read_topography`](reading.md) and plots its default channel with
`plot()` (requires matplotlib):

```
show-topography [--grid-points] filename
```

| Option | Meaning |
|-|-|
| `--grid-points`, `-g` | label axes with grid indices $i$, $j$ instead of positions |
