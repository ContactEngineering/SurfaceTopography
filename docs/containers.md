# Surface containers

```python
from SurfaceTopography import read_container, open_container, read_published_container
from SurfaceTopography.Container.SurfaceContainer import InMemorySurfaceContainer
```

A *surface container* is a collection of topographies (maps and/or line scans)
of the same surface. Typically, each topography was measured with a different
instrument or at a different magnification, so that each one covers a
different range of length scales (its *bandwidth*). Statistical properties of
the surface over the full range of scales are obtained by combining the
results of the individual topographies. Containers are the data model of
[contact.engineering](https://contact.engineering), where a container
corresponds to a *digital surface twin*.

A container behaves like a read-only sequence of topographies:
`len(container)`, `container[i]` and `for t in container` work as for a list.
The individual topographies are ordinary topography objects (see
[Topographies](topographies.md)).

## Container classes

| Class | Storage |
|-|-|
| `SurfaceContainer` | Abstract base class; holds the registry of analysis functions |
| `InMemorySurfaceContainer(topographies=[])` | A list of topography objects in memory |
| `LazySurfaceContainer(readers=[], info={})` | A list of callables, each returning a topography; `info` holds the surface metadata |

Reading a container file returns `LazySurfaceContainer`s. Each element access
calls the respective reader and reads the topography from the file again, so
that opening a container with many large measurements is cheap. If the
topographies are accessed repeatedly, `read_all()` loads all of them once and
returns an `InMemorySurfaceContainer`.

To build a container from topographies in memory, pass a list:

```python
import numpy as np
from SurfaceTopography.Container.SurfaceContainer import InMemorySurfaceContainer
from SurfaceTopography.Generation import fourier_synthesis

np.random.seed(0)
t1 = fourier_synthesis((256, 256), (10, 10), hurst=0.8, rms_height=1,
                       unit="µm", info={"name": "large scan"})
t2 = fourier_synthesis((256,), (1,), hurst=0.8, rms_height=0.1,
                       unit="µm", info={"name": "small line scan"})
surface = InMemorySurfaceContainer([t1, t2])
```

A container can mix maps and line scans, uniform and nonuniform data, and
topographies in different units.

## Reading containers

```python
read_container(fn, format=None, **kwargs)
open_container(fobj, format=None, **kwargs)
read_published_container(publication_url, **request_args)
```

A container file can hold several surfaces. `read_container` returns a list
with one container per surface:

```python
container, = read_container("container-1.zip")
print(len(container), container.info["name"])
for t in container:
    print(t.info["name"], t.nb_grid_pts, t.physical_sizes, t.unit)
```

```text
3 Self-affine synthetic surface
50000x50000_random.txt (500, 500) (100.0, 100.0) µm
5000x5000_random.txt (500, 500) (10.0, 10.0) µm
500x500_random.txt (500, 500) (1.0, 1.0) µm
```

`fn` is a file name or a binary file-like object. Without `format`, the
format is detected by trying each container reader in turn. Further keyword
arguments are passed to the reader.

| `format` | Reader | File |
|-|-|-|
| `"ce"` | `CEReader` | contact.engineering ZIP container (see [below](#the-container-file-format)) |
| `"zag"` | `ZAGReader` | Keyence ZAG container: all visible ZON measurements as one surface |

`CEReader` accepts two options:

| Argument | Default | Meaning |
|-|-|-|
| `datafile_keys` | `["original", "squeezed-netcdf"]` | Keys of the `datafile` entry of each measurement to try, in this order |
| `ignore_filters` | `False` | Do not reconstruct the pipeline from the metadata |

### Reader objects

`open_container` returns a reader object without reading any topography. It
is a context manager and provides

| Member | Meaning |
|-|-|
| `format()`, `name()`, `description()`, `mime_types()`, `file_extensions()` | Information on the file format |
| `nb_containers` | Number of surfaces in the file |
| `container(index=0)` | The `LazySurfaceContainer` of surface `index` |
| `members(index=0)` | The raw data files of surface `index` as `ContainerMember`s |

`members` lists the measurements without decoding them. A `ContainerMember`
has the fields `name`, `open` (a callable returning a new binary stream with
the verbatim bytes of the data file), `visible` (ZAG files can hide
measurements; hidden ones are listed by `members` but excluded from
`container`) and `metadata` (a validated `TopographyMeta`, or `None` if the
format or the metadata file does not provide it):

```python
with open_container("container-1.zip") as reader:
    print(reader.format(), reader.nb_containers)
    for member in reader.members():
        print(member.name, member.metadata.size, member.metadata.unit)
```

```text
ce 1
50000x50000_random.txt [100.0, 100.0] µm
5000x5000_random.txt [10.0, 10.0] µm
500x500_random.txt [1.0, 1.0] µm
```

Opening a file parses only the container's metadata. A corrupt metadata file
raises on opening; a corrupt data file of an individual measurement raises
when that topography is accessed. This keeps format detection (which
trial-opens the file with every reader) and the listing of large containers
cheap.

### Published containers

`read_published_container` downloads a container published on
contact.engineering and returns the list of its surfaces. The URL is the
publication URL or its DOI:

```python
container, = read_published_container("https://doi.org/10.57703/ce-v9qwe")
```

The publication URL is requested with the header `Accept: application/json`;
the response contains the download URL of the ZIP file, which is then read
with `read_container`. Keyword arguments are passed to `requests.get` for both
requests (a `headers` dictionary is merged with the `Accept` header); the
timeout is 60 seconds unless `timeout` is given. This function requires
network access.

## The container file format

A contact.engineering container is a ZIP archive with

- the data files of all measurements, either as originally uploaded or as
  NetCDF files with the processed heights ("squeezed" data), or both,
- a metadata file `index.json`,
- optionally `meta.yml` (a YAML file with the same content), `README.txt` and
  a license file.

`index.json` is validated against the pydantic model
`SurfaceTopography.Container.IO.Schema.ContainerMeta`; a file that does not
conform raises `CorruptFile`. If the archive contains no `index.json`,
`meta.yml` is read instead and is not validated, so that archives lacking
fields the schema requires can still be read. An archive with neither file is
not a contact.engineering container.

The metadata has the structure

```text
ContainerMeta
├── versions          {package: version}
├── created_at        str
├── extra             dict (not interpreted)
└── surfaces[]        SurfaceMeta
    ├── name, category, description, tags, created_by
    ├── is_published, publication (url, license, authors, version, date, doi_url, ...)
    ├── properties[]  (name, value, unit)
    └── topographies[] TopographyMeta
```

The fields of each measurement (`TopographyMeta`) are

| Field | Default | Meaning |
|-|-|-|
| `name` | required | Name of the measurement |
| `datafile` | required | `{"original": ..., "squeezed-netcdf": ...}`: names of the data files in the archive |
| `size` | required | Physical size, `physical_sizes` |
| `unit` | `None` | Length unit |
| `data_source` | `0` | Channel of the data file to read |
| `is_periodic` | `False` | Periodicity |
| `height_scale` | `None` | Factor applied to the heights |
| `fill_undefined_data_mode` | `"do-not-fill"` | Mode of `interpolate_undefined_data` |
| `detrend_mode` | `"center"` | Mode of `detrend` |
| `has_undefined_data`, `created_by`, `measurement_date`, `description`, `tags`, `instrument` | | Descriptive metadata |

Each measurement is read as follows. The first key of `datafile_keys` that is
present in `datafile` selects the data file, which is read with
`open_topography` (see [Reading topographies](reading.md)) from channel
`data_source`. The physical size and unit are taken from the data file if
it provides them (with a warning if they disagree with `size` and `unit`;
sizes are compared with a relative tolerance of $10^{-4}$), otherwise from the
metadata. Unless `ignore_filters` is set or the data file key starts with
`squeezed`, the pipeline (see [Pipelines](pipelines.md)) is reconstructed
from the metadata in the order `scale(height_scale)`,
`interpolate_undefined_data(fill_undefined_data_mode)` (unless
`"do-not-fill"`), `detrend(detrend_mode)`, each step only if the field is
present. Squeezed data files already contain the processed heights. All
remaining metadata fields end up in the topography's `info` dictionary; the
surface entry is the container's `info`.

## Writing containers

```python
container.to_zip(fn)
write_containers(containers, fn)   # from SurfaceTopography.Container.IO
```

`to_zip` writes a single container, `write_containers` a list of containers
(one surface each) into one archive. `fn` is a file name or a binary stream.

```python
surface.to_zip("my-surface.zip")
c, = read_container("my-surface.zip")
print([t.info["name"] for t in c])
```

```text
['large scan', 'small line scan']
```

Each topography is written as a NetCDF file `topography<n>.nc` (see
[File formats](formats.md)) that contains the heights after the pipeline,
listed under `squeezed-netcdf`; reading the archive therefore does not
reconstruct a pipeline. The metadata is written both as `index.json` and with
identical content as `meta.yml`, so that readers that only understand the YAML
file can read the archive. Measurement names are taken from
`topography.info["name"]` and surface names from `container.info["name"]`
(defaults: `topography<n>.nc` and `surface<n>`).

## Container analyses

Analysis functions are methods of every container. They are computed from
the 1D profiles along $x$ (`*_from_profile` functions of the topographies),
because a container can mix maps and line scans. All functions take a length
`unit`; each topography is converted to this unit (lateral sizes and heights,
`to_unit`) before the analysis, so that results of topographies stored in
different units are combined consistently. Results are reported in `unit`.

| Method | Result |
|-|-|
| `bandwidth(unit)` | Smallest lower and largest upper bandwidth bound of all topographies |
| `suggest_length_unit(scale)` | Length unit for plotting the container's bandwidth (`scale` is `"linear"` or `"log"`) |
| `power_spectrum(unit, ...)` | Averaged 1D PSD $C^\text{1D}(q)$ |
| `autocorrelation(unit, ...)` | Averaged height-difference ACF $A(r)$ |
| `variable_bandwidth(unit, ...)` | Averaged rms height as a function of bandwidth |
| `scale_dependent_statistical_property(func, n, unit, ...)` | Averaged statistics of scale-dependent derivatives |
| `integrate_psd_from_profile(factor, unit, ...)` | Weighted integral of the 1D PSDs of all topographies |
| `c1d_moment(order, unit, ...)` | Moment of the averaged 1D PSD |
| `ciso_moment(order, unit, ...)` | Moment of the isotropic PSD estimated from the averaged 1D PSD |

```python
container, = read_container("container-1.zip")
print(container.bandwidth(unit="µm"), container.suggest_length_unit("log"))
```

```text
(np.float64(0.002), np.float64(100.0)) nm
```

The bandwidth of a uniform topography ranges from its grid spacing $\Delta x$
to its physical size $L$ (the mean over both directions for maps); for
nonuniform line scans the lower bound is the mean point spacing.

All averages weight each topography equally, independent of its number of
grid points or its size; each topography is treated as a statistically
independent measurement of the surface.

### Averaging on a logarithmic grid

```python
container.power_spectrum(unit, nb_points_per_decade=10, reliable=True, progress_callback=None, **kwargs)
container.autocorrelation(unit, nb_points_per_decade=10, reliable=True, progress_callback=None, **kwargs)
container.variable_bandwidth(unit, nb_points_per_decade=10, reliable=True, progress_callback=None, **kwargs)
```

These functions compute, for each topography, `power_spectrum_from_profile`,
`autocorrelation_from_profile` or `variable_bandwidth_from_profile` (see
[Spectral analysis](spectral.md)) without resampling and, with
`reliable=True`, restricted to the reliable data (see
[reliability cutoff](scale-dependent.md)). Further keyword arguments are
passed to these functions. The results are combined as follows.

Let $x$ be the abscissa (wavevector $q$, distance $r$ or bandwidth) and
$y$ the property, and let $n_d$ be `nb_points_per_decade`. The raw data of
topography $i$ spans $[a_i, b_i]$ (only $x>0$ is considered). It is
bin-averaged onto logarithmic bins that cover the full decades from
$10^{\lfloor\log_{10}a_i\rfloor}$ to $10^{\lceil\log_{10}b_i\rceil}$ with $n_d$
bins per decade (the outer edges are extended by $0.01$ decades). Within each
bin, $x$ and $y$ are averaged over the data points of the topography, giving
$\bar x_{i,k}$ and $\bar y_{i,k}$. Because each topography's bins start at a
full decade, its bin $j$ is identified with the global bin

$$
k = n_d\,\lfloor\log_{10}a_i\rfloor + j,
$$

so that bins of topographies with different bandwidths coincide. The
container result in bin $k$ is the arithmetic mean over the set $S_k$ of
topographies that have data in that bin,

$$
\bar x_k = \frac{1}{|S_k|}\sum_{i\in S_k}\bar x_{i,k},
\qquad
\bar y_k = \frac{1}{|S_k|}\sum_{i\in S_k}\bar y_{i,k}.
$$

Where the bandwidths of several topographies overlap, the result is their
mean; where only one topography has data, it is that topography's result;
bins between non-overlapping bandwidths, where no topography has data, are
masked. Averages are taken of $y$ itself, not of $\log y$, because the
arithmetic mean is the maximum-likelihood estimator of the mean of noisy PSD
or ACF data. Topographies without (reliable) data are skipped with a warning;
if no topography contributes, `NoReliableDataError` is raised.

The return values are `numpy.ma.MaskedArray`s $(\bar x_k, \bar y_k)$; masked
entries are bins without data. `progress_callback(i, n)` is called before
each topography is processed.

| Function | $x$ | $y$ |
|-|-|-|
| `power_spectrum` | $q$ (1/`unit`) | $C^\text{1D}(q)$ (`unit`³) |
| `autocorrelation` | $r$ (`unit`) | $A(r)$ (`unit`²) |
| `variable_bandwidth` | bandwidth (`unit`) | rms height (`unit`) |

```python
q, C = container.power_spectrum(unit="µm", nb_points_per_decade=5)
print(q.compressed()[[0, -1]], C.count())
```

```text
[6.28318531e-02 1.28805299e+03] 22
```

### Scale-dependent statistics

```python
container.scale_dependent_statistical_property(func, n, unit, nb_points_per_decade=10,
                                               distances=None, reliable=True,
                                               progress_callback=None, **kwargs)
```

This averages the statistics `func` of the $n$-th scale-dependent derivative
over all topographies (see [Scale-dependent statistics](scale-dependent.md)
for the derivatives and the definition of the scale). `func(dx, dy=None)`
receives the derivative along $x$ and, for maps, along $y$; it must accept a
single argument for line scans and return a scalar or an array of fixed size.
Further keyword arguments (e.g. `interpolation`) are passed to the
topographies' `scale_dependent_statistical_property`.

The distance $\lambda$ is the overall length of the finite-difference
stencil of lowest truncation order, not the grid spacing used by the
stencil; for a topography with grid spacing $\Delta x$ the stencil is scaled
by the factor $\lambda/(n\,\Delta x)$. Topography $i$ contributes at distance
$\lambda$ only if

$$
n\,\lambda_{\text{min},i} < \lambda < \lambda_{\text{max},i},
$$

where $\lambda_{\text{max},i}$ is the upper bound of its bandwidth and
$\lambda_{\text{min},i}$ is the lower bound of its bandwidth, or its reliability
cutoff $\lambda_c$ if that is larger and `reliable=True`. The factor $n$
accounts for an $n$-th derivative spanning at least $n$ grid intervals
([Sanner et al., Appl. Surf. Sci. Adv. 7, 100190 (2022)](https://doi.org/10.1016/j.apsadv.2021.100190)).

The result at each distance is the mean of `func` over the contributing
topographies (NaN results are ignored). With `distances=None`, each
topography is evaluated on the logarithmic grid
$\lambda = 10^{m/n_d}$ (integer $m$) restricted to its range, so that all
topographies are evaluated at identical distances; the function returns only
distances where at least one topography contributes. With explicit
`distances`, it returns exactly these distances and a masked array whose mask
is set where no topography contributes.

The rms slope as a function of scale, $h'_\text{rms}(\lambda)$, is

```python
d, s = container.scale_dependent_statistical_property(
    lambda dx, dy=None: np.mean(dx * dx), n=1, unit="µm",
    distances=[0.01, 0.1, 1, 10, 1000])
print(np.sqrt(s))
```

```text
[0.04337405864964129 0.02623046091507904 0.017328503038542144
 0.008499900193510768 --]
```

The largest distance exceeds the size of all scans and is masked.

### Integrals and moments of the PSD

The moments of the PSD give the rms height, slope and curvature (see
[PSD moments](spectral.md)). A container provides two routes.

**Summation of the raw spectra.** `integrate_psd_from_profile` sums the raw
(unbinned) 1D PSDs of all topographies, each summand divided by the number of
topographies whose bandwidth contains that wavevector:

```python
container.integrate_psd_from_profile(factor, unit, window=None, reliable=True)
```

$$
I = \sum_i \frac{1}{L_{x,i}} \sum_{q_x} \frac{f(q_x)\,C^\text{1D}_i(q_x)}{N(q_x)}
\approx \frac{1}{2\pi}\int_{-\infty}^{\infty} dq_x\, f(q_x)\, C^\text{1D}(q_x),
$$

where $f$ is `factor`, the inner sum runs over the discrete wavevectors
$q_x = 2\pi k/L_{x,i}$ of topography $i$ (for maps, $C^\text{1D}_i$ is
averaged over all profiles), and

$$
N(q_x) = \#\{\,i : 2\pi/L_{x,i} \le |q_x| \le \min(\pi/\Delta x_i,\, 2\pi/\lambda_{c,i})\,\}
$$

is the number of topographies whose bandwidth contains $q_x$ ($\lambda_{c,i}$
only with `reliable=True`; $q_x=0$ counts for all topographies). Wavevectors
covered by several topographies thus enter as the mean of these topographies,
and for a single topography the result equals
`topography.integrate_psd_from_profile(factor)`. `factor(qx)` receives
wavevectors in 1/`unit`; `window` is applied to each topography before the
Fourier transform. Only uniform topographies are supported. The raw spectra
are summed directly because binning the PSD before integration introduces
discretization errors.

```python
# rms slope along x
print(np.sqrt(container.integrate_psd_from_profile(lambda qx: qx**2, unit="µm")))
```

```text
0.0577387812545376
```

**Integration of the averaged PSD.** `c1d_moment` and `ciso_moment` integrate
the averaged PSD from `power_spectrum` with the trapezoidal rule; masked bins
are dropped:

```python
container.c1d_moment(order=1, cumulative=False, **kwargs)
container.ciso_moment(order=1, cumulative=False, **kwargs)
```

Keyword arguments (`unit`, `nb_points_per_decade`, `reliable`) are passed to
`power_spectrum`. With `cumulative=True`, the cumulative integral up to each
wavevector is returned. The moments of order $\alpha$ are

$$
m_\alpha^\text{1D} = \frac{1}{\pi}\int_0^\infty dq\, q^\alpha\, C^\text{1D}(q),
\qquad
m_\alpha^\text{iso} = \frac{1}{2\pi}\int_0^\infty dq\, q^{\alpha+1}\, C^\text{iso}(q).
$$

$m_0$ is $h_\text{rms}^2$, $m_2$ is $h'^2_\text{rms}$ (along $x$ for
$m^\text{1D}$, the full 2D gradient for $m^\text{iso}$) and $m_4$ is related to
$h''^2_\text{rms}$.

```python
print(np.sqrt(container.ciso_moment(order=2, unit="µm")))
```

```text
0.07338183285823531
```

### Isotropic PSD from the 1D PSD

`ciso_moment` needs the isotropic PSD $C^\text{iso}(q)$, but a container only
provides the averaged 1D PSD. It uses

$$
C^\text{iso}(q) = \frac{\pi}{q}\, C^\text{1D}(q).
$$

For an isotropic surface, the 1D PSD of profiles along $x$ is the projection
of the 2D PSD,

$$
C^\text{1D}(q_x) = \frac{1}{2\pi}\int_{-\infty}^{\infty} dq_y\,
C^\text{iso}\!\left(\sqrt{q_x^2+q_y^2}\right).
$$

For a self-affine PSD $C^\text{iso}(q) = C_0\,q^{-2-2H}$ this gives

$$
\frac{\pi}{q}\,C^\text{1D}(q) = \frac{\sqrt{\pi}\,\Gamma(H+\tfrac12)}{2\,\Gamma(H+1)}\, C^\text{iso}(q),
$$

so the relation is exact for $H=\tfrac12$ and otherwise underestimates
$C^\text{iso}$ by a constant factor between $\pi/4 \approx 0.79$ ($H=1$) and
$\pi/2 \approx 1.57$ ($H\to0$, i.e. an overestimate). Its assumptions are
therefore an isotropic surface and a locally power-law PSD with Hurst
exponent close to $\tfrac12$; the error is a constant factor and does not
change the scaling of $C^\text{iso}$ with $q$. For anisotropic surfaces,
$C^\text{1D}$ depends on the scan direction and the relation does not hold.
<!-- TODO: rationale for using C^iso = (pi/q) C^1D rather than a Hurst-exponent-dependent prefactor? -->

## Custom analyses

Analysis functions are registered on the container class and become methods
of all containers:

```python
from SurfaceTopography.Container.SurfaceContainer import SurfaceContainer

def nb_line_scans(self):
    return sum(t.dim == 1 for t in self)

SurfaceContainer.register_function("nb_line_scans", nb_line_scans)
print(container.nb_line_scans())
```

`container.apply(name, *args, **kwargs)` calls a registered function by name.
