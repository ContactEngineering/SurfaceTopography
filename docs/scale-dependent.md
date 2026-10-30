# Scale-dependent parameters

```python
from SurfaceTopography.Uniform.Derivative import (
    DiscreteDerivative, FourierDerivative, first_central_1d)
```

Slopes and curvatures of rough surfaces depend on the length over which they
are measured. A scalar parameter such as `Rdq` (see
[scalar statistics](statistics.md)) is the value at the scale of the grid
spacing. Scale-dependent roughness parameters (SDRPs) evaluate derivatives
with finite-difference stencils stretched to a length $\lambda$. The result is
a function of $\lambda$ that characterizes the surface over the whole
measured bandwidth. They can also be compared across measurements with
different resolutions [^sanner2022].

All functions on this page are methods of topographies and line scans,
including results of [pipelines](pipelines.md).

## Finite-difference derivatives

`derivative(n, ...)` computes the $n$-th derivative of uniform data with a
stencil of coefficients $c_k$ at integer offsets $k$. The stencil is stretched
by a scale factor $s$:

$$
h^{(n)}(x_i) \approx \frac{1}{(s\,\Delta x)^n}\sum_k c_k\, h(x_i + k\,s\,\Delta x) .
$$

```python
derivative(n, scale_factor=None, distance=None, operator=None, periodic=None,
           mask_function=None, interpolation="linear", progress_callback=None)
```

For a line scan, the result is an array. For a map, it is a tuple
$(\partial_x^n h, \partial_y^n h)$. If `scale_factor` or `distance` is a list,
there is one entry per scale.

### Default stencils

| $n$ | Offsets $k$ | Coefficients $c_k$ | Truncation error at $x_i$ | At the stencil center |
|-|-|-|-|-|
| 1 | $0, 1$ | $-1, 1$ (upwind) | $O(\Delta x)$ | $O(\Delta x^2)$ at $x_i + \Delta x/2$ |
| 2 | $-1, 0, 1$ | $1, -2, 1$ (central) | $O(\Delta x^2)$ | same |
| 3 | $-1, 0, 1, 2$ | $-1, 3, -3, 1$ (upwind) | $O(\Delta x)$ | $O(\Delta x^2)$ at $x_i + \Delta x/2$ |

Maps use the same stencils along $x$ and along $y$. The module also defines
second-order central stencils of the first derivative
(`first_central_1d`, `first_central_2d`) and of the third derivative
(`third_central_1d`). Any `DiscreteDerivative(lbounds, stencil)` or tuple of
them can be passed as `operator`. Halving $\Delta x$ for $h=\sin x$ reduces the
error of the upwind first derivative at $x_i$ by 2. At $x_i+\Delta x/2$, it
reduces the error by 4, the same as for the central stencils.

The minimal stencils measure the derivative at one well-defined scale.
<!-- TODO: rationale for upwind (rather than central) default stencils for n = 1 and n = 3? -->

In Fourier space, a stencil multiplies $\tilde h(q)$ by
$\sum_k c_k e^{i k q s\Delta x}/(s\Delta x)^n$. For the second derivative,
this factor is $-4\sin^2(qs\Delta x/2)/(s\Delta x)^2$, which approaches the
exact $-q^2$ for $qs\Delta x \ll 1$. It vanishes at $q = 2\pi/(s\Delta x)$.
A stretched stencil therefore acts as a filter: it suppresses wavelengths
shorter than the stencil.

### Scale factor and distance

The scale is given by either argument:

- `scale_factor` $s$: the stencil spacing in units of the grid spacing. It can
  be a number, a list of numbers or a list of $(s_x, s_y)$ tuples. The default
  is $s = 1$.
- `distance` $\lambda$: the total length of the lowest-order stencil. Its
  spacing is $\lambda/n$, so $s = \lambda/(n\,\Delta x)$ (on maps, separately
  $\lambda/(n\Delta x)$ and $\lambda/(n\Delta y)$). For $n=1$, $\lambda$ is the
  distance of the two points. For $n=2$, $\lambda$ is the distance between the
  outer points of the three-point stencil.

Non-integer $s$ requires heights between grid points, so the data are
interpolated (argument `interpolation`):

| `interpolation` | Heights between grid points | Cost per scale | Notes |
|-|-|-|-|
| `"linear"` (default) | linear (bilinear on maps) interpolation | $O(N)$ | local; undefined data yield masked derivatives |
| `"fourier"` | Fourier interpolation: the stencil's Fourier multiplier is applied to $\tilde h$ | $O(N\log N)$ | exact stencil for integer $s$ on periodic data; large errors for nonperiodic data; raises `UndefinedDataError` for undefined data; required for `mask_function` and `FourierDerivative` |
| `"disable"` | none, integer $s$ only | $O(N)$ | raises `ValueError` if interpolation would be needed |

`mask_function` receives the frequencies of the Fourier grid in units of
1/length, an array of shape `(dim, ...)` equal to $\mathbf q/2\pi$. It returns
a factor that multiplies $\tilde h$ before the derivative is taken. This
implements the `short_wavelength_cutoff` of the
[scalar statistics](statistics.md#short-wavelength-cutoff).

### Periodic and nonperiodic data

The stencil is always applied as if the data were periodic. For nonperiodic
data (or `periodic=False`), the values whose stencil reaches across the
boundary are invalid and are removed. The function trims
$\lceil s\max(-k_\text{min}, 0)\rceil$ points at the start and
$\lfloor s\max(k_\text{max}, 0)\rfloor$ at the end of every axis. On maps, the
default stencils are stored as $2\times2$ ($n=1$) and $3\times3$ ($n=2$)
arrays, so both derivative components are trimmed along both axes and have
the same shape.

```python
import numpy as np
from SurfaceTopography import Topography, UniformLineScan

t = UniformLineScan(np.random.rand(100), 1.0, periodic=False)
len(t.derivative(1))                          # 99
len(t.derivative(2))                          # 98
len(t.derivative(2, distance=0.1))            # s = 5: 90
[len(d) for d in t.derivative(1, scale_factor=[1, 2, 4])]   # [99, 98, 96]

m = Topography(np.random.rand(32, 16), (1.0, 0.5))
dx, dy = m.derivative(2)                      # both of shape (30, 14)
```

### Fourier derivative

`fourier_derivative()` is the spectral first derivative of the Fourier
interpolation:

$$
\left[\frac{\partial h}{\partial x}\right]_{x_i,y_j}
= \frac{1}{L_xL_y}\sum_{kl} i q_x^k\,\tilde h_{kl}\, e^{i(q_x^k x_i + q_y^l y_j)} .
$$

For an even number of points, the Nyquist component is taken as a cosine,
so its derivative vanishes on the grid points (see
[S. G. Johnson, *Notes on FFT-based differentiation*](https://math.mit.edu/~stevenj/fft-deriv.pdf)).
The result is spectrally accurate for smooth periodic data. For $\sin x$ on 32
points, the error is $10^{-15}$, compared with $10^{-1}$ for the upwind stencil
at $x_i$. It costs $O(N\log N)$ and requires a periodic topography; on
nonperiodic data it raises `ValueError`. It is not tied to a scale, so
`scale_factor` and `distance` do not change the result. It returns one
array per direction, also for line scans (a list with one entry).

```python
x = np.arange(64) * 2 * np.pi / 64
p = UniformLineScan(np.sin(x), 2 * np.pi, periodic=True)
p.fourier_derivative()[0]                   # cos(x) to machine precision
p.derivative(1, operator=first_central_1d)  # second-order central differences
```

### Nonuniform line scans

Without `distance`, nonuniform line scans use the two-point slope
$(h_{i+1}-h_i)/\Delta x_i$, which has $N-1$ values. For $n=2$, they use the
three-point formula for the spacings $\Delta x_+ = x_{i+1}-x_i$ and
$\Delta x_- = x_i - x_{i-1}$:

$$
h''(x_i) \approx 2\,\frac{\Delta x_-\left[h_{i+1}-h_i\right] + \Delta x_+\left[h_{i-1}-h_i\right]}
{\Delta x_+\Delta x_-(\Delta x_+ + \Delta x_-)} ,
$$

which has $N-2$ values. It follows from the Taylor expansions of $h_{i\pm1}$:
the weights $\Delta x_\mp$ eliminate the first derivative. The truncation
error is $O(\Delta x_+ - \Delta x_-)$ for unequal spacings and $O(\Delta x^2)$
for equal ones.

With a `distance` $\lambda$, the line scan is interpolated linearly onto a
uniform grid. The grid spacing is $\lambda/(ns)$ with
$s = \lceil\lambda/\lambda_\text{min}\rceil$, where $\lambda_\text{min}$ is the
lower bound of `bandwidth()`. The uniform stencil with integer scale factor $s$
is then applied, so its total length is exactly $\lambda$. Scale factors and
Fourier interpolation are not supported for nonuniform data. Line scans with
overhangs raise `ReentrantDataError`.

## Scale-dependent slope and curvature

The rms of a stretched two-point difference can be expressed exactly through
the height-difference autocorrelation function
$A(r) = \tfrac12\langle[h(x+r)-h(x)]^2\rangle$ (see
[spectral analysis](spectral.md#height-difference-autocorrelation)). The rms slope at
distance $\lambda$ is

$$
h'_\text{rms}(\lambda) = \frac{\left\langle[h(x+\lambda)-h(x)]^2\right\rangle^{1/2}}{\lambda}
= \frac{\left[2A(\lambda)\right]^{1/2}}{\lambda} .
$$

Expanding the square of the three-point difference with spacing $\lambda$
gives the rms curvature:

$$
h''_\text{rms}(\lambda) = \frac{\left\langle[h(x+\lambda)-2h(x)+h(x-\lambda)]^2\right\rangle^{1/2}}{\lambda^2}
= \frac{\left[8A(\lambda) - 2A(2\lambda)\right]^{1/2}}{\lambda^2} .
$$

| Function | Autocorrelation | Returned distance |
|-|-|-|
| `scale_dependent_slope_from_profile(**kwargs)` | `autocorrelation_from_profile` (along $x$; also nonuniform) | $\lambda$ |
| `scale_dependent_slope_from_area(**kwargs)` | `autocorrelation_from_area` (radial average) | $\lambda$ |
| `scale_dependent_curvature_from_profile(**kwargs)` | `autocorrelation_from_profile` | total stencil length $2\lambda$ |
| `scale_dependent_curvature_from_area(**kwargs)` | `autocorrelation_from_area` | total stencil length $2\lambda$ |

Keyword arguments are passed on to the autocorrelation function. The
curvature functions are subject to the following restrictions:

- The value $A(2\lambda)$ is obtained by linear interpolation of the sampled
  ACF. The curvature is therefore available only up to half the largest
  distance of the ACF.
- The result is truncated at the first distance where
  $8A(\lambda)-2A(2\lambda) < 0$.
- The curvature functions return the total stencil length $2\lambda$, which is
  the `distance` convention of `derivative`.

With `resampling_method=None` (the ACF on all grid distances), the results
equal the statistics of the stretched stencils. The profile curvature at
distance $r$ equals the rms of `derivative(2, distance=r)`. At
$\lambda = \Delta x$, the slope and curvature are `Rdq()` and `Rddq()`.
Computing all scales through the ACF costs $O(N\log N)$ (one FFT), whereas
explicit stencils cost $O(N)$ per scale.

```python
from SurfaceTopography.Generation import fourier_synthesis

np.random.seed(0)
t = fourier_synthesis((4096,), (10.0,), hurst=0.8, rms_height=0.1,
                      short_cutoff=0.05, unit="um", periodic=True)
r, slope = t.scale_dependent_slope_from_profile(resampling_method=None)
r, curv = t.scale_dependent_curvature_from_profile(resampling_method=None)
np.isclose(slope[0], t.Rdq()), np.isclose(curv[0], t.Rddq())  # (True, True)
```

## Statistical properties at arbitrary scales

`scale_dependent_statistical_property` generalizes the rms to any statistic
of the derivative distribution [^sanner2022], for example the kurtosis of the
slopes or the fraction of positive curvatures.

```python
scale_dependent_statistical_property(func, n=1, scale_factor=None, distance=None,
                                     reliable=True, interpolation="linear",
                                     progress_callback=None, collocation="log",
                                     nb_points=None, nb_points_per_decade=10,
                                     threshold=4)
```

- The function computes the derivatives of order `n` with
  [`derivative`](#finite-difference-derivatives) at each distance and
  returns `(distance, func(...))`.
- `func(dx, dy=None)` receives the derivative along $x$, and on maps also
  along $y$. It must return a scalar or an array of fixed size.
- If neither `scale_factor` nor `distance` is given, the distances are
  generated between $n\lambda_\text{min}$ and $\lambda_\text{max}$ from
  `bandwidth()`. The spacing is log, quadratic or linear (`collocation`), with
  `nb_points` points or `nb_points_per_decade` points per decade.
- Derivative arrays with fewer than `threshold` values are replaced by NaN
  before `func` is called.
- If `reliable` is true and the topography has a
  [reliability cutoff](#reliability-cutoff) $\lambda_c$, only distances
  $\lambda > n\lambda_c/2$ are returned. If none remains, the function raises
  `NoReliableDataError`. $\lambda_c$ is the length of the second-derivative
  stencil, so $\lambda_c/2$ is its spacing. The criterion compares stencil
  spacings for any $n$. This filter applies only to multiple distances.

The cost is $O(N)$ per distance with linear interpolation. For $N_\lambda$
distances, it is $O(NN_\lambda)$.

```python
import scipy.stats

d, rms_slope = t.scale_dependent_statistical_property(
    lambda dx, dy=None: np.sqrt(np.mean(dx**2)), n=1)
d, kurtosis = t.scale_dependent_statistical_property(
    lambda dx, dy=None: scipy.stats.kurtosis(dx), n=1, distance=[0.01, 0.1, 1.0])
```

## Variance of the half derivative

The variance of the half derivative, $\langle|\nabla^{1/2}h|^2\rangle$, sets
the elastic energy needed to flatten a surface (see
[spectral analysis](spectral.md#half-derivative-and-elastic-energy) for the spectral definition).
It can also be estimated from the ACF:

$$
\int_0^\infty dr\,\frac{A(r)}{r^2} = \frac12\int_0^\infty dr\, \left[h'_\text{rms}(r)\right]^2 .
$$

| Function | Result |
|-|-|
| `variance_half_derivative_via_autocorrelation_from_area(**kwargs)` | uses the radially averaged ACF. Since $\int_0^\infty dr\,[1-J_0(qr)]/r^2 = q$, this equals $\int d^2q/(2\pi)^2\,q\,C^\text{2D}(\mathbf q) = \langle\lvert\nabla^{1/2}h\rvert^2\rangle$ |
| `variance_half_derivative_via_autocorrelation_from_profile(**kwargs)` | uses the profile ACF along $x$. Since $\int_0^\infty dr\,[1-\cos(q_xr)]/r^2 = \pi\lvert q_x\rvert/2$, this equals $\langle\lvert\nabla^{1/2}h\rvert^2\rangle$ for isotropic surfaces. For a line scan, the result is $\frac12\int_0^\infty dq\, q\, C^\text{1D}(q)$, the estimate for an isotropic surface |
| `variance_half_derivative_via_scale_dependent_slope(r, slope, rms_height=None)` | the integral for a sampled slope |
| `container.variance_half_derivative_from_autocorrelation(scale_dependent=False, **kwargs)` | the same for a [container](containers.md), from `container.autocorrelation` (also available as `container.variance_half_derivative_via_autocorrelation_from_profile`). With `scale_dependent=True`, it returns the cumulative integral up to each distance $r$, without the contribution beyond the largest distance |

The integrals are discretized as follows:

- Below the smallest sampled distance $r_0$, the slope is assumed constant,
  which contributes $r_0[h'_\text{rms}(r_0)]^2/2 = A(r_0)/r_0$.
- The integral between $r_0$ and the largest sampled distance $r_\text{max}$
  is evaluated with the trapezoidal rule.
- Beyond $r_\text{max}$, the heights are assumed to be uncorrelated,
  $A(r) = \langle h^2\rangle$, which contributes $\langle
  h^2\rangle/r_\text{max}$ (with `rms_height` for
  `variance_half_derivative_via_scale_dependent_slope`; see
  [spectral analysis](spectral.md#half-derivative-and-elastic-energy) for the
  choice of $\langle h^2\rangle$).
- Empty bins (masked) and NaN entries are dropped.

## Reliability cutoff

Instrument artifacts make data unreliable below some length scale $\lambda_c$.
`short_reliability_cutoff(other_cutoff=None)` determines $\lambda_c$ from the
instrument metadata, `info["instrument"]["parameters"]`:

| Parameter | $\lambda_c$ |
|-|-|
| `tip_radius` | `scanning_probe_reliability_cutoff(R)` with the tip radius $R$ |
| `resolution` | the resolution |
| neither | `None` |

The values are dictionaries `{"value": ..., "unit": ...}` and are converted to
the unit of the topography. If `other_cutoff` is given, the larger of the two
cutoffs is returned.

### Tip-radius artifacts

A rigid spherical tip of radius $R$ cannot enter features that are sharper
than the tip. The measured topography therefore cannot have curvature below
$-1/R$ at the scales where the tip determines the image.
`scanning_probe_reliability_cutoff(R, safety_factor=1/2)` finds the stencil
length $\lambda_c$ at which the most negative scale-dependent second
derivative reaches this limit [^sanner2022]:

$$
\min_i h''(x_i;\lambda_c) = -\frac{f}{R},
$$

where $h''(x_i;\lambda)$ is `derivative(n=2, distance=λ)` (along $x$ on maps)
and $f$ is the `safety_factor`. The value $f = 1/2$ was estimated from
simulated scans in [^sanner2022]. The root is found with Brent's method in
$[2\lambda_\text{min}, \lambda_\text{max}/2]$, the bandwidth reduced by a
factor 2 at each end. The search works on distances normalized by the
geometric mean of the bandwidth, with tolerances `xtol` and `rtol`. Each
evaluation costs $O(N)$.

| Outcome | Return value |
|-|-|
| criterion met already at $2\lambda_\text{min}$ (no tip artifacts) | `None` |
| criterion violated even at $\lambda_\text{max}/2$ (no reliable data) | $\lambda_\text{max}$ |
| otherwise | $\lambda_c$ |

Scale-dependent analyses (`scale_dependent_statistical_property` and the
[spectral functions](spectral.md#power-spectral-density) with `reliable=True`)
discard data below this cutoff.

### Rigid-sphere scan simulation

`scan_with_rigid_sphere(R)` emulates the measurement of a line scan with a
rigid spherical tip. At each position $x_i$, the scanned height is the lowest
apex height at which the sphere touches but does not penetrate any data
point:

$$
\tilde h_i = \max_{j:\,|x_j-x_i|\le R}\left[h_j + \sqrt{R^2-(x_j-x_i)^2} - R\right] .
$$

- Only data points are tested for contact, not the interpolation between
  them.
- Nonperiodic line scans are treated as ending at their first and last
  points.
- A periodic line scan is scanned as three periodic copies, of which the
  central one is returned.
- The cost is $O(Nw)$, where $w$ is the number of points within a window of
  width $2R$. The windows are evaluated in blocks of bounded size, which keeps
  the memory use bounded for any $N$ and $R$.
- Only line scans (uniform or nonuniform) are supported.

The result records the tip radius in its instrument metadata, so
`short_reliability_cutoff()` returns the tip-radius cutoff for the scanned
data:

```python
scanned = t.scan_with_rigid_sphere(0.5)        # tip radius in units of t
scanned.info["instrument"]["parameters"]       # {'tip_radius': {'value': 0.5, 'unit': 'um'}}
lc = scanned.short_reliability_cutoff()        # about 0.8 um
d, s = scanned.scale_dependent_statistical_property(
    lambda dx, dy=None: np.sqrt(np.mean(dx**2)), n=2)
d.min() > lc                                   # True: unreliable scales removed
```

`SurfaceTopography.ScanningProbe.scan_with_rigid_sphere(topography, radius)`
is the underlying function. It returns the array of scanned heights.

[^sanner2022]: A. Sanner, W. G. Nöhring, L. A. Thimons, T. D. B. Jacobs,
    L. Pastewka, *Scale-dependent roughness parameters for topography
    analysis*, Appl. Surf. Sci. Adv. 7, 100190 (2022),
    <https://doi.org/10.1016/j.apsadv.2021.100190>.
