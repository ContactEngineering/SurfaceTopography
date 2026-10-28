# Scalar statistics

```python
from SurfaceTopography import Topography, UniformLineScan, NonuniformLineScan
```

Scalar roughness parameters are analysis functions of topographies and line
scans: they are called as methods, e.g. `t.Sq()`, and work on raw data as well
as on the result of any [pipeline](pipelines.md) step (`t.detrend().Sq()`).
Each parameter summarizes the topography at a single scale, the grid spacing.
For parameters that depend on a chosen length, see
[scale-dependent parameters](scale-dependent.md). For the moments of
the power spectrum, see [spectral analysis](spectral.md#power-spectral-density).

| Function | Quantity | Uniform 1D | Uniform 2D | Nonuniform |
|-|-|-|-|-|
| `mean()`, `min()`, `max()` | $\langle h\rangle$, extrema | ✓ | ✓ | ✓ |
| `median()` | median height | ✓ | ✓ | ✓ |
| `Sq()` | rms height of the whole map | | ✓ | |
| `Rq()` | rms height of individual profiles | ✓ | ✓ | ✓ |
| `Rdq()` | rms slope along $x$ | ✓ | ✓ | ✓ |
| `rms_gradient()` | rms of $\lvert\nabla h\rvert$ | | ✓ | |
| `Rddq()` | rms second derivative along $x$ | ✓ | ✓ | ✓ |
| `rms_laplacian()` | rms of $\nabla^2 h$ | | ✓ | |
| `rms_curvature_from_area()` | rms mean curvature $\nabla^2 h/2$ | | ✓ | |
| `mad_height()` | robust rms height from deviations from the median | ✓ | ✓ | ✓ |
| `mad_polyfit(deg)` | polynomial fit minimizing `mad_height` | ✓ | ✓ | ✓ |
| `bearing_area(h)` | fractional area above height $h$ | ✓ | ✓ | ✓ |
| `moment(alpha)` | $\langle h^\alpha\rangle$ | | | ✓ |

The names `rms_height_from_area`, `rms_height_from_profile`,
`rms_slope_from_profile` and `rms_curvature_from_profile` are aliases of
`Sq`, `Rq`, `Rdq` and `Rddq`. They are deprecated for uniform data and emit a
`DeprecationWarning`. For nonuniform line scans they emit no warning, so that
generic code gives the same warnings for both data types.

## Conventions

- **Uniform data** are averaged over grid points. The derivatives use the
  finite-difference stencils described in
  [scale-dependent parameters](scale-dependent.md#finite-difference-derivatives)
  at the scale of a single grid spacing.
- **Undefined (masked) data** do not contribute. Averages are normalized by the
  number of defined points or elements. See
  [undefined data](topographies.md#undefined-data).
- **Nonuniform line scans** are interpolated linearly between data points.
  An average is the integral of this piecewise-linear function divided by
  $L = x_{N-1} - x_0$. The integrals are evaluated exactly, so no additional
  discretization error arises.
- **Periodic data**: finite-difference stencils wrap around the boundary.
  For nonperiodic data, values whose stencil sticks out of the data are
  dropped.

## Mean, median and extrema

For uniform data, `mean()` is the arithmetic mean of the heights, and `min()`
and `max()` are the extreme data values. For domain-decomposed
topographies, all three are reduced over the MPI processes. For nonuniform
line scans, the mean integrates the piecewise-linear interpolation:

$$
\langle h\rangle = \frac{1}{L}\sum_{i=0}^{N-2} \frac{h_i + h_{i+1}}{2}\,\Delta x_i .
$$

Nonuniform line scans also provide general moments, `moment(alpha)`. Each
segment contributes

$$
\frac{1}{L}\int_{x_i}^{x_{i+1}} dx\, h^\alpha(x)
= \frac{\Delta x_i}{(\alpha+1)L}\sum_{k=0}^{\alpha} h_i^k\, h_{i+1}^{\alpha-k} .
$$

This is the polynomial form of $(h_{i+1}^{\alpha+1}-h_i^{\alpha+1})/(h_{i+1}-h_i)$.
It stays exact for equal neighboring heights, for example in quantized
instrument data, where the quotient is $0/0$, and it avoids cancellation when
the heights are nearly equal.

`median()` is the height at which the fractional
[bearing area](#bearing-area-curve) is $1/2$. It is the median of the
piecewise-linear interpolation, not of the data points.

## RMS height

`Sq()` is the rms height of a topography map about its mean:

$$
S_q = \left[\frac{1}{N}\sum_{ij}\left(h_{ij} - \langle h\rangle\right)^2\right]^{1/2},
$$

where $N$ is the number of defined data points.

`Rq()` is the rms height of the individual profiles along $x$. Each profile
$j$ is referenced to its own mean, $\bar h_j = N_x^{-1}\sum_i h_{ij}$:

$$
R_q = \left[\frac{1}{N}\sum_{ij}\left(h_{ij} - \bar h_j\right)^2\right]^{1/2} .
$$

The difference between profile means does not enter $R_q$, so $R_q \le S_q$.
$R_q$ is therefore insensitive to height offsets between scan lines. For a line
scan, `Rq()` is the rms height about the mean.

For a nonuniform line scan, `Rq()` integrates the piecewise-linear
interpolation. With heights measured from the mean, segment $i$ contributes

$$
\int_0^{\Delta x_i} dx \left(h_i + \frac{\Delta h_i}{\Delta x_i}x\right)^2
= \frac{\Delta x_i}{3}\left(h_i^2 + h_i h_{i+1} + h_{i+1}^2\right),
\qquad \Delta h_i = h_{i+1} - h_i ,
$$

and $R_q^2$ is the sum of these contributions divided by $L$.

## RMS slope and gradient

`Rdq()` is the rms slope along $x$. It uses the two-point difference:

$$
R_{\Delta q} = \left\langle\left(\frac{h_{i+1,j}-h_{ij}}{\Delta x}\right)^2\right\rangle^{1/2} .
$$

For a nonuniform line scan, the slope of the piecewise-linear interpolation is
constant on each segment, and

$$
R_{\Delta q} = \left[\frac{1}{L}\sum_i \frac{\Delta h_i^2}{\Delta x_i}\right]^{1/2} .
$$

This raises `ReentrantDataError` if the positions are not strictly
increasing.

`rms_gradient()` uses the gradient of a topography map:

$$
h'_\text{rms} = \left\langle \left(\frac{\partial h}{\partial x}\right)^2
+ \left(\frac{\partial h}{\partial y}\right)^2\right\rangle^{1/2},
$$

with two-point differences in both directions. For an isotropic surface,
$\langle(\partial h/\partial x)^2\rangle = \langle(\partial h/\partial y)^2\rangle$.
The gradient is then larger than the slope along one direction by a factor
$\sqrt 2$:

$$
h'_\text{rms} = \sqrt{2}\,R_{\Delta q} .
$$

The two-point difference is second-order accurate at the midpoint
$x_i + \Delta x/2$ of the segment. On a grid, these parameters depend on
$\Delta x$: they are the
[scale-dependent slope](scale-dependent.md#scale-dependent-slope-and-curvature)
evaluated at the smallest scale.

### Short-wavelength cutoff

`Rdq`, `Rddq`, `rms_gradient` and `rms_laplacian` accept a
`short_wavelength_cutoff` $\lambda_s$. If it is given, the topography is
[windowed](pipelines.md) first (arguments `window` and `direction`). The
derivative is then computed with Fourier interpolation, and Fourier components
with wavelength below $\lambda_s$ are set to zero. `Rdq` and `Rddq` filter on
$|q_x| \ge 2\pi/\lambda_s$. `rms_gradient` and `rms_laplacian` filter on
$|\mathbf q| \ge 2\pi/\lambda_s$.

```python
import numpy as np
from SurfaceTopography.Generation import fourier_synthesis

t = fourier_synthesis((256, 256), (1.0, 1.0), hurst=0.8, rms_height=1e-3,
                      periodic=True)
t.Sq(), t.Rq()
t.rms_gradient() / t.Rdq()             # close to sqrt(2) for isotropic data
t.Rdq(short_wavelength_cutoff=0.05)    # slope of wavelengths above 0.05
```

## RMS curvature and Laplacian

`Rddq()` is the rms second derivative along $x$. It uses the second-order
central difference:

$$
R_{\Delta\Delta q} = \left\langle\left(\frac{h_{i+1,j}-2h_{ij}+h_{i-1,j}}{\Delta x^2}\right)^2\right\rangle^{1/2} .
$$

`rms_laplacian()` is the rms of the five-point Laplacian:

$$
\nabla^2 h \approx \frac{h_{i+1,j}-2h_{ij}+h_{i-1,j}}{\Delta x^2}
+ \frac{h_{i,j+1}-2h_{ij}+h_{i,j-1}}{\Delta y^2} .
$$

`rms_curvature_from_area()` returns half of it. In the small-slope
approximation, $\nabla^2 h/2$ is the mean curvature
$\kappa = (\kappa_1+\kappa_2)/2$. The stencils have a truncation error of
$O(\Delta x^2)$, so a paraboloid gives its curvature to machine precision.

For a nonuniform line scan, `Rddq()` evaluates the three-point second derivative
for unequal spacings at the interior points $x_1, \dots, x_{N-2}$, using the
stencil given in
[scale-dependent parameters](scale-dependent.md#nonuniform-line-scans). It
integrates the square of this derivative with the trapezoidal rule:

$$
R_{\Delta\Delta q} = \left[\frac{1}{x_{N-2}-x_1}\int_{x_1}^{x_{N-2}} dx
\left(\frac{\partial^2 h}{\partial x^2}\right)^2\right]^{1/2} .
$$

```python
t.Rddq(), t.rms_laplacian(), t.rms_curvature_from_area()
```

## Bearing area curve

The fractional bearing area (Abbott-Firestone curve) $\Phi(h)$ is the
fraction of the projected area on which the surface lies above height $h$. As
a function of $h$, it is the complementary cumulative distribution function
of the heights. `bearing_area` computes it exactly for the piecewise-linear
interpolation of the data:

- **Line scans** consist of linear segments. A segment with end heights
  $h_a \le h_b$ and length $\Delta x$ contributes
  $\Delta x\,\min\!\left(1, \max\!\left(0, \frac{h_b - h}{h_b - h_a}\right)\right)$.
- **Topography maps** split each grid cell into two triangles, with corners
  $(i,j),(i+1,j),(i,j+1)$ and $(i+1,j+1),(i+1,j),(i,j+1)$. Let the corner
  heights of a triangle be sorted, $h_1 \le h_2 \le h_3$. Its fractional area
  above $h$ is

$$
\phi(h) = \begin{cases}
1 & h \le h_1 \\
1 - \dfrac{(h-h_1)^2}{(h_2-h_1)(h_3-h_1)} & h_1 < h \le h_2 \\
\dfrac{(h_3-h)^2}{(h_3-h_2)(h_3-h_1)} & h_2 < h \le h_3 \\
0 & h > h_3 .
\end{cases}
$$

$\Phi(h)$ is the mean of the triangle fractions, or the length-weighted mean of
the segment fractions. Elements that touch undefined data are skipped.
Periodic data include the elements that wrap around the boundary.

Called with heights, `bearing_area` returns the fractional areas. Called
without arguments, it returns an object that caches the preparation, which
sorts the element minima and maxima in $O(N\log N)$. This object provides:

| Member | Returns | Cost |
|-|-|-|
| `b(h)` | exact $\Phi(h)$ (scalar or array) | $O(N)$ per height |
| `b.bounds(h)` | lower and upper bound on $\Phi(h)$: fraction of elements entirely above $h$ and fraction with any part above $h$ | $O(\log N)$ |
| `b.min`, `b.max` | smallest and largest height | |

```python
b = t.bearing_area()
b(np.array([0.0, 1e-3]))   # fractional area above 0 and above 1e-3
t.bearing_area(0.0)        # same, without caching
```

## Robust statistics

`median()` and `mad_height()` invert the bearing area curve by bisection on
the height range $[\min h, \max h]$. The bisection stops at a relative
tolerance of $10^{-6}$ of this range. Each step evaluates the $O(\log N)$
bounds first. The exact $O(N)$ bearing area is needed only when the target
value lies between the bounds.

`mad_height(percentile=p)` returns the deviation $\delta$ from the median
height $h_\text{med}$ such that a fraction $2p$ of the area deviates by more
than $\delta$:

$$
\Phi(h_\text{med}+\delta) + 1 - \Phi(h_\text{med}-\delta) = 2p .
$$

The default is $p = \Phi_\mathcal{N}(-1) \approx 0.159$, where $\Phi_\mathcal{N}$ is
the standard normal cumulative distribution function. With this default,
$\delta$ equals the standard deviation for Gaussian heights, so `mad_height()`
can be compared directly with `Sq()` and `Rq()`. It is insensitive to
outliers, which affect only the tails of $\Phi$. With $p = 1/4$, $\delta$ is
the classical median absolute deviation.

`mad_polyfit(deg)` returns the coefficients of the polynomial of degree `deg`
that minimizes `mad_height()` of the residual. The coefficients use the
convention of `polyfit` and `detrend(coeffs=...)`. The non-constant
coefficients are optimized with the Nelder-Mead method, starting from the
least-squares `polyfit`. The constant term is the median of the residual. This
makes the fit robust against outliers and spikes, which bias a least-squares
fit.

```python
t.median(), t.mad_height()

x = np.linspace(0, 1, 2000)
line = UniformLineScan(0.1 * np.random.normal(size=2000) + 2 * x + 1, 1.0)
line.mad_polyfit(1)        # approximately [1, 2]
```

## Geometry analysis

`SurfaceTopography.Uniform.GeometryAnalysis` analyzes boolean maps on the
grid, for example the region above a threshold height or a contact area. These
are module-level functions, not methods of topographies:

| Function | Result |
|-|-|
| `assign_patch_numbers_area(mask, periodic, stencil=None)` | number of connected patches and a map of patch ids (0 outside, 1, 2, … inside). Patches are found by flood fill. The default neighborhood is the 8-neighbor stencil `nnn_stencil`. A custom stencil is a flat integer array of offset pairs, e.g. `np.array(nn_stencil).ravel()` for 4 neighbors |
| `patch_areas(patch_ids)` | number of pixels of each patch; entry `i` belongs to patch `i + 1` |
| `coordination(mask, periodic, stencil=nn_stencil)` | number of neighbors inside `mask`, for each pixel |
| `inner_perimeter_area(mask, periodic, stencil=nn_stencil)` | pixels inside `mask` with at least one neighbor outside |
| `outer_perimeter_area(mask, periodic, stencil=nn_stencil)` | pixels outside `mask` with at least one neighbor inside |
| `distance_map(mask)` | Euclidean distance, in grid points, to the closest pixel inside `mask`. Distances are always periodic |
| `assign_segment_numbers_area(mask)` | connected segments along the second ($y$) axis, labeled row by row. Always periodic |
| `assign_patch_numbers_profile(mask, periodic)` | number of patches and patch ids of a 1D mask |
| `outer_perimeter_profile(mask, periodic)` | points of a 1D mask outside `mask` next to a point inside |

Patch labeling and the perimeter maps cost $O(N)$ for $N$ pixels.

```python
from SurfaceTopography.Uniform.GeometryAnalysis import (
    assign_patch_numbers_area, patch_areas)

mask = t.heights() > 2 * t.Sq()          # islands above 2 Sq
nb_patches, patch_ids = assign_patch_numbers_area(mask, t.is_periodic)
areas = patch_areas(patch_ids) * np.prod(t.pixel_size)
```

## MPI-parallel topographies

For [domain-decomposed topographies](topographies.md#mpi), support depends on
the function:

| Function | Domain-decomposed topographies |
|-|-|
| `mean`, `min`, `max`, `Sq` | supported (reduced over all processes) |
| `Rq` | raises `NotImplementedError` |
| `rms_laplacian`, `rms_curvature_from_area` | raise `NotImplementedError` whenever the communicator has more than one process. This also applies to pipeline steps on a decomposed topography, which report no decomposition but share its communicator |
| `Rdq`, `Rddq`, `rms_gradient` | reduce over processes. They use the same derivative operator as `rms_laplacian`, which is not domain-decomposition aware, but they do not check for decomposition |
| `bearing_area`, `median`, `mad_height`, `mad_polyfit` | use the heights of the local subdomain only (no reduction) |
