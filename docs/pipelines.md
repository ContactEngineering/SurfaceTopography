# Pipelines

```python
import numpy as np
from SurfaceTopography import read_topography
```

A pipeline function takes a topography and returns a new topography, e.g. a
detrended, filtered or rescaled version of it. Calls can be chained, and the
heights are computed only when they are requested. How pipelines are evaluated
and how to collapse them into a plain topography (`squeeze`) is described in
[Topographies](topographies.md#pipelines). This page describes the individual
operations and the numerics behind them.

```python
t = read_topography("test/file_format_examples/di-1.di")   # 512 x 512 AFM scan, nm
t2 = t.detrend(detrend_mode="rms-curvature").window()      # still lazy
h = t2.heights()                                           # evaluated here
```

## Overview

| Function | Uniform | Nonuniform | Result periodic? |
|-|-|-|-|
| [`detrend`](#detrending) | 1D, 2D | 1D | only for mode `"center"` (else `False`) |
| [`scan_line_align`](#scan-line-alignment) | 2D | – | `False` |
| [`window`](#windowing) | 1D, 2D | – | unchanged |
| [`filter`, `shortcut`, `longcut`](#fourier-filters) | 1D, 2D, periodic only | – | unchanged |
| [`downsample`](#downsampling) | 2D | – | unchanged |
| [`fill_undefined_data`](#filling-with-a-constant) | 1D, 2D | – | unchanged |
| [`interpolate_undefined_data`](#harmonic-interpolation) | 1D, 2D | – | unchanged |
| [`mirror_stitch`](#mirror-stitching) | 2D | – | `True` |
| [`interpolate_fourier`](#fourier-interpolation) | 2D | – | `True` |
| [`to_nonuniform`, `to_uniform`](#uniform-and-nonuniform-representation) | 1D | 1D | `False` |
| [`to_unit`, `scale`](#units-and-scaling) | 1D, 2D | 1D | unchanged |
| [`transpose`](#transposition-and-translation) | 1D, 2D | – | unchanged |
| [`translate`](#transposition-and-translation) | 1D, 2D, periodic only | – | unchanged |
| [`superpose`, `add`, `subtract`](#arithmetic) | 1D, 2D | 1D | of the first operand |

`interpolate_linear`, `interpolate_bicubic` and `interpolate_cubic` do not
return a topography but a function of the position; they are described in
[Interpolation](#interpolation) because they underlie several of the
operations above. `interpolate_fourier` and the nonuniform `add`/`subtract`
return a new (non-lazy) topography rather than a pipeline step.

The fits, Fourier filters, harmonic interpolation and interpolators work on the
complete grid; they are not written for domain-decomposed (MPI-parallel)
topographies, and `mirror_stitch` raises `NotImplementedError` for them. See
[Topographies](topographies.md#mpi) for parallel data.

## Detrending

```python
t.detrend(detrend_mode="height", coeffs=None)
t.polyfit(deg)
```

`detrend` subtracts a polynomial $p$ from the heights, $h_i - p(x_i)$. The
polynomial is either computed according to `detrend_mode` or given explicitly
as `coeffs`.

| `detrend_mode` | Polynomial | Criterion |
|-|-|-|
| `"mean"`, `"center"` | constant | mean height |
| `"median"` | constant | median height |
| `"rms-tilt"`, `"height"` (default) | linear | minimal rms height |
| `"mad-tilt"` | linear | minimal median absolute deviation |
| `"slope"` | linear | minimal rms slope |
| `"rms-curvature"`, `"curvature"` | quadratic | minimal rms height |
| `"mad-curvature"` | quadratic | minimal median absolute deviation |

`"center"`, `"height"` and `"curvature"` are aliases of `"mean"`,
`"rms-tilt"` and `"rms-curvature"`. A detrended topography is reported as
nonperiodic for every mode except `"center"`, which keeps the periodicity of
the parent; to remove only the mean of a periodic topography and keep it
periodic, use `"center"`.

The detrended topography exposes the polynomial:

| Attribute | Meaning |
|-|-|
| `coeffs` | coefficients, lowest order first (see below for the coordinates) |
| `stringify_plane(fmt=str)` | the polynomial as a string, e.g. `'0.1 + 2.3 x'` |
| `curvatures` | $\partial^2 p/\partial x^2$ for line scans; $(\partial^2 p/\partial x^2, \partial^2 p/\partial y^2, \partial^2 p/\partial x\partial y)$ for topographies, in inverse length units |
| `detrend_mode` | settable; setting it recomputes the coefficients |

```python
d = t.detrend(detrend_mode="rms-curvature")
d.stringify_plane(lambda c: f"{c:.3g}")
# '-4.26 + 48.8 x + -35.2 y + -43.1 x^2 + 47.9 y^2 + -16.7 xy'
d.curvatures      # approx. (-3.5e-04, 3.8e-04, -1.3e-04) in 1/nm
```

### Uniform data

For uniform line scans and topographies the polynomial is written in the
normalized coordinates $\xi = x/L_x$ and $\eta = y/L_y$, i.e.
$\xi_i = i/N_x$ on the grid. All coefficients then carry the unit of the
heights, and the slope of the tilt $a_0 + a_1\xi$ is $a_1/L_x$. For
topographies the coefficients are ordered

$$
p(\xi,\eta) = a_0 + a_1\xi + a_2\eta \quad\text{or}\quad
p(\xi,\eta) = a_0 + a_1\xi + a_2\eta + a_3\xi^2 + a_4\eta^2 + a_5\xi\eta .
$$

`polyfit(deg)` and the rms modes solve the linear least-squares problem

$$
\min_{\{a_k\}} \sum_{(i,j)\,\text{defined}} \bigl(h_{ij} - p(\xi_i,\eta_j)\bigr)^2 ,
$$

i.e. they minimize the rms height of the detrended data. Undefined (masked)
points are excluded from the sum. For topographies the problem is solved with
`numpy.linalg.lstsq` on the $N \times m$ design matrix ($m = 3$ or $6$
columns, $N$ defined points), at cost $O(N m^2)$; line scans use
`numpy.polyfit`, which accepts any degree, but `detrend` only handles
polynomials up to second order. Topographies support degrees 1 and 2.
Because $\xi, \eta \in [0, 1)$, the columns of the design matrix are of order
one independent of the length unit and the scan size.

The `"slope"` mode chooses the tilt whose slope is the mean of the
finite-difference slopes,
$a_1/L_x = \frac{1}{N_x-1}\sum_{i}(h_{i+1}-h_i)/\Delta x = (h_{N_x-1}-h_0)/\bigl((N_x-1)\Delta x\bigr)$
(per scan line, averaged, for topographies). A constant slope $s$ that
minimizes $\sum_i \bigl((h_{i+1}-h_i)/\Delta x - s\bigr)^2$ is exactly this mean, so the
mode minimizes the rms slope of the result. $a_0$ is chosen such that the
detrended data has zero mean.

### Nonuniform line scans

Nonuniform line scans are treated as the piecewise-linear interpolant of
their data points, the same model used for their
[statistics](statistics.md). Their trend is a polynomial in the *absolute*
position, $p(x) = \sum_{k=0}^n a_k x^k$, and `coeffs` are the $a_k$. The rms
height of a piecewise-linear profile is

$$
h_\text{rms}^2 = \frac{1}{L}\int_{x_0}^{x_{N-1}} h^2\,dx
= \frac{1}{3L} \sum_{i=0}^{N-2} \left( h_i^2 + h_{i+1}^2 + h_i h_{i+1} \right) \Delta x_i ,
$$

and `polyfit(deg)` minimizes the same expression for the residuals
$r_i = h_i - p(x_i)$:

$$
\min_{\{a_k\}} \sum_{i=0}^{N-2} \left( r_i^2 + r_{i+1}^2 + r_i r_{i+1} \right) \Delta x_i .
$$

This is the exact integral of the squared piecewise-linear interpolant of the
residual; for a tilt ($n \le 1$) it equals $\int (h - p)^2\,dx$. In contrast
to an unweighted fit to the data points, every part of the profile contributes
in proportion to its length, so densely sampled regions do not dominate the
fit. Setting $\partial/\partial a_k$ to zero yields the $(n+1)\times(n+1)$
normal equations

$$
\sum_{l=0}^n a_l \sum_{i=0}^{N-2} \left( 2x_i^{k+l} + 2x_{i+1}^{k+l} + x_i^k x_{i+1}^l + x_{i+1}^k x_i^l \right) \Delta x_i
= \sum_{i=0}^{N-2} \left( \left[ 2h_i + h_{i+1} \right] x_i^k + \left[ 2h_{i+1} + h_i \right] x_{i+1}^k \right) \Delta x_i ,
$$

assembled in $O(N n^2)$ operations and solved directly. They are assembled in
coordinates centered on the scan, $x - (x_0+x_{N-1})/2$, and the coefficients
are transformed back to monomials of $x$ afterwards: raw monomials make the
system ill-conditioned when the scan is short compared to its distance from
$x = 0$.

The `"slope"` mode minimizes $\int (h' - a_1)^2\,dx$, whose solution is the
length-weighted mean of the segment slopes,
$a_1 = (h_{N-1} - h_0)/(x_{N-1} - x_0)$; $a_0$ makes the mean of the result
vanish.

```python
l = read_topography("test/file_format_examples/dektak-1.csv")   # nonuniform, µm
dl = l.detrend(detrend_mode="rms-tilt")
dl.coeffs     # array([0.2064..., 0.0110...]): a_0 in µm, a_1 dimensionless
dl.mean()     # ~ 1e-16
```

### Robust (MAD) modes

`"mad-tilt"` and `"mad-curvature"` minimize the median absolute deviation of
the heights instead of the rms height, which reduces the influence of
outliers such as spikes or particles. The coefficients except the constant are
found with the derivative-free Nelder–Mead method, started from the rms
solution; the constant is the median of the result. The MAD itself is defined
in [Scalar statistics](statistics.md). Since this is an iterative optimization
of a non-smooth objective, it is considerably more expensive than the rms
fit and its result depends on the starting point.

### Checkerboard detrending

`checkerboard_detrend_profile` and `checkerboard_detrend_area` subdivide the
data into tiles and detrend each tile separately; they return arrays, not
pipeline steps, and are used for the variable-bandwidth analysis described in
[Scale-dependent parameters](scale-dependent.md).

## Scan line alignment

```python
t.scan_line_align(direction="x", mode="median", degree=1)
```

AFM scans often show per-line offsets and tilts from scanner drift or feedback
errors. `scan_line_align` corrects them in two steps. Let $j$ label the scan
lines and $t_i = i/N$ the normalized position along a line of $N$ points
(`direction="x"`: lines run along $x$, i.e. they are the columns
`heights()[:, j]`; `"y"`: the rows).

1. For every line, fit a polynomial $p_j(t) = \sum_{k=0}^{d} c_{jk} t^k$ of
   degree $d$ = `degree` by least squares over its defined points and subtract
   it. Lines with fewer than $d+1$ defined points are fitted with the highest
   degree their points allow.
2. Compute the center $c_j$ (median or mean, `mode`) of each residual line and
   shift line $j$ by $o_j = o_{j-1} + c_{j-1} - c_j$, $o_0 = 0$; finally
   subtract the mean of the $o_j$.

The cost is $O(N_x N_y d^2)$. Because step 1 contains a constant term, the
mean of every residual line vanishes; with `mode="mean"` step 2 is therefore a
no-op, and only `mode="median"` (default) aligns lines whose residuals are
skewed, e.g. by particles. The correction assumes that every scan line has
the same average height: a real height variation along the slow scan
direction is removed together with the artifact. The fitted coefficients are
available as `line_coeffs` (shape `(nb_lines, degree + 1)`, highest power
first, as in `numpy.polyfit`) and `offsets`. The result is reported as
nonperiodic.

```python
a = t.scan_line_align(direction="x", mode="median", degree=1)
a.line_coeffs.shape    # (512, 2)
a.Sq()
```

## Windowing

```python
t.window(window=None, direction=None)
```

Multiplies the heights by a window $w$ that decays to zero at the boundary.
This suppresses the spectral leakage caused by the discontinuity between
opposite edges of a nonperiodic scan when Fourier transforms are computed; its
effect on the PSD is discussed in [Spectral analysis](spectral.md).

| `direction` | Window |
|-|-|
| `"x"` (default for line scans) | 1D window $w_i$ along $x$, applied to every line |
| `"y"` | 1D window along $y$ |
| `"radial"` (default for topographies) | rotationally symmetric window |

For `"x"` and `"y"`, `window` is any name accepted by
`scipy.signal.get_window`. The radial window supports only `"hann"`:

$$
w(r) = \begin{cases} \tfrac12 + \tfrac12\cos(\pi r / r_\text{max}) & r \le r_\text{max} \\ 0 & r > r_\text{max} \end{cases},
\qquad r_\text{max} = \tfrac12\min(L_x, L_y),
$$

with $r$ the distance from the center of the scan in physical units, so the
window is circular also for non-square pixels. If `window` is `None`,
nonperiodic topographies get a Hann window and periodic topographies are left
unchanged.

The window is normalized to unit mean square, $\langle w^2 \rangle = 1$,
i.e. $w \leftarrow w\,\sqrt{N/\sum w^2}$ over the $N$ grid points. For heights
with zero mean that are uncorrelated with the window, this conserves
$\langle h^2 \rangle$ and therefore the integral of the PSD.

```python
w = t.window()          # Hann, radial (t is nonperiodic)
(w.window_data ** 2).mean()   # 1.0
```

## Fourier filters

```python
t.filter(filter_function=lambda q: q <= 1, isotropic=True)
t.shortcut(cutoff_wavevector=None, cutoff_wavelength=None, kind="circular step")
t.longcut(cutoff_wavevector=None, cutoff_wavelength=None, kind="circular step")
```

All three multiply the discrete Fourier transform by a transfer function
$F$ and transform back,

$$
h'_{ij} = \mathcal F^{-1}\!\left[ F(\mathbf q)\, \tilde h(\mathbf q) \right]_{ij},
\qquad q_{x,k} = \frac{2\pi k}{L_x},\; k = -\tfrac{N_x}{2},\dots,\tfrac{N_x}{2}-1 ,
$$

using real-to-complex FFTs at cost $O(N\log N)$. Wavevectors are in radians
per length unit. The discrete Fourier transform implies periodic continuation,
so these filters accept only periodic topographies and raise `ValueError`
otherwise; for nonperiodic data, make the data periodic first (e.g. with
[`mirror_stitch`](#mirror-stitching)) or construct a periodic `Topography`
from it if the edge mismatch is acceptable.

`filter` takes an arbitrary transfer function. It is called as
`filter_function(q)` with $q = |\mathbf q|$ for line scans and isotropic
filters, and as `filter_function(qx, qy)` with `isotropic=False`.

`shortcut` and `longcut` are step filters with cutoff
$q_\text{cut} = 2\pi/\lambda_\text{cut}$ (pass either `cutoff_wavevector` $q_\text{cut}$ or
`cutoff_wavelength` $\lambda_\text{cut}$):

| | `kind="circular step"` | `kind="square step"` (2D only) |
|-|-|-|
| `shortcut` (low-pass) | $F = 1$ for $\lvert\mathbf q\rvert \le q_\text{cut}$ | $F = 1$ for $\lvert q_x\rvert \le q_\text{cut}$ and $\lvert q_y\rvert \le q_\text{cut}$ |
| `longcut` (high-pass) | $F = 1$ for $\lvert\mathbf q\rvert \ge q_\text{cut}$ | $F = 1$ for $\lvert q_x\rvert \ge q_\text{cut}$ and $\lvert q_y\rvert \ge q_\text{cut}$ |

and $F = 0$ otherwise. `longcut` therefore also removes the mean height. A
sharp step in $\mathbf q$ produces ringing (Gibbs oscillations) next to sharp
features in real space; a smooth `filter_function` avoids it.

If the number of grid points along the last axis ($y$ for topographies, $x$
for line scans) is odd, the filtered heights have one grid point less along
that axis. Use grids with an even number of points there.

```python
from SurfaceTopography import Topography
p = Topography(t.heights(), t.physical_sizes, periodic=True, unit=t.unit)
s = p.shortcut(cutoff_wavelength=50)            # remove wavelengths < 50 nm
s.cutoff_wavevector                              # 0.1257 (= 2 pi / 50)
g = p.filter(filter_function=lambda q: np.exp(-(q * 10) ** 2))   # Gaussian low-pass
```

## Downsampling

```python
t.downsample(factor, mode="nth")
```

Reduces the resolution of a topography by an integer factor $f$ (an int, or a
tuple $(f_x, f_y)$). The result has
$\lfloor N_x/f_x\rfloor \times \lfloor N_y/f_y\rfloor$ grid points with spacing
$f_x\Delta x$, $f_y\Delta y$; grid points that do not fill a complete
$f_x\times f_y$ patch at the end are discarded and the physical size shrinks
accordingly.

| `mode` | Heights |
|-|-|
| `"nth"` | $h'_{ij} = h_{f_x i,\, f_y j}$ |
| `"average"` | mean over the $f_x \times f_y$ patch |

`"nth"` keeps the original values but aliases wavelengths shorter than the new
grid spacing into the spectrum; `"average"` is a box filter that attenuates
them (but does not remove them completely).

```python
t.downsample(4, mode="average").nb_grid_pts    # (128, 128)
t.downsample((3, 5)).physical_sizes            # (498.05, 498.05): trailing points dropped
```

## Undefined data

Optical instruments often report undefined data points; they are stored as
masked heights, see [Topographies](topographies.md#undefined-data). Most
analyses require complete data, and two pipeline functions provide it.

```python
o = read_topography("test/file_format_examples/opd-1.opd")
o.has_undefined_data                         # True (116 of 307200 points)
```

### Filling with a constant

```python
t.fill_undefined_data(fill_value=-np.inf)
```

Replaces every undefined point by `fill_value`. The default $-\infty$ marks
points that should never be in contact, e.g. in contact mechanics
calculations.

### Harmonic interpolation

```python
t.interpolate_undefined_data(method="harmonic")
t.interpolate_undefined_data_with_harmonic_function()
```

Replaces the undefined points by the harmonic function that matches the
defined data, i.e. the solution of the Laplace equation

$$
\nabla^2 h = 0 \quad\text{on the undefined region } \Omega,
\qquad h = h_\text{data} \quad\text{on the defined points bordering } \Omega .
$$

It is discretized with the nearest-neighbor stencil: for every undefined grid
point $p$

$$
\sum_{n\in\mathcal N(p)} (h_n - h_p) = 0 ,
$$

where $\mathcal N(p)$ are the two (line scans) or four (topographies) nearest
neighbors. Neighbors that are defined enter the right-hand side as Dirichlet
values; on periodic topographies the stencil wraps around, on nonperiodic ones
neighbors outside the domain are dropped, which corresponds to a vanishing
normal derivative (natural boundary condition) at the edge. On line scans the
result is linear interpolation between the bordering points (and constant
continuation at a free end).

All undefined points form one sparse linear system with $M$ unknowns (the
number of undefined points) and at most five entries per row, solved with
SciPy's sparse direct solver. Disconnected holes do not couple, so the matrix is
block diagonal and solving them together is equivalent to solving them one by
one, but avoids a pass over the full grid per hole. Assembly costs $O(N)$; the
cost of the direct solve grows superlinearly with $M$ because of fill-in, so
very large undefined regions are expensive. The discrete Laplacian obeys a
maximum principle: imputed heights lie between the minimum and maximum of the
bordering defined heights, so no new extrema are created. The system is
singular, and the solve fails, if a connected region of undefined points has
no defined neighbor (e.g. if all points are undefined).

```python
f = o.interpolate_undefined_data()
f.has_undefined_data                         # False
```

## Interpolation

These functions return interpolating functions of the position (in the units of
`physical_sizes`), not topographies.

### Linear (uniform)

```python
fn = t.interpolate_linear()
fn(x, periodic=None)        # line scans
fn(x, y, periodic=None)     # topographies
```

Line scans are interpolated linearly between grid points. On topographies,
each grid cell is split along the diagonal from $(i+1, j)$ to $(i, j+1)$ into
two triangles, and the heights are interpolated linearly on each triangle. In
local coordinates $\xi, \eta \in [0,1)$ of the cell:

$$
h(\xi,\eta) = \begin{cases}
(1-\xi-\eta)\,h_{i,j} + \xi\,h_{i+1,j} + \eta\,h_{i,j+1} & \xi+\eta \le 1 \\
(\xi+\eta-1)\,h_{i+1,j+1} + (1-\xi)\,h_{i,j+1} + (1-\eta)\,h_{i+1,j} & \xi+\eta > 1
\end{cases}
$$

The interpolant is continuous with piecewise-constant slopes. Evaluation is
$O(1)$ per point. For periodic data, positions wrap around; for nonperiodic
data only positions in $[0, (N_x-1)\Delta x] \times [0, (N_y-1)\Delta y]$ are
allowed and others raise `ValueError`. The `periodic` argument overrides the
periodicity of the topography. Undefined points yield NaN in the adjacent
cells.

```python
fn = t.interpolate_linear()
fn(np.array([1.3, 2.7]), np.array([4.1, 0.2]))   # heights in nm
```

### Bicubic

```python
fn = t.interpolate_bicubic(derivative="fourier")
fn(x, y, derivative=0)
```

On each cell, the interpolant is a bicubic polynomial

$$
h(\xi,\eta) = \sum_{a=0}^{3}\sum_{b=0}^{3} c_{ab}\,\xi^a\eta^b ,
$$

whose 16 coefficients are determined by the height, the two slopes
$\partial h/\partial x$, $\partial h/\partial y$, and the mixed derivative
$\partial^2 h/\partial x\partial y$ (set to zero) at the four corners.
Along every cell edge the height and its normal derivative are then cubic
Hermite polynomials of the corner data shared by the neighboring cell, so the
interpolant is continuous with continuous first derivatives (C¹); second
derivatives jump across cell boundaries. The slopes at the grid points come
from the spectral derivative (`derivative="fourier"`, exact for band-limited
periodic data, cost $O(N \log N)$) or from central differences
$(h_{i+1,j}-h_{i-1,j})/(2\Delta x)$ (`derivative="fd"`, second order).

Only periodic topographies are supported. The interpolating function returns
the heights (`derivative=0`), additionally the two slopes (`derivative=1`),
or additionally $\partial^2h/\partial x^2$, $\partial^2h/\partial y^2$,
$\partial^2h/\partial x\partial y$ (`derivative=2`). Each evaluation is
$O(1)$. The coefficients of a cell are computed when the cell is evaluated and
only those of the most recently used cell are kept: storing them for all cells
would take 128 bytes per grid point (more than 2 GB for a 4096 × 4096 map),
whereas evaluation points that arrive in order mostly hit the same cell.

```python
fn = p.interpolate_bicubic()
h, dhdx, dhdy = fn(np.array([1.3]), np.array([4.1]), derivative=1)
```

### Fourier interpolation

```python
t.interpolate_fourier(nb_grid_pts)
```

Returns a periodic `Topography` on a finer grid with `nb_grid_pts` points and
the same physical size, obtained by padding the Fourier spectrum with zeros
(trigonometric interpolation). The underlying interpolant is the band-limited
periodic function through the data; if `nb_grid_pts` is a multiple of the
original grid, every $f$-th point of the result reproduces the original
heights. `nb_grid_pts` must not be smaller than the original grid. If a grid dimension is even, the Nyquist component is split
equally between $\pm q_\text{Nyquist}$, which corresponds to a cosine at the
Nyquist frequency, the trigonometric interpolant with minimal oscillation
([S. G. Johnson, *Notes on FFT-based differentiation*](https://math.mit.edu/~stevenj/fft-deriv.pdf)).
Cost is $O(M\log M)$ in the number $M$ of output points. Only 2D topographies
are supported; the input is treated as periodic, so nonperiodic data shows
ringing near the edges. The unit and info dictionary are not carried over.

```python
fi = p.interpolate_fourier((1024, 1024))
np.allclose(fi.heights()[::2, ::2], p.heights())   # True
```

### Nonuniform line scans

```python
l.interpolate_linear()
l.interpolate_cubic()
```

Both return a `scipy.interpolate.interp1d` object on the data points:
piecewise-linear, or the cubic spline through the points (C², not-a-knot end
conditions). Positions must lie within $[x_0, x_{N-1}]$. Reentrant line scans
(with overhangs) raise `ReentrantDataError`.

### Mirror stitching

```python
t.mirror_stitch()
```

Appends mirror images in $x$ and $y$:

$$
h' = \begin{pmatrix} h & h_{:,\,\text{rev}} \\ h_{\text{rev},\,:} & h_{\text{rev},\,\text{rev}} \end{pmatrix},
$$

where "rev" reverses the order along an axis. The result has $2N_x \times
2N_y$ grid points and size $2L_x \times 2L_y$, has no jumps at the boundaries
of the doubled domain and is reported as periodic, so that FFT-based
operations can be applied without a window. The edge values are duplicated
(even symmetry about the half-sample point), so slopes change sign at the
mirror lines. 2D, serial topographies only.

```python
t.mirror_stitch().nb_grid_pts    # [1024, 1024]
```

## Uniform and nonuniform representation

```python
u.to_nonuniform()
l.to_uniform(nb_points=None, padding=0, nb_interpolate=None, pixel_size=None)
```

`to_nonuniform` wraps a nonperiodic uniform line scan as a nonuniform one with
positions $x_i = i\,\Delta x$; its physical size is the span of the data,
$(N-1)\Delta x$. No data is changed. Periodic line scans raise `ValueError`;
nonuniform line scans are always nonperiodic.

`to_uniform` resamples a nonuniform line scan on $M$ equidistant points by
linear interpolation between the data points (the piecewise-linear model used
throughout for nonuniform data). Exactly one of the following sets the grid:

| Argument | Number of points $M$ | Spacing $\Delta x$ |
|-|-|-|
| `nb_points` | given | $L/(M-1)$ |
| `nb_interpolate` $=k$ | $k\,\lfloor L/\min_i \Delta x_i\rfloor$ | $L/(M-1)$ |
| `pixel_size` | $\lfloor L/\Delta x\rfloor + 1$ | given |

The grid starts at the first data point $x_0$ (positions are reported starting
at zero) and the result is nonperiodic with physical size $(M-1)\Delta x$.
`padding` appends that many points holding the last height. With
`nb_interpolate`, the grid resolves the finest spacing of the data. Reentrant
line scans raise `RuntimeError`. Cost: $O(M + N)$.

```python
l.to_uniform(nb_points=1000).pixel_size    # (1.5013...,) µm
l.to_uniform(pixel_size=1.0).nb_grid_pts    # (1500,)
```

## Units and scaling

```python
t.to_unit(unit)
t.scale(height_scale_factor, position_scale_factor=1, unit=None)
```

`to_unit` converts heights and lateral positions to another length unit. Both
are multiplied with $s_\text{from}/s_\text{to}$, where $s$ is the length of
the unit in meters. The topography must have a unit. Known length units:

| Unit | m | Unit | m |
|-|-|-|-|
| `Gm` | $10^{9}$ | `µm` (`um`) | $10^{-6}$ |
| `Mm` | $10^{6}$ | `nm` | $10^{-9}$ |
| `km` | $10^{3}$ | `Å` (`A`) | $10^{-10}$ |
| `m` | 1 | `pm` | $10^{-12}$ |
| `mm` | $10^{-3}$ | `fm` | $10^{-15}$ |

`scale` multiplies heights by `height_scale_factor` and positions by
`position_scale_factor` without any unit logic; `unit` only sets the unit
metadata of the result. It is used, e.g., to convert raw instrument data or to
flip the sign of the heights.

```python
t.to_unit("µm").physical_sizes      # (0.5, 0.5)
t.scale(-1)                         # inverted heights, still in nm
```

The conversion factors come from `SurfaceTopography.Support.UnitConversion`:
`get_unit_conversion_factor(from_unit, to_unit)` returns the factor,
`suggest_length_unit(scale, lower, upper)` suggests the unit for displaying
data in a range given in meters on a `"linear"` or `"log"` axis.

```python
from SurfaceTopography.Support.UnitConversion import get_unit_conversion_factor
get_unit_conversion_factor("nm", "µm")      # 0.001
```

## Transposition and translation

```python
t.transpose()
t.translate(offset=(0, 0))
```

`transpose` swaps the $x$ and $y$ axes (grid points, physical sizes, heights,
subdomain information). `translate` shifts a periodic topography by an
integer number of grid points, $h'_{ij} = h_{i-o_x,\, j-o_y}$ with periodic
wrap-around; nonperiodic topographies raise `ValueError` since the shift
would move data out of the domain. For line scans `offset` may be a scalar.

```python
tr = p.translate(offset=(10, 0))
np.allclose(tr.heights()[10], p.heights()[0])   # True
```

## Arithmetic

```python
t.superpose(other)
t.add(other, interpolate=False)
t.subtract(other, interpolate=False)
```

`superpose` adds the heights of two uniform topographies with identical
dimension, number of grid points and physical size; mismatching properties
raise `ValueError`. The comparison is exact, so physical sizes that differ by
rounding (e.g. after a unit round trip) are rejected.

`add` and `subtract` (`subtract(b)` is `add(b.scale(-1))`) first convert
`other` to the unit of `t` if both have units. For uniform data, `other` must
then be on the same grid, or, with `interpolate=True`, it is evaluated with
[`interpolate_linear`](#linear-uniform) at the grid positions of `t` and must
cover the domain of `t`. A typical application is the gap between two
surfaces, or the composite topography of two surfaces in contact.

For nonuniform line scans, `add` and `subtract` return a new
`NonuniformLineScan` on the union of both sets of positions inside the
overlapping range, with both profiles linearly interpolated there. Line scans
that do not overlap or are reentrant raise an error.

```python
gap = p.subtract(p.translate(offset=(1, 0)))     # difference of two periodic maps
l.add(l.scale(-1)).heights().max()               # 0.0
```
