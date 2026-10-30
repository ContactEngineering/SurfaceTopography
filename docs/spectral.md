# Spectral analysis

```python
import numpy as np
from SurfaceTopography import Topography, UniformLineScan, NonuniformLineScan
```

The power spectral density (PSD), the height-difference autocorrelation
function (ACF) and the variable bandwidth method (VBM) describe how roughness is
distributed over length scales. All three are methods of topography objects.
This page defines them, states the conventions of their discretization and
explains the resampling of the results onto collocation points. Scalar
parameters are described in [statistics](statistics.md), scale-dependent
derivatives and the reliability cutoff in
[scale-dependent analysis](scale-dependent.md), and averages over several
topographies in [containers](containers.md).

## Conventions

| | |
|-|-|
| Wavevectors | $q_k = 2\pi k/L$, $k = 0, \dots, N-1$ (2D: $\mathbf q = (q_{x,k}, q_{y,l})$) |
| Fourier transform | $\tilde h(q_k) = \Delta x \sum_i h_i e^{-i q_k x_i}$ (2D: $\Delta x \Delta y \sum_{ij}$) |
| 1D PSD | $C^\text{1D}(q) = \lvert\tilde h(q)\rvert^2/L$ (units length³) |
| 2D PSD | $C^\text{2D}(\mathbf q) = \lvert\tilde h(\mathbf q)\rvert^2/(L_xL_y)$ (units length⁴) |
| Returned wavevectors | $0 \le q < \pi/\Delta x$; values at $\pm q$ are averaged |
| Height-difference ACF | $A(r) = \tfrac12\langle[h(x+r)-h(x)]^2\rangle$ (units length²) |
| FFT | `numpy.fft`, $O(N\log N)$ for $N$ grid points |

With these conventions, Parseval's theorem reads

$$
\langle h^2\rangle = \frac{1}{L}\sum_{k} C^\text{1D}(q_k)
\approx \frac{1}{2\pi}\int_{-\infty}^{\infty} dq\, C^\text{1D}(q)
$$

for line scans and

$$
\langle h^2\rangle = \frac{1}{L_xL_y}\sum_{\mathbf q} C^\text{2D}(\mathbf q)
\approx \frac{1}{(2\pi)^2}\int d^2q\, C^\text{2D}(\mathbf q)
$$

for topography maps, where the sums run over all wavevectors of the FFT. A
self-affine topography has $C^\text{2D}(q) = C_0 q^{-2-2H}$ and
$C^\text{1D}(q) \propto q^{-1-2H}$. The conventions follow
Jacobs, Junge & Pastewka (2017).

None of the functions on this page accept topographies with undefined data
(they raise `UndefinedDataError`); see [topographies](topographies.md) for how
to fill in missing points.

## Power spectral density

### Line scans and profiles

```python
q, C = topography.power_spectrum_from_profile(
    window=None, reliable=True, resampling_method='bin-average',
    collocation='log', nb_points=None, nb_points_per_decade=10)
```

For a line scan, the PSD is computed from the FFT of the heights,

$$
C^\text{1D}(q_k) = \frac{1}{L}\left\lvert \Delta x\sum_{i=0}^{N-1} h_i e^{-iq_kx_i}\right\rvert^2 .
$$

The heights are real, so $C^\text{1D}(q_k) = C^\text{1D}(-q_k)$; the
function returns the average of the two branches for $0 \le q_k < \pi/\Delta x$
($k < N/2$). For a topography map, each profile along $x$ (the first array
axis) is transformed and the PSDs of all $N_y$ profiles are averaged. The cost
is $O(N_xN_y\log N_x)$.

```python
from SurfaceTopography import UniformLineScan

x = np.arange(128) / 128 * 10.0                      # L = 10 µm, N = 128
line = UniformLineScan(np.sin(2 * np.pi * x / 2.5), 10.0, periodic=True, unit='µm')

q, C = line.power_spectrum_from_profile(resampling_method=None)
print(q[:5])                                          # q_k = 2πk/L
print(q[np.argmax(C)], 2 * np.pi / 2.5)               # peak at 2π/λ
# Parseval: sum over q = 0 and both branches ±q
print(np.isclose((C[0] + 2 * C[1:].sum()) / 10.0, np.mean(line.heights() ** 2)))
```

```
[0.         0.62831853 1.25663706 1.88495559 2.51327412]
2.5132741228718345 2.5132741228718345
True
```

### Topography maps

```python
q, C = topography.power_spectrum_from_area(
    window=None, reliable=True, collocation='log', nb_points=None,
    nb_points_per_decade=10, return_map=False, resampling_method='bin-average')
```

The 2D PSD is computed from the 2D FFT ($O(N\log N)$ with $N = N_xN_y$),

$$
C^\text{2D}(\mathbf q) = \frac{1}{L_xL_y}\left\lvert \Delta x\Delta y\sum_{i,j} h_{ij} e^{-i(q_xx_i + q_yy_j)}\right\rvert^2 ,
$$

and averaged over all wavevectors whose magnitude $\lvert\mathbf q\rvert$ falls
into the same bin (see [resampling](#resampling)). This yields the isotropic
PSD $C^\text{iso}(q)$. Bins extend up to the smaller of the two Nyquist
wavevectors, $\pi\min(1/\Delta x, 1/\Delta y)$, so that the result does not
depend on the orientation of the map for anisotropic pixels. With
`return_map=True`, the full map $C^\text{2D}(\mathbf q)$ is returned as a third
value in the layout of `np.fft.fft2`; `topography.fftfreq()` returns the
matching wavevectors $(q_x, q_y)$ and `topography.wavevectors_norm2()` their
norm $\lvert\mathbf q\rvert$.

```python
from SurfaceTopography.Generation import fourier_synthesis

np.random.seed(0)
topo = fourier_synthesis((512, 512), (1.0, 1.0), hurst=0.8, rms_height=1e-3,
                         short_cutoff=0.01, long_cutoff=0.2, unit='mm')

q1, C1 = topo.power_spectrum_from_profile()              # mean 1D PSD along x
q2, C2, C_map = topo.power_spectrum_from_area(return_map=True)
print(C_map.shape)
```

```
(512, 512)
```

See [generation](generation.md) for `fourier_synthesis`.

### Nonuniform line scans

```python
q, C = line_scan.power_spectrum_from_profile(
    reliable=True, algorithm='fft', wavevectors=None, nb_interpolate=5,
    short_cutoff=np.mean, window=None, resampling_method='bin-average',
    collocation='log', nb_points=None, nb_points_per_decade=10)
```

A nonuniform line scan with points $x_0 < x_1 < \dots < x_{N-1}$ is treated as
the piecewise linear function that connects its data points, with
$L = x_{N-1} - x_0$. Two algorithms compute its PSD:

| `algorithm` | Method | Cost |
|-|-|-|
| `'fft'` (default) | Linear interpolation onto a uniform grid with spacing $\min_i\Delta x_i/n$ ($n$ = `nb_interpolate`), then the uniform PSD above | $O(N_\text{int}\log N_\text{int})$, $N_\text{int} \approx nL/\min_i\Delta x_i$ |
| `'brute-force'` | Exact Fourier transform of the piecewise linear function at the given `wavevectors` | $O(NM)$ for $M$ wavevectors |

The brute-force algorithm integrates each segment analytically. With
segment length $\Delta x_i = x_{i+1}-x_i$, midpoint
$\bar x_i = (x_i + x_{i+1})/2$, mean height $\bar h_i = (h_i+h_{i+1})/2$ and
height difference $\Delta h_i = h_{i+1} - h_i$, the height on segment $i$ is
$h(x) = \bar h_i + \Delta h_i (x-\bar x_i)/\Delta x_i$, and

$$
\tilde h(q) = \sum_{i=0}^{N-2} \Delta x_i\, e^{-iq\bar x_i}
\left[\bar h_i \operatorname{sinc}\!\left(\frac{q\Delta x_i}{2}\right)
+ \frac{i}{2}\,\Delta h_i\, \operatorname{sinc}'\!\left(\frac{q\Delta x_i}{2}\right)\right],
\qquad
C^\text{1D}(q) = \frac{1}{L}\lvert\tilde h(q)\rvert^2 ,
$$

with $\operatorname{sinc}(x) = \sin(x)/x$. The two terms are the Fourier
transforms of a rectangle and of a linear ramp on $[-1/2, 1/2]$. For
$\lvert x\rvert < 10^{-2}$, $\operatorname{sinc}'(x)$ is evaluated by its
Taylor series $-x/3 + x^3/30$, because $(\cos x - \operatorname{sinc}x)/x$
loses accuracy by cancellation at small $x$. The brute-force algorithm requires
`reliable=False` and `resampling_method=None`.

Both algorithms then discard wavevectors $q \ge 2\pi/s$, where $s$ is
`short_cutoff` applied to the array of distances $\Delta x_i$ (by default their
mean); wavelengths shorter than the typical point spacing are not resolved by
the data. `short_cutoff=None` disables this cutoff. Reentrant line scans
(overhangs) raise `ReentrantDataError` for the `'fft'` algorithm.

```python
from SurfaceTopography import NonuniformLineScan

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 10, 200))
x[0], x[-1] = 0, 10
h = np.sin(2 * np.pi * x / 5) + 0.1 * rng.normal(size=200)
nonuniform = NonuniformLineScan(x, h)

q, C = nonuniform.power_spectrum_from_profile()            # 'fft'
q_bf, C_bf = nonuniform.power_spectrum_from_profile(
    algorithm='brute-force', wavevectors=np.linspace(0.1, 10, 100),
    reliable=False, resampling_method=None)
print(q_bf.max() < 2 * np.pi / np.mean(np.diff(x)))       # short_cutoff applied
```

```
True
```

### Windowing of nonperiodic data

The discrete Fourier transform treats the data as one period of a periodic
function. For a nonperiodic topography, the heights at the two ends differ and
the periodic continuation has a step. A step contributes
$\lvert\tilde h\rvert^2 \propto q^{-2}$ to the spectrum, which can exceed the
PSD of the topography at large $q$. Multiplying the heights by a window
$w$ that goes to zero at the boundary suppresses this contribution.

| Function | Default window (`window=None`) |
|-|-|
| periodic topography | none |
| nonperiodic line scan (uniform) | `scipy.signal.get_window('hann', N)` along $x$ |
| nonperiodic topography map | rotationally symmetric Hann window, $w(r) = [1+\cos(\pi r/r_\text{max})]/2$ for $r < r_\text{max} = \min(L_x, L_y)/2$ and $w = 0$ outside, $r$ measured from the center of the map |
| nonuniform line scan | $w(x) \propto 1-\cos[2\pi(x-x_0)/L]$, evaluated at the data points |

Any window name understood by `scipy.signal.get_window` can be passed for
uniform line scans, `'hann'` for maps and nonuniform line scans; `'None'` (the
string) disables windowing also for nonperiodic data. The windows are
normalized such that the mean of $w^2$ is one, which preserves
$\langle h^2\rangle$ (and hence the magnitude of the PSD) for data whose
statistics do not depend on position.

The window is chosen by the dimension of the topography, not by the function:
`power_spectrum_from_profile` on a nonperiodic *map* uses the 2D radial window.
To window each profile along $x$ instead, apply the window explicitly and
switch off the default one:

```python
q, C = topo.window('hann', direction='x').power_spectrum_from_profile(window='None')
```

The radial window is zero in the corners of the map, so these data do not
contribute to the PSD of nonperiodic maps.

<!-- TODO: rationale for applying the radial 2D window (rather than a window along x) in power_spectrum_from_profile and integrate_psd_from_profile for nonperiodic maps? -->

## Height-difference autocorrelation

The height-difference autocorrelation function

$$
A(r) = \frac12\left\langle\left[h(x+r)-h(x)\right]^2\right\rangle
= \langle h^2\rangle - \langle h(x)\,h(x+r)\rangle
$$

is the variance of height differences at distance $r$ (Wang & Müser, 2018).
It is related to the PSD by

$$
A(r) = \frac{1}{2\pi}\int_{-\infty}^{\infty} dq\, C^\text{1D}(q)\,\left[1-\cos(qr)\right] ,
$$

and grows as $r^{2H}$ for a self-affine topography. Unlike the PSD, it needs
no window for nonperiodic data.

### Uniform data

```python
r, A = topography.autocorrelation_from_profile(
    reliable=True, resampling_method='bin-average', collocation='log',
    nb_points=None, nb_points_per_decade=10)
r, A = topography.autocorrelation_from_area(
    reliable=True, collocation='log', nb_points=None, nb_points_per_decade=5,
    return_map=False, resampling_method='bin-average')
```

`autocorrelation_from_profile` computes $A(r)$ along $x$ and, for maps,
averages it over all profiles; `autocorrelation_from_area` computes the 2D
function $A(\mathbf r)$ and averages it radially. Both use the FFT
(Wiener–Khinchin theorem) at a cost of $O(N\log N)$.

*Periodic topographies.* The average runs over all $N$ points with periodic
wrap-around. The height-height correlation
$\langle h(x)h(x+r_k)\rangle = \frac1N\sum_i h_ih_{i+k}$ is the inverse FFT
of $\lvert\mathrm{FFT}(h)\rvert^2/N$, and
$A(r_k) = \langle h^2\rangle - \langle h(x)h(x+r_k)\rangle$. Since
$A(r_k) = A(L-r_k)$, the two values are averaged and $0\le r_k < L/2$ is
returned.

*Nonperiodic topographies.* Only the $N-k$ pairs of points that lie inside the
scan enter the average at distance $r_k = k\Delta x$:

$$
A(r_k) = \frac{1}{2(N-k)}\sum_{i=0}^{N-k-1}\left(h_i-h_{i+k}\right)^2
= \frac{1}{2(N-k)}\sum_{i=0}^{N-k-1}\left(h_i^2+h_{i+k}^2\right)
- \frac{1}{N-k}\sum_{i=0}^{N-k-1}h_ih_{i+k} .
$$

The last sum is a linear (not circular) correlation; it is computed with an FFT
of the heights zero-padded to $2N-1$ points, so that periodic images do not
overlap. The first sum is obtained from cumulative sums of $h_i^2$. Distances
$0 \le r_k < L$ are returned. In 2D, the map is padded to
$(2N_x-1)\times(2N_y-1)$ points and normalized by $(N_x-k)(N_y-l)$; only
distance vectors with nonnegative components are computed, and radial bins
extend up to $(L_x+L_y)/2$ (periodic: $(L_x+L_y)/4$).

```python
r, A = line.autocorrelation_from_profile(resampling_method=None)
print(np.allclose(A, 0.5 * (1 - np.cos(2 * np.pi * r / 2.5))))

r, A = topo.autocorrelation_from_area()
```

```
True
```

### Nonuniform line scans

```python
r, A = line_scan.autocorrelation_from_profile(
    reliable=True, algorithm='fft', distances=None, nb_interpolate=5,
    short_cutoff=np.mean, resampling_method='bin-average', collocation='log',
    nb_points=None, nb_points_per_decade=10)
```

As for the PSD, the line scan is a piecewise linear function. The `'fft'`
algorithm interpolates it onto a uniform grid (see above) and computes the
nonperiodic ACF of the interpolated data. The `'brute-force'` algorithm
computes

$$
A(r) = \frac{1}{2(L-r)}\int_{x_0}^{x_{N-1}-r} dx\,\left[h(x+r)-h(x)\right]^2
$$

exactly at the given `distances` (all smaller than $L$): for each pair of
segments $(i, j)$, the integrand is a quadratic polynomial on the interval
where $x$ lies on segment $i$ and $x+r$ on segment $j$, and is integrated
analytically. This costs $O(N^2M)$ for $M$ distances and is implemented in
C++. Distances $r \le s/2$, with $s$ the `short_cutoff` of the point spacings,
are discarded.

```python
r_bf, A_bf = nonuniform.autocorrelation_from_profile(
    algorithm='brute-force', distances=[0.5, 1.0, 2.5],
    reliable=False, resampling_method=None)
print(A_bf)
```

```
[0.09647622 0.31834826 0.99761992]
```

## Variable bandwidth method

```python
bandwidths, rms_heights = topography.variable_bandwidth_from_profile(
    quantities='bh', reliable=True, resampling_method=None,
    nb_grid_pts_cutoff=4, func=lambda h: np.sqrt(np.mean(h * h)))
bandwidths, rms_heights = topography.variable_bandwidth_from_area(
    quantities='bh', reliable=True, resampling_method=None,
    nb_grid_pts_cutoff=4, func=lambda h: np.sqrt(np.mean(h * h)))
```

The variable bandwidth method measures the rms height as a function of the
size $\ell$ (the bandwidth) of the region it is computed on. At magnification
$M = 2^n$, $n = 0, 1, \dots$, the topography is divided into sections of size
$\ell = L/M$, a least-squares plane (mean and tilt) is subtracted in each
section, and the rms height of the detrended heights is computed. Subdivision
stops when a section would contain fewer than `nb_grid_pts_cutoff` points. For
a self-affine topography, $h_\text{rms}(\ell) \propto \ell^H$. Each
magnification costs $O(N)$, the analysis $O(N\log N)$.

| Variant | Sections | Statistic |
|-|-|-|
| `variable_bandwidth_from_profile` (uniform) | $M$ segments of each profile along $x$ | `func` of all detrended heights; default rms |
| `variable_bandwidth_from_area` | checkerboard of rectangles; the shorter side is divided into $M$, the longer one into $M\,\mathrm{round}(L_\text{long}/L_\text{short})$ | `func` of all detrended heights; bandwidth = mean side length |
| nonuniform `variable_bandwidth_from_profile` | $M$ segments; a point is inserted by linear interpolation at segment boundaries | mean of the rms heights of the segments |

The `quantities` string selects the returned arrays: `'m'` magnification,
`'b'` bandwidth, `'h'` statistic (`func`), and, for uniform data, `'g'` rms
slope (profiles) or rms gradient (maps) of the subtracted planes. Resampling is
not supported (`resampling_method` must be None).

```python
magnifications, bandwidths, rms_heights = topo.variable_bandwidth_from_profile(quantities='mbh')
print(magnifications)
```

```
[  1   2   4   8  16  32  64 128]
```

The detrending itself is available as `checkerboard_detrend_profile` and
`checkerboard_detrend_area`.

## Resampling

The FFT yields the PSD and ACF on $O(N)$ points, linearly spaced and, for
self-affine topographies, noisy (the PSD of a single random realization has a
relative standard deviation of order one at each $q$). By default, the
results are averaged within bins, which reduces the noise and gives points that
are equally spaced on the logarithmic axes in which PSDs are usually plotted.

### Collocation

`collocation` determines the bin edges between `min_value` and `max_value`
(the range of wavevectors or distances, see [reliability](#reliability-cutoff)):

| `collocation` | Bin edges | Number of bins |
|-|-|-|
| `'log'` (default) | equally spaced in $\log_{10}$, extended by $0.01$ decades at both ends | `nb_points` *edges*, i.e. `nb_points - 1` bins; if None, $\lceil(\log_{10}(\text{max}/\text{min})+1)\,n_\text{dec}\rceil$ edges with $n_\text{dec}$ = `nb_points_per_decade` |
| `'linear'` | equally spaced | `nb_points` bins (required) |
| `'quadratic'` | equally spaced in the square of the value; on a 2D radial grid, each bin contains a similar number of points | `nb_points` bins (required) |

For `'log'` collocation, $q = 0$ and $r = 0$ are excluded from the data.

### Methods

| `resampling_method` | Result | Cost |
|-|-|-|
| `None` | raw values on the FFT grid (not available for radial averages) | — |
| `'bin-average'` (default) | arithmetic mean of all values within a bin; returned as `numpy.ma.MaskedArray`, empty bins are masked | $O(n\log B)$ for $n$ values and $B$ bins |

For `'bin-average'`, the returned abscissa of each bin is the mean of the $q$
(or $r$) values of the data in that bin, not the bin center; use
`q.compressed()` and `C.compressed()` to drop empty bins. The arithmetic mean
is the maximum-likelihood estimator of the expected PSD from noisy samples.

```python
q, C = topo.power_spectrum_from_profile(collocation='linear', nb_points=50)
q, C = topo.power_spectrum_from_area(nb_points_per_decade=20)
```

The underlying functions `resample`, `resample_radial` and `bin_average` are
in `SurfaceTopography.Support.Regression`.

### Reliability cutoff

With `reliable=True` (the default), all functions on this page discard the
scales below the short reliability cutoff $\lambda_c$, which is determined from
the instrument metadata of the topography (resolution or tip radius, see
[scale-dependent analysis](scale-dependent.md)). If no cutoff is known, only the
grid limits apply. `reliable=False` returns all data within the grid limits.

| Function | Returned range |
|-|-|
| `power_spectrum_from_profile` | $q < 2\pi/\max(\lambda_c, 2\Delta x)$, i.e. at most the Nyquist wavevector $\pi/\Delta x$ |
| `power_spectrum_from_area` | $q \le 2\pi/\lambda_c$; without cutoff $q \le \pi\min(1/\Delta x, 1/\Delta y)$ |
| `autocorrelation_from_profile` | $r > \lambda_c/2$; with resampling, $r \ge \max(\lambda_c, 2\Delta x)/2$ |
| `autocorrelation_from_area` | $r \ge \max(\lambda_c, 2\overline{\Delta x})/2$, $\overline{\Delta x}$ the mean pixel size |
| `variable_bandwidth_from_*` | $\ell > \lambda_c$ |
| `integrate_psd`, `moment_power_spectrum`, … | $\lvert\mathbf q\rvert < 2\pi/\lambda_c$ |
| nonuniform, `'fft'` | as for the interpolated uniform line scan, plus the `short_cutoff` of the point spacings |
| nonuniform, `'brute-force'` | not supported, requires `reliable=False` |

With `'log'` collocation, the outermost bins extend $0.01$ decades beyond
these limits.

The ACF is cut at $\lambda_c/2$ because $\lambda_c$ is the length of the
second-derivative stencil used to determine it, whose points are spaced by
$\lambda_c/2$, while $A(r)$ compares heights at distance $r$ (Sanner et al.,
2022). If no reliable data remains, `NoReliableDataError` is raised.

## Moments of the PSD

```python
m = topography.moment_power_spectrum(order=0, window=None, reliable=True)
I = topography.integrate_psd(factor=lambda q: 1, window=None, reliable=True)
I = topography.integrate_psd_from_profile(factor=lambda qx: 1, window=None, reliable=True)
```

The moment of order $\alpha$ of the PSD is

$$
m_\alpha = \frac{1}{(2\pi)^2}\int d^2q\, \lvert\mathbf q\rvert^\alpha\, C^\text{2D}(\mathbf q)
\approx \frac{1}{L_xL_y}\sum_{\mathbf q}\lvert\mathbf q\rvert^\alpha\, C^\text{2D}(\mathbf q)
$$

for maps and

$$
m_\alpha = \frac{1}{2\pi}\int_{-\infty}^{\infty} dq\, \lvert q\rvert^\alpha\, C^\text{1D}(q)
\approx \frac{1}{L}\sum_{q}\lvert q\rvert^\alpha\, C^\text{1D}(q)
$$

for line scans. The sums run over all wavevectors of the FFT, i.e. the
full rectangular Brillouin zone including $q = 0$. They use the unbinned PSD,
since integrating a binned PSD adds the error of the binning.
`integrate_psd` evaluates the sum for an arbitrary weight $f$, called with
$\lvert\mathbf q\rvert$ or, if it takes two arguments, with $(q_x, q_y)$;
`integrate_psd_from_profile` does the same with the 1D PSDs of the profiles
along $x$ and averages over profiles.

Because the Fourier transform of a derivative is $iq\,\tilde h$, Parseval's
theorem relates the moments to the rms of the topography and its derivatives:

| Moment | Line scan | Map |
|-|-|-|
| $m_0$ | $\langle h^2\rangle$ | $\langle h^2\rangle$ |
| $m_2$ | $(h'_\text{rms})^2$ | $\langle\lvert\nabla h\rvert^2\rangle$ (rms gradient squared) |
| $m_4$ | $(h''_\text{rms})^2$ | $\langle(\nabla^2 h)^2\rangle = 4\,(h''_\text{rms})^2$, with mean curvature $\nabla^2 h/2$ |

$m_0$ includes the $q = 0$ term and equals $h_\text{rms}^2$ only for
topographies with zero mean. The moments correspond to spectral derivatives;
the real-space parameters in [statistics](statistics.md) use finite
differences, whose values differ at the shortest wavelengths.

```python
print(np.isclose(topo.moment_power_spectrum(0), np.mean(topo.heights() ** 2)))
print(np.sqrt(topo.moment_power_spectrum(2)), topo.rms_gradient())
print(np.sqrt(topo.moment_power_spectrum(4)) / 2, topo.rms_curvature_from_area())
print(np.sqrt(topo.integrate_psd_from_profile(lambda qx: qx ** 2)), topo.Rdq())
```

```
True
0.07157361109307538 0.07078263935587487
10.975631867211009 10.426450766162928
0.04973938025862655 0.049180525300477014
```

For a PSD that is only available on collocation points, e.g. a resampled PSD,
`SurfaceTopography.Generic.Moments` integrates with the trapezoidal rule:

| Function | Moment |
|-|-|
| `compute_1d_moment(q, C, order)` | $\frac{1}{\pi}\int_0^\infty dq\, q^\alpha C^\text{1D}(q)$ (one-sided $q$, both branches) |
| `compute_iso_moment(q, C, order)` | $\frac{1}{2\pi}\int_0^\infty dq\, q^{\alpha+1} C^\text{iso}(q)$ |

Both accept `cumulative=True` to return the integral up to each $q$.

## Half-derivative and elastic energy

```python
var = topography.variance_half_derivative(window=None, reliable=True)
U = topography.elastic_energy(contact_modulus, window=None, reliable=True)
```

The variance of the half derivative is the moment of order one,

$$
\langle\lvert\nabla^{1/2}h\rvert^2\rangle
= \frac{1}{(2\pi)^2}\int d^2q\, \lvert\mathbf q\rvert\, C^\text{2D}(\mathbf q) = m_1 .
$$

It determines the elastic energy per unit area stored in an elastic half-space
with contact modulus $E^*$ that is pressed into full contact with the
topography,

$$
\frac{U}{A} = \frac{E^*}{4}\,\langle\lvert\nabla^{1/2}h\rvert^2\rangle .
$$

The variance of the half derivative is defined by the spectral expression
above, in analogy to the first and second derivatives
($\langle\lvert\nabla h\rvert^2\rangle = m_2$,
$\langle\lvert\nabla^2 h\rvert^2\rangle = m_4$). The energy has the units of
`contact_modulus` times length.

**Maps.** On a topography map, the same quantity follows from the radially
averaged height-difference autocorrelation function $A(r)$ (see
[Height-difference autocorrelation](#height-difference-autocorrelation)),

$$
\langle\lvert\nabla^{1/2}h\rvert^2\rangle = \int_0^\infty dr\,\frac{A(r)}{r^2},
$$

because $\int_0^\infty dr\,[1-J_0(qr)]/r^2 = q$. This identity holds for any
surface; it does not require isotropy. It is computed by
`variance_half_derivative_via_autocorrelation_from_area`.

**Line scans.** A line scan does not contain the areal quantity, so any
estimate from line scans needs an assumption about the surface perpendicular
to the scan. SurfaceTopography assumes an isotropic surface, which gives

$$
\langle\lvert\nabla^{1/2}h\rvert^2\rangle
= \frac{1}{2}\int_0^\infty dq\, q\, C^\text{1D}(q)
$$

for `variance_half_derivative` and `elastic_energy` of line scans,
`variance_half_derivative_via_autocorrelation_from_profile` and
`ciso_moment(order=1)` of containers. (A surface that is invariant
perpendicular to the scan would give $2/\pi$ times this value.) The estimate
is the half derivative of the map from which the line scans were taken. It can
be written
as the moment of order one of the isotropic PSD obtained with the relation
$C^\text{iso}(q) = \frac{\pi}{q}C^\text{1D}(q)$ (see
[Isotropic PSD from the 1D PSD](containers.md#isotropic-psd-from-the-1d-psd)).
That relation holds pointwise only for a self-affine surface with Hurst
exponent $H=\tfrac12$; it is used for all surfaces, because a measured surface
is not self-affine and has no well-defined Hurst exponent. For the moment of
order one, its error cancels: inserting the projection of an isotropic PSD,
$C^\text{1D}(q_x) = \frac{1}{\pi}\int_{q_x}^\infty dq\,
q\,C^\text{iso}(q)/\sqrt{q^2-q_x^2}$, and exchanging the order of integration
gives

$$
\frac{1}{2}\int_0^\infty dq_x\, q_x\, C^\text{1D}(q_x)
= \frac{1}{2\pi}\int_0^\infty dq\, q\, C^\text{iso}(q)
\underbrace{\int_0^q dq_x\, \frac{q_x}{\sqrt{q^2-q_x^2}}}_{=\,q}
= \frac{1}{2\pi}\int_0^\infty dq\, q^2\, C^\text{iso}(q),
$$

which is the areal $\langle\lvert\nabla^{1/2}h\rvert^2\rangle$ of any isotropic
surface, independent of $H$ and of the shape of the PSD. For the moments of
order zero and two, the same relation is off by the constant factors $\pi/2$
and $\pi/4$.

The profile autocorrelation function gives the isotropic estimate directly:
with $\int_0^\infty dr\,[1-\cos(q_xr)]/r^2 = \frac{\pi}{2}\lvert q_x\rvert$,

$$
\int_0^\infty dr\,\frac{A(r)}{r^2}
= \frac{1}{2}\int_0^\infty dq\, q\, C^\text{1D}(q).
$$

`variance_half_derivative_via_autocorrelation_from_profile` evaluates this
integral. Unlike the PSD of a nonperiodic line scan, $A(r)$ needs no window.

**Finite data.** The autocorrelation and spectral routes agree in the
continuum limit. On a finite measurement they differ because

- the integral over $A(r)$ ends at the largest distance $r_\text{max}$, which
  leaves out a contribution of order $\langle h^2\rangle/r_\text{max}$;
- below the smallest distance $r_0$ the slope is assumed constant, which adds
  $A(r_0)/r_0$, while the spectral route drops wavevectors above the
  reliability cutoff;
- the PSD of nonperiodic data is computed with a window.

Periodic surfaces need no window, so comparing both routes on periodic
surfaces isolates the first two effects. The autocorrelation functions raise `UndefinedDataError` for data
with undefined (masked) points.

```python
print(topo.variance_half_derivative())                 # mm
print(topo.elastic_energy(contact_modulus=1e3))        # with E* in MPa: N/mm
```

```
4.1522715744081076e-05
0.010380678936020269
```

## References

- T. D. B. Jacobs, T. Junge, L. Pastewka, *Quantitative characterization of
  surface topography using spectral analysis*, Surf. Topogr. Metrol. Prop. 5,
  013001 (2017), [doi:10.1088/2051-672X/aa51f8](https://doi.org/10.1088/2051-672X/aa51f8)
  (PSD conventions; `power_spectrum_from_profile`, `power_spectrum_from_area`)
- A. Wang, M. H. Müser, *On the usefulness of the height-difference-autocorrelation
  function for contact mechanics*, Tribol. Int. 123, 224–233 (2018),
  [doi:10.1016/j.triboint.2018.02.002](https://doi.org/10.1016/j.triboint.2018.02.002)
  (`autocorrelation_from_profile`, `autocorrelation_from_area`)
- A. Sanner, W. G. Nöhring, L. A. Thimons, T. D. B. Jacobs, L. Pastewka,
  *Scale-dependent roughness parameters for topography analysis*, Appl. Surf.
  Sci. Adv. 7, 100190 (2022),
  [doi:10.1016/j.apsadv.2021.100190](https://doi.org/10.1016/j.apsadv.2021.100190)
  (relation between reliability cutoff and ACF distance)
