# Generating topographies

```python
from SurfaceTopography.Generation import fourier_synthesis
from SurfaceTopography.Models.SelfAffine import SelfAffine
from SurfaceTopography import make_sphere, PlasticTopography
```

Synthetic topographies are used to test analysis methods against known
statistics and as input for simulations, e.g. of contact mechanics. This page
describes random self-affine topographies with a prescribed power-spectral
density (PSD), the analytic model `SelfAffine` of such a PSD, and
deterministic geometries. The definitions of the PSD and of the Fourier
transform are those of [spectral analysis](spectral.md).

## Fourier synthesis

```python
fourier_synthesis(nb_grid_pts, physical_sizes, hurst=None, rms_height=None,
                  rms_slope=None, c0=None, psd=None, short_cutoff=None,
                  long_cutoff=None, rolloff=1.0,
                  amplitude_distribution=lambda n: np.random.normal(size=n),
                  periodic=True, rfn=None, kfn=None, progress_callback=None,
                  unit=None, info={})
```

`fourier_synthesis` returns a `Topography` for `nb_grid_pts` $=(N_x, N_y)$
and a `UniformLineScan` for `nb_grid_pts` $=(N,)$. It implements the Fourier
filtering algorithm of Ramisetti et al.,
[J. Phys.: Condens. Matter 23, 215004 (2011)](https://doi.org/10.1088/0953-8984/23/21/215004),
and Jacobs, Junge, Pastewka,
[Surf. Topogr.: Metrol. Prop. 5, 013001 (2017)](https://doi.org/10.1088/2051-672X/aa51f8).

### Algorithm

The heights are the inverse discrete Fourier transform of random
coefficients on the wavevectors of the periodic grid,
$\mathbf q = (2\pi k/L_x,\, 2\pi l/L_y)$ with integers $k$, $l$ in
$(-N_x/2, N_x/2]$, $(-N_y/2, N_y/2]$. Each coefficient is

$$
\tilde h(\mathbf q) = \sqrt{A\,C(q)}\; a_\mathbf q\, e^{i\phi_\mathbf q},
$$

with $A = L_xL_y$ ($A = L$ for line scans), $q = |\mathbf q|$, random
amplitudes $a_\mathbf q$ drawn from `amplitude_distribution` (default:
standard normal) and phases $\phi_\mathbf q$ uniformly distributed in
$[0, 2\pi)$. With $\tilde h(\mathbf q)=\Delta x\Delta y\sum_{ij}h_{ij}e^{-i\mathbf q\cdot\mathbf r_{ij}}$
and $C^\text{2D}(\mathbf q) = |\tilde h(\mathbf q)|^2/A$, the expectation
value of the PSD of the result is

$$
\langle C^\text{2D}(\mathbf q)\rangle = \langle a^2\rangle\, C(q) = C(q),
$$

i.e. the result has the requested PSD on average; any amplitude distribution
with $\langle a^2\rangle = 1$ preserves this. For line scans the same holds
for $C^\text{1D}(q) = |\tilde h(q)|^2/L$. Coefficients are generated for
the half-space $q_y \ge 0$ only; the other half follows from
$\tilde h(-\mathbf q) = \tilde h(\mathbf q)^*$ (real heights), and
coefficients that must be real ($\mathbf q = 0$ and Nyquist wavevectors) are
replaced by their real part. A single inverse real-to-complex FFT yields the
heights.

### The PSD

The target PSD $C(q)$ is either

- **self-affine**,
  $C(q) = C_0 q^{-2-2H}$ for maps and $C(q) = C_0 q^{-1-2H}$ for line scans,
  with Hurst exponent $H$ (`hurst`); or
- an arbitrary function `psd(q)` of the magnitude of the wavevector, which
  must return $C^\text{2D}$ for maps and $C^\text{1D}$ for line scans.

It is modified by the cutoffs:

$$
C_\text{synth}(q) = \begin{cases}
\text{rolloff}^2\, C(q_L) & q < q_L\\
C(q) & q_L \le q \le q_s\\
0 & q > q_s
\end{cases}
$$

with $q_s = 2\pi/\lambda_s$ from `short_cutoff` (default: $q_s = \pi/\Delta x$,
the Nyquist wavevector; for maps the smaller of the two directions) and
$q_L = 2\pi/\lambda_L$ from `long_cutoff` (default: no long cutoff). `rolloff=1`
gives a PSD that is flat below $q_L$, `rolloff=0` one that vanishes there.
For the self-affine PSD, the coefficient at $\mathbf q = 0$ is zero, so the
mean height is zero; `psd` is evaluated at $q=0$ and determines the
coefficient there. `psd` cannot be combined with `hurst`, `c0`,
`rms_height`, `rms_slope` or `long_cutoff`.

### Normalization

The prefactor $C_0$ is given directly (`c0`) or determined from `rms_height`
or `rms_slope` by `self_affine_prefactor`. For maps,

$$
h_\text{rms}^2 = \frac{1}{2\pi}\int_{q_L}^{q_s} q\,C(q)\,\mathrm dq
= \frac{C_0}{4\pi H}\left(q_L^{-2H} - q_s^{-2H}\right),\qquad
{h'_\text{rms}}^2 = \frac{1}{2\pi}\int_{q_L}^{q_s} q^3 C(q)\,\mathrm dq
= \frac{C_0}{4\pi(1-H)}\left(q_s^{2-2H} - q_L^{2-2H}\right),
$$

and for line scans

$$
h_\text{rms}^2 = \frac{1}{\pi}\int_{q_L}^{q_s} C(q)\,\mathrm dq
= \frac{C_0}{2\pi H}\left(q_L^{-2H} - q_s^{-2H}\right),\qquad
{h'_\text{rms}}^2 = \frac{C_0}{2\pi(1-H)}\left(q_s^{2-2H} - q_L^{2-2H}\right),
$$

which are solved for $C_0$. Here $q_L = 2\pi/\lambda_L$ if `long_cutoff` is
given and otherwise $q_L = 2\pi/L$ (for maps with the larger of
$2\pi/L_x$, $2\pi/L_y$). $h'_\text{rms}$ is the rms gradient for maps and the
rms slope for line scans. The rms height requires $H>0$, the rms slope
$H<1$; at these limits the integrals diverge logarithmically. The roll-off
region $q<q_L$ is not included, so `rolloff > 0` together with `long_cutoff`
adds to the rms values.

The normalization uses the continuous integral, while a realization contains
the discrete wavevectors of the grid. By Parseval's theorem the expected
variance of a realization is

$$
\langle h_\text{rms}^2 \rangle = \frac{1}{A}\sum_{\mathbf q \ne 0} C_\text{synth}(q),
$$

which differs from the integral when few long-wavelength modes dominate,
i.e. for large $H$ and without a long cutoff (for $N=128$, $H=0.8$ and no
long cutoff, $\langle h_\text{rms}^2\rangle$ is about 1.7 times the
requested value). Moreover, the rms height of an individual realization
fluctuates strongly because it is dominated by these few modes. The rms
slope is dominated by the many short-wavelength modes and is accurate. If a
topography with an exact rms height is needed, rescale it:

```python
import numpy as np
from SurfaceTopography.Generation import fourier_synthesis

np.random.seed(0)
t = fourier_synthesis((256, 256), (1.0, 1.0), hurst=0.8, rms_height=1.0)
t = t.scale(1.0 / t.Sq()).squeeze()
print(t.Sq())  # 1.0
```

### Periodicity

The heights are periodic by construction because they are a discrete Fourier
series on the grid. `periodic` only sets the `is_periodic` flag of the
result (default `True`). A nonperiodic topography with the same statistics is
obtained by generating a larger periodic one and cutting out a part, e.g.
with numpy slicing of `heights()`.

### Randomness, cost and memory

Phases and amplitudes are drawn from numpy's global random number
generator; seed it with `np.random.seed` for reproducible results. The cost
is dominated by the inverse FFT, $O(N_xN_y\log(N_xN_y))$. The real-space
array ($N_xN_y$ floats) and the Fourier-space array
($N_x(N_y/2+1)$ complex numbers) can be memory-mapped files (`rfn`,
`kfn`), which allows maps that do not fit into memory. The 2D inverse FFT is
carried out as one-dimensional transforms column by column and row by row,
which works on memory-mapped arrays and allows `progress_callback(i, n)` to
report progress.

`fourier_synthesis` runs serially; for MPI-parallel computations generate
the topography on every process with the same seed and
[decompose it](topographies.md#mpi-parallel-topographies).

### Examples

Topography map with rms slope 0.1, Hurst exponent 0.8 and a short-wavelength
cutoff at four pixels:

```python
np.random.seed(1)
t = fourier_synthesis((512, 512), (1.0, 1.0), hurst=0.8, rms_slope=0.1,
                      short_cutoff=4 / 512, unit="µm")
print(t.rms_gradient())  # ≈ 0.1
```

Line scan with given prefactor, long-wavelength cutoff and flat roll-off:

```python
l = fourier_synthesis((4096,), (10.0,), hurst=0.5, c0=1e-3, long_cutoff=2.0)
print(type(l).__name__, l.is_periodic)  # UniformLineScan True
```

Prescribed PSD, here a Gaussian $C^\text{2D}(q) = e^{-q^2/q_0^2}$:

```python
t = fourier_synthesis((256, 256), (1.0, 1.0), psd=lambda q: np.exp(-(q / 50) ** 2))
```

## The SelfAffine model

```python
SelfAffine(cr, rolloff_wavelength, hurst_exponent, longcut_wavelength=np.inf,
           shortcut_wavelength=0, unit=None)
```

`SelfAffine` describes the isotropic PSD of a self-affine topography with
roll-off,

$$
C^\text{iso}(q) = \begin{cases}
C_r & q_L < q < q_r\\
C_r\,(q/q_r)^{-2-2H} & q_r \le q < q_s\\
0 & \text{otherwise}
\end{cases}
$$

with roll-off wavevector $q_r = 2\pi/\lambda_r$, long-wavelength cutoff
$q_L = 2\pi/\lambda_L$ (default $\lambda_L = \infty$, $q_L = 0$) and
short-wavelength cutoff $q_s = 2\pi/\lambda_s$ (default $\lambda_s = 0$,
$q_s = \infty$). In the self-affine region $C^\text{iso}(q) = C_0
q^{-2-2H}$ with $C_0 = C_r q_r^{2+2H}$. `power_spectrum_isotropic(q)`
evaluates $C^\text{iso}(q)$.

### Moments and rms parameters

The variance of the derivative of (fractional) order $\alpha$ is

$$
\left[h^{(\alpha)}_\text{rms}\right]^2 = \frac{1}{2\pi}\int_{q_L}^{q_s}
q^{1+2\alpha}\, C^\text{iso}(q)\,\mathrm dq ,
$$

where $\alpha=0$ gives $h_\text{rms}^2$, $\alpha=1$ the mean square gradient
${h'_\text{rms}}^2 = \langle|\nabla h|^2\rangle$ and $\alpha=2$ the mean
square Laplacian $\langle(\nabla^2 h)^2\rangle$. For $q_L \le q_r \le q_s$
the integral splits into the roll-off and the self-affine region,

$$
\left[h^{(\alpha)}_\text{rms}\right]^2 =
\frac{C_r}{2\pi}\,\frac{q_r^{2+2\alpha} - q_L^{2+2\alpha}}{2+2\alpha}
+ \frac{C_0}{2\pi}\,\frac{q_s^{2\alpha-2H} - q_r^{2\alpha-2H}}{2\alpha-2H},
$$

where the second term becomes $\frac{C_0}{2\pi}\ln(q_s/q_r)$ for
$\alpha = H$. It is finite for $q_s\to\infty$ only if $\alpha < H$. For the
rms height ($\alpha=0$) this is

$$
h_\text{rms}^2 = \frac{C_r q_r^2}{4\pi}\left[1 + \frac{1}{H}
- \left(\frac{q_L}{q_r}\right)^2 - \frac{1}{H}\left(\frac{q_s}{q_r}\right)^{-2H}\right],
$$

and for the rms gradient ($\alpha=1$, $H<1$)

$$
{h'_\text{rms}}^2 = \frac{C_r q_r^4}{4\pi}\left[\frac{1}{2}
- \frac{1}{2}\left(\frac{q_L}{q_r}\right)^4
+ \frac{(q_s/q_r)^{2-2H} - 1}{1-H}\right].
$$

`variance_derivative(order, shortcut_wavelength=None,
longcut_wavelength=None)` evaluates $[h^{(\alpha)}_\text{rms}]^2$; it also
handles cutoffs outside the roll-off region. The optional wavelengths
restrict the integration range further (only a larger short cutoff or a
smaller long cutoff than the model's takes effect), which gives scale-dependent
parameters. `rms_height(shortcut_wavelength=None, longcut_wavelength=None)` is
the square root of `variance_derivative(0)` with the same cutoff arguments.

```python
from SurfaceTopography.Models.SelfAffine import SelfAffine

m = SelfAffine(cr=1.0, rolloff_wavelength=0.2, hurst_exponent=0.8,
               longcut_wavelength=1.0, shortcut_wavelength=0.01)
print(np.sqrt(m.variance_derivative(0)))  # rms height, 13.14
print(np.sqrt(m.variance_derivative(1)))  # rms gradient, 967.3
```

### Profiles

`power_spectrum_profile(q)` approximates the 1D PSD of line scans through
the isotropic topography by

$$
C^\text{1D}(q) \approx \frac{q}{\pi}\, C^\text{iso}(q)\sqrt{1-(q/q_s)^2}
$$

in the self-affine region and by $C_r q_r/\pi$ in the roll-off region. The
exact projection,
$C^\text{1D}(q_x) = \frac{1}{2\pi}\int \mathrm dq_y\,C^\text{iso}\big(\sqrt{q_x^2+q_y^2}\big)$,
gives for a pure power law
$C_0 q^{-1-2H}\,\Gamma(H+\frac12)/\big(2\sqrt\pi\,\Gamma(H+1)\big)$; the
approximation equals it for $H = 1/2$ and $q_s\to\infty$.

### Realizations

`generate_roughness(pixel_size, n_pixels, seed, shortcut_wavelength=None,
longcut_wavelength=None, **kwargs)` returns a square periodic `Topography`
with $N_x = N_y =$ `n_pixels` and $\Delta x =$ `pixel_size` by passing
`power_spectrum_isotropic` as `psd` to `fourier_synthesis`, after seeding
numpy's random number generator with `seed`:

```python
t = m.generate_roughness(pixel_size=0.005, n_pixels=512, seed=0)
print(t.physical_sizes)  # (2.56, 2.56)
```

## Special geometries

### Spheres and paraboloids

```python
make_sphere(radius, nb_grid_pts, physical_sizes, centre=None,
            standoff="undefined", offset=0, periodic=False, kind="sphere",
            nb_subdomain_grid_pts=None, subdomain_locations=None,
            communicator=MPI.COMM_WORLD)
```

`make_sphere` returns a spherical cap with apex at height `offset` (default
zero),

$$
h(\mathbf r) = \begin{cases}
\sqrt{R^2 - r^2} - R & r \le R\\
-R - \text{standoff} & r > R
\end{cases}
$$

where $R$ is `radius` and $r$ the lateral distance from `centre` (default:
the centre of the domain). `kind="paraboloid"` gives the paraboloid
$h = -r^2/(2R)$, the small-slope approximation of the sphere, defined
everywhere. With `standoff="undefined"` (default) the points outside the
sphere are [undefined](topographies.md#undefined-data). With `periodic=True`,
$r$ is the distance to the nearest periodic image of the centre, so the
sphere wraps around the boundaries. A scalar `nb_grid_pts` returns a
`UniformLineScan`, two values a `Topography`; the subdomain arguments create
a [domain-decomposed](topographies.md#mpi-parallel-topographies) map
directly.

```python
from SurfaceTopography import make_sphere

s = make_sphere(2.0, (64, 64), (4.0, 4.0), kind="paraboloid")
c = make_sphere(2.0, (64, 64), (4.0, 4.0), standoff=10.0)
print(c.heights().min())  # -12.0
```

### Topographies from functions

```python
from SurfaceTopography.Special import make_topography_from_function

t = make_topography_from_function(lambda x, y: np.cos(2 * np.pi * x),
                                  (1.0, 1.0), nb_grid_pts=(32, 32),
                                  centre="middle", periodic=True)
```

`make_topography_from_function(fun, physical_sizes, nb_grid_pts=None,
subdomain_locations=None, nb_subdomain_grid_pts=None, centre=(0, 0), **kwargs)`
evaluates $h(x, y) = f(x - c_x, y - c_y)$ on the grid positions;
`centre="middle"` sets $(c_x, c_y) = (L_x/2, L_y/2)$. Further keyword
arguments (`periodic`, `unit`, `info`, `communicator`) are passed to
`Topography`.

### Plastic deformation

`PlasticTopography(topography, hardness, plastic_displ=None)` is a pipeline
step that adds a plastic displacement field $u_\text{pl}$ to the heights of a
uniform topography, $h = h_0 + u_\text{pl}$. It is meant for contact
mechanics codes that limit the local pressure to the `hardness` and store
the resulting permanent deformation in the attribute `plastic_displ` (an
array of shape `nb_subdomain_grid_pts`). `plastic_area` is the area of the
grid points with nonzero plastic displacement, `undeformed_profile()` returns
$h_0$.

```python
from SurfaceTopography import PlasticTopography

p = PlasticTopography(s, hardness=1.0)
p.plastic_displ = np.where(s.heights() < -0.5, 0.1, 0.0)
print(p.plastic_area)
```
