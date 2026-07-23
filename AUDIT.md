# SurfaceTopography codebase audit

Date: 2026-07-23. Scope: full repository at commit `119cb66` — core topography
classes, `Uniform/`, `Nonuniform/`, `Generic/`, `IO/`, `Container/`,
`Support/`, `Models/`, the C++ extensions in `cpp/`, and packaging. Findings
were verified by re-reading code, cross-checking against the test suite, and —
where marked *(verified by execution)* — by running the code against the
installed extension. Line numbers refer to this commit.

Severity legend:
- **HIGH** — silently wrong scientific results or data corruption on realistic inputs.
- **MEDIUM** — crashes on valid inputs, wrong results under non-default but supported options, resource/metadata correctness.
- **LOW** — edge cases, performance, documentation drift, code health.

> **Status update (2026-07-23):** All findings in section 1 (the
> wrong-results tier, §1.1–§1.17) and sections §2.1–§2.4 have been
> **fixed** on this branch, with regression tests; see the commit history
> following the commit that added this report. Exceptions: the latent
> non-square axis-pairing in DATX/LEXT noted under §1.3 is *not* changed
> (unprovable without a non-square reference file); of the packaging items
> (T8), the broken `pytest-flake8` plugin and the `tiffile` alias package
> were fixed because they broke CI, while the remaining build-system
> hardening (meson `python` alias, glob check for meson file lists) is
> still open. Section 3 (low severity) has also been **fixed**, with two
> deliberate exceptions: the NMS height scale divisor `2**16 - 2` (§3,
> IO) was left untouched because no reference implementation or fixture
> is available to decide between `2**16 - 2` and `2**16 - 1`, and the
> DATX/LEXT axis-pairing exception above applies here too. Of section 4,
> the actionable items have been addressed: property-based invariant
> tests (transpose/translate/scale/origin invariance, decorator geometry
> consistency, bearing-area CDF properties) were added in
> `test/test_invariants.py` (T1/T3/T6). The meson version-discovery
> interpreter (T8) must remain the bare `python`: switching it to
> `python3` broke all Windows wheel builds, because pip's isolated build
> environment provides no `python3.exe` on Windows and the name resolved
> to an interpreter without the DiscoverVersion build requirement. This
> is now documented in `meson.build`; building with bare meson outside a
> venv requires a `python` alias. The
> proposed dedup of the three `bandwidth` implementations and of the
> `compute_1d_moment`/`compute_iso_moment` pair was assessed and
> deliberately rejected: the `bandwidth` variants differ semantically
> (pixel size vs. mean point spacing vs. unit-converting container
> variant) and the moment helpers are seven lines each, so a shared
> helper would add indirection without reducing risk. The larger
> refactoring themes (shared row-major reader helper for IO, unifying
> the Uniform/Nonuniform/Container analysis-function triplication)
> remain proposals — see section 4. The performance tier (§2.5) remains
> open. Note that the masked-data normalization fix
> (§2.2) changed the reference values of several IO tests whose fixtures
> contain undefined data points; the old values were dominated by a
> mean-normalization artifact.

---

## 1. High severity — silent data corruption / wrong results

### 1.1 Lazy `transpose()` does not swap `pixel_size` → wrong derivatives and PSDs *(verified by execution)*

`SurfaceTopography/UniformLineScanAndTopography.py:669` — `TransposedUniformTopography`
overrides `nb_grid_pts` and `physical_sizes` but inherits `pixel_size`,
`nb_subdomain_grid_pts` and `subdomain_locations` from
`DecoratedUniformTopography`, which delegates to the parent. For anisotropic
pixels every derivative-based quantity of a transposed topography is wrong,
with no error:

```
t = Topography(np.random.rand(64, 32), (2., 4.), periodic=True)
t.rms_gradient()                        -> 13.8586
t.transpose().rms_gradient()            -> 10.3769   (silently wrong)
t.transpose().squeeze().rms_gradient()  -> 13.8586   (correct)
```

`Uniform/Derivative.py:349,374` divides FFT frequencies by `pixel_size`. The
un-swapped `nb_subdomain_grid_pts` additionally breaks shape checks under MPI.
**Fix:** override `pixel_size`, `nb_subdomain_grid_pts`, `subdomain_locations`
to swap components.

### 1.2 `downsample()` reports parent pixel size → derivatives/PSD normalization wrong by the downsampling factor *(verified by execution)*

`SurfaceTopography/Uniform/Filtering.py:80-93` — `DownsampledUniformTopography`
overrides `nb_grid_pts` but inherits `pixel_size`/`area_per_pt` from the
parent. A 64×64 map of size 1.0 downsampled by 2 reports `pixel_size = 1/64`
(should be 1/32) and `area_per_pt` wrong by 4×; `derivative()` on the result is
wrong by ~the same factor. For non-divisible factors, sample positions are also
inconsistent with `positions()`. **Fix:** override `pixel_size`/`area_per_pt`;
shrink `physical_sizes` to `nb_grid_pts*fx*px` or reject non-divisible factors.

### 1.3 Non-square scans scrambled by wrong reshape order in four readers

The raw buffer of these formats is stored scan-line-major, i.e. C-order shape
`(ny, nx)`, but the readers reshape to `(nx, ny)`. Whenever `nx != ny` (partial
/ aborted scans, rectangular acquisition settings) scan lines are interleaved
into garbage, silently. All test fixtures for these formats are square, so the
test suite cannot see it:

- `SurfaceTopography/IO/DI.py:308-310` (Bruker Nanoscope)
- `SurfaceTopography/IO/EZD.py:221-223` (NanoSurf easyScan)
- `SurfaceTopography/IO/MI.py:156,181` (Molecular Imaging; additionally pairs the transposed shape with un-swapped sizes)
- `SurfaceTopography/IO/PS.py:211` (Park Systems TIFF)

**Fix:** `reshape(ny, nx)` then `.T` (the pattern FRT.py:623 and GWY.py:290
already use correctly). Add a non-square fixture per format.

Related latent axis-pairing issues for non-square maps (untested, ambiguous
rather than provably wrong): `IO/DATX.py:101,122,130,243-248` pairs the HDF5
row count with the X converter and hands data to `Topography` untransposed;
`IO/LEXT.py:114-117,231` same pattern (contrast JPK.py:350, which transposes
correctly and is verified by a non-square fixture).

### 1.4 MI reader: lateral sizes in meters labeled with the height unit (µm) — mixed quantities off by ~10⁶

`SurfaceTopography/IO/MI.py:117,181` — `xLength`/`yLength` are meters per the
MI spec (2e-05 in the test file), but the topography is constructed with
`unit=bufferUnit` (µm, which correctly applies to the heights via
`bufferRange`). A 20 µm × 20 µm scan is reported as 2×10⁻⁵ µm wide; rms slope,
PSD, curvature are off by ~10⁶. `test/IO/test_mi.py:64` asserts the wrong
value. **Fix:** convert lengths from m into `bufferUnit`.

### 1.5 BCR reader misinterprets `voidpixels` as the invalid-data marker value

`SurfaceTopography/IO/BCR.py:217-219` — per the BCR-STM spec (and Gwyddion),
`voidpixels` is the *count* of void pixels; the void marker is 32767 (int16)
or ~3.4e38 (float). Consequences: a file with `voidpixels = 0` masks every
pixel whose value is 0 (mass corruption); real void pixels are never masked;
a missing key crashes `topography()` with `KeyError`. Only works for the
single test file because its writer non-conformantly stored the marker value
in that field. **Fix:** treat as count, mask against the format's marker
constant, set `undefined_data` on `ChannelInfo`.

### 1.6 Channel selection returns the wrong channel's data in three readers

- `SurfaceTopography/IO/NMM.py:256-258` — every `ChannelInfo` is created with
  index 0 inside the scan loop, so `channels[i].topography()` always loads
  scan 1.
- `SurfaceTopography/IO/OIR.py:523-528` — late-binding closure: the reader
  lambda captures loop variables `uuid`/`nx`/`ny`/`dtype` by reference; with
  ≥2 height channels every channel returns the last channel's data. **Fix:**
  bind via default arguments.
- `SurfaceTopography/IO/OIR.py:563-597` — `POIRReader` concatenates
  sub-readers' channels but keeps per-file `_index`; `channels[1].topography()`
  on a two-file container returns file 0's data. **Fix:** renumber `_index`.

### 1.7 C++ triangle moments are mathematically wrong — all `uniform2d_*` moment bindings return garbage *(verified by execution)*

`cpp/moments.h:115-175` — `_TriangleMoment` has a sign error and a missing
`2/(h3−h1)` prefactor relative to the correct moment of the linearly
interpolated triangle distribution. Live results: `uniform2d_mean(const 5.0) =
0.0`; `uniform2d_variance` of a ±1 terraced surface = 0.0 (should be 1.0). The
generic order-4 path divides by `(h2−h1)`/`(h3−h2)` with no degeneracy guard →
NaN for any triangle with two equal corner heights (ubiquitous in quantized
data). `test/test_moments.py:140-166,354-358` *codifies* the broken behavior
("returns 0 due to triangle integration"). The bindings are exported but
currently unused inside the library. **Fix:** rederive the integral, add the
order-4 specialization, handle degenerate corners analytically, fix the tests.

### 1.8 Order-4 line-scan moments return 0 for flat segments *(verified by execution)*

`cpp/moments.h:34-40` — the `|h2−h1| < 1e-12` guard returns 0 where the
correct limit is `(order+1)·h^order`, and no order-4 specialization exists, so
`nonuniform_moment4`/`uniform1d_moment4` hit this path: `uniform1d_moment4(2.0
· ones) = 0.0` (true 16). Kurtosis of surfaces with flat segments is
systematically underestimated; the absolute epsilon is also unit-dependent
(meters-scale heights zero out entirely). **Fix:** exact factored form
`h1⁴+h1³h2+h1²h2²+h1h2³+h2⁴`; drop the epsilon guard.

The same removable-singularity bug exists in pure Python:
`SurfaceTopography/Nonuniform/ScalarParameters.py:74` — `moment()` evaluates
`(h₂^{α+1} − h₁^{α+1})/(h₂ − h₁)` directly; any two equal adjacent heights
(quantized instrument data) give 0/0 → NaN which propagates to the whole
result. **Fix:** `np.where(dh == 0, (alpha+1)*h**alpha, ...)`.

### 1.9 `Bicubic.__call__` reads non-contiguous arrays as contiguous — wrong values and out-of-bounds reads *(verified by execution)*

`cpp/bicubic.cpp:362-363,385-388,409-421` — `py::array_t<double>::ensure` does
not force C-contiguity and strides are never consulted. A strided view
(`x[::2]`) gives values wrong by up to 0.31; a negative-stride view walks up
to `n−1` doubles past the allocation (heap OOB read / UB). **Fix:**
`py::array_t<double, py::array::c_style | py::array::forcecast>`.

Related copy-paste bug on the Python side, `SurfaceTopography/Uniform/Interpolation.py:167`:

```python
interp_derxy = interp_derxx / dx / dy   # should be interp_derxy
```

Every `derivative=2` user of `interpolate_bicubic` gets the xx-derivative
(rescaled with the wrong units) in place of the cross-derivative.

### 1.10 Periodic 2D bearing-area bounds built with flattened `np.roll` → corrupts `median()`, MAD statistics, robust detrending *(verified by execution)*

`SurfaceTopography/Uniform/BearingArea.py:126-133` — `np.roll(h, (-1, 0))`
without `axis=` flattens the array and rolls by the *sum* of the shifts, so
the element min/max bounds pair wrong pixels. On periodic 8×8 maps, 252/3400
tested heights fell outside `[lower, upper]`; `bisect()` in
`Generic/RobustStatistics.py` relies on these bounds and returned a "median"
with `bearing_area(median) = 0.626` instead of 0.5. Affects `median()`,
`mad_height()`, `mad_polyfit()` and `'median'`/`'mad-tilt'`/`'mad-curvature'`
detrending for periodic 2D topographies. **Fix:** `np.roll(h, -1, axis=0)`,
`np.roll(h, -1, axis=1)`, `np.roll(h, (-1, -1), axis=(0, 1))`.

### 1.11 `to_uniform()` produces garbage for line scans whose x-axis does not start at 0

`SurfaceTopography/Nonuniform/Converters.py:158-165` — the uniform grid always
spans `[0, s]` while the parent's positions may start anywhere; `np.interp`
clamps outside the parent range, so a scan with `x[0] = 5` gets most of the
grid filled with the constant `h[0]`. This silently poisons the default
`'fft'` paths of `power_spectrum_from_profile`, `autocorrelation_from_profile`,
`derivative(distance=...)`, scale-dependent statistics, and the
scanning-probe reliability cutoff. Masked in practice only because the
XYZ/Mitutoyo readers shift `x -= min(x)`; the public `NonuniformLineScan(x, y)`
constructor does not. **Fix:** interpolate at `x + parent.positions()[0]`.

Two more `x[0] == 0` assumptions in the same subsystem:

- `SurfaceTopography/Nonuniform/Detrending.py:107` — `'slope'` detrend uses
  `a0 = mean(h)` (missing the `−a1·⟨x⟩` correction) and an unweighted
  `derivative(1).mean()` where the rms-minimizing constant is the
  length-weighted mean `(h[-1]−h[0])/L`. Example: `x=[0,1,10], h=[0,1,1]` →
  slope 0.5 instead of 0.1.
- `SurfaceTopography/Nonuniform/PowerSpectrum.py:107-109` — the Hann window is
  evaluated at absolute `x`, not `x − x.min()`; for offset scans the window is
  ≈1.5 at the endpoints instead of 0, defeating leakage suppression (and
  nonperiodic scans get this window by default).

### 1.12 Scale-dependent curvature stencil silently wrong with default arguments

`SurfaceTopography/Generic/Curvature.py:64-72,104-112` — the pairing
`8·A[1+i] − 2·A[2+2i]` implements `8A(λ) − 2A(2λ)` only for a linearly spaced,
origin-anchored ACF grid, but `autocorrelation_from_profile` defaults to
`resampling_method='bin-average', collocation='log'` (log-spaced `r`), and
with `resampling_method=None` a reliability cutoff trims leading points and
shifts the origin. Both default and reliability paths therefore compute
curvatures from mismatched distance pairs. Existing tests only check
`np.max` loosely / "finishes without failing". **Fix:** compute the ACF on a
linear grid internally, apply the reliability mask after forming the stencil,
or interpolate `A(2λ)`.

### 1.13 ZAG containers return topographies backed by an already-closed stream

`SurfaceTopography/Container/IO/ZAG.py:156-175` +
`Container/IO/__init__.py:128-131` — `read_container()` does `with
open_container(fn) as reader:` and returns the lazy container after `__exit__`
closed the file stream; every subsequent element access reads from a closed
`ZipExtFile`. `CEReader` avoids this by re-opening the ZIP per read; ZAG does
not. Unnoticed because `test/IO/test_zag.py` is `@pytest.mark.skip` with a
hardcoded path `/home/pastewka/Downloads/zag-1.zag`. **Fix:** a re-opening
opener object like `CEFileOpener`; check in a fixture and unskip the test.

### 1.14 `fourier_synthesis` populates the q=0 mode with a spurious, unit-dependent amplitude *(verified by execution)*

`SurfaceTopography/Generation.py:283-292` — `q_sq[0] = 1.` (to avoid
`0**negative`) causes the DC amplitude to equal `C(q=1)`, which depends on the
arbitrary length unit. For 128×128, `rms_height=1.0`, realization means come
out at `4.4, −22.9, −13.7, 4.9, 19.4` — a mean offset that dwarfs the
requested roughness for any consumer that does not first subtract the mean.
**Fix:** zero the DC mode instead.

### 1.15 `make_sphere(..., periodic=True)` silently returns a nonperiodic topography *(verified by execution)*

`SurfaceTopography/Special.py:216-223` — neither constructor call passes
`periodic=periodic` (the wrapped radii are computed, but the flag is dropped).
The flag controls detrending defaults, PSD/ACF windowing and derivative
stencils. The existing test checks only height values. **Fix:** pass the flag
(and `communicator` in the 1D branch).

### 1.16 `scale_dependent_statistical_property` reports wrong distances with `scale_factor`

`SurfaceTopography/Generic/ScaleDependentStatistics.py:130-131` — `distance =
scale_factor * np.mean(physical_sizes)` contradicts the module's own
convention (`distance = scale_factor · n · pixel_size`); the returned abscissa
is too large by ~`nb_grid_pts/n` (e.g. 512×), and since this distance feeds
the reliability mask, unreliable scales are never filtered when `scale_factor`
is combined with reliability metadata.

### 1.17 Scan-line alignment `direction` semantics inverted

`SurfaceTopography/Uniform/ScanLineAlignment.py:66-68,171-181` — the docstring
says `'x'` means the fast scan runs along x, but the implementation iterates
`heights[i, :]`, and in this library the first index *is* x. Users with
standard fast-scan-x AFM maps (the default) get the per-line correction
applied across scan lines. The tests added with the feature encode the same
inversion. **Fix:** transpose for `'x'` instead of `'y'`.

---

## 2. Medium severity

### 2.1 Analysis correctness under non-default options

- `Support/__init__.py:50-52` (`fold_fft_half`, callers
  `Uniform/PowerSpectrum.py:105`, `Uniform/Autocorrelation.py:118`) — the
  final `/= 2` also halves the q=0 row, which has no mirrored partner;
  `power_spectrum_from_profile(resampling_method=None)` returns `C[0]` exactly
  half the true DC power *(verified by execution)*. Fix: halve `result[1:]`
  only.
- `Uniform/PowerSpectrum.py:129` — `min_value=q[1]` is applied after q=0 was
  already stripped, discarding the fundamental mode from 1D log-collocation
  resampling *(verified)*. Fix: `q[0]`.
- `Uniform/Derivative.py:449` — `"disabled"` vs accepted option `"disable"`:
  the guard is dead; `interpolation='disable'` with a fractional scale factor
  silently truncates the stencil step *and* divides by the fractional pixel
  size — doubly wrong instead of raising *(verified)*.
- `Uniform/Interpolation.py:53-68` — linear interpolator truncates toward zero
  instead of flooring; negative fractional positions on periodic topographies
  extrapolate (`interp(-1.5) → 10.5` outside data range) and corrupt the first
  pixels of scale-dependent slope/curvature with fractional scale factors
  *(verified)*. Fix: `np.floor(...).astype(int)`.
- `Uniform/Interpolation.py:205-227` — `interpolate_fourier` halves Nyquist
  rows/columns even when that dimension is not enlarged; interpolation onto
  the same grid is not the identity (max error 0.49 on unit-variance 8×8)
  *(verified)*. Fix: halve only when actually padding that dimension.
- `Uniform/Autocorrelation.py:227,242` — factor-2 inconsistency in the
  reliability cutoff between profile (`short_cutoff / 2`) and area
  (`short_cutoff`) ACF paths; the area path discards a factor-2 band of
  reliable short-distance data (the `other_cutoff` defaults also differ).
- `Nonuniform/VariableBandwidth.py:158-166` — the loop condition tests the
  *previous* magnification's minimum point count, so the last recorded level
  can contain segments below `nb_grid_pts_cutoff` (down to 2 points, rms
  exactly 0), biasing the smallest-bandwidth datum low. The uniform version
  checks before recording.
- `Nonuniform/VariableBandwidth.py:39,85,94` — absolute tolerance `tol=1e-6`
  compared against position differences is unit-dependent: in SI meters it
  exceeds entire subdivisions; in nm-scale coordinates it is effectively zero.
- `Models/SelfAffine.py:246-266` — `generate_roughness` resolves
  `longcut_wavelength` and then never uses it (always passes the model's
  rolloff to `fourier_synthesis`); the `order == hurst_exponent` branch of
  `variance_derivative` (line 218) also uses the wrong lower limit.
- `Generation.py:140-148` — `Hurst=1` with `rms_slope` (and `Hurst=0` with
  `rms_height`) produce 0/0 → all-NaN surfaces with only a RuntimeWarning
  *(verified)*. Fix: raise, or implement the logarithmic limits.

### 2.2 Masked / undefined data handled inconsistently

- `Uniform/Detrending.py:47-50` — `polyfit_line_scan` fits the raw buffer
  under the mask (`np.polyfit` on a masked array); the 2D counterpart
  correctly compresses. Verified: junk under 10 masked points shifts a
  unit-slope fit to coefficients ~1e5.
- `Uniform/VariableBandwidth.py:88-92,169-173` — `np.bincount(region_index,
  h * x**i)` strips masks; masked points contribute raw fill values with no
  `has_undefined_data` guard *(verified: detrended heights of magnitude 3e6)*.
- `Uniform/ScalarParameters.py:60-68,91-95` — `Rq`/`Sq` normalize by the total
  grid-point count including masked ones; a half-masked constant profile
  returns `Rq = 0.354` instead of 0 *(verified)*.
- `Uniform/GeometryAnalysis.py:112-114` — periodic wrap merge in
  `assign_patch_numbers_profile` uses `patch_ids[0]` (the *second* pixel's
  id); a masked patch spanning the wrap boundary comes back from
  `interpolate_undefined_data()` with one pixel still masked while
  `has_undefined_data` reports `False` *(verified)*.
- `Uniform/Imputation.py:110-118` — the Laplace-matrix neighbor lookup uses
  unconditional `np.roll`, wrapping across the boundary of *nonperiodic*
  topographies and coupling edge pixels to per-patch unknown #0 *(verified:
  wrong harmonic fill values for edge-touching patches)*.
- `IO/FromFile.py:196-215` (HGT) — SRTM voids (−32768) are not masked (they
  enter statistics as −32768 m spikes), and fabricated
  `physical_sizes=data.shape` makes it impossible to pass real sizes later
  (`MetadataAlreadyFixedByFile`).
- `IO/SDF.py:105-115,304-308` — ASCII variant never maps integer invalid
  markers (−32768/−2³¹) to NaN; only the literal `BAD` token. Binary path is
  correct. Also uses deprecated `np.fromstring`.
- `IO/X3P.py` — ignores the ISO 5436-2 `ValidPointsLink` mask and the CZ
  `Offset`; integer invalid points read as real heights. The writer casts NaN
  through `astype(uint16)`.

### 2.3 Crashes on valid inputs

- `Special.py:205-228` — `make_sphere(kind='paraboloid', standoff=1.0)` →
  `UnboundLocalError` (`standoff_val` only set in the sphere branch)
  *(verified)*.
- `UniformLineScanAndTopography.py:745-750` — `translate()` crashes for 1D
  line scans (`axis=1` roll), even with the default offset *(verified)*.
- `UniformLineScanAndTopography.py:707-709` — `transpose()` of a 1D line scan
  crashes in `positions()` (`too many values to unpack`) although 1D transpose
  is intended as identity *(verified)*.
- `HeightContainer.py:285-295,348-357` — `__eq__` raises (broadcast error /
  `AttributeError`) instead of returning `False` for mismatched grids or
  foreign types, and defining `__eq__` without `__hash__` makes all
  topographies unhashable *(verified)*.
- `Uniform/Filtering.py:229,274-276` — the default `filter_function` of
  `FourierFilteredUniformTopography` crashes with `isotropic=True` (wrong
  arity), has an operator-precedence bug if called with two args, and error
  paths `raise ("ValueError: ...")` raise a string (→ `TypeError`)
  *(verified)*.
- `Uniform/Interpolation.py:265-272` — `MirrorStichedTopography.positions()`
  is shape-inconsistent with `heights()` (10×10 vs 20×20), breaking
  `positions_and_heights()` *(verified)*.
- `IO/GWY.py:149-158` — any GWY file containing an object array (graphs /
  spectra) crashes twice over: the reader-dict entry doesn't accept
  `skip_arrays` and a missing comma makes `range(nb_items)` receive a tuple.
- `IO/GWY.py:204,250` — channel-index regex `([0-9])` accepts one digit;
  files with ≥11 channels crash with `TypeError`.
- `IO/FRT.py:460` — invalid struct format `'gint32'` (copied from Gwyddion C
  source); any FRT file containing block 0x00ae fails to open. Should be `'i'`.
- `IO/EZD.py:70-72` — magic check compares `bytes` to `str` (always False)
  *and* has inverted logic; net effect: no format validation at all.
- `IO/MI.py:146-152` — ASCII-data branch references undefined `encode_length`
  → `NameError` for text-format MI files the header parser explicitly detects.
- `IO/MI.py:162` — `if meta['scanUp']:` tests the *string* `'FALSE'` (truthy):
  scan-down images are also flipped.
- `IO/NC.py:287-296` — nonuniform line scans report `nb_grid_pts=(None,)`
  (reads the wrong dimension attribute) *(verified by write/read round trip)*.
- `IO/NC.py:452` — MPI guard `rank > 1` should be `rank > 0`: ranks 0 and 1
  both write the same file concurrently in non-decomposed runs.
- `IO/PLUX.py:118-150,177-180` — multi-layer files advertise N channels but
  `topography()` raises for any `channel_index != 0`.
- `IO/H5.py:64-65` — crashes with `AttributeError` on HDF5 files containing
  groups (`Group` has no `.shape`).
- `IO/DATX.py:119,127,135` — `and` where `or` is needed in converter
  validation (both conditions must fail to raise), plus a copy-paste error
  message in the Z-converter check.
- `Nonuniform/ScaleDependentStatistics.py:162-164` — the 2D single-distance
  fallback assigns `np.nan` into a *tuple* → `TypeError` when the threshold
  trips; a Python-list `distances` argument crashes against the reliability
  mask (`list > float`).
- `Generic/Slope.py:107` — `scale_dependent_slope_from_area` is registered on
  `NonuniformLineScanInterface`, which has no `autocorrelation_from_area` →
  guaranteed `AttributeError`.
- `HeightContainer.py:306-307` — `is_reentrant` crashes for scans with < 2
  points *(verified)*.
- `IO/Mitutoyo.py:101,139,163` — bare `IndexError` for spreadsheets without a
  roughness-metric cell or 1-point scans; locale-dependent `%b` date parsing;
  workbook handle never closed.
- `Support/UnitConversion.py:181` — `suggest_length_unit` crashes with
  `OverflowError` on all-zero data (`log10(0)`); `mangle_length_unit_ascii('Å')
  = 'A'` round-trips to an unconvertible unit.
- `Container/ScaleDependentStatistics.py:89-119,158-174` — the docstring
  example (`distance=[...]`) crashes with `TypeError: multiple values for
  'distance'`; scalar `distances` unsupported despite docs; inner loop reuses
  the outer loop variable.
- `Container/IO/__init__.py:130` — `read_container(**kwargs)` forwards kwargs
  that neither `CEReader.container` nor `ZAGReader.container` accepts.

### 2.4 Infrastructure and metadata correctness

- `SurfaceTopography/Metadata.py:33-44` — `InfoModel` silently *discards*
  arbitrary `info` keys (pydantic `extra='ignore'`) although docstrings
  throughout promise a free-form dict for third-party codes; explicit `None`
  values raise `ValidationError` (fields typed `str = None` instead of
  `Optional[str]`) *(verified)*. Fix: `extra='allow'` (or `'forbid'` to fail
  loudly) and `Optional[...]` annotations.
- `HeightContainer.py:184-189` — the base-class `DecoratedTopography.info`
  passes a `BaseModel` to `model_copy(update=)` which requires a dict →
  `AttributeError` under pydantic 2; currently shadowed by two copy-pasted
  correct overrides in the uniform/nonuniform subclasses *(verified)*. Fix in
  the base, delete both overrides.
- `HeightContainer.py:67-68`, `Container/SurfaceContainer.py:43-44` —
  `apply()` discards the function's return value in both dispatch bases
  *(verified)*. Add `return`.
- `Support/Bibliography.py:59-88` — `doi.dois`/`doi._n` are class-level
  globals with no try/finally: any exception inside a wrapped analysis leaves
  the decorator permanently "on", silently dumping DOIs into a user's set on
  all later calls; thread-unsafe *(verified)*. Fix: try/finally, or a
  `contextvars.ContextVar`.
- `Support/JSON.py:15-29,121-125` — `json.dump` bypasses the NaN handling
  (only `encode()` is overridden, not `iterencode()`; `IO/DZI.py:197` uses
  exactly `json.dump`); `nan_to_none` crashes on `nomask` and on 2-D masked
  arrays; Inf passes through as invalid JSON *(all verified)*.
- `Support/UnitConversion.py:242-247` — `find_length_unit_in_string` matches
  substrings of alias keys only: `'X Axis'` → `'Å'` (wrong unit, wrong
  scaling), `'Height (nm)'` → `None`; result feeds the XYZ reader's
  unit/height-scale *(verified)*.
- `IO/JPK.py:279-280` — every channel's `raw_metadata` contains the *last*
  TIFF page's metadata (stale loop variable `channel_metadata`; should be
  `channel`).
- `IO/ZON.py:227,243` — mutable default `info={}` is mutated
  (`info.update(...)`) and permanently polluted across calls — including into
  ZAG containers; the merge direction also lets file metadata overwrite
  user-supplied info, opposite of every other reader *(verified)*.
- `IO/MetroPro.py:124,167-239` — little-endian and native-endian struct fields
  in a big-endian format: heights are unaffected (offsets verified to line
  up), but `raw_metadata` values and the reported instrument serial are
  garbage. `_HEADER_FORMATS`/`_HEADER_SIZE*` are dead code.
- `IO/OPD.py:142-151,157` — named metadata blocks read fixed 2–4 bytes
  regardless of the directory-declared length (silent offset shift if they
  differ); missing `Wavelength` block → `NameError`.
- `IO/VK.py:258-261` — physical size uses `(n−1)·pixel` (interval convention)
  where the library convention (and Gwyddion) is `n·pixel`.
- `Container/IO/__init__.py:134-158` — `read_published_container` has no
  `raise_for_status()` (404 → confusing `KeyError`), no timeout, buffers the
  whole download in memory, and applies `**request_args` to only one of the
  two requests.
- `Container/IO/CE.py:74-80` — `CEFileOpener` never closes the `ZipFile`
  (relies on refcount GC) and re-parses the central directory on every lazy
  topography access.
- cpp — `eigen_helper.h:30`: `Eigen::Array<long,...>` breaks Windows wheels
  (LLP64 `long` is 32-bit; non-const `Ref` needs exact dtype match →
  `TypeError` from every `NonuniformLineScan.bearing_area()` call). Fix:
  `std::int64_t`.
- cpp — all bindings take non-const `Eigen::Ref`: read-only, int-dtype or
  F-ordered inputs are rejected although nothing is written; a read-only
  input array breaks `t.mean()`, `t.rms_height_from_profile()` etc. for
  nonuniform scans *(verified)*. Fix: `Eigen::Ref<const ...>`.
- cpp — `patchfinder.cpp:66-82`: single-`if` periodic wrap → OOB reads (and
  potentially writes) for stencil offsets ≥ grid dimension *(verified)*;
  `shortest_distance` correctly uses `while` loops.
- cpp — `moments.h:186`, `bearing_area.cpp:140`, `patchfinder.cpp:213,251`:
  `int` accumulators / flat indices overflow at `nx·ny ≥ 2³¹` (32768²), within
  the project's stated large-memory scope.
- cpp — `stack.h:272-298`: unchecked `malloc` → null `memcpy` (segfault
  instead of `MemoryError`); diagnostic `printf` writes to stdout during
  normal `distance_map` operation *(verified)*.
- cpp — `autocorrelation.cpp:90-92`: `nonuniform_autocorrelation` divides by
  `physical_size − distance` with no validation → NaN/garbage for user-supplied
  distances ≥ physical size *(verified)*; distances are mandatory user input
  on this path.
- cpp — `module.cpp`: no `py::call_guard<py::gil_scoped_release>` on any
  kernel; O(N²) autocorrelation and full-map flood fills hold the GIL for
  seconds to minutes.
- `Pipeline.py:80-87` — the generated class's `__getattr__` shadows the
  `_functions` dispatch and silently applies chained pipeline functions to the
  *parent*, discarding the wrapped transformation *(verified: scale(3) of a
  doubling wrapper returns 3×, not 6×)*; the module has zero callers and its
  pickle support is dead code. Delete or fix.
- Registration system (`HeightContainer.py:199-208`,
  `Container/SurfaceContainer.py:33,65-67`) — `register_function` mutates the
  *shared* base-class `_functions` dict (2-D-only functions like
  `interpolate_bicubic`/`mirror_stitch` appear on 1-D line scans and fail with
  unrelated errors); the `deprecated=` flag is accepted and ignored at ~8 call
  sites; the `func_with_doi` double-wrap check is defeated by
  `functools.wraps` *(all verified)*.
- Loggers — `logging.Logger(__name__)` instead of `getLogger` in
  `Support/Regression.py:37`, `Container/Averaging.py:37`,
  `Container/ScaleDependentStatistics.py:35`, `Container/IO/ZAG.py:60`,
  `IO/XYZ.py:47`: detached from the logging hierarchy, user configuration
  never applies.
- `IO/__init__.py:323`, `Container/IO/__init__.py:90` — `raise
  FileExistsError("file not found")`: wrong builtin (`FileNotFoundError`).
- `IO/Reader.py:1342-1349` — `DeclarativeReaderBase.topography` calls
  `.scale(None)` when a format has no height-scale metadata, and its signature
  lacks `channel_id`/`height_channel_index`.
- `Special.py:107` — `make_topography_from_function` assigns `_heights`
  directly, bypassing NaN-masking/float conversion/shape validation
  *(verified: NaNs with `has_undefined_data == False`)*.

### 2.5 Performance

- `Container/Integration.py:51-68,119-136` — container PSD integration is
  O(N²) in topography reads: the per-topography `average(qx)` callback loops
  over all topographies again, and `LazySurfaceContainer.__getitem__`
  re-reads the file from the ZIP each time (own TODO admits it). Precompute
  the bandwidth-count table once and/or memoize container elements.
- `IO/binary.py:843,862-878` + `IO/MNT.py:720-753` — zlib block scanning reads
  the whole file and walks it byte-by-byte in pure Python with speculative
  `zlib.decompress` on every `0x78` false positive; MNT additionally
  reimplements the scan inline (never advancing past decoded blocks) instead
  of using the shared `ZlibBlockChain`, and retains the whole file plus all
  decompressed blocks for the reader's lifetime.
- `IO/XYZ.py:199-320` — the file is fully parsed twice (once at `__init__`
  for channel info, again in `topography()`), with a per-token pure-Python
  accumulator loop (`data[key] += [float(value)]`) — orders of magnitude
  slower than `numpy.loadtxt`/`pandas` for large point clouds.
- `IO/MI.py:133-155` — data block converted bytes → `[chr(_) for _ in
  buffer]` → join → re-encode → `np.frombuffer`: an identity transform
  creating ~64M temporary objects for a 4k×4k map; the whole file is retained
  in `self.lines`.
- `IO/IBW.py:52-58` — `loadibw` reads the entire wave (all channels) at open
  time and keeps it for the reader's lifetime, violating the documented lazy
  design rule; `assert`-based file validation vanishes under `python -O`
  (same anti-pattern in FRT, BCR, JPK, LEXT, ZON).
- `Nonuniform/Autocorrelation.py:74-105` — double Python loop over all segment
  pairs (own `# FIXME!!! This is slow`) while the height-difference variant
  already delegates to C++; the function is unregistered and used only by
  tests.
- cpp — `patchfinder.cpp:42,54`: fresh 16 MB `malloc`/`free` per patch;
  `patchfinder.cpp:499-520`: O(max_dist⁴) correlation integration where a
  prefix sum is O(max_dist²); `bearing_area.cpp:138-163`: full-grid sweep per
  query height with no pruning (the nonuniform variant prunes); the Python
  wrapper already computes sorted bounds the C++ never uses.
- cpp — `bicubic.cpp:142-148`: eager precomputation of all 16 coefficients per
  node costs ~128 bytes/pixel (~2.1 GB for a 4096² map) before the first
  evaluation; `n1_*n2_` is `int × int` (overflow ≥ 46341²).
- `Support/Regression.py:278-293` — GP regression factorizes the covariance
  twice per solve (no Cholesky reuse); log-space branch returns log-space
  variance next to exponentiated values (FIXME acknowledges).
- `ScanningProbe/RigidScan.py:53-72` — per-pixel Python loop, vectorizable.

---

## 3. Low severity (selected)

- `IO/AL3D.py:100,104-105` — `rowstride` is already in bytes but is multiplied
  by `itemsize` again (4× over-read into the texture image; works by luck);
  invalid-pixel mask misses `abs()` on the marker.
- `IO/WSXM.py:153,237` — leftover debug `print(...)` on every load;
  ZeroDivisionError for perfectly flat scans.
- `IO/DZI.py:153-154` — division by zero for flat topographies → all-NaN
  image pyramid.
- `IO/ZON.py:83-87` — multi-entry `as_strided` strides scrambled (currently
  unreachable; guarded only by an `assert`).
- `IO/NMS.py:166` — full-scale mapping divides by 65534 instead of 65535 (or
  0xFFFF is an unhandled invalid marker); no undefined-data handling.
- `IO/Text.py:124-141,270,467` — x/y regex mapping swaps instrument axes for
  Wyko/SPIP flavors (Aspect factor lands on the wrong axis); `defaultdict(None)`
  no-op; off-by-one channel bounds check.
- `IO/JPK.py:332-337` — `except KeyError` around list indexing (`IndexError`)
  — friendly error is dead code.
- `IO/binary.py:45` + `IO/FRT.py:516` — `decode()` defaults to native byte
  order; FRT relies on it for little-endian structures (breaks on big-endian
  hosts).
- `IO/binary.py:626-633` — `TLVContainer._parse_entry` temporarily mutates a
  shared class-level layout object (not reentrant/thread-safe).
- `FFTTricks.py:61-72` — `_mufft` cache ignores the `communicator` argument.
- `UniformLineScanAndTopography.py:738-743,782-795` — `CompoundTopography`
  validates with `assert` and carries dead state; `offset` setter has an
  unreachable `offsety` branch.
- `Generic/ScanningProbe.py:68` — misplaced parenthesis in the geometric-mean
  normalization `exp((log l + log u / 2))` (root unaffected; intent defeated).
- `Nonuniform/PowerSpectrum.py:43-59` — `dsinc` branch threshold at 1e-6 sits
  in the cancellation zone (~1e-3 relative error band); raise to ~1e-2. Integer
  input inherits integer dtype → truncation. Line 209-210 dead code.
- `Uniform/PowerSpectrum.py:186` — radial-average `qmax` uses only the
  x-Nyquist; result changes under transposition for anisotropic pixels.
- `Uniform/Derivative.py:517` — per-operator normalization assumes operator
  index equals Cartesian direction (single y-operator divides by the x pixel
  size).
- `Uniform/Integration.py:82-97` — docstring formula off by 2 vs the two-sided
  sum; `try/except TypeError` dispatch can mask user errors.
- `Uniform/VariableBandwidth.py:199,267,320` — documented quantity `'s'` is
  registered as `'g'` (`KeyError: 's'`); from_area error message says
  from_profile.
- `Uniform/ScalarParameters.py:57-67` — dead NuMPI/MPI machinery behind an
  unconditional `NotImplementedError`.
- `Support/Regression.py:85` — `make_grid('log')` treats `nb_points` as edge
  count, linear/quadratic as point count; callers compensate (+1) — API trap.
- `Support/Deprecation.py:47` — warnings lack `stacklevel=2`, pointing at the
  decorator instead of the caller.
- `Generic/Moments.py:33,81,92,120` — syntactically invalid doctest; docstring
  formulas differ from the code by `(2π)^d` prefactors (code is right).
- `Nonuniform/` docstring drift: `Converters.py:59` (padding "zeros" is
  actually edge-clamped), `Interpolation.py:43` ("linear" on the cubic
  routine), `__init__.py:27` ("uniform"), `ScalarParameters.py:66,145`
  (return descriptions wrong).
- cpp `stack.h` — rule-of-three violation (double free if copied); misnamed
  (used exclusively as a FIFO); `get_size()` asserts a legitimately occurring
  state.
- cpp `patchfinder.cpp:131-155` — `assign_segment_numbers`, `distance_map`,
  `closest_patch_map`, `perimeter_length` are always periodic, with no
  `periodic` flag, unlike `assign_patch_numbers` — silent contract gap.
- cpp misc — stale "column-major" comments on row-major loops; ACF ½
  convention undocumented; 3-way corner sort copy-pasted 5×; bearing-area
  bindings lack `py::arg` names; `distance_map` fills a `next` map it never
  returns.
- `cpp/bicubic.cpp:350-359` — scalar path ignores `derivative=1/2` and returns
  a bare float where callers unpack a tuple *(verified)*.

---

## 4. Cross-cutting themes and architectural recommendations

**T1. The decorated-topography pattern has a systematic property-forwarding
hazard.** `transpose()` (1.1) and `downsample()` (1.2) both broke the same
way: a decorator overrides some geometry properties but inherits others that
are no longer consistent. Recommendation: give `DecoratedUniformTopography` a
single source of truth (derive `pixel_size`/`area_per_pt` from
`physical_sizes` and `nb_grid_pts` of *self*, not of the parent), and add a
generic invariant test run against every decorator class
(`pixel_size == physical_sizes / nb_grid_pts`, `positions().shape ==
heights().shape`, etc.).

**T2. Row-major files vs. the x-first array convention is re-decided in every
reader — and decided wrong in four of them.** DI, EZD, MI, PS scramble
non-square scans; DATX and LEXT are ambiguous; FRT, GWY, JPK are correct. The
per-reader copy-pasted `topography()` boilerplate (metadata-override checks,
masked-array wrap, reshape, scale) is what allows this. Recommendation: a
shared helper ("read row-major block → transpose → mask → scale") in
`IO/binary.py`, plus one non-square fixture per binary format (the square-only
fixtures are why none of this is caught).

**T3. The nonuniform code path repeatedly assumes `x[0] == 0`.** `to_uniform`
(1.11), slope detrending, Hann windowing — all masked by IO readers that shift
the origin, all exposed by the public constructor. Recommendation: normalize
the origin in one place (or make analyses origin-invariant) and add tests with
offset x grids; `Nonuniform/Detrending.py:79-88` should also center x before
solving the normal equations (ill-conditioned for large offsets).

**T4. Masked (undefined) data is a partially supported concept.** PSD/ACF
guard with `has_undefined_data`; detrending of line scans, variable bandwidth,
`Rq`/`Sq`, checkerboard detrend and several readers do not (2.2).
Recommendation: decide a contract per function (raise `UndefinedDataError` or
handle the mask), and enforce it with a shared parametrized test over the
registered-function table.

**T5. The string-keyed registration system needs hardening.**
`register_function` writes into shared base-class dicts (functions leak across
classes), `deprecated=` is silently ignored, `__getattr__` dispatch swallows
internal `AttributeError`s and re-reports them as "no analysis function
registered" (masking real bugs like the broken base-class `info` property),
and nothing is visible to IDEs/type checkers or `dir()`. Recommendation:
per-class dicts via `__init_subclass__`, implement or remove `deprecated`,
re-raise internal `AttributeError`s with the original traceback, and consider
generating real methods with `setattr`.

**T6. Tests validate "runs without error" more than correctness in exactly
the areas that broke.** Scale-dependent statistics, nonuniform detrending,
container IO (the only ZAG test is skipped with a hardcoded developer path),
`uniform2d_*` moments (tests assert the broken values), scan-line alignment
(tests encode the inverted axis). CLI has no tests. Recommendation:
property-based invariants (Parseval for PSD, `bearing_area(median()) == 0.5`,
transpose/flip invariance of scalar analyses, non-square and offset-origin
fixtures) would have caught most of the high-severity findings here.

**T7. Analyses are implemented three times with drifting signatures.**
Uniform/Nonuniform/Container triplets of PSD, ACF, variable bandwidth, scalar
parameters; `bandwidth` exists in three `common.py` files;
`ciso_moment`/`c1d_moment` are copy-paste identical except one call. The
`Generic/` package shows the better pattern (one implementation registered on
both interfaces) but is applied inconsistently. Long-term: converge on
`Generic/`-style single implementations.

**T8. Packaging risks.** `pyproject.toml:61` depends on the deprecated
misspelled alias package `tiffile` (and code imports `from tiffile import
TiffFile` in JPK/PS/LEXT) — switch to `tifffile`; `matplotlib>=1.0.0` is a
meaningless floor; `pytest-flake8` is unmaintained (with a `flake8<8` pin
papering over it); `meson.build:6` runs `run_command('python', ...)` which
fails on distros without a `python` alias; every `.py` must be hand-listed in
per-directory `meson.build` files with no completeness check (a forgotten
entry ships a silently broken wheel) — add a glob-based CI check.

---

## 5. Suggested triage order

1. **Wrong-results bugs users cannot detect** (§1.1–1.4, 1.7–1.12, 1.14,
   1.17): decorator geometry, non-square reshapes + MI units, triangle/flat
   moments, bearing-area bounds, `to_uniform` origin, curvature stencil,
   `fourier_synthesis` DC.
2. **Reader crashes and channel misdirection** (§1.5, 1.6, 2.3): cheap,
   contained fixes; add fixtures while touching each format.
3. **Masked-data contract** (§2.2, T4) and the **registration system** (T5).
4. **Container IO** (§1.13, 2.4, 2.5): ZAG stream lifetime, caching layer to
   kill the O(N²) reads, HTTP hardening.
5. **C++ hygiene** (§2.4 cpp items): const refs, int64 sizes/dtypes, GIL
   release, contiguity — mostly mechanical.
6. Structural work per themes T1–T7, guarded by the new invariant tests.
