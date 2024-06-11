Change log for SurfaceTopography
================================

v1.14.0 (11June24)
------------------

- MAINT: Switched from muSpectre to muFFT (v0.91.2) in dependencies

v1.13.13 (16Apr24)
------------------

- BUG: Fixed reader for MetroPro v3 files, close #364
- BUG: Wrong data layout in SUR files
- TST: scipy.signal.triang -> scipy.signal.windows.triang, closes #365

v1.13.12 (21Mar24)
------------------

- MAINT: More robust CSV reader

v1.13.11 (07Mar24)
------------------

- BUG: Further fixes for bearing are of topographies with undefined value

v1.13.10 (05Mar24)
------------------

- BUG: Correct bounds for bearing area of topography with undefined values

v1.13.9 (05Mar24)
-----------------

- BUG: Need to use relative tolerance in bisection for robust statistics

v1.13.8 (05Mar24)
-----------------

- BUG: Fixed MAD height for data with undefined values

v1.13.7 (05Mar24)
-----------------

- BUG: Avoid always warning about unit

v1.13.6 (04Mar24)
-----------------

- MAINT: Added `ignore_filters` option to `CEReader` which disables
  construction of the filter pipeline from container metadata
- MAINT: Don't fail but warn if meta.yml has wrong metadata

v1.13.5 (04Mar24)
-----------------

- BUG: Implemented CSV parser and removed pandas dependency (pandas
  had issues reading CSV-style data from ZIP containers)

v1.13.4 (29Feb24)
-----------------

- MAINT: Performance optimization for MAD analysis

v1.13.3 (19Feb24)
-----------------

- BUG: Scale-dependent statistics for containers with distances outside
  any bandwidth

v1.13.2 (19Feb24)
-----------------

- BUG: Scale-dependent statistics with multiple components now works for
  containers
- BUG: Fixed issue with I/O in Python 3.12

v1.13.1 (13Feb24)
-----------------

- MAINT: Start Nelder-Mead optimization for MAD fits from the initial guess
  obtained from an RMS fit

v1.13.0 (13Feb24)
-----------------

- API: Standard RMS height functions are now called `Rq` and `Sq`;
  `rms_height` returns the RMS height of the linearly-interpolated
  topography
- API: Translating nonperiodic topography raises error
- ENH: Median-absolute-deviation (MAD) calculation
- ENH: Median calculation
- ENH: Exact bearing area model based on linear interpolation of topography
- ENH: Arbitrary statistical functions for VBM
- BUG: Fixed translation of decorated topographies not possible
- BUG: Added missing test of periodicity before translation
- MAINT: Flipped Gwyddion images, #351
- MAINT: C++ extension module (`_SurfaceTopographyPP`)
- MAINT: Use new API response for contact.engineering to get download URLs

v1.12.2 (12Dec23)
-----------------

- BUG: Fixed issue with DI reader (#343)
- BUG: Fixed reading of some file formats inside containers

v1.12.1 (14Nov23)
-----------------

- BUG: Fixed info dictionary in Dektak CSV reader

v1.12.0 (14Nov23)
-----------------

- ENH: Reader for Dektak CSV files

v1.11.3 (8Nov23)
----------------

- BUG: `x`-variable of NetCDF output contained random data for uniform line
  scans

v1.11.2 (28Oct23)
-----------------

- BUG: Honor `data_source` when reading contact.engineering containers

v1.11.1 (18Oct23)
-----------------

- BUG: HGT, NC and VK readers returned integer `physical_sizes`
- MAINT: Flipped MetroPro images, #332

v1.11.0 (11Sep23)
-----------------

- ENH: Reader for JPK (.jpk) files
- BUG: Mask missing data point in DATX files

v1.10.1 (30Aug23)
-----------------

- BUG: Support for Dektak OPDx 16-bit integer values 

v1.10.0 (29Aug23)
-----------------

- API: Do not return None as `ChannelInfo` (this was only the GWY reader)
- ENH: Reader for WSxM (.stp, .top) files
- ENH: Reader for Sensorfar XML SPM (.plux) files
- BUG: Fixed loading of non-square Gwyddion (.gwy) files
- MAINT: More robust XY (line scan) and XYZ reader

v1.9.0 (28Aug23)
----------------

- ENH: Reader for Olympus OIR (.oir) and POIR (.poir)
- ENH: Output of Deep Zoom data files (NetCDF or Numpy)
- BUG: Fixed loading of SUR files that report different units for lateral and
  height information
- BUG: Fixed loading some LEXT files

v1.8.0 (24Jul23)
----------------

- API: Generalized container readers to support multiple file formats
- API: `SurfaceContainer` is now `InMemorySurfaceContainer`
- ENH: Reader for Zygo DATX (.datx)
- ENH: Reader for Olympus LEXT (.lext)
- ENH: Reader for ZAG containers (.zag)
- BUG: Fixed `scale_dependent_curvature_from_area`
- BUG: More robust date parsing for Mitutoyo reader
- MAINT: Reader infrastructure now supports declarative readers that define
  the file layout directly
- MAINT: Switched SUR and PLU readers to declarative style
- CI: macOS wheels (arm64 and x86_64)

v1.7.0 (27Jun23)
----------------

- API: Renamed `UnknownFileFormatGiven` exception to `UnknownFileFormat`
- ENH: Support for Gwyddion's native file format (#307)
- ENH: Support for Sensofar SPM files (.plu, .apx)
- ENH: Support for Micrprof files (.frt)
- ENH: Support for HFM files (.hfm, text files)
- ENH: Readers now report MIME types and typical file extensions
- BUG: Contents of ZIP files should be detected as binary streams
- MAINT: Added pandas as a dependency for text file parsing

v1.6.3 (19Jun23)
----------------

- MAINT: More robust date parsing for DI reader
- MAINT: Updated URL for container downloads

v1.6.2 (17Apr23)
----------------

- BUG: MetroPro info dictionary is now JSON serializable

v1.6.1 (10Apr23)
----------------

- BUG: More robust catching of invalid values in AL3D reader

v1.6.0 (05Apr23)
----------------

- ENH: Reader for the Zygo MetroPro

v1.5.0 (25Mar23)
----------------

- ENH: analytical computation of scalar roughness parameters of ideal self-affine psd
- ENH: weighted integrals of the PSDs of containers
- ENH: scanning probe artefact simulations
- TST: remove dependency of test-suite on ContactMechanics
- MAINT: Switched from igor to igor2 which supports latest numpy

v1.4.1 (6Mar23)
---------------

- BUG: Do not use `peek`, since some stream objects do not support this
  operation
- BUG: DI files sometimes report more data than they hold

v1.4.0 (6Mar23)
---------------

- MAINT: Dropped Python 3.7 support (missing importlib)
- MAINT: Unified file naming schemes in file format examples
- ENH: Convenience decorator for creating simple pipeline functions
- ENH: Reader for NanoSurf easyScan (NID/EZD)
- ENH: Reader for BCR-STM/BCRF files
- BUG: Wrong data layout in AL3D reader
- BUG: Do not fail if SUR files have not acquisition time

v1.3.3 (29Jan23)
----------------

- BUG: Fixed discover of version when in a git repository that is not the
  source directory of SurfaceTopography

v1.3.2 (18Jan23)
----------------

- ENH: Pipeline function superpose
- BUG: Repair CompoundTopography and TranslatedTopography

v1.3.1 (14Jan23)
----------------

- BUG: AL3D reader reported wrong unit in channel info (#276)
- BUG: SUR reader reported wrong info dictionary in channel info
- BUG: Increased numerical robustness of scanning probe artifact analysis
  (#275)

v1.3.0 (29Dec22)
----------------

- BUILD: Replaced LAPACK with Eigen3, removing external dependency
- CI: Building and publishing of binary wheels (currently manylinux
  and Python 3.7, 3.8, 3.9 and 3.10 only)

v1.2.6 (29Dec22)
----------------

- BUG: SUR reader reported wrong `height_scale_factor` in channel info
- BUILD: Yet another fix for version discovery when installing from source
  package

v1.2.5 (05Dec22)
----------------

- BUILD: Another fix for version discovery when installing from source package

v1.2.4 (04Dec22)
----------------

- BUILD: Fixed version discovery when installing from source package

v1.2.3 (02Dec22)
----------------

- BUG: Make sure all readers return the same info dictionary for channels
  and the final topography
- MAINT: Automatic version discovery for Meson build

v1.2.2 (28Nov22)
----------------

- MAINT: Switched build system to Meson
- BUG: Fixed multiple bugs in handling of `height_scale_factor` by readers
- BUG: Fixed handling of no scale factor in ASC reader
- TST: Added test for proper handling of `height_scale_factor` by readers

v1.2.1 (22Nov22)
----------------

- ENH: Extract X3P metadata
- MAINT: Converted X3P reader to new style
- MAINT: Bumped muFFT require to 0.24.0 - this avoids installation problems
  downstream, but disables automatic MPI detection
- BUG: X3P data was read in the wrong storage order

v1.2 (15Nov22)
--------------

- ENH: Reader for Park Systems TIFF files
- ENH: Reader for Keyence VK3 and VK6 files
- ENH: Reader for Alicona Imaging AL3D files
- ENH: Reader for Digital Surf SUR files
- MAINT: Removed pandas dependency
- BUG: Fixed bug in VK4 reader

v1.1 (20Oct22)
--------------

- API: `scale_dependent_statistical_property` of a `SurfaceContainer` now
  returns either `np.ndarray` or `np.ma.masked_array`
- ENH: Reader for the Mitutoyo SurfTest Excel spread sheet format
- ENH: Reader for Keyence VK4
- MAINT: Added contact.engineering paper to bibliography
- BUG: Check for reentrant topographies in slope calculation
- BUG: Fixed aspect ratio used in convenience plotting function

v1.0 (06May22)
--------------

- ENH: Generalized HDF5 reader
- MAINT: Removed `unit` entry of `info` dictionary
- MAINT: Removed `scale_factor` property

v0.101.3 (24Jan22)
------------------

- BUG: Ignore NaNs when suggesting units for data

v0.101.2 (09Dec21)
------------------

- MAINT: Explicity convert to CSR matrix to suppress spsolve warning
- MAINT: Detect file corruption in DI reader when opening the file

v0.101.1 (30Nov21)
------------------

- BUG: Ignore if analysis fails because of undefined data when computing
  averages

v0.101.0 (25Nov21)
------------------

- API: ChannelInfo now has `is_uniform` and `has_undefined_data` properties
  that indicate the structure of the underlying data (if known after reading
  file headers)

v0.100.3 (24Nov21)
------------------

- BUG: Allow manual specification of unit if IBW file does not provide unit
  information
- BUG: Missing `unit` keyword argument for HDF5 reader

v0.100.2 (23Nov21)
------------------

- API: Add `has_undefined_data` property to `NonuniformLineScan`
- MAINT: Avoid reading full file in Matlab reader
- MAINT: Avoid reading full file in OPD reader

v0.100.1 (22Nov21)
------------------

- MAINT: Moved geometry analysis to separate module

v0.100.0 (22Nov21)
------------------

- ENH: Filling undefined data points with harmonic functions

v0.99.3 (15Nov21)
-----------------

- BUG: Date format in OPDx can vary; use dateutil for parsing

v0.99.2 (15Nov21)
-----------------

- BUG: Date format in OPDx is month first

v0.99.1 (15Nov21)
-----------------

- BUG: Fixed unicode conversion issue in OPDx reader
- BUG: Always suggest a unit (#164)

v0.99.0 (11Nov21)
-----------------

- ENH: Self-report citations (#135)
- ENH: Raise exception if missing data points are unsupported (#106)
- ENH: Parallel implementation of scalar roughness parameters (#55)
- ENH: DZI writer has option to write metadata as JSON
- ENH: DZI writer dumps colorbar information into metadata file
- ENH: Support for line scans in OPDx files
- BUG: __eq__ for topographies did not compare height information (#96)
- BUG: NetCDF reader checks consistency of units (#82)
- BUG: Storing topographies with undefined data to NetCDF files was
  broken in some cases; now an explicit mask is stored to the NetCDF file
- MAINT: Compute Fourier derivative with muFFT (#111)
- MAINT: Refactored OPDx reader (attention: height scale factor differs from
  old reader)
- MAINT: Variable bandwidth raises `UndefinedDataError` when topography has
  undefined data points

v0.98.2 (25Oct21)
-----------------

- BUG: Raise NoReliableDataError for container only if none of the
  topographies contribute data (#152)
- MAINT: DZI writer return list of written files

v0.98.1 (08Oct21)
-----------------

- ENH: Allow specification of a minimum number of data points for
  scale-dependent statistical properties
- BUG: Properly raise `NoReliableDataError` for variable bandwidth
  method and scale-dependent statistical properties
- BUG: Multiple bug fixes for corner cases
- BUG: Fixed shape of return array for derivatives

v0.98.0 (05Oct21)
-----------------

- API: Analysis function now raise errors upon encountering reentrant
  topographies or topographies with no reliable data

v0.97.5 (03Oct21)
-----------------

- ENH: Added kilometers to units

v0.97.4 (30Sep21)
-----------------

- ENH: Interpret 'resolution' instrument entry as reliability cutoff

v0.97.3 (28Sep21)
-----------------

- ENH: Apply reliability cutoff to scale-dependent statistical analysis
- BUG: Filtered wrong end of autocorrelation for reliability cutoff

v0.97.2 (24Sep21)
-----------------

- BUG: Fix segmentation faults due to integer overflow for large bicubic
  interpolation

v0.97.1 (21Sep21)
-----------------

- BUG: Prefer micrometer with Greek 'mu' sign
- BUG: Surface is already reentrant if two x values are identical

v0.97.0 (21Sep21)
-----------------

- ENH: `bandwidth` function for containers
- ENH: Suggesting ideal length units for containers
- ENH: Progress callbacks for some analysis functions
- ENH: Gaussian process regression for resampling noisy data onto an
  arbitrary grid
- ENH: Added `is_reentrant` property
- ENH: Computing averaged scale-dependent roughness parameteras,
  autocorrelation, power spectrum and variable bandwidth analysis for
  containers
- BUG: Multiple bugs fixed in scale-dependent analyses
- API: Reconciled radial averages into its own regression module;
  default behavior has changed
- API: All analysis functions now resample data onto a reasonable grid
- API: Resampling methods no longer discard bins with no data; those are
  filled with NaNs

v0.96.0 (13Sep21)
-----------------

- ENH: Generic scale-dependent statistical properties for containers (#116)
- ENH: Linear interpolation for derivatives at arbitrary distances (#128)
- ENH: Tip radius reliability analysis (#116)
- ENH: Third derivatives
- DOC: Include description for PyPI package (#102)
- API: `to_nonuniform` now only works with nonperiodic uniform line scans

v0.95.2 (9Sep21)
----------------

- ENH: Read instrument name from DI files (#124)
- ENH: Specify PSD as a function in Fourier synthesis
- BUG: Avoid loosing mask information when reading/writing NetCDF (#126)
- DOC: Mention .spm file extension in doc string of DI reader (#123)
- DOC: Clearer error message for reentrant topographies (#108)
- DOC: Updated docstring of `bandwidth` (#109)
- DOC: Converted docstrings to numpydoc (#5)

v0.95.1 (12Jul21)
-----------------

- Fixed missing dependency for matplotlib in setup.py (#120)
- Fixed building wheel when numpy is not installed yet (#119)

v0.95.0 (07Jul21)
-----------------

- API: Changed the meaning of `distance` in scale dependent derivative.
  A distance is now the width of the stencil of lowest truncation order
  of a certain derivative, i.e. n * dx where n is the order of the
  derivative
- API: `variable_bandwidth` is now `variable_bandwidth_from_area` and an
  additional `variable_bandwidth_from_profile` was added for a line-by-line
  analysis
- API: `checkerboard_detrend` is now `checkerboard_detrend_area` and an
  additional `checkerboard_detrend_profile` was added for a line-by-line
  analysis
- API: Topographies now have a `unit` property. For backwards compatibility,
  this information is still returned in the `info` dictionary but this
  behavior will be deprecated in version 1.0. (#83)
- API: `physical_sizes` can no longer be set after the topography has been
  created
- API: Renamed `ninterpolate` keyword argument to `nb_interpolate`
- API: `to_uniform` for nonuniform line scans now accepts an `nb_interpolate`
  argument specifying the number of grid points put between closest points.
- ENH: Writing Deep Zoom Image (DZI) files
- ENH: Checkerboard detrending can now be carried out to arbitrary polynomial
  order
- ENH: `scale` pipeline function is now able to rescale lateral dimensions
  (positions)
- ENH: Add `to_unit` pipeline function for unit conversion
- ENH: `SurfaceContainer` class including pipeline functions for collections
  of topographies
- ENH: Reading and writing of surface containers
- ENH: Retrieving surface containers from https://contact.engineering
- ENH: Added analysis functions for scale-dependent statistical properties of
  containers
- ENH: Added `scanning_probe_reliability_cutoff` pipeline function to estimate
  a lateral length scale below which scanning probe data become unreliable
  because of tip artifacts (see arXiv:2106.16013)
- ENH: Support for Digital Instruments Nanoscope (DI) files with 32-bit
  integer data

v0.94.0 (10Jun21)
-----------------

- ENH: Added API tests for checking existence of functions
  and properties for different kinds of topographies (#100)
- ENH: Added height_scale_factor to reader channels (#98)
- ENH: Disallow to provide physical_sizes or height_scale_factor
  as argument to .topography() if fixed by file contents
- BUG: Fixed missing function for RMS curvature for uniform
  line scans (#95)
- BUG: Make sure that .topography(height_scale_factor=..) is effective
- BUG: Fixed loading of 2D measurements from xyz data (#93)

v0.93.0 (27Apr21)
-----------------

- API: Replace `_1D` suffix with `_from_profile` and `_2D` suffix with
  `_from_area`
- API: `save` is now called `to_matrix`
- ENH: Reader for Keyence ZON files
- ENH: Derivative can be computed for different scales
- ENH: Scale-dependent slope and curvature
- ENH: Pipeline function for windowing
- ENH: Reader for NetCDF can read from streams
- BUG: Fixes that info dict was not always taken when topography
  was initialized from a reader's channel

v0.92.0 (15Mar21)
-----------------

- Drop support for python3.5
- ENH: rms_slope, rms_curvature and rms_laplace can now filter short
  wavelength roughness
- MAINT: Filters follow pipeline model
- MAINT: Derivatives now use muFFT derivative operators
- MAINT: Bumped muFFT dependency to v0.12.0
- API: Return nonuniform PSD only up to a reasonable wavevector

v0.91.5 (19Oct20)
-----------------

- BUG: Fixed differing versions in setup.py and requirements

v0.91.4 (16Oct20)
-----------------

- BUG: Fixed issue with reading npy from binary stream
- BUG: Fixed periodic flag for npy files in some cases

v0.91.3 (22Sep20)
-----------------

- BUG: UniformLineScan.area_per_pt is now a scalar, not a tuple
- BUG: supress assertion that discarded imaginary part in fourier
  derivative is small

v0.91.2 (31Jul20)
-----------------

- BUG: Memory leak in Python bindings for bicubic interpolation

v0.91.1 (23Jul20)
-----------------

- MAINT: Bumped muFFT dependency to v0.10.0

v0.91.0 (2Jul20)
----------------

- ENH: return topography with undefined (masked) data filled (#19)
- ENH: make_sphere has option to put undefined data in standoff or not
- BUG: Fixed missing dependencies when installed as package (#15)
- BUG: Fixed error when loading text file with trailing newline (#16)
- BUG: Fixed wrong error message when loading unknown text file format (#14)
- BUG: renamed CompoundTopography.array to CompoundTopography.heights (#18)

v0.90.0 (17Jun20)
-----------------

- Refactored PyCo code into three separate Python modules:
  SurfaceTopography, ContactMechanics and Adhesion
- muFFT dependency updated to muFFT-0.9.1
- Moved documentation from README.md to the docs folder

Change log for PyCo (previous name of the package)
==================================================

v0.57.0 (15May20)
-----------------

- MAINT: Support for new, rewritten muFFT bindings
- ENH: Bicubic interpolation of two-dimensional topography maps
- ENH: Fourier derivative of topography
- BUG: Computation of plastic area is now parallelized (#303)
- BUG: Info dictionary mutable from user

v0.56.0 (26Feb20)
-----------------

- ENH: Change orientation of some readers such that all topographies
  look like the image in Gwyddion when plotted with
  "pcolormesh(t.heights().T)" (#295)
- BUG: Fixes unknown unit "um" when reading mi file (#296)
- BUG: Fixes missing channel name for mi files (#294)
- ENH: generate self-affine random surfaces by specifying the self-affine prefactor (#261, #278, #279)
- BUG: now fourier synthesis can generate Linescans again (#277, #279)

v0.55.0 (14Feb20)
-----------------

- API: Readers now report channel info in ChannelInfo class,
  fixes inconsistencies in reporting channel information (#190, #192, #236)
- ENH: Readers report format identifier and are self-documented (#229, #238)
- ENH: Readers now support Gwyddion's text export format for English and German locale (#230)
- ENH: DI reader now read acquisition date and stores it in the info dictionary
- BUG: DI reader autodetection did not work (#258)
- BUG: Fixes orientation for DI files (#291)
- DOC: Added notebook showing how 2D topographies can be plotted
- TST: Added demo notebook which shows how to plot 2D topographies
- ENH: adhesive ideal plastic simulations with Softwall system (#260, #283)

v0.54.4 (20Dec19)
-----------------

- BUG: Fixes missing 'nb_grid_pts' key in channels from IBW reader

v0.54.3 (20Dec19)
-----------------

- BUG: Fixes assertion because of wrong number of channel names (#252)

v0.54.2 (13Dec19)
-----------------

- BUG: fix rms_laplacian for periodic topographies (#247)

v0.54.1 (13Dec19)
-----------------

- ENH: higher order derivative for periodic surface (#234,#227)
- ENH: new reader for Igor Binary Wave files (IBW) (#224)
- BUG: opdx reader can now handle binary filestreams (#209)
- BUG: store and restore periodic flag in NonuniformTopography (#240)

v0.54.0 (06Dec19)
-----------------

- MAINT: correct installation problems because Eigen repository has moved
- ENH: anisotropic cubic Green's Functions
- BUG: NPY reader can now handle filestreams (#209, NUMPI/#24)
- BUG: opdx reader can now handle filestreams (#209)

v0.53.1 (21Nov19)
-----------------

- API: Detrended Topographies with mode "center" keep is_periodic property. Other modes lead to is_periodic=False.
  See pastewka/TopoBank/#347

v0.53.0 (20Nov19)
-----------------

- API: ability to set periodic property of HeightContainer in reader.topography (#198)
- API: default window for computing PSD is choosen according to topography.is_periodic (#217)
- Feature: interpolate_fourier pipeline function
- API: Default of check_boundaries in FreeSystem is False
- Bug fix: fourier synthesis had a padding line in the generation of topographies with odd number of points (#202)
- Bug fix: `topography.rms_curvature` no returns rms_curvature, previously rms_laplacian (#200)
- gnuplot scripts to plot logger output

v0.52.0 (25Aug19)
-----------------

- API: Return contact map (the 'active set') from constrained conjugate gradient
- Bug fix: `assign_patch_numbers` was broken on some configurations since v0.51.2

v0.51.2 (8Aug19)
----------------

- Bug fix: `assign_patch_numbers` crashed for maps larger that 64k x 64k (#191)

v0.51.1 (7Aug19)
----------------

- Bug fix: Setting physical_sizes argument in readers (#188)
- Bug fix: physical_sizes should be None for surfacs without a physical size (#189)
- Bug fix: Running and testing without mpi4py is now possible (#179)
- Bug fix: Multiple calls to `topography` method of readers (#187)
- Method to inspect pipeline (#175)
- CI: All tests (serial and MPI parallel) pass in Travis CI

v0.51.0 (5Aug19)
----------------

- Cleanup of new reader API

v0.50.2 (1Aug19)
----------------

- Bug fix: Missing `channel` argument for `topography` method of `WrappedReader` (#181)
- `WrappedReader` now uses 'Default' as channel name

v0.50.1 (1Aug19)
----------------

- Bug fix: Running without an MPI installation
- Bug fix: Reading DI files with non-topographic data (#338)

v0.50.0 (31Jul19)
-----------------

Overview:

- MPI parallelization of topographies, substrates and interaction.
- Updated reader framework that supports loading files in parallel. This requires to peek at the files (without
  loading them) to understand the number of grid points to decide on a domain decomposition strategy.

Technical:

- Use MPI wrapper provided by NuMPI (https://github.com/IMTEK-Simulation/NuMPI) for serial calculations.
- Switch to parallel L-BFGS of NuMPI.
- Removed Cython dependencies. (Parallel) FFT is now handled by muFFT (https://github.com/muSpectre/muFFT.git).
- Tests have been partially converted to pytest. Parallel tests are run through run-tests
  (https://github.com/AntoineSIMTEK/runtests).

v0.32.0 (15Jul19)
-----------------

- Autocorrelation and power-spectrum updates. Both now have an option 'algorithm' that let's the user select
  between a (fast) FFT and a (slow) brute-force implementation.

v0.31.3 (7Jul19)
----------------

- Removed check for existing forces on boundaries (nonperiodic calculations only).

v0.31.1 (20May19)
-----------------

- Bug fix: Contact calculations now also run with detrended/scaled topographies.
- Updated hard wall command line script to new topography interface.

v0.31.0 (5Mar19)
----------------

- Added height-difference autocorrelation and variable bandwidth analysis for nonuniform
  line scans.
- Added wrapper 'to_nonuniform' function that turns uniform into nonuniform line scans.
- Bug fix: 'center' detrend mode for nonunform line scans now minimizes rms height.

v0.30.0 (15Feb19)
-----------------

Overview:

- Added non-uniform line scans, which can be loaded from text files or constructed from arrays.
- New class structure for topographies and line scans (for easier maintenance).
- Major API changes and several bug fixes (see below).
- Added Hardwall simulation tutorial.
- Added calculation for second derivative and RMS curvature for nonuniform topographies.
- Added coordination counting for contact patches.
- Simplified computation of perimeter using coordination counting.
- Started Sphinx documentation with notes how to use the package.

API Changes:

- New API for generating topographies and line scans (height containers) from data,
  please use "from PyCo Topography import Topography, NonlinearLineScan, UniformLineScan" now.
- New API for building pipelines using methods on height containers, e.g. "topography.scale(2).detrend()".
- Uniform topographies and line scans can be periodic.
- Removed unit property from height containers. Units are now stored in the info dictionary,
  which has to be set on generation of the height container.
- All topographies must have a physical_sizes. Readers use the resolution as the default physical_sizes
  if the files contain no physical_sizes information.
- Removed 'shape' alias to 'resolution' property for height containers.
- Size + shape are now always tuples, physical_sizes is also always set as tuple.
- Topographies can now be pickled and unpickled.
- Replaced class 'Sphere' with generator function 'make_sphere'.
- Contact with "FreeFFTElasticHalfSpace":
  Now an error is raised when points at the outer ring of the surface are interacting.
  See notebook "examples/Hardwall_Simulation.ipynb".

Bug fixes:

- periodicity was ignored in calculation of the distance between contact patches in `distance_map`
- computation of energy in fourier space didn't match the computation of energy in real space
  (however it is not used in actual simulation)
- Removed keyword "full_output" from shift_and_tilt().
- Text files without spaces at beginning of line can be read now.
- Enable reading topography data from memory buffers and from binary streams.
- Calculation of 2D autocorrelation function was broken, e.g. radial average.
- 1D autocorrelation was broken for nonperiodic calculations.

v0.18.0 (31Oct18)
-----------------

- Refactored "Surface" to "Topography".
- Bug fix: Corrected computation of attractive contact area in Smooth contact system.
- Bug fix: Corrected computation of inflexion point in LJ93 and VW82 smoothed potentials.

v0.17.0 (06Jul18)
-----------------

- Height-difference autocorrelation function.

v0.16.0 (23Oct17)
-----------------

- PyCo now licensed under MIT license.

v0.15.0 (06Sep17)
-----------------

- Implemented substrates of finite thickness.
- Support for additional DI file formats.
- More clever unit conversion in DI files.

v0.14.1 (16Jun17)
-----------------

- macOS compatibility fixes.
- Automatic conversion from hardness value (given in units of pressure)
  into internal units in constrained CG solver.

v0.14.0 (14Mar17)
-----------------

- Added penetration hardness model for simple plastic calculations.

v0.13.1 (07Mar17)
-----------------

- Bug fix: Periodic Green's function offset by one lattice constant.

v0.13.0 (13Jan17)
-----------------

- Added further adhesive reference models (Maugis-Dugdale type models for
  cylinder and wedge).
- Added callback option for Polonsky & Keer optimizer.
- setup.py now has '--openmp' option that triggers compilation of shared-memory
  (OpenMP) parallel code.

v0.12.0 (05Dec16)
-----------------

- Main enhancement: Support for masked_arrays in NumpySurface. This allows to
  have undefined (missing) data points in surfaces. Polonsky & Keer can handle
  this now.
- Polonsky & Keer can now optimize at constant pressure (in addition to
  constant displacement)
- Updated hard wall script to accept command line arguments.
- Moved scripts to new 'commandline' folder.
- Added plotmap.py, tool for plotting surfaces from the command line.
- Added plotpsd.py, tool for plotting the PSD of a surface from the command
  line.

v0.11.0 (21Sep16)
-----------------

- Renamed TiltedSurface to DetrendedSurface.

v0.10.3 (16Sep16)
-----------------

- Added reader for HGT files (topography data from NASA Shuttle Radar Topography
  Mission).
- Bug fix in deprecated 'set_size' that broke hard wall example.

v0.10.2 (29Aug16)
-----------------

- Added reader for MATLAB files.

v0.10.1 (22Aug16)
-----------------

- Added 'center' detrending mode which just subtracts the mean value.
- Added getter and setter for detrend_mode.
- Added function to return string representation of subtracted plane.
- Added area_per_pt property to Surface object.

v0.10.0 (31Jul16)
-----------------

- Exponential adhesion potential from Martin's contact mechanics challenge, to
  be used in combination with hard-wall (bounded L-BFGS). Added tests for this
  potential. Thanks go to Joe Monty for implementing this.
- Surfaces now have a *unit* property, that can be any object but will likely
  be a string in many cases.
- Readers now create NumpySurface with *raw* data and wrap it into a
  ScaledSurface to convert to proper unit.
- Travis-CI integration

v0.9.4 (27Apr16)
----------------

- Greenwood-Tripp reference solution
- Many bug fixes in topography file readers

v0.9.3 (20Mar16)
----------------

- Wyko OPD reader (.opd)
- Digital Instruments Nanoscope reader (.di)
- Igor Binary Wave reader (.ibw)
- Detrending

v0.9.2 (06Mar16)
----------------

- X3P reader (.x3p)
- Automatic file format detection
