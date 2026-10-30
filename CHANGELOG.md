<!--
Style for this changelog (keep to it so entries stay uniform):

- Newest version first. Heading `vX.Y.Z (DDMonYY)`, underlined with dashes.
  A version not yet released is headed `vX.Y.Z (not yet released)`.
- One bullet per user-visible change, at most two lines (about 200
  characters). Say what changed for the user and, if useful, why or by how
  much. Omit internals: no function, class, file or variable names, no
  environment variables, no commit or PR references.
- Start every bullet with one of: `ENH:` (new feature), `BUG:` (fix),
  `API:` (interface change), `PERF:` (speed or memory), `MAINT:`, `TST:`,
  `DOC:`, `BUILD:`, `CI:`.
- Small related fixes may share one bullet. No trailing period on bullets.
-->

Change log for SurfaceTopography
================================

v1.28.0 (not yet released)
--------------------------

- BUG: Profile PSD of nonperiodic maps windowed the profiles radially instead of
  along x; low-wavevector values were off by up to a factor of six
- BUG: Subtracting the mean or median of a periodic topography kept it periodic only for
  mode "center"; spectra of mean- and median-detrended topographies were windowed
- BUG: The rms height of the self-affine PSD model omitted a factor 1/H, e.g. 31% too
  large for H=0.3; cutoff overrides can only narrow the model's bandwidth, as for its derivatives
- BUG: Synthetic topographies with a requested rms height were 10-45% too rough on average;
  the normalization now sums over the wavevectors of the grid, including the roll-off region
- ENH: Synthetic topographies accept a seed or random number generator
- API: The scale-dependent curvature of profiles reports the total stencil length as
  distance, like the areal curvature and derivatives; distances doubled
- BUG: Fourier filters (short and long cutoff) failed or lost a grid point for odd numbers
  of grid points along y
- BUG: Nonuniform line scans resampled on a uniform grid reported a physical size one pixel
  too short, which distorted their spectra and autocorrelation functions
- BUG: The areal PSD of topographies with a lateral resolution finer than the pixel size
  extended beyond the Nyquist wavevector into the corners of the Fourier grid
- API: Gaussian-process resampling of spectra and autocorrelation functions was removed; it
  was biased by about a factor of two for noisy spectra. Bin averaging remains
- BUG: Explicit bin edges (arrays or lists) for spectra and autocorrelation functions failed
  with an error about the truth value of an array
- BUG: Half-derivative variance from autocorrelation functions counted empty bins as zero,
  e.g. 40% too small at 50 points per decade
- BUG: The profile half-derivative variance of containers failed with an attribute error
- BUG: Binned scale-dependent slopes dropped the smallest distance; the half-derivative
  variance computed from them was up to 9% too small
- BUG: Half-derivative variance from autocorrelation functions ignored distances beyond the
  scan and was up to 21% too small; heights are now taken as uncorrelated there
- BUG: Half-derivative variance and elastic energy of line scans assume an isotropic
  surface, like the estimate from the autocorrelation function; values increase by pi/2
- BUG: Adding topographies in different units failed because converted physical sizes
  differed by rounding; unit conversion factors are exact powers of ten

v1.27.0 (28Oct26)
-----------------

- BUILD: MPI and muGrid are optional; the default install runs serially, install the `mpi`
  extra for domain-decomposed topographies
- BUG: Areal rms height of domain-decomposed (MPI-parallel) topographies was computed per process
- API: The rms curvature of domain-decomposed (MPI-parallel) topographies raises an error
  instead of failing with an index error; it is not supported in parallel
- BUG: NPY files in big-endian byte order or half precision failed to load when mpi4py is
  installed
- CI: Parallel tests run under MPI again; they had silently been skipped

v1.26.0 (27Oct26)
-----------------

- ENH: New pipeline functions `add` and `subtract` for topographies and line scans, e.g.
  for the gap between two surfaces; optionally interpolates onto a common grid
- BUG: Bicubic interpolation no longer crashes for maps of 46341 x 46341 points and larger
- DOC: Definition of the distance in the scale-dependent statistics of containers
- API: Instrument metadata reports the scanner serial number; serial numbers are
  normalized and placeholders such as "not available" become empty
- ENH: Declarative readers support line-scan channels and channels that do not hold
  heights
- ENH: Unit detection can treat "A" as ampere instead of ångström
- BUG: Loop conditions of format descriptions are serialized as expressions
- BUG: Channel grid sizes are plain integers and serialize to JSON
- ENH: New reader for NumPy NPZ archives (.npz)
- ENH: AL3D: the comment is reported in the metadata
- BUG: BCR: void pixels and NaN are masked; defaults for missing header size, units and
  byte order; the height scale applies to integer data only
- BUG: DATX: x and y are no longer swapped; non-finite values are masked
- ENH: DI: files from version 9.2, alternative section names, slow axis size,
  single-number scan size; scanner serial number
- BUG: EZD/NID: height offset and lateral units are applied; controller and scan head
  serial numbers
- BUG: FRT: rows are no longer flipped; multi-image block and acquisition time
- BUG: GWY: the writer stores SI base units; missing title and units are tolerated;
  MetroPro serial numbers from Gwyddion metadata
- ENH: H5: non-numeric datasets are no longer listed as channels
- BUG: HGT: x runs west to east (was north to south)
- BUG: IBW: per-channel data units (e.g. phase in degrees); 1D and non-square waves as
  line scans; acquisition time
- BUG: JPK: height offset is applied; rows are flipped as in Gwyddion
- BUG: LEXT: x and y are no longer swapped
- ENH: MAT: MATLAB v7.3 files; non-numeric variables are skipped
- BUG: MDT: rows are flipped as in Gwyddion; negative lateral steps
- BUG: MetroPro: correct serial number; little-endian header fields; header size check
- BUG: MI: rows are always flipped; implicit binary data; acquisition time
- BUG: MNT: rewritten; correct height scale and offset, mask of non-measured points and
  physical sizes; software metadata
- BUG: NC: files without heights are rejected as a different file format
- BUG: NMM: physical size is number of points times spacing
- ENH: NPY: 1D arrays as line scans; 3D and complex arrays are rejected
- BUG: OIR/POIR: z calibration is applied; 8-bit data; only OIR members of POIR archives
  are read
- ENH: OPD: sample data arrays, byte arrays, metadata and acquisition time
- BUG: OPDx: physical sizes were too small by n/(n-1); all height channels; metadata
- BUG: PLU: files with several layers; objective names; metadata and acquisition time
- ENH: PLUX: the recipe is optional
- BUG: PS: height offset is applied; rows are flipped as in Gwyddion
- ENH: SDF: data types 0 to 4; compressed files are rejected
- BUG: SUR: non-measured points are masked; inversion flag; legacy µ encoding of units
- BUG: Text: x/y pixels of SPIP files were swapped; SPIP, Attocube and Nova headers;
  per-channel metadata
- BUG: TMD: unit is mm (was µm)
- ENH: VK: second and third height images
- BUG: WSxM: images are rotated as in Gwyddion; float data is not rescaled; Dulcinea
  serial number
- BUG: X3P: integer data is signed
- ENH: XYZ: unit header of Gwyddion exports
- BUG: ZMG: heights are unsigned; recipe name
- API: The schema of container metadata is public; the original file of a datafile is
  optional
- ENH: Containers are read from a schema-validated index, with the older YAML metadata as
  a lenient fallback for archives written by older versions
- PERF: Opening containers and detecting their format is cheap, since datafiles are opened
  on access; a corrupt member fails there instead of misleading detection
- ENH: Written containers include a schema-valid index next to the older YAML metadata and
  keep the periodicity of topographies
- BUG: Containers written by SurfaceTopography could not be read back
- ENH: Container readers list their members with name, visibility and access to the
  verbatim bytes
- ENH: ZAG: hidden measurements are excluded from the container
- PERF: ZAG: measurements are opened on access; opening a 9.6 GB archive takes 0.9 s
  instead of 127 s
- DOC: Updated file format descriptions

v1.25.0 (13Oct26)
-----------------

- ENH: Variance of the half derivative and elastic energy of topographies and line scans:
  the energy per area to press a half-space into full contact with the topography
- PERF: Harmonic interpolation of undefined data solves all patches in one sparse system;
  orders of magnitude faster for measurements with many small dropouts
- DOC: Updated file format descriptions
- MAINT: The change log moved to the main directory

v1.24.0 (25Aug26)
-----------------

- ENH: New reader for NT-MDT AFM files (.mdt), fully declarative and exported to a format
  description
- ENH: New reader for Mahr MarWin QEP profile archives (.qep); points flagged invalid by
  the instrument are undefined data
- BUG: OPDx: 2D height maps stored as anonymous matrices, as in some Dektak files, failed
  to load

v1.23.0 (05Aug26)
-----------------

- BUG: ZON: out-of-range and operator-excluded pixels are undefined data instead of
  placeholder heights
- MAINT: ZON reader rewritten as a declarative file layout
- ENH: Declarative file layouts parse members of ZIP archives, optionally decompressed,
  and XML documents
- ENH: Conversion functions of binary arrays can access the parser context
- BUG: Validation failures of readers are reported as corrupt files and caught like other
  read errors
- MAINT: Channel information omits entries without a value instead of reporting null; raw
  metadata is kept verbatim
- ENH: Serializable expression language for declarative file layouts
- ENH: Declarative layouts, channel bindings and format descriptions serialize to and from
  JSON; new switch and seek primitives
- ENH: Declarative channel bindings, including per-layer loops and masks of undefined
  data, and declarative detection by magic bytes
- ENH: ZMG, TMD, AL3D, SUR, MetroPro, PLU and ZON readers are fully declarative and export
  language-neutral format descriptions
- ENH: Declarative parsing of text headers with sections, comments and several delimiters,
  of number matrices and of checks across structures
- ENH: BCR, MI, WSxM and SDF readers are fully declarative and export format descriptions
- ENH: PS, LEXT and JPK readers are fully declarative and export format descriptions;
  declarative layouts read TIFF pages with their tags and images
- ENH: OPD, PLUX and X3P readers are fully declarative and export format descriptions;
  declarative layouts gained loops and computed values
- API: OS3D, NMS, VK and FRT readers are fully declarative and export format descriptions;
  their raw metadata is keyed by the parsed structures
- TST: Format descriptions must be complete, and a generic reader built from them must
  reproduce the readers on the example files
- DOC: Design document and format description contract (schema version 1) for declarative
  readers in any language
- BUG: Transposed and downsampled topographies report the correct pixel size; this
  affected derivatives and PSDs for anisotropic pixels
- BUG: Statistics of data with undefined points are normalized by the number of defined
  points; changes RMS height and moments
- BUG: The power spectrum no longer halves the q=0 entry when folding the +q and -q
  branches
- BUG: DI, EZD, MI and PS: non-square scans are reshaped in the correct order
- BUG: MI: lateral sizes are converted from meters to the requested unit
- BUG: BCR: void pixels are a count and are masked against the format's markers
- BUG: NMM and OIR: fixed channel selection
- BUG: EZD: fixed format detection
- BUG: ZAG: the ZIP file is reopened for each read
- BUG: Height moments are computed exactly; there were sign errors, missing prefactors and
  divisions by zero
- BUG: Bicubic interpolation accepts non-contiguous input arrays
- BUG: Fixed bounds of the bearing area of periodic topographies
- BUG: Conversion to a uniform grid interpolates at the correct positions for scans not
  starting at zero
- BUG: Slope detrending uses the length-weighted slope
- BUG: The Hann window of the nonuniform power spectrum does not depend on the origin
- BUG: Fourier synthesis sets the mean to zero instead of to C(q=1)
- BUG: Spheres created as periodic are flagged periodic
- BUG: Fixed the direction argument of scan-line alignment
- BUG: Polynomial fits of nonuniform scans use centered coordinates; detrending no longer
  depends on the x origin
- BUG: Fixed Nyquist handling in Fourier interpolation and in the 2D power spectrum for
  anisotropic pixels
- BUG: Registering an analysis function on a subclass no longer modifies the base class
- BUG: Fixed scale-dependent curvature on log-spaced and reliability-trimmed grids
- PERF: Bicubic interpolation computes spline coefficients on demand instead of storing
  128 bytes per pixel
- PERF: Integrating the PSDs of containers reads each file once instead of once per
  topography
- PERF: IBW, MI and MNT readers no longer keep file data in memory after opening
- PERF: Height-difference autocorrelation of nonuniform line scans is computed in C++
- PERF: Rigid-sphere scans are vectorized over blocks of bounded memory
- PERF: Gaussian process regression reuses a single Cholesky factorization
- PERF: The patch finder allocates its flood-fill stack once per call instead of once per
  patch
- TST: Property-based tests for invariance under transposition, translation, scaling and
  change of origin
- BUILD: Removed a broken test dependency; the TIFF library is required under its correct
  name

v1.22.0 (08May26)
-----------------

- BUG: DZI tiling no longer drops the last pixel row and column
- ENH: The DZI writer stores global physical coordinates and correct physical sizes in
  NetCDF tiles
- ENH: NetCDF tiles of DZI output carry units
- ENH: More than 16 readers (MetroPro, DATX, IBW, JPK, SUR, Bruker/DI, ...) report
  standardized instrument metadata
- ENH: Declarative readers can be serialized
- API: Instrument metadata has vendor, serial number and software fields
- TST: Regression tests for the instrument metadata of 17 readers

v1.21.1 (18Jan26)
-----------------

- BUG: Reading from gzip streams works again

v1.21.0 (16Jan26)
-----------------

- ENH: New reader for Digital Surf Mountains MNT files
- ENH: New readers for NMS, OS3D, SDF, TMF and ZMG files
- ENH: Declarative binary parsing of OLE containers and zlib-compressed data
- BUG: MNT: the height unit is read from the file instead of being fixed
- BUG: MNT: masking of undefined values depends on the data type
- MAINT: FRT reader converted to declarative style
- MAINT: Uses the 1D capabilities of muGrid 0.105

v1.20.0 (09Jan26)
-----------------

- API: Readers check from magic bytes whether they can read a file
- API: New `scan_line_align` pipeline function: line-by-line polynomial correction of AFM
  scans, e.g. of scanner bow
- API: Reader channels report their data kind and a stable channel identifier
- API: Channels can be selected by stable identifier or height channel index when reading
  a topography
- API: Readers list their height channels separately, for backwards-compatible access
- API: Conversion between height channel indices and channel identifiers, e.g. for
  database migration
- API: Channel information reports data kind, data unit, lateral unit, whether it holds
  heights, its identifier and height index
- API: X3P writer for ISO 5436-2 / ISO 25178-72 files
- API: Gwyddion (GWY) writer
- PERF: File format detection pre-filters by magic bytes before full parsing
- DOC: List of all 35+ supported file formats
- DOC: Container autocorrelation, power spectrum and variable bandwidth have proper
  docstrings
- BUG: Opening a file no longer passes a communicator to readers that do not support one
- BUILD: PyPI keywords for instrument manufacturers and file formats
- BUILD: Requires Python 3.10 or newer
- BUILD: Supports Python 3.14
- BUILD: Requires muGrid 0.103.0 or newer, which replaces muFFT
- MAINT: NumPy-based FFT for 1D topographies, which muGrid does not support
- MAINT: Pipeline functions keep docstrings and source code for IPython introspection

v1.19.1 (19Aug25)
-----------------

- MAINT: OPD: thresholds for masked data are those of Gwyddion
- BUG: JPK: files with thumbnails can be read

v1.19.0 (19Jul25)
-----------------

- ENH: ZON: Zstandard-compressed files
- ENH: Plot pipeline function for line scans
- BUG: ZIP files with multiple containers can be read

v1.18.3 (13Apr25)
-----------------

- BUILD: Removed support for Python 3.8; requires NumPy 2.0 or newer
- ENH: Encoding of masked arrays

v1.18.2 (16Mar25)
-----------------

- MAINT: Scale-dependent statistics of containers pass keyword arguments on to the
  functions for single topographies

v1.18.1 (04Mar25)
-----------------

- BUG: Single-column text files can be read again

v1.18.0 (03Mar25)
-----------------

- ENH: Reader for Wyko text files
- MAINT: Class-based text reader
- MAINT: Requires muFFT 0.93.0

v1.17.1 (09Oct24)
-----------------

- BUG: The ångström unit is converted to ASCII
- BUILD: Command line tools were missing from the Meson build

v1.17.0 (26Sep24)
-----------------

- API: Removed the unit entry of the info dictionary, deprecated since v0.95.0
- ENH: NMM: 2D topographic maps
- ENH: Command line tool for visualizing topographies
- ENH: JSON encoder that handles dates and NumPy arrays
- MAINT: Info metadata is validated
- MAINT: Plots are identical to Gwyddion's, with the origin in the top left corner
- BUG: MetroPro: fixed orientation
- BUG: GWY: fixed handling of undefined data
- BUG: POIR/OIR: fixed physical size
- BUG: SUR: files with comment or private sections can be read

v1.16.2 (03Aug24)
-----------------

- BUG: Derivatives return a fully masked array instead of raising an error for undefined
  data

v1.16.1 (03Aug24)
-----------------

- BUG: OPD: undefined data points are masked again
- BUG: Fixed derivatives of topographies with missing data

v1.16.0 (02Aug24)
-----------------

- ENH: Derivatives of topographies with missing data

v1.15.1 (31Jul24)
-----------------

- ENH: Convenience function for reading NMM profiles from separate files
- MAINT: Compatibility with SciPy 1.14

v1.15.0 (31Jul24)
-----------------

- ENH: Variable bandwidth analysis for topographies with missing data
- ENH: NMM (Nanomeasuring Machine) reader (.zip)
- MAINT: Supports NumPy 2.0 and newer
- MAINT: Version discovery through DiscoverVersion
- CI: macOS and Linux ARM wheels

v1.14.2 (24Jul24)
-----------------

- BUG: AL3D: fixed data layout

v1.14.1 (16Jun24)
-----------------

- MAINT: Requires NumPy below 2.0, which is not yet compatible

v1.14.0 (11Jun24)
-----------------

- MAINT: Depends on muFFT 0.91.2 instead of muSpectre

v1.13.13 (16Apr24)
------------------

- BUG: Fixed reader for MetroPro v3 files
- BUG: SUR: fixed data layout
- TST: Compatibility with the window functions of newer SciPy

v1.13.12 (21Mar24)
------------------

- MAINT: More robust CSV reader

v1.13.11 (07Mar24)
------------------

- BUG: Further fixes for the bearing area of topographies with undefined values

v1.13.10 (05Mar24)
------------------

- BUG: Correct bounds of the bearing area of topographies with undefined values

v1.13.9 (05Mar24)
-----------------

- BUG: Robust statistics use a relative tolerance in bisection

v1.13.8 (05Mar24)
-----------------

- BUG: Fixed MAD height for data with undefined values

v1.13.7 (05Mar24)
-----------------

- BUG: No longer always warns about the unit

v1.13.6 (04Mar24)
-----------------

- MAINT: The contact.engineering container reader can skip building the filter pipeline
  from container metadata
- MAINT: Wrong metadata in containers raises a warning instead of an error

v1.13.5 (04Mar24)
-----------------

- BUG: Own CSV parser replaces pandas, which failed to read CSV data from ZIP containers

v1.13.4 (29Feb24)
-----------------

- PERF: Faster MAD analysis

v1.13.3 (19Feb24)
-----------------

- BUG: Scale-dependent statistics of containers with distances outside any bandwidth

v1.13.2 (19Feb24)
-----------------

- BUG: Scale-dependent statistics with multiple components work for containers
- BUG: Fixed I/O with Python 3.12

v1.13.1 (13Feb24)
-----------------

- MAINT: Nelder-Mead optimization of MAD fits starts from the result of an RMS fit

v1.13.0 (13Feb24)
-----------------

- API: Standard RMS heights are `Rq` and `Sq`; `rms_height` returns the RMS height of the
  linearly interpolated topography
- API: Translating a nonperiodic topography raises an error
- ENH: Median absolute deviation (MAD)
- ENH: Median
- ENH: Exact bearing area model based on linear interpolation of the topography
- ENH: Arbitrary statistical functions for variable bandwidth analysis
- BUG: Decorated topographies can be translated
- BUG: Periodicity is checked before translation
- MAINT: Gwyddion images are flipped
- MAINT: C++ extension module
- MAINT: Uses the new contact.engineering interface to obtain download URLs

v1.12.2 (12Dec23)
-----------------

- BUG: Fixed issue with DI reader
- BUG: Fixed reading of some file formats inside containers

v1.12.1 (14Nov23)
-----------------

- BUG: Dektak CSV: fixed info dictionary

v1.12.0 (14Nov23)
-----------------

- ENH: Reader for Dektak CSV files

v1.11.3 (08Nov23)
-----------------

- BUG: NetCDF output of uniform line scans contained random x positions

v1.11.2 (28Oct23)
-----------------

- BUG: The data source is honored when reading contact.engineering containers

v1.11.1 (18Oct23)
-----------------

- BUG: HGT, NC and VK readers returned integer physical sizes
- MAINT: MetroPro images are flipped

v1.11.0 (11Sep23)
-----------------

- ENH: Reader for JPK (.jpk) files
- BUG: DATX: missing data points are masked

v1.10.1 (30Aug23)
-----------------

- BUG: Dektak OPDx: 16-bit integer values

v1.10.0 (29Aug23)
-----------------

- API: Channel information is never empty (was only the case for the GWY reader)
- ENH: Reader for WSxM (.stp, .top) files
- ENH: Reader for Sensofar XML SPM (.plux) files
- BUG: Non-square Gwyddion (.gwy) files can be read
- MAINT: More robust XY (line scan) and XYZ readers

v1.9.0 (28Aug23)
----------------

- ENH: Reader for Olympus OIR (.oir) and POIR (.poir) files
- ENH: Output of Deep Zoom data files (NetCDF or NumPy)
- BUG: SUR files with different units for lateral and height information can be read
- BUG: Fixed reading of some LEXT files

v1.8.0 (24Jul23)
----------------

- API: Container readers support multiple file formats
- API: `SurfaceContainer` is now `InMemorySurfaceContainer`
- ENH: Reader for Zygo DATX (.datx) files
- ENH: Reader for Olympus LEXT (.lext) files
- ENH: Reader for ZAG containers (.zag)
- BUG: Fixed scale-dependent curvature of areas
- BUG: Mitutoyo: more robust date parsing
- MAINT: Readers can be declarative, defining the file layout directly
- MAINT: SUR and PLU readers are declarative
- CI: macOS wheels (arm64 and x86_64)

v1.7.0 (27Jun23)
----------------

- API: The exception for unknown file formats is now `UnknownFileFormat`
- ENH: Reader for Gwyddion's native file format
- ENH: Reader for Sensofar SPM files (.plu, .apx)
- ENH: Reader for MicroProf files (.frt)
- ENH: Reader for HFM files (.hfm, text files)
- ENH: Readers report MIME types and typical file extensions
- BUG: Contents of ZIP files are detected as binary streams
- MAINT: Uses pandas for parsing text files

v1.6.3 (19Jun23)
----------------

- MAINT: DI: more robust date parsing
- MAINT: Updated URL for container downloads

v1.6.2 (17Apr23)
----------------

- BUG: MetroPro: the info dictionary is JSON serializable

v1.6.1 (10Apr23)
----------------

- BUG: AL3D: more robust detection of invalid values

v1.6.0 (05Apr23)
----------------

- ENH: Reader for Zygo MetroPro files

v1.5.0 (25Mar23)
----------------

- ENH: Analytical scalar roughness parameters of ideal self-affine PSDs
- ENH: Weighted integrals of the PSDs of containers
- ENH: Simulation of scanning probe artifacts
- TST: The test suite no longer depends on ContactMechanics
- MAINT: Switched from igor to igor2, which supports the latest NumPy

v1.4.1 (06Mar23)
----------------

- BUG: Readers no longer peek into streams, which some streams do not support
- BUG: DI: files may report more data than they hold

v1.4.0 (06Mar23)
----------------

- BUILD: Dropped Python 3.7 support
- MAINT: Unified naming of the file format examples
- ENH: Convenience decorator for creating simple pipeline functions
- ENH: Reader for NanoSurf easyScan (NID/EZD) files
- ENH: Reader for BCR-STM/BCRF files
- BUG: AL3D: wrong data layout
- BUG: SUR: files without acquisition time can be read

v1.3.3 (29Jan23)
----------------

- BUILD: Fixed version discovery inside a git repository other than the source of
  SurfaceTopography

v1.3.2 (18Jan23)
----------------

- ENH: Pipeline function for superposing topographies
- BUG: Repaired compound and translated topographies

v1.3.1 (14Jan23)
----------------

- BUG: AL3D: wrong unit in the channel information
- BUG: SUR: wrong info dictionary in the channel information
- BUG: Scanning probe artifact analysis is numerically more robust

v1.3.0 (29Dec22)
----------------

- BUILD: Eigen3 replaces LAPACK, removing an external dependency
- CI: Binary wheels for manylinux and Python 3.7 to 3.10

v1.2.6 (29Dec22)
----------------

- BUG: SUR: wrong height scale factor in the channel information
- BUILD: Yet another fix for version discovery when installing from the source package

v1.2.5 (05Dec22)
----------------

- BUILD: Another fix for version discovery when installing from the source package

v1.2.4 (04Dec22)
----------------

- BUILD: Fixed version discovery when installing from the source package

v1.2.3 (02Dec22)
----------------

- BUG: All readers report the same info dictionary for channels and topographies
- BUILD: Automatic version discovery for the Meson build

v1.2.2 (28Nov22)
----------------

- BUILD: Switched build system to Meson
- BUG: Fixed several bugs in the handling of height scale factors by readers
- BUG: ASC: files without scale factor
- TST: Readers handle height scale factors correctly

v1.2.1 (22Nov22)
----------------

- ENH: X3P: metadata
- MAINT: Class-based X3P reader
- MAINT: Requires muFFT 0.24.0, which avoids installation problems downstream but disables
  automatic MPI detection
- BUG: X3P: data was read in the wrong storage order

v1.2.0 (15Nov22)
----------------

- ENH: Reader for Park Systems TIFF files
- ENH: Reader for Keyence VK3 and VK6 files
- ENH: Reader for Alicona Imaging AL3D files
- ENH: Reader for Digital Surf SUR files
- MAINT: Removed pandas dependency
- BUG: Fixed VK4 reader

v1.1.0 (20Oct22)
----------------

- API: Scale-dependent statistical properties of containers return arrays or masked arrays
- ENH: Reader for Mitutoyo SurfTest Excel spreadsheets
- ENH: Reader for Keyence VK4 files
- DOC: contact.engineering paper in the bibliography
- BUG: Slope calculation checks for reentrant topographies
- BUG: Fixed aspect ratio of the convenience plotting function

v1.0.0 (06May22)
----------------

- ENH: Generalized HDF5 reader
- API: Removed the unit entry of the info dictionary
- API: Removed the scale factor property

v0.101.3 (24Jan22)
------------------

- BUG: NaNs are ignored when suggesting units

v0.101.2 (09Dec21)
------------------

- MAINT: Suppressed a warning of the sparse solver
- MAINT: DI: corrupt files are detected when opening

v0.101.1 (30Nov21)
------------------

- BUG: Averages ignore analyses that fail because of undefined data

v0.101.0 (25Nov21)
------------------

- API: Channel information reports whether data is uniform and has undefined points, if
  known from the file headers

v0.100.3 (24Nov21)
------------------

- BUG: IBW: the unit can be specified if the file has none
- BUG: HDF5: missing unit argument

v0.100.2 (23Nov21)
------------------

- API: Nonuniform line scans report whether they have undefined data
- PERF: MATLAB and OPD readers no longer read the full file when opening

v0.100.1 (22Nov21)
------------------

- MAINT: Geometry analysis in a separate module

v0.100.0 (22Nov21)
------------------

- ENH: Filling undefined data points with harmonic functions

v0.99.3 (15Nov21)
-----------------

- BUG: OPDx: dates in varying formats

v0.99.2 (15Nov21)
-----------------

- BUG: OPDx: dates are month first

v0.99.1 (15Nov21)
-----------------

- BUG: OPDx: fixed Unicode conversion
- BUG: A unit is always suggested

v0.99.0 (11Nov21)
-----------------

- ENH: Self-reported citations
- ENH: An error is raised if missing data points are unsupported
- ENH: Parallel computation of scalar roughness parameters
- ENH: The DZI writer can write metadata as JSON
- ENH: The DZI writer stores colorbar information in the metadata
- ENH: OPDx: line scans
- BUG: Equality of topographies compares heights
- BUG: The NetCDF reader checks consistency of units
- BUG: Topographies with undefined data are stored to NetCDF with an explicit mask; this
  was broken in some cases
- MAINT: Fourier derivatives are computed with muFFT
- MAINT: Rewritten OPDx reader; its height scale factor differs from the old reader
- MAINT: Variable bandwidth raises an error for topographies with undefined data points

v0.98.2 (25Oct21)
-----------------

- BUG: Containers raise a no-reliable-data error only if no topography contributes data
- MAINT: The DZI writer returns the list of written files

v0.98.1 (08Oct21)
-----------------

- ENH: Minimum number of data points for scale-dependent statistical properties
- BUG: Variable bandwidth and scale-dependent statistics raise the no-reliable-data error
  properly
- BUG: Several fixes for corner cases
- BUG: Fixed shape of derivative arrays

v0.98.0 (05Oct21)
-----------------

- API: Analysis functions raise errors for reentrant topographies or topographies without
  reliable data

v0.97.5 (03Oct21)
-----------------

- ENH: Kilometers as unit

v0.97.4 (30Sep21)
-----------------

- ENH: The resolution entry of the instrument is used as reliability cutoff

v0.97.3 (28Sep21)
-----------------

- ENH: The reliability cutoff applies to scale-dependent statistical analysis
- BUG: The reliability cutoff filtered the wrong end of the autocorrelation

v0.97.2 (24Sep21)
-----------------

- BUG: Segmentation faults from integer overflow in large bicubic interpolations

v0.97.1 (21Sep21)
-----------------

- BUG: Micrometers are written with the Greek mu sign
- BUG: A surface is reentrant already if two x values are identical

v0.97.0 (21Sep21)
-----------------

- ENH: Bandwidth of containers
- ENH: Ideal length units for containers
- ENH: Progress callbacks for some analysis functions
- ENH: Gaussian process regression for resampling noisy data onto an arbitrary grid
- ENH: Topographies report whether they are reentrant
- ENH: Averaged scale-dependent roughness parameters, autocorrelation, power spectrum and
  variable bandwidth of containers
- BUG: Several fixes in scale-dependent analyses
- API: Radial averages use a separate regression module; the default behavior has changed
- API: All analysis functions resample data onto a reasonable grid
- API: Resampling keeps bins without data, filled with NaN

v0.96.0 (13Sep21)
-----------------

- ENH: Generic scale-dependent statistical properties of containers
- ENH: Derivatives at arbitrary distances by linear interpolation
- ENH: Tip radius reliability analysis
- ENH: Third derivatives
- DOC: Description of the PyPI package
- API: Conversion to nonuniform line scans requires nonperiodic uniform line scans

v0.95.2 (09Sep21)
-----------------

- ENH: DI: instrument name
- ENH: The PSD for Fourier synthesis can be given as a function
- BUG: The mask is kept when reading and writing NetCDF
- DOC: The DI reader mentions the .spm extension
- DOC: Clearer error message for reentrant topographies
- DOC: Updated docstring of the bandwidth
- DOC: Docstrings in numpydoc format

v0.95.1 (12Jul21)
-----------------

- BUILD: Missing dependency on matplotlib
- BUILD: Wheels build when NumPy is not yet installed

v0.95.0 (07Jul21)
-----------------

- API: The distance in scale-dependent derivatives is the width of the lowest-order
  stencil, n times the grid spacing for the n-th derivative
- API: `variable_bandwidth` is now `variable_bandwidth_from_area`; new
  `variable_bandwidth_from_profile` for line-by-line analysis
- API: `checkerboard_detrend` is now `checkerboard_detrend_area`; new
  `checkerboard_detrend_profile` for line-by-line analysis
- API: Topographies have a unit property; the unit in the info dictionary will be removed
  in version 1.0
- API: Physical sizes can no longer be set after creating a topography
- API: Keyword argument `ninterpolate` renamed to `nb_interpolate`
- API: Conversion of nonuniform line scans to a uniform grid takes the number of points
  between closest points
- ENH: Writing Deep Zoom Image (DZI) files
- ENH: Checkerboard detrending to arbitrary polynomial order
- ENH: Scaling can rescale lateral dimensions
- ENH: Pipeline function for unit conversion
- ENH: Surface containers with pipeline functions for collections of topographies
- ENH: Reading and writing surface containers
- ENH: Retrieving surface containers from https://contact.engineering
- ENH: Scale-dependent statistical properties of containers
- ENH: Length scale below which scanning probe data are unreliable because of tip
  artifacts (arXiv:2106.16013)
- ENH: DI: 32-bit integer data

v0.94.0 (10Jun21)
-----------------

- TST: Tests for the existence of functions and properties for each kind of topography
- ENH: Height scale factor in reader channels
- ENH: Physical sizes or height scale factor cannot be overridden if fixed by the file
- BUG: RMS curvature of uniform line scans was missing
- BUG: A height scale factor passed when reading a topography takes effect
- BUG: Fixed reading 2D measurements from XYZ data

v0.93.0 (27Apr21)
-----------------

- API: Suffixes `_1D` and `_2D` are now `_from_profile` and `_from_area`
- API: `save` is now `to_matrix`
- ENH: Reader for Keyence ZON files
- ENH: Derivatives at different scales
- ENH: Scale-dependent slope and curvature
- ENH: Pipeline function for windowing
- ENH: The NetCDF reader reads from streams
- BUG: The info dictionary of a reader channel was not always used for the topography

v0.92.0 (15Mar21)
-----------------

- BUILD: Dropped support for Python 3.5
- ENH: RMS slope, curvature and Laplacian can filter short-wavelength roughness
- MAINT: Filters follow the pipeline model
- MAINT: Derivatives use muFFT derivative operators
- MAINT: Requires muFFT 0.12.0
- API: Nonuniform PSDs only up to a reasonable wavevector

v0.91.5 (19Oct20)
-----------------

- BUILD: Fixed differing versions in setup and requirements

v0.91.4 (16Oct20)
-----------------

- BUG: Reading NPY from binary streams
- BUG: Periodic flag of NPY files in some cases

v0.91.3 (22Sep20)
-----------------

- BUG: The area per point of uniform line scans is a scalar, not a tuple
- BUG: Suppressed assertion on the discarded imaginary part of the Fourier derivative

v0.91.2 (31Jul20)
-----------------

- BUG: Memory leak in the Python bindings of bicubic interpolation

v0.91.1 (23Jul20)
-----------------

- MAINT: Requires muFFT 0.10.0

v0.91.0 (02Jul20)
-----------------

- ENH: Topographies with undefined data filled in
- ENH: Spheres can put undefined data into the standoff or not
- BUG: Missing dependencies when installed as package
- BUG: Loading text files with trailing newline
- BUG: Wrong error message when loading an unknown text format
- BUG: The heights of compound topographies are called heights, not array

v0.90.0 (17Jun20)
-----------------

- MAINT: PyCo split into SurfaceTopography, ContactMechanics and Adhesion
- MAINT: Requires muFFT 0.9.1
- DOC: Documentation moved from the README to the docs folder

Change log for PyCo (previous name of the package)
==================================================

v0.57.0 (15May20)
-----------------

- MAINT: Supports the rewritten muFFT bindings
- ENH: Bicubic interpolation of 2D topography maps
- ENH: Fourier derivative of topographies
- BUG: The plastic area is computed in parallel
- BUG: The info dictionary could be modified by the user

v0.56.0 (26Feb20)
-----------------

- ENH: Readers orient topographies so that plots look like Gwyddion's
- BUG: MI: unknown unit "um"
- BUG: MI: missing channel name
- ENH: Self-affine random surfaces from the self-affine prefactor
- BUG: Fourier synthesis generates line scans again

v0.55.0 (14Feb20)
-----------------

- API: Readers report channel information consistently in a dedicated class
- ENH: Readers report a format identifier and are self-documenting
- ENH: Gwyddion text exports in English and German locales
- ENH: DI: acquisition date in the info dictionary
- BUG: DI: format detection did not work
- BUG: DI: fixed orientation
- DOC: Notebook showing how to plot 2D topographies
- TST: Demo notebook for plotting 2D topographies
- ENH: Adhesive ideal plastic simulations with soft-wall systems

v0.54.4 (20Dec19)
-----------------

- BUG: IBW: channels lacked the number of grid points

v0.54.3 (20Dec19)
-----------------

- BUG: Assertion from a wrong number of channel names

v0.54.2 (13Dec19)
-----------------

- BUG: RMS Laplacian of periodic topographies

v0.54.1 (13Dec19)
-----------------

- ENH: Higher-order derivatives of periodic surfaces
- ENH: New reader for Igor Binary Wave (IBW) files
- BUG: OPDx: binary streams
- BUG: Nonuniform topographies store and restore the periodic flag

v0.54.0 (06Dec19)
-----------------

- BUILD: Fixed installation after the Eigen repository moved
- ENH: Anisotropic cubic Green's functions
- BUG: NPY: streams
- BUG: OPDx: streams

v0.53.1 (21Nov19)
-----------------

- API: Detrending in center mode keeps periodicity; other modes make topographies
  nonperiodic

v0.53.0 (20Nov19)
-----------------

- API: Periodicity can be set when reading a topography
- API: The default window of the PSD depends on periodicity
- ENH: Fourier interpolation pipeline function
- API: Free systems no longer check boundaries by default
- BUG: Fourier synthesis padded topographies with an odd number of points by one line
- BUG: RMS curvature returned the RMS Laplacian
- ENH: gnuplot scripts for plotting logger output

v0.52.0 (25Aug19)
-----------------

- API: Constrained conjugate gradient returns the contact map (active set)
- BUG: Patch numbering was broken on some configurations since v0.51.2

v0.51.2 (08Aug19)
-----------------

- BUG: Patch numbering crashed for maps larger than 64k x 64k

v0.51.1 (07Aug19)
-----------------

- BUG: Physical sizes passed to readers
- BUG: Physical sizes are None for surfaces without physical size
- BUG: Runs and tests without mpi4py
- BUG: Multiple reads of a topography from one reader
- ENH: Inspection of pipelines
- CI: All serial and MPI tests pass on Travis CI

v0.51.0 (05Aug19)
-----------------

- API: Cleanup of the new reader interface

v0.50.2 (01Aug19)
-----------------

- BUG: Wrapped readers lacked the channel argument
- API: Wrapped readers name their channel "Default"

v0.50.1 (01Aug19)
-----------------

- BUG: Runs without an MPI installation
- BUG: DI: files with non-topographic data

v0.50.0 (31Jul19)
-----------------

- ENH: MPI parallelization of topographies, substrates and interactions
- ENH: Readers load files in parallel; they inspect files without loading them to choose a
  domain decomposition
- MAINT: Serial calculations use the MPI wrapper of NuMPI
- MAINT: Parallel L-BFGS of NuMPI
- BUILD: Removed Cython dependency; (parallel) FFTs are handled by muFFT
- TST: Tests partially converted to pytest; parallel tests run through runtests

v0.32.0 (15Jul19)
-----------------

- ENH: Autocorrelation and power spectrum can use FFT (fast) or brute force (slow)

v0.31.3 (07Jul19)
-----------------

- MAINT: Removed check for existing forces on boundaries (nonperiodic calculations only)

v0.31.1 (20May19)
-----------------

- BUG: Contact calculations run with detrended or scaled topographies
- MAINT: The hard-wall command line script uses the new topography interface

v0.31.0 (05Mar19)
-----------------

- ENH: Height-difference autocorrelation and variable bandwidth analysis of nonuniform
  line scans
- ENH: Conversion of uniform to nonuniform line scans
- BUG: Center detrending of nonuniform line scans minimizes the RMS height

v0.30.0 (15Feb19)
-----------------

- ENH: Nonuniform line scans, loaded from text files or constructed from arrays
- MAINT: New class structure for topographies and line scans
- DOC: Hard-wall simulation tutorial
- ENH: Second derivative and RMS curvature of nonuniform topographies
- ENH: Coordination counting for contact patches
- MAINT: Perimeter computed by coordination counting
- DOC: Sphinx documentation with usage notes
- API: New interface for creating topographies and line scans from data
- API: Pipelines are built from methods of topographies, e.g. scaling followed by
  detrending
- ENH: Uniform topographies and line scans can be periodic
- API: Units are stored in the info dictionary, set when creating a topography, instead of
  a property
- API: All topographies have physical sizes; readers default to the number of grid points
  if the file has none
- API: Removed the shape alias of the resolution
- API: Sizes and shapes are always tuples
- ENH: Topographies can be pickled
- API: Spheres are created by a generator function instead of a class
- API: Contact with a free elastic half-space raises an error when points at the boundary
  interact
- BUG: The distance between contact patches ignored periodicity
- BUG: The energy in Fourier space did not match the energy in real space (not used in
  simulations)
- API: Removed the full output option of shift and tilt
- BUG: Text files without leading spaces can be read
- ENH: Topographies can be read from memory buffers and binary streams
- BUG: The 2D autocorrelation, e.g. its radial average, was broken
- BUG: The 1D autocorrelation was broken for nonperiodic calculations

v0.18.0 (31Oct18)
-----------------

- API: Surfaces are now called topographies
- BUG: Attractive contact area of smooth contact systems
- BUG: Inflexion point of the smoothed LJ93 and VW82 potentials

v0.17.0 (06Jul18)
-----------------

- ENH: Height-difference autocorrelation function

v0.16.0 (23Oct17)
-----------------

- MAINT: Licensed under the MIT license

v0.15.0 (06Sep17)
-----------------

- ENH: Substrates of finite thickness
- ENH: Further DI file formats
- ENH: Smarter unit conversion of DI files

v0.14.1 (16Jun17)
-----------------

- BUILD: macOS compatibility fixes
- ENH: Hardness given as pressure is converted to internal units in the constrained CG
  solver

v0.14.0 (14Mar17)
-----------------

- ENH: Penetration hardness model for simple plastic calculations

v0.13.1 (07Mar17)
-----------------

- BUG: The periodic Green's function was offset by one lattice constant

v0.13.0 (13Jan17)
-----------------

- ENH: Further adhesive reference models (Maugis-Dugdale type for cylinder and wedge)
- ENH: Callback for the Polonsky & Keer optimizer
- BUILD: Option to compile shared-memory (OpenMP) parallel code

v0.12.0 (05Dec16)
-----------------

- ENH: Surfaces support masked arrays for undefined (missing) data points; Polonsky & Keer
  handles them
- ENH: Polonsky & Keer optimizes at constant pressure as well as constant displacement
- ENH: The hard-wall script accepts command line arguments
- MAINT: Scripts moved to a separate folder
- ENH: Command line tools for plotting surfaces and their PSD

v0.11.0 (21Sep16)
-----------------

- API: Tilted surfaces are now called detrended surfaces

v0.10.3 (16Sep16)
-----------------

- ENH: Reader for HGT files (NASA Shuttle Radar Topography Mission)
- BUG: The deprecated size setter broke the hard-wall example

v0.10.2 (29Aug16)
-----------------

- ENH: Reader for MATLAB files

v0.10.1 (22Aug16)
-----------------

- ENH: Center detrending, which subtracts the mean
- ENH: Getter and setter for the detrend mode
- ENH: String representation of the subtracted plane
- ENH: Area per point of surfaces

v0.10.0 (31Jul16)
-----------------

- ENH: Exponential adhesion potential from Martin's contact mechanics challenge, for
  hard-wall calculations; thanks to Joe Monty
- API: Surfaces have a unit property
- MAINT: Readers wrap raw data in a scaled surface for unit conversion
- CI: Travis CI integration

v0.9.4 (27Apr16)
----------------

- ENH: Greenwood-Tripp reference solution
- BUG: Many fixes in topography file readers

v0.9.3 (20Mar16)
----------------

- ENH: Wyko OPD reader (.opd)
- ENH: Digital Instruments Nanoscope reader (.di)
- ENH: Igor Binary Wave reader (.ibw)
- ENH: Detrending

v0.9.2 (06Mar16)
----------------

- ENH: X3P reader (.x3p)
- ENH: Automatic file format detection
