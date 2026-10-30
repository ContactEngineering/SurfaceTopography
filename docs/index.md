# SurfaceTopography

*Read and analyze surface topographies with Python.* SurfaceTopography
represents line scans and topography maps on uniform and nonuniform grids,
reads the file formats of most commercial profilometers and scanning probe
microscopes, and computes standard and scale-dependent roughness parameters.

## Highlights

- **One interface for all measurements.** Topography maps, uniform line scans
  and nonuniform line scans expose the same methods, independent of the
  instrument the data came from.
- **More than 40 file formats.** Readers detect the format from the file
  content and report channels, physical sizes, units and instrument metadata.
- **Statistics with defined discretizations.** Root-mean-square height, slope
  and curvature, power spectral density, height-difference autocorrelation,
  variable bandwidth analysis and bearing area, for maps and for profiles.
- **Scale-dependent roughness parameters** from finite-difference derivatives
  at arbitrary scales, with a reliability cutoff for scanning probe artifacts.
- **Pipelines.** Detrending, scan-line alignment, filtering, interpolation and
  filling of missing data return new topographies that record how they were
  derived from the raw data.
- **Containers** combine several measurements of the same surface and average
  their statistics over the combined bandwidth.
- **Optional MPI.** Topographies can be domain-decomposed across MPI processes;
  serial use needs no MPI installation.

## A first example

```python
from SurfaceTopography import read_topography

# The file format is detected from the file content
topography = read_topography("surface.opd")

print(topography.nb_grid_pts, topography.physical_sizes, topography.unit)
print("Sq =", topography.Sq())

# Remove a tilt and compute the power spectral density
detrended = topography.detrend(detrend_mode="height")
q, C = detrended.power_spectrum_from_profile()

# Plot the topography; x is the first array axis
import matplotlib.pyplot as plt
plt.pcolormesh(*detrended.positions_and_heights())
plt.colorbar(label=f"Height ({detrended.unit})")
plt.show()
```

## Where to go next

- [Installation](installation.md): `pip install SurfaceTopography`, and the
  optional `mpi` extra.
- [Topographies](topographies.md): the topography classes, grids, units,
  orientation, missing data, pipelines and MPI.
- [Reading and writing files](reading.md): readers, channels and output
  formats; [supported file formats](formats.md).
- [Pipelines](pipelines.md): detrending, scan-line alignment, filtering,
  interpolation and imputation.
- [Scalar statistics](statistics.md), [spectral analysis](spectral.md) and
  [scale-dependent parameters](scale-dependent.md): the analysis functions and
  their definitions.
- [Containers](containers.md): collections of measurements of one surface.
- [Synthetic topographies](generation.md): self-affine random surfaces and
  simple geometries.
- [Declarative readers](declarative-readers.md): how file formats are
  described and executed.
- [Contributing](contributing.md), [Testing](testing.md),
  [Changelog](changelog.md).

## Citing

If you use this code, please cite

- T. D. B. Jacobs, T. Junge, L. Pastewka,
  *Quantitative characterization of surface topography using spectral analysis*,
  [Surf. Topogr. Metrol. Prop. 5, 013001 (2017)](https://doi.org/10.1088/2051-672X/aa51f8)
- M. C. Röttger et al.,
  *Contact.engineering—Create, analyze and publish digital surface twins from topography measurements across many scales*,
  [Surf. Topogr. Metrol. Prop. 10, 035032 (2022)](https://doi.org/10.1088/2051-672X/ac860a)

Every analysis function records the publications it is based on; see
[Topographies](topographies.md#bibliography) for how to retrieve them.

## Funding

Development of this project is funded by the
[European Research Council](https://erc.europa.eu) within
[Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the
[Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within projects
[PA 2023/2](https://gepris.dfg.de/gepris/projekt/258153560) and
[EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).
