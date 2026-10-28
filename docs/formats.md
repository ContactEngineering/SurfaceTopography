# Supported file formats

```python
from SurfaceTopography.IO import readers, lookup_reader_by_format
```

`readers` is the list of registered reader classes, `lookup_reader_by_format`
maps format identifiers to them. Every reader class describes its format:

| Class method | Returns |
|-|-|
| `format()` | format identifier, the value of the `format` argument of [`open_topography`](reading.md#opening-a-file) |
| `name()` | short name of the format |
| `description()` | description of the format (Markdown) |
| `file_extensions()` | typical file extensions, without the dot |
| `mime_types()` | MIME types |

The formats are detected from the file contents, not from the file extension
(see [format detection](reading.md#format-detection)). The extensions listed
below are those that the instruments typically write.

## Topography files

| Format | Identifier | Extensions | Description |
|-|-|-|-|
| Alicona Imaging AL3D | `al3d` | `.al3d` | Bruker Alicona focus-variation instruments, e.g. InfiniteFocus |
| BCR-STM | `bcr` | `.bcr`, `.bcrf` | BCR-STM and BCRF interchange files for scanning probe data |
| Bruker/Veeco/DI Nanoscope | `di` | `.spm`, `.001`–`.005` | Digital Instruments, Veeco and Bruker Nanoscope/Dimension AFMs, version 4.3 and later |
| Dektak OPDx | `opdx` | `.opdx` | Vision64 software of Bruker Dektak XT stylus profilometers |
| Digital Metrology OmniSurf3D | `os3d` | `.os3d` | OmniSurf3D surface analysis software |
| Digital Surf Mountains | `mnt` | `.mnt` | Mountains documents; see [Digital Surf MNT](file-formats/mnt.md) |
| Digital Surf SUR | `sur` | `.sur` | Native format of MountainsMap, written directly by instruments of several vendors |
| FRT MicroProf | `frt` | `.frt` | MicroProf optical profilometers by FRT |
| Gwyddion | `gwy` | `.gwy` | Native format of [Gwyddion](http://gwyddion.net/) |
| Hierarchical data format (HDF5) | `h5` | `.h5` | Generic HDF5; every two-dimensional numerical dataset is a channel |
| Igor binary wave | `ibw` | `.ibw` | Igor Pro waves, used by Asylum Research (Oxford Instruments) AFMs |
| ISO 25178-71 SDF | `sdf` | `.sdf` | Surface Data File, ASCII and binary variants |
| JPK image scan | `jpk` | `.jpk` | TIFF-based image scans of JPK (Bruker) AFMs |
| Keyence VK | `vk` | `.vk3`, `.vk4`, `.vk6`, `.vk7` | Keyence VK laser scanning confocal microscopes |
| Keyence ZON | `zon` | `.zon` | Keyence VR one-shot 3D measuring macroscopes |
| KLA Zeta ZMG | `zmg` | `.zmg` | KLA Zeta optical profilers |
| Mahr MarWin QEP | `qep` | `.qep` | ZIP archive with measured and reference profile (line scans) of Mahr MarWin instruments |
| MATLAB | `mat` | `.mat` | MATLAB workspace files including version 7.3 (HDF5); every two-dimensional numerical array is a channel |
| Mitutoyo SurfTest | `mitutoyo` | `.xlsx` | Excel export of Mitutoyo SurfTest roughness testers (line scan) |
| Molecular Imaging (MI) | `mi` | `.mi` | Agilent Technologies (Molecular Imaging) AFMs |
| NanoFocus NMS | `nms` | `.nms` | NanoFocus confocal microscopes, e.g. µsurf |
| Nanomeasuring machine (NMM) | `nmm` | `.zip` | ZIP archive of SIOS nanomeasuring machine scans (DSC and DAT files) |
| Nanosurf easyScan | `ezd` | `.ezd`, `.nid` | Nanosurf scanning probe microscopes |
| NASA SRTM | `hgt` | `.hgt` | Shuttle Radar Topography Mission elevation data |
| Network Common Data Format (NetCDF) | `nc` | `.nc` | Classic NetCDF (version 3) written by [`to_netcdf`](reading.md#writing-files); parallel I/O |
| NT-MDT | `mdt` | `.mdt` | NT-MDT Spectrum Instruments AFMs, e.g. Solver and Ntegra; scanned-data frames only |
| NumPy array (NPY) | `npy` | `.npy` | A single one- or two-dimensional array; parallel I/O |
| NumPy array archive (NPZ) | `npz` | `.npz` | Archive written by `numpy.savez`; every two-dimensional array is a channel |
| Olympus LEXT | `lext` | `.lext` | TIFF-based files of Olympus LEXT laser scanning confocal microscopes |
| Olympus OIR | `oir` | `.oir` | Olympus (Evident) laser scanning microscopes |
| Olympus packed OIR | `poir` | `.poir` | ZIP archive of OIR files |
| Park Systems TIFF | `ps` | `.tiff`, `.tif` | TIFF-based format of Park Systems AFMs |
| Plain text | `asc` | `.txt`, `.asc`, `.dat` | Height matrix with optional header (Wyko, SPIP, Attocube, Nova, Gwyddion exports) |
| Plain text (x,y,z coordinates) | `xyz` | `.xyz`, `.hfm`, `.csv` | Two columns (line scan, possibly nonuniform) or three columns (map on a regular grid); HFM and Dektak headers |
| Sensofar PLU | `plu` | `.plu`, `.apx` | Sensofar 3D optical profilometers |
| Sensofar PLUX | `plux` | `.plux` | ZIP archive with XML metadata and binary layers, Sensofar 3D optical profilometers |
| TrueMap TMD | `tmd` | `.tmd` | TrueGage TrueMap surface metrology software |
| WSxM | `wsxm` | `.cur`, `.stp`, `.tom`, `.top` | [WSxM](http://www.wsxm.eu/) scanning probe microscopy software |
| Wyko OPD | `opd` | `.opd` | Vision software of Bruker Wyko white-light interferometers |
| XML 3D surface profile (X3P) | `x3p` | `.x3p` | ISO 25178-72 (ISO 5436-2) XML container, see [openGPS](http://www.opengps.eu/) |
| Zygo DATX | `datx` | `.datx` | HDF5-based files of the Zygo Mx software |
| Zygo MetroPro DAT | `metropro` | `.dat` | Binary files of the Zygo MetroPro software |

The full description of a format is returned by `description()`, e.g.

```python
from SurfaceTopography.IO import lookup_reader_by_format

print(lookup_reader_by_format["nc"].description())
```

### Formats without lateral size or unit

The following formats store only the height values. Physical sizes (and,
optionally, the unit and a height scale factor) must be passed when reading
the topography, see [overriding metadata](reading.md#metadata-from-the-file-and-from-arguments):

| Identifier | Stored in the file |
|-|-|
| `npy`, `npz` | heights |
| `h5` | heights |
| `mat` | heights |
| `hgt` | heights as 16-bit integers (meters in SRTM data, but no unit is set); voids (value $-32768$) are undefined data |
| `asc` | heights; size and unit only if the file has a header that provides them |

### Parallel I/O

The NetCDF (`nc`) and NPY (`npy`) readers accept an MPI communicator and read
only the subdomain of every process, see
[MPI-parallel reading](reading.md#mpi-parallel-reading). All other readers
read the whole file on every process.

### Order of detection

During format detection, the readers are tried in the order of the list
`readers`. Formats that are specializations of others come first:

- SDF before plain text, because ASCII SDF files are valid plain text files.
- x,y,z coordinates before plain text, because three-column files are a
  special case of a height matrix.
- NetCDF and Zygo DATX before HDF5, because NetCDF 4 and DATX files are HDF5
  files.
- NPZ after the readers of other ZIP-based formats, because it may accept any
  ZIP archive.
- NanoFocus NMS and NASA SRTM last, because their files have no
  identifying signature (magic bytes).

To bypass detection, pass the identifier from the table as `format` to
[`open_topography`](reading.md#opening-a-file).

## Container files

Containers bundle several topographies, e.g. the measurements of a single
sample. They are read with `open_container` and `read_container`, see
[containers](containers.md).

| Format | Identifier | Extensions | Description |
|-|-|-|-|
| contact.engineering | `ce` | `.zip` | Digital surface twins exported from [contact.engineering](https://contact.engineering/) |
| Keyence ZAG | `zag` | `.zag` | Keyence containers |

## Writers

Uniform topographies can be written to the following formats; see
[writing files](reading.md#writing-files).

| Method | Format | Readable by |
|-|-|-|
| `to_netcdf` | NetCDF (also nonuniform line scans; parallel for domain-decomposed topographies) | `nc` |
| `to_x3p` | X3P (two-dimensional maps) | `x3p` |
| `to_gwy` | Gwyddion (two-dimensional maps) | `gwy` |
| `to_matrix` | plain text matrix of heights | `asc` |
| `to_dzi` | Deep Zoom Image (tiled images for web viewers) | — |
| `SurfaceTopography.IO.NPY.save_npy` | NPY (parallel) | `npy` |
