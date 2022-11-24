    # https://mesonbuild.com/Python-module.html

# Pure Python sources
python_sources = [
    '__init__.py',
    'Bibliography.py',
    'Deprecation.py',
    'Interpolation.py',
    'Regression.py',
    'UnitConversion.py'
]

# Install pure Python
python.install_sources(
    python_sources,
    pure: false, # Will be installed next to binaries
    subdir: 'SurfaceTopography'  # Folder relative to site-packages to install to
)
