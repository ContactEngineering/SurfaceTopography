# https://mesonbuild.com/Python-module.html

# Pure Python sources
python_sources = [
    '__init__.py',
    'Version.py',
    'Exceptions.py',
    'FFTTricks.py',
    'Generation.py',
    'HeightContainer.py',
    'Metadata.py',
    'NonuniformLineScan.py',
    'Special.py',
    'UniformLineScanAndTopography.py'
]

# Install pure Python
python.install_sources(
    python_sources,
    subdir: 'SurfaceTopography'
)

# Also copy data files
install_data(
    'ChangeLog.md'
)

subdir('CLI')
subdir('Container')
subdir('Generic')
subdir('IO')
subdir('Models')
subdir('Nonuniform')
subdir('ScanningProbe')
subdir('Support')
subdir('Uniform')

