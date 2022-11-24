# https://mesonbuild.com/Python-module.html

# Build and install the extension module
module = python.extension_module(
    '_SurfaceTopography', # Name of the module
    [  # List of sources
        'autocorrelation.cpp',
        'bicubic.cpp',
        'module.cpp',
        'patchfinder.cpp'
    ],
    install: true,  # Install it
    dependencies: [ # List of dependencies
        python.dependency() # Add Python.h as dependency
    ],
    subdir: 'SurfaceTopography', # Folder relative to site-packages to install to
)
