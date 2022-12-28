# https://mesonbuild.com/Python-module.html

eigen3_incdir = include_directories('../external/eigen3')

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
    include_directories: [eigen3_incdir],
    dependencies: [ # List of dependencies
        python.dependency() # Add Python.h as dependency
    ]
)
