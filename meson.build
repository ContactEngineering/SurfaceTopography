# https://mesonbuild.com/
project(
    'SurfaceTopography', # Project name
    'cpp', # Project type. We need a C compiler.
    version: '1.3', # Project version
)

# https://mesonbuild.com/Python-module.html
pymod = import('python')
python = pymod.find_installation(
    required: true,
)

subdir('c')
subdir('SurfaceTopography')