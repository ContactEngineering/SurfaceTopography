# https://mesonbuild.com/
project(
    'SurfaceTopography', # Project name
    'cpp', # Project type. We need a C++ compiler.
    default_options : ['cpp_std=c++17'], # Yes, we need C++17, at least for constexpr
    # This must be 'python', not 'python3': builds run inside a virtual
    # environment (pip/mesonpy build isolation), which provides 'python' on
    # all platforms but no 'python3.exe' on Windows -- there, 'python3'
    # resolves to some other interpreter on PATH that does not have the
    # DiscoverVersion build requirement installed. When building with bare
    # meson outside a venv (e.g. on Debian without python-is-python3),
    # create a venv first.
    version: run_command('python', '-m', 'DiscoverVersion', check: true).stdout().strip(),
)

# https://mesonbuild.com/Python-module.html
pymod = import('python')
python = pymod.find_installation('python3',
    required: true,
)

eigen = dependency('eigen3', version: '>=3.4.0')
pybind11 = dependency('pybind11')

subdir('cpp')
subdir('SurfaceTopography')