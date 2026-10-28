#
# Copyright 2020-2021, 2023 Lars Pastewka
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import abc
from functools import update_wrapper

from ..Support.Deprecation import deprecated as deprecation_warning


class SurfaceContainer(metaclass=abc.ABCMeta):
    """
    Abstract base class for a collection of topographies (maps and/or line
    scans) that describe the same surface, typically measured with different
    instruments or at different resolutions.

    A container behaves like a read-only sequence of topographies
    (`len(container)`, `container[i]`, iteration). Analysis functions are
    registered on the class with `register_function` and become available as
    methods of all containers, e.g. `container.power_spectrum(unit='µm')`.
    """

    _functions = {}

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    def _function_registry(self):
        """
        Return the merged dictionary of functions registered on this class
        and its bases. See `AbstractTopography._function_registry`.
        """
        functions = {}
        for klass in reversed(type(self).__mro__):
            functions.update(klass.__dict__.get('_functions', {}))
        return functions

    def apply(self, name, *args, **kwargs):
        """
        Call the registered analysis function `name` on this container.
        Equivalent to `getattr(self, name)(*args, **kwargs)`.
        """
        return self._function_registry()[name](self, *args, **kwargs)

    def __getattr__(self, name):
        functions = self._function_registry()
        if name in functions:

            def func(*args, **kwargs):
                return functions[name](self, *args, **kwargs)

            update_wrapper(func, functions[name])
            return func
        else:
            raise AttributeError(
                "Unkown attribute '{}' and no analysis or pipeline function of this name registered"
                "(class {}). Available functions: {}".format(
                    name, self.__class__.__name__, ", ".join(functions.keys())
                )
            )

    def __dir__(self):
        return sorted(super().__dir__() + [*self._function_registry()])

    @classmethod
    def register_function(cls, name, function, deprecated=False):
        """
        Register an analysis function that becomes available as a method
        `name` of all instances of this class and its subclasses.

        Parameters
        ----------
        name : str
            Name of the method.
        function : callable
            Function taking the container as first argument.
        deprecated : bool, optional
            Emit a deprecation warning when the function is called.
            (Default: False)
        """
        if deprecated:
            function = deprecation_warning()(function)
        if '_functions' not in cls.__dict__:
            # Copy on write: registering on a subclass must not leak into
            # the shared base-class registry
            cls._functions = {}
        cls._functions[name] = function


class InMemorySurfaceContainer(SurfaceContainer):
    """
    A container that holds a list of topography objects in memory.

    Parameters
    ----------
    topographies : list of topographies, optional
        The topographies (maps and/or line scans) of this container.
        (Default: [])
    """

    def __init__(self, topographies=[]):
        self._topographies = topographies

    def __len__(self):
        return len(self._topographies)

    def __getitem__(self, item):
        return self._topographies[item]


class LazySurfaceContainer(SurfaceContainer):
    """
    A container that holds callables that construct the topographies on
    demand. Each element access (`container[i]`, iteration) calls the
    respective callable and therefore reads the topography again; use
    `read_all` to load all topographies into memory once.

    Parameters
    ----------
    readers : list of callables, optional
        Each callable takes no arguments and returns a topography.
        (Default: [])
    info : dict, optional
        Metadata of the surface, e.g. the surface entry of the container
        metadata file. (Default: {})
    """

    def __init__(self, readers=[], info={}):
        self._readers = readers
        self._info = info

    def __len__(self):
        return len(self._readers)

    def __getitem__(self, item):
        return self._readers[item]()

    def read_all(self):
        """
        Load all topographies into memory.

        Returns
        -------
        container : :obj:`InMemorySurfaceContainer`
        """
        return InMemorySurfaceContainer([x for x in self])

    @property
    def info(self):
        """Metadata dictionary of the surface."""
        return self._info
