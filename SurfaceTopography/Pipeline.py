#
# Copyright 2023 Lars Pastewka
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

"""
Decorator for defining pipeline functions.
"""


def pipeline_function(parent):
    """
    Decorator that turns a function into a pipeline class with delayed
    (lazy) execution.

    The decorated function `func(topography, *args, **kwargs)` must return
    the heights computed from the parent topography. The decorator returns
    a class derived from `parent` (e.g. `DecoratedUniformTopography`) whose
    constructor stores the parent topography and the arguments; `heights()`
    calls `func` on every invocation. Keyword arguments are accessible as
    attributes of the pipeline object; the special keyword arguments
    `unit` and `info` are passed to the base class. The class must be
    registered with `register_function` to be available as a method.

    Parameters
    ----------
    parent : type
        Base class of the pipeline step, typically
        `DecoratedUniformTopography` or `DecoratedNonuniformTopography`.

    Returns
    -------
    decorator : callable
        Decorator that maps a function onto the pipeline class.

    Examples
    --------
    ::

        @pipeline_function(DecoratedUniformTopography)
        def scale_by(topography, factor):
            return topography.heights() * factor

        UniformTopographyInterface.register_function("scale_by", scale_by)
    """

    def _pipeline_function(func):
        class PipelineClass(parent):
            def __init__(self, topography, *args, **kwargs):
                """
                This topography wraps a parent topography and applies the pipeline
                function.

                Arguments
                ---------
                topography : :obj:`UniformTopographyInterface`
                    Parent topography
                unit : str, optional
                    Target unit. This is simply used to update the metadata, not for
                    determining scale factors. (Default: None)
                info : dict, optional
                    Updated entries to the info dictionary. (Default: {})
                """
                if 'unit' in kwargs:
                    unit = kwargs['unit']
                    del kwargs['unit']
                else:
                    unit = None

                if 'info' in kwargs:
                    info = kwargs['info']
                    del kwargs['info']
                else:
                    info = {}

                super().__init__(topography, unit=unit, info=info)
                self._args = args
                self._kwargs = kwargs

            def __getstate__(self):
                """ is called and the returned object is pickled as the contents for
                    the instance
                """
                state = super().__getstate__(), self._args, self._kwargs
                return state

            def __setstate__(self, state):
                """ Upon unpickling, it is called with the unpickled state
                Keyword Arguments:
                state -- result of __getstate__
                """
                superstate, self._args, self._kwargs = state
                super().__setstate__(superstate)

            def __getattr__(self, name):
                # Expose keyword arguments as attributes; everything else
                # goes through the regular dispatch of the parent class.
                # (Delegating to `getattr(self.parent_topography, name)`
                # here would silently apply chained pipeline functions to
                # the *parent*, discarding this transformation.)
                if name != '_kwargs':
                    try:
                        kwargs = object.__getattribute__(self, '_kwargs')
                    except AttributeError:
                        kwargs = {}
                    if name in kwargs:
                        return kwargs[name]
                return super().__getattr__(name)

            def heights(self):
                return func(self.parent_topography, *self._args, **self._kwargs)

        return PipelineClass

    return _pipeline_function
