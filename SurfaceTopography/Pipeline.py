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


def pipeline_function(parent):
    """
    Simple convenience decorator that turns a function into a pipeline
    function with delayed (lazy) execution of the pipeline.
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
                state = super().__getstate__(), self._kwargs
                return state

            def __setstate__(self, state):
                """ Upon unpickling, it is called with the unpickled state
                Keyword Arguments:
                state -- result of __getstate__
                """
                superstate, self._kwargs = state
                super().__setstate__(superstate)

            def __getattr__(self, name):
                if name in self._kwargs:
                    return self._kwargs[name]
                else:
                    return getattr(self.parent_topography, name)

            def heights(self):
                return func(self.parent_topography, *self._args, **self._kwargs)

        return PipelineClass

    return _pipeline_function
