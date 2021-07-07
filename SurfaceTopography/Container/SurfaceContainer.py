#
# Copyright 2020-2021 Lars Pastewka
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


class SurfaceContainer(object):
    _functions = {}

    def __init__(self, topographies=[]):
        self._topographies = topographies

    def __len__(self):
        return len(self._topographies)

    def __getitem__(self, item):
        return self._topographies[item]

    def apply(self, name, *args, **kwargs):
        self._functions[name](self, *args, **kwargs)

    def __getattr__(self, name):
        if name in self._functions:
            def func(*args, **kwargs):
                return self._functions[name](self, *args, **kwargs)

            func.__doc__ = self._functions[name].__doc__
            return func
        else:
            raise AttributeError("Unkown attribute '{}' and no analysis or pipeline function of this name registered"
                                 "(class {}). Available functions: {}".format(name, self.__class__.__name__,
                                                                              ', '.join(self._functions.keys())))

    def __dir__(self):
        return sorted(super().__dir__() + [*self._functions])

    @classmethod
    def register_function(cls, name, function):
        cls._functions.update({name: function})
