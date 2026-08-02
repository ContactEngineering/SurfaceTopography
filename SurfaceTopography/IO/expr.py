#
# Copyright 2026 Lars Pastewka
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
Serializable expression mini-language for the declarative file layouts.

PROTOTYPE. The declarative readers currently express data-dependent
quantities (array shapes, validation conditions, conversions) as Python
lambdas. Lambdas cannot be serialized, which ties the format descriptions to
Python. This module provides drop-in replacements: expression objects that
are callable with the same signatures as the lambdas, but that can also be
serialized to a JSON-compatible AST and rehydrated, e.g. by an expression
interpreter written in another language.

The normative definition of the AST, its operator semantics and the
function registry is `docs/format_description_contract.rst`; the design
rationale is `docs/declarative_readers_design.rst`. Where this module and
the contract disagree, the contract wins and this module needs fixing.

Authoring vocabulary
--------------------
- `C`: the parser context, e.g. `C.header.nb_grid_pts_x`
- `V`: the value currently being processed (in validators, converters and
  array conversion functions)
- `F`: registered named functions, e.g. `F.dtype("<i2")`; the function
  registry is the fixed set of primitives that a foreign-language
  interpreter must provide
- `Tup(...)`: a tuple of expressions, e.g. an array shape
- `Cond(condition, then, otherwise)`: conditional expression

Standard Python operators (`+`, `-`, `*`, `/`, `//`, `%`, `&`, `|`, `^`,
`<<`, `>>`, comparisons, unary `-`), indexing and slicing are overloaded on
expression objects. Note that `and`/`or`/`not` cannot be overloaded in
Python; use `&`/`|` on boolean subexpressions (parenthesize, `&`/`|` bind
tighter than comparisons) or `Cond`.

Calling convention
------------------
Expression objects are callable so that they can stand in for the lambdas
the layout classes accept. The layout classes call these hooks either with
`(context)` or with `(value, context)`; a single dict-like argument is
interpreted as the context, a single non-dict argument as the value.

Examples
--------
>>> from SurfaceTopography.IO.binary import AttrDict
>>> shape = Tup(C.header.nb_grid_pts_y, C.header.row_bytes // 4)
>>> shape(AttrDict({'header': AttrDict({'nb_grid_pts_y': 3, 'row_bytes': 16})}))
(3, 4)
>>> is_valid = (V & 0x03) == 0
>>> is_valid(0x80, {})
True
>>> from_dict(is_valid.to_dict())(0x02, {})
False
"""

import base64
import operator

import dateutil.parser
import numpy as np

_BINARY_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "in": lambda a, b: a in b,
}

_UNARY_OPS = {
    "neg": operator.neg,
    "not": operator.not_,
}


def _unit_conversion_factor(from_unit, to_unit):
    from ..Support.UnitConversion import get_unit_conversion_factor

    return get_unit_conversion_factor(from_unit, to_unit)


def _mangle_length_unit(unit):
    from ..Support.UnitConversion import mangle_length_unit_utf8

    return mangle_length_unit_utf8(unit)


def _is_length_unit(unit):
    from ..Support.UnitConversion import is_length_unit

    return is_length_unit(unit)


def _make_datetime(
    year, month, day, hour, minute, second, utc_offset_minutes=None
):
    import datetime

    try:
        if utc_offset_minutes is None:
            return datetime.datetime(year, month, day, hour, minute, second)
        return datetime.datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            tzinfo=datetime.timezone(
                datetime.timedelta(minutes=utc_offset_minutes)
            ),
        )
    except ValueError:
        # E.g. all-zero date fields; the file carries no acquisition time
        return None


def _isnan(value):
    # The binary decoder sanitizes scalar NaNs in parsed metadata to None
    # (`null` in a document); the format description contract defines
    # `isnan(null)` as true.
    if value is None:
        return True
    return np.isnan(value)


def _zstd_reader(stream_obj):
    import zstandard

    return zstandard.ZstdDecompressor().stream_reader(stream_obj)


def _zlib_reader(stream_obj):
    import io
    import zlib

    return io.BytesIO(zlib.decompress(stream_obj.read()))


# Named function registry. This is the fixed set of primitives that an
# expression interpreter in another language must provide (or reject).
# The normative list, including which functions are capability-gated,
# is `docs/format_description_contract.rst`.
_FUNCTIONS = {
    "dtype": np.dtype,
    "float": float,
    "int": int,
    "str": str,
    "len": len,
    "abs": np.abs,
    "isnan": _isnan,
    "transpose": np.transpose,
    "flip": lambda arr, axis: np.flip(arr, axis),
    "reshape": lambda arr, shape: np.reshape(arr, shape),
    "isfinite": np.isfinite,
    "logical_not": np.logical_not,
    # Unpack a uint8 array into `count` bits, least-significant bit first
    # (e.g. ISO 5436-2 per-point validity masks)
    "unpackbits": lambda arr, count: np.unpackbits(arr, bitorder="little")[
        :count
    ],
    "strip": lambda s: s.strip(),
    "split": lambda s, separator: s.split(separator),
    "parse_datetime": dateutil.parser.parse,
    "make_datetime": _make_datetime,
    # POSIX timestamp to datetime (in local time, matching historic
    # reader behavior)
    "from_timestamp": lambda ts: __import__("datetime").datetime.fromtimestamp(ts),
    "unit_conversion_factor": _unit_conversion_factor,
    "mangle_length_unit": _mangle_length_unit,
    "is_length_unit": _is_length_unit,
    # Mapping utilities: build a mapping from a list of records (e.g. a
    # tag list), look up with a default, merge two mappings, drop a key
    "to_map": lambda records, key, value: {r[key]: r[value] for r in records},
    "get": lambda mapping, key, default: mapping.get(key, default),
    "merge": lambda a, b: {**a, **b},
    "omit": lambda mapping, key: {k: v for k, v in mapping.items() if k != key},
    # List utilities: collect the values of a key from those records that
    # have it; collect the values of all keys starting with a prefix (in
    # mapping order)
    "pluck": lambda records, key: [r[key] for r in records if key in r],
    "values_with_prefix": lambda mapping, prefix: [
        v for k, v in mapping.items() if k.startswith(prefix)
    ],
    # Capability-gated stream filters
    "zstd_reader": _zstd_reader,
    "zlib_reader": _zlib_reader,
}

# Registry functions that require a capability beyond `core`, per the
# format description contract. Used to compute a description's
# capability list.
FUNCTION_CAPABILITIES = {
    "zstd_reader": "zstd",
    "zlib_reader": "zlib",
}


def register_function(name, fun):
    """Register a named function usable as `F.<name>(...)` in expressions."""
    _FUNCTIONS[name] = fun


class Expr:
    """Base class of all expression nodes."""

    def evaluate(self, context, value=None):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # Dispatch the calling conventions used by the layout classes:
        # `(context)` for shapes, sizes, dtypes, names and conditions;
        # `(value, context, ...)` for validators and converters; a single
        # non-dict argument is a bare value (e.g. XML converters).
        if len(args) >= 2:
            value, context = args[0], args[1]
        elif len(args) == 1 and isinstance(args[0], dict):
            value, context = None, args[0]
        elif len(args) == 1:
            value, context = args[0], {}
        else:
            value, context = None, {}
        return self.evaluate(context, value)

    def __getitem__(self, index):
        return GetItem(self, index)

    def __getattr__(self, name):
        # Attribute access on an expression is mapping lookup, so paths can
        # continue past an index: `C.entries[0].prefix`. Underscore names
        # are reserved for the implementation (and special attribute
        # protocols such as copy/pickle probing must see AttributeError).
        if name.startswith("_"):
            raise AttributeError(name)
        return GetItem(self, name)

    def __neg__(self):
        return UnaryOp("neg", self)

    def isin(self, *values):
        """Membership test, e.g. `V.isin('KPK0', 'KPK1')`."""
        # A tuple expression, so that each value is encoded individually
        # (bytes values must become base64 `bytes` nodes)
        return BinaryOp("in", self, TupleExpr(*values))

    def __bool__(self):
        raise TypeError(
            "The truth value of an expression is undefined; expressions are "
            "evaluated lazily against a parser context."
        )


def _make_binop(op, reflected):
    def method(self, other):
        if reflected:
            return BinaryOp(op, ensure_expr(other), self)
        else:
            return BinaryOp(op, self, ensure_expr(other))

    return method


for _op, _name in [
    ("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "truediv"),
    ("//", "floordiv"), ("%", "mod"), ("&", "and"), ("|", "or"),
    ("^", "xor"), ("<<", "lshift"), (">>", "rshift"),
]:
    setattr(Expr, f"__{_name}__", _make_binop(_op, False))
    setattr(Expr, f"__r{_name}__", _make_binop(_op, True))
for _op, _name in [
    ("==", "eq"), ("!=", "ne"), ("<", "lt"), ("<=", "le"),
    (">", "gt"), (">=", "ge"),
]:
    setattr(Expr, f"__{_name}__", _make_binop(_op, False))


def ensure_expr(value):
    """Coerce a value into an expression node."""
    if isinstance(value, Expr):
        return value
    elif isinstance(value, bytes):
        return BytesLit(value)
    elif isinstance(value, (tuple, list)) and any(
        isinstance(v, Expr) for v in value
    ):
        return TupleExpr(*value)
    else:
        return Lit(value)


def _is_json_value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return True
    elif isinstance(value, (list, tuple)):
        return all(_is_json_value(v) for v in value)
    elif isinstance(value, dict):
        # JSON object keys are strings; other key types would be
        # silently stringified on serialization and no longer match on
        # lookup after rehydration
        return all(
            isinstance(key, str) and _is_json_value(v)
            for key, v in value.items()
        )
    else:
        return False


class Lit(Expr):
    """Literal (JSON-representable) value."""

    def __init__(self, value):
        self._value = value

    def evaluate(self, context, value=None):
        return self._value

    def to_dict(self):
        if not _is_json_value(self._value):
            raise ValueError(
                f"The literal {self._value!r} does not survive a JSON "
                f"round trip."
            )
        return {"kind": "lit", "value": self._value}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["value"])

    def __repr__(self):
        return repr(self._value)


class BytesLit(Expr):
    """Bytes literal, serialized as base64."""

    def __init__(self, value):
        if not isinstance(value, bytes):
            raise TypeError("BytesLit requires a bytes value.")
        self._value = value

    def evaluate(self, context, value=None):
        return self._value

    def to_dict(self):
        return {
            "kind": "bytes",
            "value": base64.b64encode(self._value).decode("ascii"),
        }

    @classmethod
    def _from_dict(cls, d):
        return cls(base64.b64decode(d["value"]))

    def __repr__(self):
        return repr(self._value)


class DictExpr(Expr):
    """Mapping with static string keys, e.g. for context restructuring."""

    def __init__(self, items):
        self._items = {key: ensure_expr(value) for key, value in items.items()}

    def evaluate(self, context, value=None):
        return {
            key: item.evaluate(context, value) for key, item in self._items.items()
        }

    def to_dict(self):
        return {
            "kind": "dict",
            "items": {key: item.to_dict() for key, item in self._items.items()},
        }

    @classmethod
    def _from_dict(cls, d):
        return cls({key: from_dict(item) for key, item in d["items"].items()})

    def __repr__(self):
        inner = ", ".join(f"{k!r}: {v!r}" for k, v in self._items.items())
        return f"DictExpr({{{inner}}})"


class Val(Expr):
    """The value currently being processed (validated or converted)."""

    def evaluate(self, context, value=None):
        return value

    def to_dict(self):
        return {"kind": "val"}

    @classmethod
    def _from_dict(cls, d):
        return cls()

    def __repr__(self):
        return "V"


class CtxRef(Expr):
    """Reference to an entry of the parser context, e.g. `C.header.magic`."""

    def __init__(self, path=()):
        self._path = tuple(path)

    def __getattr__(self, name):
        # Do not swallow special attribute protocols (copy, pickle, ...);
        # `__parent__` is a legitimate context key used by `CompoundLayout`.
        if name.startswith("__") and name.endswith("__") and name != "__parent__":
            raise AttributeError(name)
        return CtxRef(self._path + (name,))

    def evaluate(self, context, value=None):
        obj = context
        for segment in self._path:
            if isinstance(obj, dict):
                obj = obj[segment]
            else:
                obj = getattr(obj, segment)
        return obj

    def to_dict(self):
        return {"kind": "ctx", "path": list(self._path)}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["path"])

    def __repr__(self):
        return ".".join(("C",) + self._path)


class BinaryOp(Expr):
    def __init__(self, op, left, right):
        if op not in _BINARY_OPS:
            raise ValueError(f"Unknown binary operator `{op}`.")
        self._op = op
        self._left = ensure_expr(left)
        self._right = ensure_expr(right)

    def evaluate(self, context, value=None):
        return _BINARY_OPS[self._op](
            self._left.evaluate(context, value),
            self._right.evaluate(context, value),
        )

    def to_dict(self):
        return {
            "kind": "binop",
            "op": self._op,
            "args": [self._left.to_dict(), self._right.to_dict()],
        }

    @classmethod
    def _from_dict(cls, d):
        left, right = d["args"]
        return cls(d["op"], from_dict(left), from_dict(right))

    def __repr__(self):
        return f"({self._left!r} {self._op} {self._right!r})"


class UnaryOp(Expr):
    def __init__(self, op, arg):
        if op not in _UNARY_OPS:
            raise ValueError(f"Unknown unary operator `{op}`.")
        self._op = op
        self._arg = ensure_expr(arg)

    def evaluate(self, context, value=None):
        return _UNARY_OPS[self._op](self._arg.evaluate(context, value))

    def to_dict(self):
        return {"kind": "unop", "op": self._op, "arg": self._arg.to_dict()}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["op"], from_dict(d["arg"]))

    def __repr__(self):
        return f"{self._op}({self._arg!r})"


class Call(Expr):
    """Call to a function from the named function registry."""

    def __init__(self, name, args):
        self._func_name = name
        self._args = [ensure_expr(arg) for arg in args]

    def evaluate(self, context, value=None):
        try:
            fun = _FUNCTIONS[self._func_name]
        except KeyError:
            raise KeyError(
                f"Unknown function `{self._func_name}`; register it with "
                f"`register_function`."
            )
        return fun(*[arg.evaluate(context, value) for arg in self._args])

    def to_dict(self):
        return {
            "kind": "call",
            "name": self._func_name,
            "args": [arg.to_dict() for arg in self._args],
        }

    @classmethod
    def _from_dict(cls, d):
        return cls(d["name"], [from_dict(arg) for arg in d["args"]])

    def __repr__(self):
        return f"F.{self._func_name}({', '.join(repr(a) for a in self._args)})"


class TupleExpr(Expr):
    """Tuple of expressions, e.g. an array shape."""

    def __init__(self, *items):
        self._items = [ensure_expr(item) for item in items]

    def evaluate(self, context, value=None):
        return tuple(item.evaluate(context, value) for item in self._items)

    def to_dict(self):
        return {"kind": "tuple", "items": [item.to_dict() for item in self._items]}

    @classmethod
    def _from_dict(cls, d):
        return cls(*[from_dict(item) for item in d["items"]])

    def __repr__(self):
        return f"Tup({', '.join(repr(i) for i in self._items)})"


class Cond(Expr):
    """Conditional expression with short-circuit evaluation."""

    def __init__(self, condition, then, otherwise):
        self._condition = ensure_expr(condition)
        self._then = ensure_expr(then)
        self._otherwise = ensure_expr(otherwise)

    def evaluate(self, context, value=None):
        if self._condition.evaluate(context, value):
            return self._then.evaluate(context, value)
        else:
            return self._otherwise.evaluate(context, value)

    def to_dict(self):
        return {
            "kind": "cond",
            "condition": self._condition.to_dict(),
            "then": self._then.to_dict(),
            "otherwise": self._otherwise.to_dict(),
        }

    @classmethod
    def _from_dict(cls, d):
        return cls(
            from_dict(d["condition"]), from_dict(d["then"]), from_dict(d["otherwise"])
        )

    def __repr__(self):
        return f"Cond({self._condition!r}, {self._then!r}, {self._otherwise!r})"


class GetItem(Expr):
    """Indexing and slicing, e.g. `V[:, :C.header.nb_grid_pts_x]`."""

    def __init__(self, base, index):
        self._base = ensure_expr(base)
        self._index = index if isinstance(index, tuple) else (index,)
        self._was_tuple = isinstance(index, tuple)

    def evaluate(self, context, value=None):
        def resolve(part):
            if isinstance(part, slice):
                return slice(
                    *(
                        None if bound is None else ensure_expr(bound).evaluate(context, value)
                        for bound in (part.start, part.stop, part.step)
                    )
                )
            else:
                return ensure_expr(part).evaluate(context, value)

        parts = tuple(resolve(part) for part in self._index)
        if not self._was_tuple:
            (parts,) = parts
        return self._base.evaluate(context, value)[parts]

    @staticmethod
    def _encode_part(part):
        if isinstance(part, slice):
            return {
                "kind": "slice",
                "bounds": [
                    None if bound is None else ensure_expr(bound).to_dict()
                    for bound in (part.start, part.stop, part.step)
                ],
            }
        else:
            return ensure_expr(part).to_dict()

    @staticmethod
    def _decode_part(d):
        if isinstance(d, dict) and d.get("kind") == "slice":
            return slice(
                *(None if bound is None else from_dict(bound) for bound in d["bounds"])
            )
        else:
            return from_dict(d)

    def to_dict(self):
        return {
            "kind": "getitem",
            "base": self._base.to_dict(),
            "index": [self._encode_part(part) for part in self._index],
            "tuple": self._was_tuple,
        }

    @classmethod
    def _from_dict(cls, d):
        parts = tuple(cls._decode_part(part) for part in d["index"])
        if not d["tuple"]:
            (parts,) = parts
        return cls(from_dict(d["base"]), parts)

    def __repr__(self):
        return f"{self._base!r}[{', '.join(repr(p) for p in self._index)}]"


class _FunctionNamespace:
    """Builder for calls into the named function registry."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def make_call(*args):
            return Call(name, args)

        return make_call


_NODE_KINDS = {
    "lit": Lit,
    "bytes": BytesLit,
    "dict": DictExpr,
    "val": Val,
    "ctx": CtxRef,
    "binop": BinaryOp,
    "unop": UnaryOp,
    "call": Call,
    "tuple": TupleExpr,
    "cond": Cond,
    "getitem": GetItem,
}


def from_dict(d):
    """Rehydrate an expression from its JSON-compatible AST."""
    try:
        node_class = _NODE_KINDS[d["kind"]]
    except (TypeError, KeyError):
        raise ValueError(f"Cannot decode expression node: {d!r}")
    return node_class._from_dict(d)


# Authoring roots
C = CtxRef()
V = Val()
F = _FunctionNamespace()
Tup = TupleExpr
