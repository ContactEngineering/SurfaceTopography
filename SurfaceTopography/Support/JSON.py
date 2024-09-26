import datetime
import decimal
import uuid
from json import JSONEncoder

import numpy as np

try:
    from jaxlib.xla_extension import ArrayImpl
except ModuleNotFoundError:
    ArrayImpl = np.ndarray


def nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, set):
        return [nan_to_none(v) for v in obj]
    elif isinstance(obj, np.ndarray) or isinstance(obj, ArrayImpl):
        if obj.ndim == 0:
            return nan_to_none(obj.item())
        else:
            return [nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# From Django:
# https://github.com/django/django/blob/main/django/utils/duration.py
def _get_duration_components(duration):
    days = duration.days
    seconds = duration.seconds
    microseconds = duration.microseconds

    minutes = seconds // 60
    seconds %= 60

    hours = minutes // 60
    minutes %= 60

    return days, hours, minutes, seconds, microseconds


# From Django:
# https://github.com/django/django/blob/main/django/utils/duration.py
def duration_iso_string(duration):
    if duration < datetime.timedelta(0):
        sign = "-"
        duration *= -1
    else:
        sign = ""

    days, hours, minutes, seconds, microseconds = _get_duration_components(duration)
    ms = ".{:06d}".format(microseconds) if microseconds else ""
    return "{}P{}DT{:02d}H{:02d}M{:02d}{}S".format(
        sign, days, hours, minutes, seconds, ms
    )


class ExtendedJSONEncoder(JSONEncoder):
    """
    Customized JSON encoder that gracefully handles:
    * numpy arrays, which will be converted to JSON arrays
    * NaNs and Infs, which will be converted to null
    * dates and times
    """

    _TYPE_MAP = {
        np.int_: int,
        np.intc: int,
        np.intp: int,
        np.int8: int,
        np.int16: int,
        np.int32: int,
        np.int64: int,
        np.uint8: int,
        np.uint16: int,
        np.uint32: int,
        np.uint64: int,
        np.float16: float,
        np.float32: float,
        np.float64: float,
        np.bool_: bool,
    }

    def default(self, obj):
        # Datetime portion takes from Django's JSON encoder:
        # https://github.com/django/django/blob/main/django/core/serializers/json.py
        # See "Date Time String Format" in the ECMA-262 specification.
        if isinstance(obj, datetime.datetime):
            r = obj.isoformat()
            if obj.microsecond:
                r = r[:23] + r[26:]
            if r.endswith("+00:00"):
                r = r.removesuffix("+00:00") + "Z"
            return r
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, datetime.time):
            if obj.utcoffset() is not None:
                raise ValueError("JSON can't represent timezone-aware times.")
            r = obj.isoformat()
            if obj.microsecond:
                r = r[:12]
            return r
        elif isinstance(obj, datetime.timedelta):
            return duration_iso_string(obj)
        elif isinstance(obj, (decimal.Decimal, uuid.UUID)):
            return str(obj)
        else:
            try:
                return self._TYPE_MAP[type(obj)](obj)
            except KeyError:
                # Pass it on the Django encoder
                return super().default(obj)

    def encode(self, obj, *args, **kwargs):
        # Solution suggested here:
        # https://stackoverflow.com/questions/28639953/python-json-encoder-convert-nans-to-null-instead
        obj = nan_to_none(obj)
        return super().encode(obj, *args, **kwargs)
