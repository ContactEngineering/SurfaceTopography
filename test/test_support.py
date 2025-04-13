import json
import numpy as np

from SurfaceTopography.Support.JSON import ExtendedJSONEncoder


def test_json_encoder():
    assert (
        json.dumps({"a": np.array([3, 1])}, cls=ExtendedJSONEncoder) == '{"a": [3, 1]}'
    )
    assert (
        json.dumps({"a": np.ma.masked_invalid([np.nan, 1])}, cls=ExtendedJSONEncoder)
        == '{"a": [null, 1.0]}'
    )
