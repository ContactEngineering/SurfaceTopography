"""
pydantic models for the `info` dictionary which stores metadata
"""

from datetime import datetime
from typing import Optional, Union

import pydantic


class ValueAndUnitModel(pydantic.BaseModel):
    value: float
    # The unit may be None for data without unit information
    unit: Optional[str] = None


class InstrumentParametersModel(pydantic.BaseModel):
    # Name of the instrument
    name: Optional[str] = None
    # Measurement resolution (as a simple cutoff of lateral scales)
    resolution: Optional[ValueAndUnitModel] = None
    # Tip radius (for scanning probe measurements)
    tip_radius: Optional[ValueAndUnitModel] = None


class InstrumentModel(pydantic.BaseModel):
    name: Optional[str] = None
    vendor: Optional[str] = None
    serial: Optional[str] = None
    software: Optional[str] = None
    parameters: Optional[InstrumentParametersModel] = None


class InfoModel(pydantic.BaseModel):
    # The `info` dictionary is documented as free form: it can carry
    # auxiliary data that is never interpreted by this library but used by
    # third-party code. Unknown keys must therefore be preserved. (pydantic's
    # default `extra='ignore'` would silently discard them.)
    model_config = pydantic.ConfigDict(extra='allow')

    # Date and time of the measurement
    acquisition_time: Optional[datetime] = None
    # Instrument information
    instrument: Optional[InstrumentModel] = None
    # Finally, allow attachment of raw metadata that will depend on the reader
    raw_metadata: Optional[Union[dict, list]] = None

    # Name of channel
    channel_name: Optional[str] = None
    # Datafile info is attached by container readers
    datafile: Optional[dict] = None
