"""
pydantic models for the `info` dictionary which stores metadata
"""

from datetime import datetime
from typing import Union

import pydantic


class ValueAndUnitModel(pydantic.BaseModel):
    value: float
    unit: str


class InstrumentParametersModel(pydantic.BaseModel):
    # Name of the instrument
    name: str = None
    # Measurement resolution (as a simple cutoff of lateral scales)
    resolution: ValueAndUnitModel = None
    # Tip radius (for scanning probe measurements)
    tip_radius: ValueAndUnitModel = None


class InstrumentModel(pydantic.BaseModel):
    name: str = None
    parameters: InstrumentParametersModel = None


class InfoModel(pydantic.BaseModel):
    # Date and time of the measurement
    acquisition_time: datetime = None
    # Instrument information
    instrument: InstrumentModel = None
    # Finally, allow attachment of raw metadata that will depend on the reader
    raw_metadata: Union[dict, list] = None

    # Name of channel
    channel_name: str = None
    # Datafile info is attached by container readers
    datafile: dict = None
