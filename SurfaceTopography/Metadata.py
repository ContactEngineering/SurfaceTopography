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


# Values that file formats write when the serial number is not known
_SERIAL_PLACEHOLDERS = {"", "0", "not available", "n/a", "na", "none", "unknown"}


class InstrumentModel(pydantic.BaseModel):
    name: Optional[str] = None
    vendor: Optional[str] = None
    # Serial number of the instrument itself: the whole system or, for
    # modular instruments such as scanning probe microscopes, the controller.
    # This identifies the measurement station.
    serial: Optional[str] = None
    # Serial number of the scanner (scan head). Scanners are exchangeable
    # between controllers and carry their own calibration, so this does not
    # identify the station.
    scanner_serial: Optional[str] = None
    software: Optional[str] = None
    parameters: Optional[InstrumentParametersModel] = None

    @pydantic.field_validator("serial", "scanner_serial", mode="before")
    @classmethod
    def _normalize_serial(cls, value):
        """Strip padding and map placeholders for unknown serials to None."""
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        value = str(value).strip().strip("\x00").strip()
        if value.lower() in _SERIAL_PLACEHOLDERS:
            return None
        return value


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
