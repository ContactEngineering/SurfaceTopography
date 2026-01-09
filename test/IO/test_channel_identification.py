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
Tests for channel identification and selection functionality.
"""

import numpy as np
import pytest

from SurfaceTopography.IO.Reader import ChannelInfo, DataKind, ReaderBase


class MockReader(ReaderBase):
    """Mock reader for testing channel functionality."""

    _format = "mock"
    _mime_types = ["application/x-mock"]
    _file_extensions = ["mock"]
    _name = "Mock Reader"
    _description = "Mock reader for testing"

    def __init__(self, channel_specs):
        """
        Parameters
        ----------
        channel_specs : list of dict
            Each dict should have: name, data_kind (optional), and optionally
            other ChannelInfo parameters.
        """
        self._channels = []
        for i, spec in enumerate(channel_specs):
            name = spec.get("name", f"Channel{i}")
            data_kind = spec.get("data_kind", DataKind.HEIGHT)
            data_unit = spec.get("data_unit", None)
            self._channels.append(
                ChannelInfo(
                    self,
                    i,
                    name=name,
                    dim=2,
                    nb_grid_pts=(64, 64),
                    physical_sizes=(1.0, 1.0),
                    uniform=True,
                    unit="um",
                    data_kind=data_kind,
                    data_unit=data_unit,
                )
            )

    @property
    def channels(self):
        return self._channels

    def topography(
        self,
        channel_index=None,
        channel_id=None,
        height_channel_index=None,
        physical_sizes=None,
        height_scale_factor=None,
        unit=None,
        info={},
        periodic=False,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
    ):
        from SurfaceTopography import Topography

        channel, channel_index = self._resolve_channel(
            channel_index, channel_id, height_channel_index
        )
        h = np.zeros((64, 64))
        return Topography(h, (1.0, 1.0), unit="um", periodic=periodic)


class TestChannelId:
    """Tests for channel_id property."""

    def test_unique_channel_names(self):
        """Test channel_id with unique names."""
        reader = MockReader([
            {"name": "Height"},
            {"name": "Phase"},
            {"name": "Amplitude"},
        ])
        assert reader.channels[0].channel_id == "Height"
        assert reader.channels[1].channel_id == "Phase"
        assert reader.channels[2].channel_id == "Amplitude"

    def test_duplicate_channel_names(self):
        """Test channel_id disambiguation with duplicate names."""
        reader = MockReader([
            {"name": "Height"},
            {"name": "Height"},
            {"name": "Phase"},
        ])
        assert reader.channels[0].channel_id == "Height#1"
        assert reader.channels[1].channel_id == "Height#2"
        assert reader.channels[2].channel_id == "Phase"

    def test_multiple_duplicates(self):
        """Test channel_id with multiple groups of duplicates."""
        reader = MockReader([
            {"name": "Height"},
            {"name": "Phase"},
            {"name": "Height"},
            {"name": "Phase"},
        ])
        assert reader.channels[0].channel_id == "Height#1"
        assert reader.channels[1].channel_id == "Phase#1"
        assert reader.channels[2].channel_id == "Height#2"
        assert reader.channels[3].channel_id == "Phase#2"

    def test_single_channel(self):
        """Test channel_id with single channel."""
        reader = MockReader([{"name": "Height"}])
        assert reader.channels[0].channel_id == "Height"


class TestHeightIndex:
    """Tests for height_index property."""

    def test_all_height_channels(self):
        """Test height_index when all channels are height channels."""
        reader = MockReader([
            {"name": "Height1", "data_kind": DataKind.HEIGHT},
            {"name": "Height2", "data_kind": DataKind.HEIGHT},
            {"name": "Height3", "data_kind": DataKind.HEIGHT},
        ])
        assert reader.channels[0].height_index == 0
        assert reader.channels[1].height_index == 1
        assert reader.channels[2].height_index == 2

    def test_mixed_channels(self):
        """Test height_index with mixed height and non-height channels."""
        reader = MockReader([
            {"name": "Height", "data_kind": DataKind.HEIGHT},
            {"name": "Phase", "data_kind": DataKind.PHASE},
            {"name": "Height2", "data_kind": DataKind.HEIGHT},
            {"name": "Voltage", "data_kind": DataKind.VOLTAGE},
        ])
        assert reader.channels[0].height_index == 0
        assert reader.channels[1].height_index is None  # Non-height
        assert reader.channels[2].height_index == 1
        assert reader.channels[3].height_index is None  # Non-height

    def test_no_height_channels(self):
        """Test height_index when there are no height channels."""
        reader = MockReader([
            {"name": "Phase", "data_kind": DataKind.PHASE},
            {"name": "Voltage", "data_kind": DataKind.VOLTAGE},
        ])
        assert reader.channels[0].height_index is None
        assert reader.channels[1].height_index is None

    def test_height_channels_property(self):
        """Test height_channels property on reader."""
        reader = MockReader([
            {"name": "Height", "data_kind": DataKind.HEIGHT},
            {"name": "Phase", "data_kind": DataKind.PHASE},
            {"name": "Height2", "data_kind": DataKind.HEIGHT},
        ])
        height_channels = reader.height_channels
        assert len(height_channels) == 2
        assert height_channels[0].name == "Height"
        assert height_channels[1].name == "Height2"


class TestChannelSelection:
    """Tests for _resolve_channel and topography channel selection."""

    def test_select_by_channel_index(self):
        """Test selecting channel by channel_index."""
        reader = MockReader([
            {"name": "Height"},
            {"name": "Phase", "data_kind": DataKind.PHASE},
        ])
        channel, idx = reader._resolve_channel(channel_index=1)
        assert channel.name == "Phase"
        assert idx == 1

    def test_select_by_channel_id(self):
        """Test selecting channel by channel_id."""
        reader = MockReader([
            {"name": "Height"},
            {"name": "Phase", "data_kind": DataKind.PHASE},
        ])
        channel, idx = reader._resolve_channel(channel_id="Phase")
        assert channel.name == "Phase"
        assert idx == 1

    def test_select_by_height_channel_index(self):
        """Test selecting channel by height_channel_index."""
        reader = MockReader([
            {"name": "Phase", "data_kind": DataKind.PHASE},
            {"name": "Height1", "data_kind": DataKind.HEIGHT},
            {"name": "Voltage", "data_kind": DataKind.VOLTAGE},
            {"name": "Height2", "data_kind": DataKind.HEIGHT},
        ])
        # height_channel_index=0 should select Height1 (channel_index=1)
        channel, idx = reader._resolve_channel(height_channel_index=0)
        assert channel.name == "Height1"
        assert idx == 1

        # height_channel_index=1 should select Height2 (channel_index=3)
        channel, idx = reader._resolve_channel(height_channel_index=1)
        assert channel.name == "Height2"
        assert idx == 3

    def test_select_default_channel(self):
        """Test selecting default channel when no argument given."""
        reader = MockReader([
            {"name": "Height"},
            {"name": "Phase", "data_kind": DataKind.PHASE},
        ])
        channel, idx = reader._resolve_channel()
        assert idx == 0  # Default is first channel

    def test_multiple_selection_methods_raises(self):
        """Test that specifying multiple selection methods raises error."""
        reader = MockReader([{"name": "Height"}])
        with pytest.raises(ValueError, match="Only one of"):
            reader._resolve_channel(channel_index=0, channel_id="Height")

        with pytest.raises(ValueError, match="Only one of"):
            reader._resolve_channel(channel_id="Height", height_channel_index=0)

    def test_invalid_channel_id_raises(self):
        """Test that invalid channel_id raises error."""
        reader = MockReader([{"name": "Height"}])
        with pytest.raises(ValueError, match="No channel with id"):
            reader._resolve_channel(channel_id="NonExistent")

    def test_invalid_height_channel_index_raises(self):
        """Test that invalid height_channel_index raises error."""
        reader = MockReader([
            {"name": "Height", "data_kind": DataKind.HEIGHT},
        ])
        with pytest.raises(ValueError, match="No height channel"):
            reader._resolve_channel(height_channel_index=5)

    def test_topography_with_channel_id(self):
        """Test loading topography using channel_id."""
        reader = MockReader([
            {"name": "Height"},
            {"name": "Phase", "data_kind": DataKind.PHASE},
        ])
        # Should not raise
        topo = reader.topography(channel_id="Height")
        assert topo is not None

    def test_topography_with_height_channel_index(self):
        """Test loading topography using height_channel_index."""
        reader = MockReader([
            {"name": "Phase", "data_kind": DataKind.PHASE},
            {"name": "Height", "data_kind": DataKind.HEIGHT},
        ])
        # height_channel_index=0 should select Height (channel_index=1)
        topo = reader.topography(height_channel_index=0)
        assert topo is not None


class TestDataKind:
    """Tests for DataKind enum and is_height_channel property."""

    def test_data_kind_values(self):
        """Test DataKind enum values."""
        assert DataKind.HEIGHT.value == "height"
        assert DataKind.VOLTAGE.value == "voltage"
        assert DataKind.PHASE.value == "phase"

    def test_is_height_channel(self):
        """Test is_height_channel property."""
        reader = MockReader([
            {"name": "Height", "data_kind": DataKind.HEIGHT},
            {"name": "Phase", "data_kind": DataKind.PHASE},
            {"name": "Voltage", "data_kind": DataKind.VOLTAGE},
        ])
        assert reader.channels[0].is_height_channel is True
        assert reader.channels[1].is_height_channel is False
        assert reader.channels[2].is_height_channel is False

    def test_default_data_kind_is_height(self):
        """Test that default data_kind is HEIGHT."""
        reader = MockReader([{"name": "Default"}])
        assert reader.channels[0].data_kind == DataKind.HEIGHT
        assert reader.channels[0].is_height_channel is True


class TestDataUnit:
    """Tests for data_unit property."""

    def test_data_unit_for_height_channel(self):
        """Test data_unit defaults to lateral unit for height channels."""
        reader = MockReader([{"name": "Height", "data_kind": DataKind.HEIGHT}])
        # data_unit should fall back to unit for height channels
        assert reader.channels[0].data_unit == "um"
        assert reader.channels[0].lateral_unit == "um"

    def test_data_unit_for_non_height_channel(self):
        """Test data_unit for non-height channels."""
        reader = MockReader([
            {"name": "Voltage", "data_kind": DataKind.VOLTAGE, "data_unit": "V"},
        ])
        assert reader.channels[0].data_unit == "V"
        assert reader.channels[0].lateral_unit == "um"

    def test_data_unit_none_for_non_height(self):
        """Test data_unit can be None for non-height channels."""
        reader = MockReader([
            {"name": "Phase", "data_kind": DataKind.PHASE},
        ])
        assert reader.channels[0].data_unit is None
        assert reader.channels[0].lateral_unit == "um"


class TestBackwardsCompatibility:
    """Tests for backwards compatibility."""

    def test_channel_index_still_works(self):
        """Test that channel_index parameter still works."""
        reader = MockReader([
            {"name": "Height1"},
            {"name": "Height2"},
        ])
        topo = reader.topography(channel_index=1)
        assert topo is not None

    def test_height_channel_index_matches_old_behavior(self):
        """Test that height_channel_index gives same result as old channel_index
        when all channels are height channels."""
        reader = MockReader([
            {"name": "Height1"},
            {"name": "Height2"},
            {"name": "Height3"},
        ])
        # When all channels are height channels, height_channel_index should
        # behave identically to channel_index
        for i in range(3):
            ch_by_index, _ = reader._resolve_channel(channel_index=i)
            ch_by_height_index, _ = reader._resolve_channel(height_channel_index=i)
            assert ch_by_index.name == ch_by_height_index.name
