"""Tests for the core framework interfaces."""

from typing import Any

import pytest

from quant_frame.core.interfaces import BaseProvider, BaseRepository
from quant_frame.core.models import TimeSeriesObservation


class TestBaseProvider:
    """Test suite for :class:`BaseProvider`."""

    def test_direct_instantiation_raises_typeerror(self) -> None:
        """Instantiating ``BaseProvider`` directly must raise ``TypeError``."""
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]

    def test_subclass_without_extract_raises_typeerror(self) -> None:
        """A subclass that omits ``extract()`` must be non-instantiable."""

        class DummyProvider(BaseProvider):
            pass

        with pytest.raises(TypeError):
            DummyProvider()  # type: ignore[abstract]

    def test_valid_subclass_instantiates(self) -> None:
        """A fully implemented subclass must instantiate without error."""

        class ValidProvider(BaseProvider):
            def extract(self) -> list[TimeSeriesObservation]:
                return []

        provider = ValidProvider()
        assert isinstance(provider, BaseProvider)
        assert provider.extract() == []


class TestBaseRepository:
    """Test suite for :class:`BaseRepository`."""

    def test_direct_instantiation_raises_typeerror(self) -> None:
        """Instantiating ``BaseRepository`` directly must raise ``TypeError``."""
        with pytest.raises(TypeError):
            BaseRepository()  # type: ignore[abstract]

    def test_subclass_without_save_raises_typeerror(self) -> None:
        """A subclass that omits ``save()`` must be non-instantiable."""

        class DummyRepository(BaseRepository):
            pass

        with pytest.raises(TypeError):
            DummyRepository()  # type: ignore[abstract]

    def test_valid_subclass_instantiates(self) -> None:
        """A fully implemented subclass must instantiate without error."""

        class ValidRepository(BaseRepository):
            def save(self, observations: list[TimeSeriesObservation]) -> None:
                return None

        repo = ValidRepository()
        assert isinstance(repo, BaseRepository)
