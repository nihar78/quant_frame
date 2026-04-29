"""Data quality validators for the quant_frame library."""

from __future__ import annotations

from quant_frame.core.models import TimeSeriesObservation


class ThresholdValidator:
    """Validates time-series observations against configurable per-feature bounds.

    This validator detects anomalous data points by checking whether every
    feature in an observation falls within the ``min`` / ``max`` thresholds
    defined at instantiation time.  Observations that violate any bound for
    any feature are discarded.

    If a feature exists in an observation but is **not** present in the
    threshold dictionary, it is allowed to pass through safely.

    Attributes:
        thresholds: Mapping of feature names to their respective lower and/or
            upper bounds.  Each inner dictionary may contain ``"min"`` and/or
            ``"max"`` keys with ``float`` values.

    Example:
        >>> thresholds = {"heart_rate": {"min": 30.0, "max": 220.0}}
        >>> validator = ThresholdValidator(thresholds=thresholds)
        >>> # Pass a list of TimeSeriesObservation objects to filter_anomalies
        >>> clean = validator.filter_anomalies(observations)
    """

    def __init__(self, thresholds: dict[str, dict[str, float]]) -> None:
        """Initialise a :class:`ThresholdValidator`.

        Args:
            thresholds: Dictionary mapping feature names to bound
                dictionaries.  Each bound dictionary may optionally contain
                ``"min"`` and/or ``"max"`` keys specifying the lower and
                upper thresholds for that feature.
        """
        self.thresholds: dict[str, dict[str, float]] = thresholds

    def filter_anomalies(
        self, observations: list[TimeSeriesObservation]
    ) -> list[TimeSeriesObservation]:
        """Return only observations whose features are within defined bounds.

        Each observation in *observations* is inspected feature-by-feature.
        If a feature has a corresponding threshold definition and the value
        falls outside the allowed range, the observation is discarded.
        Features with no threshold definition are ignored and do not cause
        the observation to be discarded.

        Args:
            observations: List of :class:`TimeSeriesObservation` objects to
                validate.

        Returns:
            A new list containing only the observations that satisfy **all**
            applicable thresholds.
        """
        result: list[TimeSeriesObservation] = []
        for obs in observations:
            if self._is_valid(obs):
                result.append(obs)
        return result

    def _is_valid(self, observation: TimeSeriesObservation) -> bool:
        """Check whether a single observation satisfies all thresholds.

        Args:
            observation: The :class:`TimeSeriesObservation` to evaluate.

        Returns:
            ``True`` if every feature in the observation falls within its
            defined bounds (or has no bounds defined), ``False`` otherwise.
        """
        for feature_name, value in observation.features.items():
            if feature_name not in self.thresholds:
                continue
            bounds = self.thresholds[feature_name]
            if "min" in bounds and value < bounds["min"]:
                return False
            if "max" in bounds and value > bounds["max"]:
                return False
        return True
