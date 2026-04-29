"""Time-series cross-validation splitters."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import pandas as pd


class WalkForwardSplitter:
    """Strict chronologically-ordered walk-forward cross-validation splitter.

    The splitter yields ``(train, test)`` tuples where ``train`` appears strictly
    before ``test`` in the original ``DataFrame``, guaranteeing zero leakage.
    Two window strategies are supported:

    * ``"rolling"``  -- the training window has a fixed size of ``train_size``.
    * ``"expanding"`` -- the training window grows by ``test_size`` after each
      split, starting from ``train_size``.

    Slicing is performed with ``iloc`` so the implementation is safe regardless
    of the ``DataFrame`` index type.

    Attributes:
        train_size: Number of rows in the initial training slice (and the fixed
            training slice when ``window_type="rolling"``).
        test_size: Number of rows in each test slice.
        window_type: Either ``"rolling"`` or ``"expanding"``.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"price": range(20)})
        >>> splitter = WalkForwardSplitter(train_size=5, test_size=3)
        >>> for train, test in splitter.split(df):
        ...     print(len(train), len(test))
        5 3
        8 3
        11 3
    """

    def __init__(
        self,
        *,
        train_size: int,
        test_size: int,
        window_type: Literal["rolling", "expanding"] = "expanding",
    ) -> None:
        """Initialise a :class:`WalkForwardSplitter`.

        Args:
            train_size: Number of rows in each (or initial) training slice.
                Must be a strictly positive integer.
            test_size: Number of rows in each test slice.
                Must be a strictly positive integer.
            window_type: ``"rolling"`` keeps the training slice at a fixed
                length; ``"expanding"`` grows it by ``test_size`` on each
                subsequent split.  Defaults to ``"expanding"``.

        Raises:
            ValueError: If ``train_size`` or ``test_size`` is not a positive
                integer, or if ``window_type`` is not one of the supported
                values.
        """
        if not isinstance(train_size, int) or train_size <= 0:
            raise ValueError("train_size must be a strictly positive integer.")
        if not isinstance(test_size, int) or test_size <= 0:
            raise ValueError("test_size must be a strictly positive integer.")
        if window_type not in {"rolling", "expanding"}:
            raise ValueError(
                f"window_type must be 'rolling' or 'expanding', got '{window_type}'."
            )
        self.train_size: int = train_size
        self.test_size: int = test_size
        self.window_type: Literal["rolling", "expanding"] = window_type

    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield chronologically-safe ``(train_df, test_df)`` slices.

        The method walks forward through the DataFrame row-by-row.  The first
        training slice ends just before the first test slice, and on each
        subsequent step the walk advances by ``test_size`` rows.

        Args:
            df: The input DataFrame, assumed to be sorted in chronological
                order.

        Yields:
            Tuples ``(train_df, test_df)`` where ``train_df.index.max()`` is
            strictly less than ``test_df.index.min()``.

        Note:
            If ``len(df) < train_size + test_size``, nothing is yielded.
        """
        n_rows = len(df)
        if n_rows < self.train_size + self.test_size:
            return

        step = self.test_size
        start = 0

        if self.window_type == "rolling":
            # For rolling: train is always fixed-size [i, i+train_size)
            # test immediately follows.
            i = self.train_size
            while i + step <= n_rows:
                train = df.iloc[start:i]
                test = df.iloc[i : i + step]
                yield (train, test)
                start += step
                i += step
        else:
            # For expanding: train grows by test_size each split
            # first split ends at train_size, then train_size+step, etc.
            i = self.train_size
            while i + step <= n_rows:
                train = df.iloc[start:i]
                test = df.iloc[i : i + step]
                yield (train, test)
                i += step
