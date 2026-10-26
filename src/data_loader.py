"""Dataset-agnostic loader: every provider is read into one `Match`.

The `Match` dataclass is the single contract the whole pipeline depends on, so
adding a dataset means writing a reader that returns one -- nothing downstream
(cleaning/sampling/features/SUVAT) changes. Readers never clean or smooth; that
is the cleaning stage's job.

  * ffpc_match (PFF FC JSONL)     -> `src/ffpc.py`, many matches.
  * bdb_match  (NFL BDB 2019 CSV) -> `src/bdb.py`, one match per game.
  * nba_match  (NBA SportVU 7z)   -> `src/nba.py`, one match per game.

`load_match()` returns a single Match for the active dataset (used by the eval
scripts); `iter_matches()` yields every match (used by build_dataset).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from . import config


@dataclass
class Match:
    positions: np.ndarray   # (T, P, 2) metres, NaN where untracked
    in_field: np.ndarray    # (T, P) int8, 1 when the player is on the pitch
    segment: np.ndarray     # (T,) int, segment/half id per frame
    frame: np.ndarray       # (T,) global frame index
    time_in_match: np.ndarray  # (T,) seconds
    player_id: np.ndarray   # (P,)
    player_name: list[str]
    team_id: np.ndarray     # (P,) 0/1
    side_id: np.ndarray     # (P,)
    recording_id: str

    @property
    def T(self) -> int:
        return self.positions.shape[0]

    @property
    def P(self) -> int:
        return self.positions.shape[1]

    def segment_slices(self):
        """Yield (segment_id, lo, hi) contiguous frame ranges per segment."""
        seg = self.segment
        bounds = np.where(np.diff(seg) != 0)[0] + 1
        edges = [0, *bounds.tolist(), len(seg)]
        for a, b in zip(edges[:-1], edges[1:]):
            yield int(seg[a]), a, b


def iter_matches(dataset: str | None = None) -> Iterator[Match]:
    """Yield every `Match` for a dataset (default: the active config.DATASET).

    ffpc_match yields one per PFF game; bdb/nba one per game.
    """
    dataset = dataset or config.DATASET
    loader = config.DATASETS[dataset]["loader"]
    if loader == "pff":
        from .ffpc import iter_ffpc_matches  # lazy: avoids import cycle
        yield from iter_ffpc_matches()
    elif loader == "bdb":
        from .bdb import iter_bdb_matches  # lazy: avoids import cycle
        yield from iter_bdb_matches()
    elif loader == "nba":
        from .nba import iter_nba_matches  # lazy: avoids import cycle
        yield from iter_nba_matches()
    else:
        raise ValueError(f"unknown loader {loader!r} for dataset {dataset!r}")


def load_match(dataset: str | None = None) -> Match:
    """Return one `Match` for the active dataset.

    For multi-match datasets this is the first match -- enough for the
    single-match eval/visualisation scripts. Use `iter_matches()` to process
    all matches (e.g. dataset building).
    """
    return next(iter_matches(dataset))
