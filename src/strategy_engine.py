from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------

@dataclass
class StrategyConfig:
    base_lap_s: float          # baseline green-flag lap time (race pace)
    pit_loss_s: float          # time lost by a pit stop on green
    deg_per_lap: float = 0.03  # tyre degradation (s added per stint lap)
    caution_mult: float = 1.20 # FCY laps are this factor slower
    caution_pit_factor: float = 0.6  # pit loss factor under FCY


# ---------------------------------------------------------------------
# Pace estimation helpers
# ---------------------------------------------------------------------

def estimate_race_pace(
    lap_df: pd.DataFrame,
    lap_time_col: str = "lap_time_s",
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> float:
    """
    Estimate a 'race pace' lap time by trimming outlier laps.
    """
    lf = lap_df.dropna(subset=[lap_time_col]).copy()
    low_q, high_q = lf[lap_time_col].quantile([q_low, q_high])
    race_pace = lf[(lf[lap_time_col] >= low_q) & (lf[lap_time_col] <= high_q)]
    return float(race_pace[lap_time_col].median())


def make_config_from_meta(
    lap_df: pd.DataFrame,
    pit_lane_time_s: float,
    lap_time_col: str = "lap_time_s",
    deg_per_lap: float = 0.03,
    caution_mult: float = 1.20,
    caution_pit_factor: float = 0.6,
) -> StrategyConfig:
    base = estimate_race_pace(lap_df, lap_time_col=lap_time_col)
    return StrategyConfig(
        base_lap_s=base,
        pit_loss_s=pit_lane_time_s,
        deg_per_lap=deg_per_lap,
        caution_mult=caution_mult,
        caution_pit_factor=caution_pit_factor,
    )


# ---------------------------------------------------------------------
# Core lap / stint model
# ---------------------------------------------------------------------

def lap_time_for_stint_lap(stint_lap: int, cfg: StrategyConfig) -> float:
    """
    Compute lap time for a given lap within a stint.
    stint_lap: 1,2,3,... within a stint.
    """
    return cfg.base_lap_s + cfg.deg_per_lap * (stint_lap - 1)


# ---------------------------------------------------------------------
# Strategy simulators
# ---------------------------------------------------------------------

def simulate_strategy_with_deg(
    n_laps: int,
    pit_laps: List[int],
    cfg: StrategyConfig,
) -> float:
    """
    Clean race (no caution), with tyre degradation.
    pit_laps: list of 1-based lap numbers where we pit.
    """
    pit_laps = sorted(pit_laps)
    pit_set = set(pit_laps)

    total_time = 0.0
    stint_lap = 1

    for lap in range(1, n_laps + 1):
        lap_time = lap_time_for_stint_lap(stint_lap, cfg)

        if lap in pit_set:
            lap_time += cfg.pit_loss_s
            stint_lap = 1
        else:
            stint_lap += 1

        total_time += lap_time

    return total_time


def simulate_strategy_with_caution(
    n_laps: int,
    pit_laps: List[int],
    cfg: StrategyConfig,
    caution_start: Optional[int] = None,  # first FCY lap (1-based)
    caution_len: int = 0,                 # number of FCY laps
) -> float:
    """
    Race with optional caution window.
    Pit loss is reduced if we stop during caution.
    """
    pit_laps = sorted(pit_laps)
    pit_set = set(pit_laps)

    caution_laps: set[int] = set()
    if caution_start is not None and caution_len > 0:
        caution_laps = set(range(caution_start, caution_start + caution_len))

    total_time = 0.0
    stint_lap = 1

    for lap in range(1, n_laps + 1):
        lap_time = lap_time_for_stint_lap(stint_lap, cfg)

        # FCY slow-down
        if lap in caution_laps:
            lap_time *= cfg.caution_mult

        # Pit stop
        if lap in pit_set:
            extra = cfg.pit_loss_s
            if lap in caution_laps:
                extra *= cfg.caution_pit_factor
            lap_time += extra
            stint_lap = 1
        else:
            stint_lap += 1

        total_time += lap_time

    return total_time


# ---------------------------------------------------------------------
# Mini Multiverse Monte Carlo simulator
# ---------------------------------------------------------------------

def simulate_random_race(
    n_laps: int,
    pit_laps: List[int],
    cfg: StrategyConfig,
    caution_prob: float = 0.5,
    min_green_lap: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    One random 'universe': maybe there is a caution, maybe not.
    """
    if rng is None:
        rng = np.random.default_rng()

    has_fcy = rng.random() < caution_prob

    if has_fcy:
        # sample start & length
        start = int(rng.integers(min_green_lap, n_laps - 3))
        length = int(rng.choice([1, 2, 3]))
    else:
        start, length = None, 0

    return simulate_strategy_with_caution(
        n_laps=n_laps,
        pit_laps=pit_laps,
        cfg=cfg,
        caution_start=start,
        caution_len=length,
    )


def run_multiverse(
    strategies: Dict[str, List[int]],
    n_laps: int,
    cfg: StrategyConfig,
    n_sims: int = 500,
    caution_prob: float = 0.5,
) -> pd.DataFrame:
    """
    Run many random races and compute mean time & win probability
    for each strategy.
    """
    rng = np.random.default_rng()
    results: Dict[str, List[float]] = {name: [] for name in strategies}

    for _ in range(n_sims):
        # simulate each strategy in this universe
        times = {}
        for name, pits in strategies.items():
            t = simulate_random_race(
                n_laps=n_laps,
                pit_laps=pits,
                cfg=cfg,
                caution_prob=caution_prob,
                rng=rng,
            )
            times[name] = t
            results[name].append(t)

    df = pd.DataFrame(results)

    # who wins each universe?
    winner = df.idxmin(axis=1)
    win_counts = winner.value_counts().rename("wins")

    mean_time = df.mean().rename("mean_time_s")
    win_prob = (win_counts / n_sims).rename("win_prob")

    summary = pd.concat([mean_time, win_counts, win_prob], axis=1)
    return summary.sort_values("mean_time_s")