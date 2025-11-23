from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
# Pace / degradation estimation helpers
# ---------------------------------------------------------------------

def estimate_race_pace(
    lap_df: pd.DataFrame,
    lap_time_col: str = "lap_time_s",
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> float:
    """Estimate a trimmed-median 'race pace' lap time.

    We drop NaNs, trim the top/bottom quantiles, then take the median.
    This is robust against out-laps, in-laps, and big mistakes.
    """
    lf = lap_df.dropna(subset=[lap_time_col]).copy()
    if lf.empty:
        return float("nan")

    low_q, high_q = lf[lap_time_col].quantile([q_low, q_high])
    race_pace = lf[(lf[lap_time_col] >= low_q) & (lf[lap_time_col] <= high_q)]
    if race_pace.empty:
        return float(lf[lap_time_col].median())
    return float(race_pace[lap_time_col].median())


def estimate_deg_from_laps(
    lap_df: pd.DataFrame,
    lap_time_col: str = "lap_time_s",
    stint_col: str = "stint_lap",
) -> float:
    """Estimate degradation (s per lap) from a set of laps.

    1. Ensure there is a 'stint_lap' index (1,2,3,... within a stint).
       - If 'stint_lap' exists, use it directly.
       - Else, try to build it from 'lap' or row order.
    2. Group by stint lap and take mean lap time.
    3. Fit a line y = m*x + b using np.polyfit; return m as deg_per_lap.

    If not enough data, falls back to 0.0.
    """
    df = lap_df.dropna(subset=[lap_time_col]).copy()
    if df.empty:
        return 0.0

    # Ensure stint_lap exists
    if stint_col not in df.columns:
        if "race" in df.columns and "lap" in df.columns:
            df = df.sort_values(["race", "lap"])
            df[stint_col] = df.groupby("race").cumcount() + 1
        elif "lap" in df.columns:
            df = df.sort_values("lap")
            df[stint_col] = np.arange(1, len(df) + 1)
        else:
            df = df.sort_index()
            df[stint_col] = np.arange(1, len(df) + 1)

    grouped = (
        df.groupby(stint_col)[lap_time_col]
        .mean()
        .reset_index()
        .sort_values(stint_col)
    )

    if len(grouped) < 2:
        return 0.0

    x = grouped[stint_col].to_numpy()
    y = grouped[lap_time_col].to_numpy()

    # Fit y = m*x + b
    try:
        m, _b = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        return 0.0

    return float(m)


def make_config_from_meta(
    lap_df: pd.DataFrame,
    pit_lane_time_s: float,
    lap_time_col: str = "lap_time_s",
    deg_per_lap: Optional[float] = 0.03,
    caution_mult: float = 1.20,
    caution_pit_factor: float = 0.6,
) -> StrategyConfig:
    """Build a :class:`StrategyConfig` from laps + known pit lane time.

    - ``base_lap_s`` is computed via trimmed-median race pace.
    - ``deg_per_lap``:
        * if a float is passed, use it directly;
        * if ``None`` is passed, estimate from laps via
          :func:`estimate_deg_from_laps`.
    """
    base = estimate_race_pace(lap_df, lap_time_col=lap_time_col)

    if deg_per_lap is None:
        est_deg = estimate_deg_from_laps(lap_df, lap_time_col=lap_time_col)
        # small safety: if estimate fails or is weird, fall back to 0.03
        if not np.isfinite(est_deg) or abs(est_deg) < 1e-4:
            est_deg = 0.03
        deg_value = est_deg
    else:
        deg_value = float(deg_per_lap)

    return StrategyConfig(
        base_lap_s=base,
        pit_loss_s=pit_lane_time_s,
        deg_per_lap=deg_value,
        caution_mult=caution_mult,
        caution_pit_factor=caution_pit_factor,
    )


# ---------------------------------------------------------------------
# Core lap / stint model
# ---------------------------------------------------------------------

def lap_time_for_stint_lap(stint_lap: int, cfg: StrategyConfig) -> float:
    """Compute lap time for a given lap within a stint.

    ``stint_lap`` is 1,2,3,... within a tyre stint.
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
    """Clean race (no caution), with tyre degradation.

    ``pit_laps`` is a list of 1-based lap numbers where we pit.
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
    """Race with optional caution window.

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
    """Simulate one random 'universe' with optional caution.

    With probability ``caution_prob`` we draw a random FCY starting
    between ``min_green_lap`` and ``n_laps - 3`` and a length of
    1–3 laps. Otherwise the race stays green.
    """
    if rng is None:
        rng = np.random.default_rng()

    has_fcy = rng.random() < caution_prob

    if has_fcy:
        start = int(rng.integers(min_green_lap, max(min_green_lap + 1, n_laps - 3)))
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
    """Run many random races and compute mean time & win probability.

    Parameters
    ----------
    strategies:
        Mapping from strategy name to list of 1-based pit-lap numbers.
    n_laps:
        Total laps in the race.
    cfg:
        StrategyConfig describing pace, degradation, and caution effects.
    n_sims:
        Number of Monte Carlo universes.
    caution_prob:
        Probability that any given universe has a caution period.
    """
    rng = np.random.default_rng()
    results: Dict[str, List[float]] = {name: [] for name in strategies}

    for _ in range(n_sims):
        times: Dict[str, float] = {}
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


# ---------------------------------------------------------------------
# Real-time analytics helpers for Streamlit + animation
# ---------------------------------------------------------------------

TYRE_PHASE_WARMUP_LAPS = 2
TYRE_PHASE_STABLE_LAPS = 10


def classify_tyre_phase(stint_lap: int, typical_stint_len: int | None = None) -> str:
    """Classify the current tyre phase.

    Heuristic only, but useful for quick labelling in the UI.
    """
    if stint_lap <= TYRE_PHASE_WARMUP_LAPS:
        return "warm-up"
    if typical_stint_len is not None and stint_lap >= typical_stint_len:
        return "late-deg"
    if stint_lap <= TYRE_PHASE_STABLE_LAPS:
        return "stable"
    return "degradation"


def project_future_laps(
    cfg: StrategyConfig,
    current_stint_lap: int,
    n_future: int,
) -> np.ndarray:
    """Project lap times for the next ``n_future`` laps if we stay out."""
    laps = np.arange(current_stint_lap, current_stint_lap + n_future)
    return cfg.base_lap_s + cfg.deg_per_lap * (laps - 1)


def basic_live_metrics(
    cfg: StrategyConfig,
    current_lap: int,
    total_laps: int,
    stint_lap: int,
    lap_history: List[float],
    gaps: Optional[Dict[str, float]] = None,
    fuel_laps_remaining: Optional[float] = None,
    planned_final_stop_lap: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute lightweight real-time metrics for the engineer console.

    This returns values only – Streamlit / Gemini can turn them into
    natural-language insights.
    """
    if gaps is None:
        gaps = {}

    n_laps_done = len(lap_history)
    last_lap_s = lap_history[-1] if lap_history else float("nan")
    best_lap_s = min(lap_history) if lap_history else float("nan")

    last_5 = lap_history[-5:]
    mean_last_5 = float(np.mean(last_5)) if last_5 else float("nan")

    # Pace deltas
    delta_vs_best = last_lap_s - best_lap_s if np.isfinite(best_lap_s) else float("nan")
    delta_vs_base = last_lap_s - cfg.base_lap_s if np.isfinite(last_lap_s) else float("nan")

    # Very rough tyre life estimate: assume a typical stint of 22 laps
    typical_stint_len = 22
    tyre_life_pct = max(0.0, 1.0 - (stint_lap - 1) / typical_stint_len)
    tyre_phase = classify_tyre_phase(stint_lap, typical_stint_len)

    # Project next 5 laps if we stay out
    proj_next_5 = project_future_laps(cfg, stint_lap, 5)
    proj_mean_next_5 = float(np.mean(proj_next_5))

    # Pit-window style helpers
    laps_remaining = total_laps - current_lap + 1
    in_final_window = False
    if planned_final_stop_lap is not None:
        in_final_window = current_lap >= planned_final_stop_lap

    fuel_ok_to_end = None
    if fuel_laps_remaining is not None:
        fuel_ok_to_end = fuel_laps_remaining >= laps_remaining

    return {
        "current_lap": current_lap,
        "total_laps": total_laps,
        "stint_lap": stint_lap,
        "n_laps_done": n_laps_done,
        "last_lap_s": last_lap_s,
        "best_lap_s": best_lap_s,
        "mean_last_5_s": mean_last_5,
        "delta_vs_best_s": delta_vs_best,
        "delta_vs_base_s": delta_vs_base,
        "tyre_life_pct": tyre_life_pct,
        "tyre_phase": tyre_phase,
        "proj_mean_next_5_s": proj_mean_next_5,
        "gaps": gaps,
        "laps_remaining": laps_remaining,
        "in_final_window": in_final_window,
        "fuel_laps_remaining": fuel_laps_remaining,
        "fuel_ok_to_end": fuel_ok_to_end,
    }


def evaluate_pit_options(
    cfg: StrategyConfig,
    total_laps: int,
    current_lap: int,
    current_pit_laps: List[int],
    candidate_offsets: List[int],
    caution_prob: float = 0.5,
    n_sims: int = 200,
) -> pd.DataFrame:
    """Evaluate a small set of "box now / earlier / later" options.

    ``candidate_offsets`` are offsets from the current lap, e.g. ``[-2, 0, 2]``.
    Only laps within ``[1, total_laps]`` are kept.

    Returns a DataFrame compatible with ``run_multiverse`` output, with an
    extra column ``pit_lap`` for convenience.
    """
    strategies: Dict[str, List[int]] = {}

    base_pits = sorted(set(current_pit_laps))
    for off in candidate_offsets:
        lap = current_lap + off
        if lap < 1 or lap > total_laps:
            continue
        pits = sorted(set(base_pits + [lap]))
        name = f"pit_L{lap}" if off == 0 else f"pit_L{lap}_off{off:+d}"
        strategies[name] = pits

    if not strategies:
        return pd.DataFrame()

    summary = run_multiverse(
        strategies=strategies,
        n_laps=total_laps,
        cfg=cfg,
        n_sims=n_sims,
        caution_prob=caution_prob,
    )

    # Attach the actual pit-lap list as a column for UI display
    strat_to_pits = {name: strategies[name] for name in summary.index}
    summary = summary.copy()
    summary["pit_laps"] = summary.index.map(lambda k: strat_to_pits[k])
    summary["primary_pit_lap"] = summary["pit_laps"].apply(lambda xs: xs[-1])

    return summary