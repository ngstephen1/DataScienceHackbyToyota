from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# make sure we can import our local modules
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from track_meta import TRACK_METAS
from strategy_engine import (
    make_config_from_meta,
    simulate_strategy_with_caution,
    run_multiverse,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GR RaceCraft – simple pit strategy recommender"
    )
    p.add_argument(
        "--track",
        type=str,
        default="barber-motorsports-park",
        help="track_id (e.g. barber-motorsports-park)",
    )
    p.add_argument(
        "--race",
        type=str,
        default="R2",
        help="race label used in filenames (e.g. R1 or R2)",
    )
    p.add_argument(
        "--car-id",
        type=str,
        default="GR86-002-000",
        help="vehicle_id for the car of interest",
    )
    p.add_argument(
        "--caution-lap",
        type=int,
        default=None,
        help="if set, treat this lap as start of caution window",
    )
    p.add_argument(
        "--caution-len",
        type=int,
        default=2,
        help="length of caution in laps (if caution-lap is set)",
    )
    p.add_argument(
        "--sims",
        type=int,
        default=500,
        help="number of Monte Carlo universes to run",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    track_id = args.track
    race = args.race
    car_id = args.car_id

    meta = TRACK_METAS[track_id]

    # we only have Barber wired for now
    data_dir = ROOT / "data" / "processed" / "barber"
    lap_path = data_dir / f"barber_{race.lower()}_{car_id}_lap_features.csv"

    if not lap_path.exists():
        raise SystemExit(f"Lap feature file not found: {lap_path}")

    lap_df = pd.read_csv(lap_path)

    # build strategy config from lap data + meta pit-lane time
    cfg = make_config_from_meta(
        lap_df,
        pit_lane_time_s=meta.pit_lane_time_s,
    )
    total_laps = int(lap_df["lap"].max())

    # choose 3 candidate strategies (can tweak later)
    one_stop_mid = [total_laps // 2]
    one_stop_early = [max(3, total_laps // 2 - 4)]
    two_stop = [total_laps // 3, 2 * total_laps // 3]

    strategies = {
        "1-stop_early": one_stop_early,
        "1-stop_mid": one_stop_mid,
        "2-stop": two_stop,
    }

    print(f"Track: {meta.name} ({track_id})")
    print(f"Race:  {race}, Car: {car_id}")
    print(f"Laps:  {total_laps}")
    print(f"Base race pace: {cfg.base_lap_s:.3f} s, pit loss: {cfg.pit_loss_s:.3f} s")
    print()

    # If user gives a specific caution scenario, quickly compare strategies
    if args.caution_lap is not None:
        print(
            f"== Deterministic comparison with caution from lap "
            f"{args.caution_lap} for {args.caution_len} laps =="
        )
        for name, pits in strategies.items():
            t = simulate_strategy_with_caution(
                n_laps=total_laps,
                pit_laps=pits,
                cfg=cfg,
                caution_start=args.caution_lap,
                caution_len=args.caution_len,
            )
            print(f"{name:>12}: {t/60:.2f} min")

        print()

    # Mini Multiverse: random cautions
    print(f"== Mini Multiverse (n={args.sims} universes, random cautions) ==")
    summary = run_multiverse(
        strategies=strategies,
        n_laps=total_laps,
        cfg=cfg,
        n_sims=args.sims,
    )

    # pick best
    best_name = summary["win_prob"].idxmax()
    best_row = summary.loc[best_name]

    print(summary)
    print()
    print(
        f"Recommended baseline strategy: {best_name} "
        f"(win_prob={best_row['win_prob']:.2%}, "
        f"mean_time={best_row['mean_time_s']/60:.2f} min)"
    )


if __name__ == "__main__":
    main()