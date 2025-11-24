from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import textwrap

import pandas as pd
import google.generativeai as genai


def _summarise_driver_pace(lap_df: pd.DataFrame) -> str:
    """Build a short text summary of driver pace from lap_df."""
    df = lap_df.copy()

    # Drop pit laps if we have the flag
    if "is_pit_lap" in df.columns:
        df = df[~df["is_pit_lap"]]

    if df.empty or "lap_time_s" not in df.columns:
        return "Lap-time summary unavailable (no non-pit laps with lap_time_s)."

    best = float(df["lap_time_s"].min())
    med = float(df["lap_time_s"].median())
    std = float(df["lap_time_s"].std())
    n_laps = int(df.shape[0])

    first_lap = int(df["lap"].min())
    last_lap = int(df["lap"].max())

    return textwrap.dedent(
        f"""
        Driver pace summary:
        - Clean laps counted (no pit laps): {n_laps}
        - Best lap: {best:.3f} s
        - Median lap: {med:.3f} s
        - Lap-time spread (std dev): {std:.3f} s
        - Laps range from {first_lap} to {last_lap}
        """
    ).strip()


def build_chat_context(
    lap_df: pd.DataFrame,
    track_meta: Any,
    cfg: Any,
    strategies: Dict[str, List[int]],
    race: str,
    car_id: str,
) -> str:
    """
    Build a textual context block for the strategy chat.

    Parameters
    ----------
    lap_df:
        Per-lap telemetry features for this car / race.
    track_meta:
        Entry from TRACK_METAS for the selected track (has .name, .pit_lane_time_s, etc.).
    cfg:
        Strategy config object returned by make_config_from_meta.
    strategies:
        Dict of strategy_name -> list of pit laps.
    race, car_id:
        Current race label (e.g. 'R2') and car identifier.
    """
    track_name = getattr(track_meta, "name", "Unknown track")
    pit_loss = getattr(cfg, "pit_loss_s", None)
    base_lap = getattr(cfg, "base_lap_s", None)
    total_laps = int(lap_df["lap"].max()) if "lap" in lap_df.columns else None

    strat_lines = []
    for name, pits in strategies.items():
        pit_str = ", ".join(str(p) for p in pits)
        strat_lines.append(f"- {name}: pit on laps {pit_str}")

    strat_block = "\n".join(strat_lines) if strat_lines else "(no strategies defined)"

    pace_block = _summarise_driver_pace(lap_df)

    header = textwrap.dedent(
        f"""
        Track & session:
        - Track: {track_name}
        - Race: {race}
        - Car: {car_id}
        - Total race laps: {total_laps if total_laps is not None else "unknown"}
        - Base clean lap (trimmed median): {base_lap:.3f} s
        - Pit lane loss (green): {pit_loss:.3f} s
        """
    ).strip()

    strategies_text = textwrap.dedent(
        f"""
        Strategies currently on the table:
        {strat_block}
        """
    ).strip()

    context = "\n\n".join([header, pace_block, strategies_text])
    return context


def _format_live_state_for_prompt(live_state: Optional[Dict[str, Any]]) -> str:
    if not live_state:
        return "No current live state snapshot was provided."
    try:
        # remove noisy keys we don't need
        cleaned = dict(live_state)
        cleaned.pop("timestamp", None)
        return json.dumps(cleaned, indent=2)
    except Exception:
        return "Live state provided but could not be serialised cleanly."


def answer_engineer(
    model: Optional[genai.GenerativeModel],
    question: str,
    context: str,
    live_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Use Gemini to answer a race-engineering question, given static context and
    optional current live_state.

    Returns a short, radio-style explanation.
    """
    if model is None:
        # Fallback if Gemini is not configured
        return (
            "Gemini is not configured in this environment, so I can only answer in a generic way.\n\n"
            f"Context summary:\n{context}\n\n"
            f"Question: {question}\n\n"
            "As a rule of thumb, compare expected time gain/loss from the pit stop against "
            "tyre degradation over the remaining laps and the risk of a safety car. "
            "If tyres are stable and there is low safety-car risk, staying out is usually safer; "
            "if degradation is high or an undercut is available, boxing sooner can pay off."
        )

    live_text = _format_live_state_for_prompt(live_state)

    prompt = textwrap.dedent(
        f"""
        You are a highly experienced GT race engineer working with a Toyota GR86 Cup car.

        ### Static race context
        {context}

        ### Current live snapshot (may be approximate)
        {live_text}

        ### Task
        - Read the engineer's question carefully.
        - Use the context + live snapshot to reason about tyre state, pit windows,
          safety-car risk, and track position.
        - Respond **as if you were talking on the radio** to the race engineer.
        - Prefer **3–6 short bullet points**, focusing on:
          - What matters most strategically,
          - Clear recommendation (e.g. *box now*, *stay out 3–4 laps*, *hold track position*),
          - Any key risks / what-ifs to watch.

        Engineer's question:
        {question}
        """
    ).strip()

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            return "No answer returned by the model."
        return text
    except Exception as e:
        return f"(Gemini chat error: {e})"