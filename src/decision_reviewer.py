from __future__ import annotations

"""
AI Decision Reviewer for race-engineer calls.

Used by `streamlit_app.py` in the Live Race Copilot tab:

    from src.decision_reviewer import review_decision

    review_text = review_decision(
        GEMINI_MODEL,
        decision_text,
        feats_review,
    )

`GEMINI_MODEL` is a google.generativeai.GenerativeModel (or None),
`decision_text` is the engineer's proposed call, and `feats_review`
comes from `make_live_feature_vector(...)`.
"""

from typing import Any, Mapping
import json
import textwrap


def _format_features_for_prompt(feats: Mapping[str, Any]) -> str:
    """
    Keep only the most relevant fields for the LLM prompt so we don't
    drown it in data and to keep the context readable.
    """
    keys_of_interest = [
        "current_lap",
        "total_laps",
        "strategy_name",
        "pit_laps",
        "caution_lap",
        "caution_len",
        "best_lap_s",
        "median_lap_s",
        "last5_avg_s",
        "last5_delta_vs_best_s",
        "stint_lap",
        "push_factor",
        "risk_pref",
        "base_lap_s",
        "pit_loss_s",
    ]
    trimmed = {k: feats.get(k) for k in keys_of_interest if k in feats}
    return json.dumps(trimmed, indent=2, sort_keys=True)


def _fallback_review(decision_text: str, feats: Mapping[str, Any]) -> str:
    """
    Simple rule-based reviewer used if Gemini is unavailable or errors.

    This gives you *something* sensible even if the model call fails,
    and is safe to run offline.
    """
    lap = feats.get("current_lap")
    total = feats.get("total_laps")
    last5 = feats.get("last5_delta_vs_best_s")
    stint_lap = feats.get("stint_lap")
    pit_laps = feats.get("pit_laps", [])
    caution_lap = feats.get("caution_lap")
    caution_len = feats.get("caution_len")

    lines: list[str] = []
    lines.append("## Verdict")
    lines.append(
        f"Provisional **OK** on: _{decision_text.strip()}_ "
        "(rule-based check only – Gemini not used here)."
    )
    lines.append("")

    lines.append("## Rationale")
    lines.append(f"- Race context: lap **{lap}/{total}**.")

    if last5 is not None:
        try:
            last5_f = float(last5)
        except (TypeError, ValueError):
            last5_f = None
        if last5_f is not None:
            if last5_f > 1.0:
                lines.append(
                    f"- Last 5 laps are about **{last5_f:.1f}s** slower than best "
                    "→ tyres look **tired**."
                )
            elif last5_f > 0.4:
                lines.append(
                    f"- Last 5 laps are **{last5_f:.1f}s** away from best "
                    "→ mild degradation, but still usable."
                )
            else:
                lines.append(
                    "- Pace is close to best laps → tyres still look **strong**."
                )

    if stint_lap is not None:
        lines.append(f"- Current stint age: **{stint_lap}** laps on this tyre set.")

    if pit_laps:
        lines.append(f"- Planned pit window per strategy: laps **{pit_laps}**.")

    if caution_lap is not None and caution_len:
        lines.append(
            f"- Modelled caution window around lap **{caution_lap}** "
            f"for about **{caution_len}** lap(s)."
        )

    lines.append("")
    lines.append("## Risks / watch-outs")

    if last5 is not None:
        try:
            last5_f = float(last5)
        except (TypeError, ValueError):
            last5_f = None
    else:
        last5_f = None

    if last5_f is not None and last5_f > 1.0:
        lines.append(
            "- Main risk: staying out too long on a heavily degraded tyre and "
            "bleeding lap time before the stop."
        )
    else:
        lines.append(
            "- Main risk: boxing too early and giving up tyre life that could "
            "have been converted into pace if no late caution appears."
        )

    lines.append(
        "- This is a heuristic sanity check only – always cross-check with "
        "your tyre life model, fuel numbers, and on-track traffic."
    )

    lines.append("")
    lines.append("## Safer alternative")
    lines.append(
        "- If unsure, favour **slightly later** stops when tyres still look strong, "
        "or **earlier** stops if degradation is clearly above 1.0s vs best."
    )

    return "\n".join(lines)


def review_decision(
    model: Any | None,
    decision_text: str,
    live_features: Mapping[str, Any],
) -> str:
    """
    Ask Gemini to act as a senior race engineer and review the proposed call.

    Parameters
    ----------
    model
        A configured `google.generativeai.GenerativeModel` instance, or None.
        In `streamlit_app.py` this is `GEMINI_MODEL`.
    decision_text
        Short, engineer radio-style description of the intended call
        (e.g. "Box now for 4 tyres and fuel to the end.").
    live_features
        Dict returned by `make_live_feature_vector(...)` describing the current
        stint, pace, tyre state, strategy, and caution context.

    Returns
    -------
    str
        Markdown-formatted review text (Gemini-based if possible,
        otherwise a rule-based fallback).
    """
    # If no model wired, or google-generativeai is not available, fall back.
    if model is None:
        return _fallback_review(decision_text, live_features)

    context_str = _format_features_for_prompt(live_features)

    prompt = textwrap.dedent(
        f"""
        You are the **most senior race engineer** on a GT team.
        Another engineer proposes this decision over the radio:

        >>> {decision_text.strip()}

        Here is a compact summary of the current race situation as Python dict:

        {context_str}

        Please review the call and respond in **clear, structured markdown** with:

        1. `## Verdict` – one line: **Agree**, **Lean agree**, **Concerned**, or **Disagree**.
        2. `## Rationale` – 3–6 bullet points referencing:
           - pace vs best (last 5 laps),
           - stint age / tyre state,
           - pit-loss vs expected gain,
           - interaction with any upcoming caution window.
        3. `## Risks if this is wrong` – 2–4 bullet points.
        4. `## Safer alternative` – a concise alternative call if you do not fully agree.

        Use short, race-engineering language, not essays.
        Do *not* invent numbers that are not implied by the context.
        """
    )

    try:
        # We don't import google.generativeai here; the model is already constructed
        # in streamlit_app.py via `init_gemini_model()`.
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            return _fallback_review(decision_text, live_features)
        return text
    except Exception:
        # Any API / network error → fall back to rule-based version
        return _fallback_review(decision_text, live_features)