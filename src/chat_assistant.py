from __future__ import annotations

"""
1:06am 11/25

Conversational race-engineer assistant for the Streamlit app.

"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import json
import math
import statistics
import textwrap

import pandas as pd
import google.generativeai as genai


# Data structures

RoleLiteral = Literal["user", "assistant"]
IntentLiteral = Literal[
    "race_core",        # direct strategy / telemetry / lap questions
    "race_education",   # general race engineering questions ("what is a race engineer?")
    "tool_help",        # questions about how to use this particular app
    "smalltalk",        # greetings / light chat
    "offtopic",         # everything else
]


@dataclass
class ChatTurn:
    """A single turn in the conversation history.

    This mirrors the structure used in Streamlit session state.
    """

    role: RoleLiteral
    content: str

# @to do: handle data return later

def _safe_float(value: Any) -> Optional[float]:
    """Convert *value* to ``float`` if possible, otherwise ``None``.

    This is used throughout the summarisation helpers so that missing
    columns never trigger exceptions.
    """

    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_opt(value: Optional[float], suffix: str = "", ndigits: int = 3) -> str:
    """Format an optional float.

    If ``value`` is ``None`` return ``"unknown"``; otherwise format to
    *ndigits* decimal places and append *suffix*.
    """

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    return f"{value:.{ndigits}f}{suffix}"


def _percent(frac: Optional[float]) -> str:
    """Format a fraction (0-1) as a percentage string.

    ``None`` or NaN become ``"unknown"``.
    """

    if frac is None or (isinstance(frac, float) and math.isnan(frac)):
        return "unknown"
    return f"{frac * 100:.1f}%"


# race summarisation utilities


def _drop_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with pit laps removed when an ``is_pit_lap`` column exists."""

    if "is_pit_lap" in df.columns:
        return df[~df["is_pit_lap"]].copy()
    return df.copy()


def _summarise_driver_pace(lap_df: pd.DataFrame) -> str:
    """Build a short text summary of driver pace from ``lap_df``.

    The function is defensive: it tolerates missing columns and
    returns a clear message instead of raising.

    
    
    damn so good
    """

    df = _drop_pit_laps(lap_df)

    if df.empty or "lap_time_s" not in df.columns:
        return "Lap-time summary unavailable (no non-pit laps with lap_time_s)."

    lap_times = [float(x) for x in df["lap_time_s"].values if pd.notna(x)]
    if not lap_times:
        return "Lap-time summary unavailable (lap_time_s is all NaN)."

    best = min(lap_times)
    med = statistics.median(lap_times)
    std = statistics.pstdev(lap_times) if len(lap_times) > 1 else 0.0
    n_laps = len(lap_times)

    lap_numbers = [int(x) for x in df["lap"].values if pd.notna(x)] if "lap" in df.columns else []
    first_lap = min(lap_numbers) if lap_numbers else None
    last_lap = max(lap_numbers) if lap_numbers else None

    first_last = (
        f"Laps range from {first_lap} to {last_lap}" if first_lap is not None and last_lap is not None else "Lap range unknown"
    )

    return textwrap.dedent(
        f"""
        Driver pace summary:
        - Clean laps counted (no pit laps): {n_laps}
        - Best lap: {best:.3f} s
        - Median lap: {med:.3f} s
        - Lap-time spread (std dev): {std:.3f} s
        - {first_last}
        """
    ).strip()


def _summarise_stints(lap_df: pd.DataFrame) -> str:
    """Summarise tyre stints when ``stint_index`` / ``stint_lap`` exist.

    The summary intentionally stays high-level so that it can be
    embedded in the prompt without blowing context length.
    """

    if "stint_index" not in lap_df.columns:
        return "Stint summary: not available (stint_index column missing)."

    df = _drop_pit_laps(lap_df)
    if df.empty:
        return "Stint summary: no non-pit laps available."

    parts: List[str] = ["Stint overview:"]

    grouped = df.groupby("stint_index", sort=True)
    for stint_idx, g in grouped:
        laps = g["lap"].tolist() if "lap" in g.columns else []
        if not laps:
            continue
        n = len(laps)
        first, last = int(min(laps)), int(max(laps))
        lt = [float(x) for x in g.get("lap_time_s", []) if pd.notna(x)]
        best = min(lt) if lt else None
        med = statistics.median(lt) if lt else None
        parts.append(
            "- Stint {idx}: laps {first}-{last} ({n} laps), best {best}, median {med}".format(
                idx=int(stint_idx),
                first=first,
                last=last,
                n=n,
                best=_fmt_opt(best, " s"),
                med=_fmt_opt(med, " s"),
            )
        )

    return "\n".join(parts) if len(parts) > 1 else "Stint summary: could not compute."


def _summarise_sectors(lap_df: pd.DataFrame) -> str:
    """Return a compact summary of sector performance if present.

    Expected columns: ``sector1_time_s``, ``sector2_time_s``,
    ``sector3_time_s``. Missing columns are ignored.
    """

    df = _drop_pit_laps(lap_df)
    cols = [c for c in ["sector1_time_s", "sector2_time_s", "sector3_time_s"] if c in df.columns]
    if not cols or df.empty:
        return "Sector summary: not available."

    lines = ["Sector overview (clean laps):"]
    for col in cols:
        vals = [float(x) for x in df[col].values if pd.notna(x)]
        if not vals:
            continue
        best = min(vals)
        med = statistics.median(vals)
        lines.append(f"- {col.replace('_time_s', '').title()}: best {best:.3f} s, median {med:.3f} s")

    return "\n".join(lines) if len(lines) > 1 else "Sector summary: not available."


def _summarise_recent_form(lap_df: pd.DataFrame, window: int = 5) -> str:
    """Summarise the most recent *window* laps.

    The goal is to give the mmodel a quick view of whether things are
    trending up or down.
    """

    if "lap" not in lap_df.columns or "lap_time_s" not in lap_df.columns:
        return "Recent form: lap and lap_time_s columns not available."

    df = _drop_pit_laps(lap_df)
    if df.empty:
        return "Recent form: no clean laps available."

    df = df.sort_values("lap")
    tail = df.tail(window)

    laps = tail["lap"].tolist()
    times = [float(x) for x in tail["lap_time_s"].tolist() if pd.notna(x)]

    if not times:
        return "Recent form: lap_time_s missing in the last laps."

    first, last = int(min(laps)), int(max(laps))
    first_t, last_t = times[0], times[-1]
    delta = last_t - first_t

    trend = "stable"
    if abs(delta) > 0.8:
        trend = "improving" if delta < 0 else "worsening"
    elif abs(delta) > 0.3:
        trend = "slightly improving" if delta < 0 else "slightly worsening"

    return (
        f"Recent form (last {len(times)} clean laps, {first}-{last}): "
        f"from {first_t:.3f}s to {last_t:.3f}s ({delta:+.3f}s, {trend})."
    )


# ---------------------------------------------------------------------------
# Context builder used by Streamlit
# ---------------------------------------------------------------------------


def build_chat_context(
    lap_df: pd.DataFrame,
    track_meta: Any,
    cfg: Any,
    strategies: Dict[str, List[int]],
    race: str,
    car_id: str,
) -> str:
    """Build a textual context block for the strategy chat.

    Parameters
    HIHHIHIHIH
    lap_df:
        Per-lap telemetry features for this car / race.
    track_meta:
        Entry from TRACK_METAS for the selected track (has .name,
        .pit_lane_time_s, etc.).
    cfg:
        Strategy config object returned by ``make_config_from_meta``.
    strategies:
        Mapping of ``strategy_name -> list of pit laps``.
    race, car_id:
        Current race label (e.g. ``"R2"``) and car identifier.
    """

    track_name = getattr(track_meta, "name", "Unknown track")
    pit_loss = _safe_float(getattr(cfg, "pit_loss_s", None))
    base_lap = _safe_float(getattr(cfg, "base_lap_s", None))
    total_laps = int(lap_df["lap"].max()) if "lap" in lap_df.columns else None

    strat_lines: List[str] = []
    for name, pits in strategies.items():
        pit_str = ", ".join(str(p) for p in pits) if pits else "no scheduled stops"
        strat_lines.append(f"- {name}: pit on laps {pit_str}")

    strat_block = "\n".join(strat_lines) if strat_lines else "(no strategies defined)"

    pace_block = _summarise_driver_pace(lap_df)
    stint_block = _summarise_stints(lap_df)
    sector_block = _summarise_sectors(lap_df)
    recent_block = _summarise_recent_form(lap_df)

    header = textwrap.dedent(
        f"""
        Track & session:
        - Track: {track_name}
        - Race: {race}
        - Car: {car_id}
        - Total race laps: {total_laps if total_laps is not None else 'unknown'}
        - Base clean lap (trimmed median): {_fmt_opt(base_lap, ' s')}
        - Pit lane loss (green): {_fmt_opt(pit_loss, ' s')}
        """
    ).strip()

    strategies_text = textwrap.dedent(
        f"""
        Strategies currently on the table:
        {strat_block}
        """
    ).strip()

    context_blocks = [header, pace_block, stint_block, sector_block, recent_block, strategies_text]
    context = "\n\n".join(context_blocks)
    return context


# ---------------------------------------------------------------------------
# Live state serialisation helpers
# ---------------------------------------------------------------------------


def _format_live_state_for_prompt(live_state: Optional[Dict[str, Any]]) -> str:
    """Convert a live-state dict (or ``None``) into a human-readable snippet.

    We strip out high-churn keys such as timestamps and try to keep the
    JSON small so that it doesn’t dominate the prompt.
    """

    if not live_state:
        return "No current live state snapshot was provided."
    try:
        cleaned = dict(live_state)
        # Remove keys that usually don’t change the meaning of the state
        for noisy_key in ["timestamp", "last_updated", "hash"]:
            cleaned.pop(noisy_key, None)
        return json.dumps(cleaned, indent=2, sort_keys=True)
    except Exception:
        return "Live state provided but could not be serialised cleanly."


# ---------------------------------------------------------------------------
# Lightweight intent classification
# ---------------------------------------------------------------------------

_GREETING_KEYWORDS = [
    "hi",
    "hi there",
    "hello",
    "hey",
    "hey there",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
]


def _is_greeting(question: str) -> bool:
    """Return ``True`` if *question* looks like a greeting or smalltalk."""

    q = question.strip().lower()
    if q in _GREETING_KEYWORDS or q in ["how are you", "how are you?"]:
        return True
    for greet in _GREETING_KEYWORDS:
        if q.startswith(greet + " "):
            return True
    return False


def _looks_race_related(question: str) -> bool:
    """Heuristic: does *question* look like a race / telemetry question?"""

    q = question.lower()
    keywords = [
        "lap",
        "pit",
        "pits",
        "box",
        "tyre",
        "tire",
        "safety car",
        "yellow flag",
        "caution",
        "strategy",
        "stint",
        "race",
        "barber",
        "vir",
        "gr86",
        "gap",
        "sector",
        "undercut",
        "overcut",
        "fuel save",
        "push now",
        "protect the tyres",
    ]
    return any(kw in q for kw in keywords)


def _looks_race_education(question: str) -> bool:
    """Detect general racing / engineering explainer questions.

    Examples:
    - "what is a race engineer?"
    - "how does an undercut work?"
    - "why do we save tyres behind a safety car?"
    """

    q = question.strip().lower()
    edu_keywords = [
        "what is a race engineer",
        "what is race engineer",
        "what does a race engineer",
        "explain undercut",
        "explain overcut",
        "what is a safety car",
        "what is tyre deg",
        "what is tire deg",
        "why do we pit",
        "why do we save fuel",
    ]
    if any(k in q for k in edu_keywords):
        return True

    # Generic pattern: "what is" / "how does" + race-y word
    if (q.startswith("what is") or q.startswith("how does")) and _looks_race_related(q):
        return True
    return False


_TOOL_KEYWORDS = [
    "streamlit",
    "app",
    "dashboard",
    "tool",
    "button",
    "tab",
    "predictive model",
    "decision reviewer",
    "chatbot",
    "vision",
    "computer vision",
    "live race copilot",
    "strategy brain",
    "driver insights",
]


def _classify_question(question: str) -> IntentLiteral:
    """Rough intent classification.

    This is rule-based on purpose so that behaviour is predictable and
    easy to adjust without retraining any model.
    """

    q = question.strip().lower()

    if _is_greeting(q):
        return "smalltalk"

    if _looks_race_education(q):
        return "race_education"

    if _looks_race_related(q):
        return "race_core"

    if any(kw in q for kw in _TOOL_KEYWORDS):
        return "tool_help"

    return "offtopic"


# helper like a pro


def _summarise_scope() -> str:
    """Return a human-readable description of what the assistant can do."""

    return (
        "I’m your GT race engineer assistant. I can:\n"
        "- Explain laps, stints, tyre phases, and gaps\n"
        "- Compare pit timing options (box now vs stay out)\n"
        "- Reason about cautions / safety car risk\n"
        "- Interpret predictive models used in this app\n"
        "- Review race-engineer decisions and strategy options\n"
        "- Explain how to use the app tabs:\n"
        "  * Strategy Brain\n"
        "  * Driver Insights\n"
        "  * Predictive Models\n"
        "  * Strategy Chat\n"
        "  * Live Race Copilot\n"
        "  * Vision (including computer vision features)\n"
        "- Handle light smalltalk but I’m specialised for racing and strategy."
    )


def _summarise_app_tabs() -> str:
    """Return a brief description of each major Streamlit tab.

    This text is reused both in model prompts and in the non-model
    fallback path so that behaviour stays consistent.
    """

    lines = [
        "This app is organised into several tabs:",
        "- **Strategy Brain** – compare pit strategies, caution windows, and Monte Carlo simulations.",
        "- **Driver Insights** – analyse lap times, sector performance, and tyre degradation.",
        "- **Predictive Models** – train and visualise lap-time models on per-lap features.",
        "- **Strategy Chat** – this conversational assistant, grounded in your data.",
        "- **Live Race Copilot** – live feed that combines heuristics with Gemini for radio-style calls.",
        "- **Vision** – computer-vision experiments analysing on-track images of the car and circuit.",
    ]
    return "\n".join(lines)



# answer_engineer



def _summarise_history(chat_history: Optional[Iterable[Dict[str, str]]]) -> str:
    """Turn a list of chat turns into a compact text snippet.

    Only the last few turns are kept and each turn is truncated. This
    is enough for coherence without consuming the whole context window.
    """

    if not chat_history:
        return "No prior conversation in this session."

    # We expect a list of dicts with keys 'role' and 'content'.
    # We walk from the end backwards and collect up to 3 pairs.
    pairs: List[Tuple[str, str]] = []  # (user, assistant)
    current_user: Optional[str] = None
    current_assistant: Optional[str] = None

    for turn in reversed(list(chat_history)):
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "assistant":
            if current_assistant is None:
                current_assistant = content
        elif role == "user":
            if current_user is None:
                current_user = content
        # When we have at least a user message, close the pair
        if current_user is not None:
            pairs.append((current_user, current_assistant or ""))
            current_user = None
            current_assistant = None
        if len(pairs) >= 3:
            break

    if not pairs:
        return "No prior conversation in this session."

    pairs.reverse()  # chronological order

    lines: List[str] = []
    for u, a in pairs:
        u_short = (u[:300] + "...") if len(u) > 300 else u
        a_short = (a[:300] + "...") if len(a) > 300 else a
        lines.append(f"User: {u_short}")
        if a_short:
            lines.append(f"Engineer: {a_short}")

    return "\n".join(lines)


def answer_engineer(
    model: Optional[genai.GenerativeModel],
    question: str,
    context: str,
    live_state: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Answer a question as the race-engineer assistant.

    Parameters
    ----------
    model:
        Configured ``google.generativeai.GenerativeModel`` instance, or
        ``None`` if Gemini is unavailable. The behaviour is still
        useful in the ``None`` case but answers become more generic.
    question:
        Engineer's free-form question.
    context:
        Static context built by :func:`build_chat_context`.
    live_state:
        Optional live state snapshot from :mod:`live_state`.
    chat_history:
        Optional list of prior conversation turns. Each turn is a dict
        with keys ``"role"`` ("user" or "assistant") and
        ``"content"``.

    Returns
    -------
    str
        Markdown-formatted answer ready for display in Streamlit.
    """

    question = (question or "").strip()
    if not question:
        return (
            "I didn’t catch a question. Try asking something like:\n"
            "- *Should we box now or stay out if there’s a safety car risk?*\n"
            "- *How are our lap times trending in the last 5 laps?*\n"
            "- *What does the Predictive Models tab actually do?*"
        )

    mode: IntentLiteral = _classify_question(question)

    # ------------------------------------------------------------------
    # Smalltalk is handled locally without the model for snappy replies.
    # ------------------------------------------------------------------
    if mode == "smalltalk":
        greeting = (
            "Hey! 👋 I’m your race engineer assistant.\n\n"
            + _summarise_scope()
            + "\n\nAsk me anything about your race, strategy, telemetry, or how to use this app."
        )
        return greeting

    # ------------------------------------------------------------------
    # Gemini unavailable -> graceful fallback paths per intent.
    # ------------------------------------------------------------------
    if model is None:
        if mode in ("race_core", "race_education"):
            return (
                "Gemini isn’t configured in this environment, so I can’t "
                "provide data-aware answers.\n\n"
                + _summarise_scope()
                + "\n\nHere are some general principles you can use right now:\n"
                "- Always relate pit decisions to tyre state, fuel, and track position.\n"
                "- Compare expected time loss of a pit stop vs degradation from staying out.\n"
                "- Treat safety cars as cheap stops but beware of track position risk.\n"
                "- Use your sector times and consistency to judge whether the driver can push or should manage."
            )
        if mode == "tool_help":
            return (
                _summarise_app_tabs()
                + "\n\nAI-specific features (chat, decision review, vision analysis) "
                "require a configured Gemini API key. Right now I can only "
                "describe the tools, not run them."
            )
        # offtopic
        return (
            "That question is outside the scope of my race engineer role.\n\n"
            + _summarise_scope()
        )

    # ------------------------------------------------------------------
    # Gemini path starts here.
    # ------------------------------------------------------------------

    history_text = _summarise_history(chat_history)
    live_text = _format_live_state_for_prompt(live_state)

    # We include the intent so that the model can branch behaviour in a
    # controllable way.
    prompt = textwrap.dedent(
        f"""
        You are a highly experienced GT race engineer working with a Toyota GR86 Cup car
        and this specific Streamlit app.

        ### Static race context
        {context}

        ### Current live snapshot (may be approximate)
        {live_text}

        ### Recent conversation (last few turns)
        {history_text}

        ### Assistant capabilities
        {_summarise_scope()}

        ### App structure
        {_summarise_app_tabs()}

        ### Intent classification
        The user question has been pre-classified as: **{mode}**
        where intents mean:
        - race_core: direct questions about lap times, stints, pit windows, tyres, or cautions.
        - race_education: general questions about race engineering concepts.
        - tool_help: questions about how to use the app, its tabs, or models.
        - smalltalk: greetings or casual chatter.
        - offtopic: questions that are not related to racing or the app.

        ### Instructions per intent

        - For **race_core**:
          - Provide a structured answer with headings:
            ### Recommendation
            ### Rationale
            ### Risks & what to watch
            ### Next checks
          - Use 3–8 concise bullet points overall.
          - Ground everything in the numeric context or live snapshot where possible.
          - If specific numeric values (lap times, gaps, tyre ages) are not present, do NOT invent them.
          - Instead, speak qualitatively ("faster", "slower", "high risk") and describe what you would check.

        - For **race_education**:
          - Give a clear, beginner-friendly explanation of the concept.
          - Relate the explanation back to GR Cup / GT-style racing when relevant.
          - Keep the answer within 4–8 short paragraphs or bullet points.

        - For **tool_help**:
          - Explain how to use the relevant parts of the app
            (Strategy Brain, Driver Insights, Predictive Models,
             Strategy Chat, Live Race Copilot, Vision).
          - Use concrete steps and references to sliders / buttons when helpful.
          - Do not make up track or race data.

        - For **offtopic**:
          - Politely say the question is not race-related.
          - Optionally provide a short generic answer if it is safe.
          - Then steer the conversation back to race-engineering topics with 1–3 suggestions.

        In **all** cases:
        - If critical information is missing (e.g., laps remaining, tyre age, gaps),
          include a short section called "Missing info" listing what you would need.
        - You may propose simple "if/then" branches when the answer depends on unknowns
          (e.g., "If we expect a safety car in the next 3 laps, then box now; otherwise stay out").
        - Never claim that you can see live video or images unless such data is explicitly described
          in the context or live snapshot.
        - When the user question is ambiguous, explain the ambiguity and give 1–3 concrete follow-up questions.

        User question:
        {question}
        """
    ).strip()

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            return "No answer returned by the model."
        return text
    except Exception as e:  # pragma: no cover - defensive path
        return f"(Gemini chat error: {e})"