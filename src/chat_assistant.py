from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
import json
import textwrap

import pandas as pd
import google.generativeai as genai


def _summarise_driver_pace(lap_df: pd.DataFrame) -> str:
    """Build a short text summary of driver pace from lap_df."""
    df = lap_df.copy()

    # damn beautiful race
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


def _is_greeting(question: str) -> bool:
    greetings = [
        "hi", "hi there", "hello", "hey", "hey there", "yo",
        "good morning", "good afternoon", "good evening"
    ]
    q = question.strip().lower()
    if q in greetings or q in ["how are you", "how are you?"]:
        return True
    for greet in greetings:
        if q.startswith(greet + " "):
            return True
    return False


def _looks_race_related(question: str) -> bool:
    q = question.lower()
    keywords = [
        "lap", "pit", "pits", "box", "tyre", "tire", "safety car",
        "yellow flag", "caution", "strategy", "stint", "race", "barber",
        "vir", "gr86", "gap", "sector"
    ]
    return any(kw in q for kw in keywords)


def _classify_question(question: str) -> Literal["race_core", "tool_help", "smalltalk", "offtopic"]:
    q = question.strip().lower()
    if _is_greeting(q):
        return "smalltalk"
    if _looks_race_related(q):
        return "race_core"
    tool_keywords = [
        "streamlit", "app", "dashboard", "tool", "button", "tab",
        "predictive model", "decision reviewer", "chatbot", "vision", "computer vision"
    ]
    if any(kw in q for kw in tool_keywords):
        return "tool_help"
    return "offtopic"


def _summarise_scope() -> str:
    return (
        "I’m your GT race engineer assistant. I can:\n"
        "- Explain laps, stints, tyre phases, and gaps\n"
        "- Compare pit timing options\n"
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
        "- Handle light smalltalk but I’m specialized for racing and strategy."
    )


def answer_engineer(
    model: Optional[genai.GenerativeModel],
    question: str,
    context: str,
    live_state: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Use Gemini to answer a race-engineering question, given static context and
    optional current live_state and chat_history.

    Parameters
HIHIHIHIHII
    model:
        The Gemini generative model instance or None if unavailable.
    question:
        The engineer's question as a string.
    context:
        Static race context text.
    live_state:
        Optional current live state snapshot dict.
    chat_history:
        Optional list of dicts with keys 'role' and 'content', representing recent conversation turns.
        Can be used to keep the conversation grounded and support multi-turn dialogue.

    Returns a short, radio-style explanation or a friendly response.
    """
    mode = _classify_question(question)

    if mode == "smalltalk":
        greeting = (
            "Hey! 👋 I’m your race engineer assistant.\n\n"
            + _summarise_scope()
            + "\n\nAsk me anything about your race, strategy, or telemetry."
        )
        # Return greeting without callng model regardless of availability
        return greeting

    if model is None:
        if mode == "race_core":
            return (
                "Gemini is not configured in this environment.\n\n"
                + _summarise_scope()
                + "\n\nWhile I can't access live data, here are some general race strategy principles:\n"
                "- Consider your current stint and tyre condition.\n"
                "- Compare expected pit stop time loss against tyre degradation.\n"
                "- Factor in safety car risk and track position.\n"
                "- Use your data and telemetry to refine decisions."
            )
        elif mode == "tool_help":
            return (
                "This app provides several tools to support race engineering:\n"
                "- Strategy Brain: analyse and compare pit strategies.\n"
                "- Driver Insights: telemetry and pace analysis.\n"
                "- Predictive Models: forecasts and risk assessments.\n"
                "- Strategy Chat: interactive assistant for race questions.\n"
                "- Decision Reviewer: review and evaluate past decisions.\n"
                "- Vision: computer vision features for track and car analysis.\n\n"
                "Currently, AI features are offline, so I cannot provide interactive answers."
            )
        else:  # offtopic
            return (
                "That question is outside the scope of my race engineer assistant role.\n\n"
                + _summarise_scope()
            )

    # model is not None path

    # Summarise recent chat history if provided
    if chat_history and len(chat_history) > 0:
        # Take up to last 3 user/assistant pairs (6 messages)
        # We' ll group by pairs: user then assistant
        # chat_history assumed to be list of dicts with 'rolee' and 'content
        # roles expected: 'user' and 'assistant'
        pairs = []
        temp_pair = {}
        count_pairs = 0
        for turn in reversed(chat_history):
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "assistant":
                temp_pair["assistant"] = content
            elif role == "user":
                temp_pair["user"] = content
                if "assistant" in temp_pair:
                    # We have a full pair
                    pairs.append(temp_pair)
                    temp_pair = {}
                    count_pairs += 1
                    if count_pairs >= 3:
                        break
                else:
                    # No assistant reply yet, just user message
                    pairs.append({"user": content})
                    temp_pair = {}
                    count_pairs += 1
                    if count_pairs >= 3:
                        break
        # Format pairs in chronological order (oldest first)
        pairs = list(reversed(pairs))
        history_lines = []
        for pair in pairs:
            user_text = pair.get("user", "")
            assistant_text = pair.get("assistant", "")
            user_text = (user_text[:300] + "...") if len(user_text) > 300 else user_text
            assistant_text = (assistant_text[:300] + "...") if len(assistant_text) > 300 else assistant_text
            history_lines.append(f"User: {user_text}")
            if assistant_text:
                history_lines.append(f"Engineer: {assistant_text}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "No prior conversation in this session."

    live_text = _format_live_state_for_prompt(live_state)

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

        ### Task
        - Interpret the mode of this question: {mode} (race_core, tool_help, smalltalk, offtopic).
        - For "race_core":
          - Provide a structured answer with headings:
            ### Recommendation
            ### Rationale
            ### Risks & what to watch
            ### Next checks
          - Use 3–8 concise bullet points overall.
        - For "tool_help":
          - Explain how to use relevant parts of the app (Strategy Brain, Driver Insights, Predictive Models, Strategy Chat, Decision Reviewer, Vision).
          - Do not make up track or race data.
        - For "offtopic":
          - Politely state the question is not race-related.
          - Optionally provide a short generic answer.
          - Then steer the conversation back to race-engineering topics.
        - If information is missing (e.g., laps remaining, tyre age, gaps), explicitly list missing pieces under a short "Missing info" bullet.
          Provide conditional logic (e.g., "If X then…, if Y then…").
        - Never invent precise lap times, gaps, or tyre ages that are not present in the context or live snapshot.
        - If clarification is needed, end the answer with a "Follow-up questions:" list of specific questions back to the engineer.

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