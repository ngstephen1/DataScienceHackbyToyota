from __future__ import annotations

"""
Chat assistant for the Racing Hokies / DataScienceHackbyToyota project.

Goals:
- Act as a production-style race engineer copilot.
- Use static race context, live state, and recent chat history.
- Support different modes: core race strategy, race education, tool help,
  smalltalk, off-topic.
- Provide refinement actions on top of the last answer:
  * shorten
  * more_detail
  * visualize
  * simplify (explain simply)

This module is intentionally verbose and heavily commented to make the logic
easy to reason about and extend.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

import json
import textwrap

import pandas as pd
import google.generativeai as genai


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ChatRole = Literal["user", "assistant", "system"]
QuestionMode = Literal["race_core", "race_education", "tool_help", "smalltalk", "offtopic"]
RefineAction = Literal["shorten", "more_detail", "visualize", "simplify"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """Simple representation of a chat message for prompt construction."""
    role: ChatRole
    content: str
    meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ChatMessage":
        """Normalise a dict from Streamlit session_state into ChatMessage."""
        role = raw.get("role", "user")
        if role not in ("user", "assistant", "system"):
            role = "user"
        content = str(raw.get("content", "") or "")
        meta = raw.get("meta", None)
        return ChatMessage(role=role, content=content, meta=meta)


# ---------------------------------------------------------------------------
# Pace and context helpers (from your original file, extended but compatible)
# ---------------------------------------------------------------------------


def _summarise_driver_pace(lap_df: pd.DataFrame) -> str:
    """Build a short text summary of driver pace from lap_df."""
    df = lap_df.copy()

    # Drop pit laps if flagged
    if "is_pit_lap" in df.columns:
        df = df[~df["is_pit_lap"]]

    if df.empty or "lap_time_s" not in df.columns:
        return "Lap-time summary unavailable (no non-pit laps with lap_time_s)."

    best = float(df["lap_time_s"].min())
    med = float(df["lap_time_s"].median())
    std = float(df["lap_time_s"].std())
    n_laps = int(df.shape[0])

    first_lap = int(df["lap"].min()) if "lap" in df.columns else 0
    last_lap = int(df["lap"].max()) if "lap" in df.columns else 0

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
    total_laps = int(lap_df["lap"].max()) if "lap" in lap_df.columns and not lap_df.empty else None

    strat_lines: List[str] = []
    for name, pits in strategies.items():
        pit_str = ", ".join(str(p) for p in pits) if pits else "no scheduled stops"
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
    """Make a compact JSON-like block from the current live_state."""
    if not live_state:
        return "No current live state snapshot was provided."
    try:
        cleaned = dict(live_state)
        # Remove noisy or fast-changing keys that add little value to prompting
        cleaned.pop("timestamp", None)
        cleaned.pop("debug", None)
        return json.dumps(cleaned, indent=2)
    except Exception:
        return "Live state provided but could not be serialised cleanly."


# ---------------------------------------------------------------------------
# Classification and scope helpers
# ---------------------------------------------------------------------------


def _is_greeting(question: str) -> bool:
    greetings = [
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
        "lap",
        "pit",
        "pits",
        "box",
        "tyre",
        "tire",
        "safety car",
        "yellow flag",
        "full course yellow",
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
        "protect tyres",
    ]
    return any(kw in q for kw in keywords)


def _looks_race_education(question: str) -> bool:
    """Questions like 'what is an undercut', 'explain tyre deg'."""
    q = question.strip().lower()
    edu_prefixes = [
        "what is",
        "what's",
        "explain",
        "why do",
        "why does",
        "how does",
        "how do",
        "can you explain",
        "could you explain",
        "what does",
    ]
    if any(q.startswith(p) for p in edu_prefixes) and _looks_race_related(q):
        return True
    return False


def _looks_tool_help(question: str) -> bool:
    q = question.lower()
    tool_keywords = [
        "streamlit",
        "app",
        "dashboard",
        "tab",
        "button",
        "predictive model",
        "predictive models",
        "decision reviewer",
        "chatbot",
        "vision",
        "computer vision",
        "strategy chat",
        "live race copilot",
        "driver insights",
        "strategy brain",
        "data sense",
        "toyota datasense",
    ]
    return any(kw in q for kw in tool_keywords)


def _classify_question(question: str) -> QuestionMode:
    """Classify the question into one of our modes."""
    q = question.strip().lower()

    if _is_greeting(q):
        return "smalltalk"
    if _looks_race_education(q):
        return "race_education"
    if _looks_tool_help(q):
        # Explicit tool-help wins over generic race-core in this heuristic
        return "tool_help"
    if _looks_race_related(q):
        return "race_core"
    return "offtopic"


def _summarise_scope() -> str:
    """Short description of what this assistant can and cannot do."""
    return (
        "I’m your GT race engineer assistant. I can:\n"
        "- Explain laps, stints, tyre phases, and gaps\n"
        "- Compare pit timing options and 1-stop vs 2-stop strategies\n"
        "- Reason about cautions / safety car windows\n"
        "- Interpret simple predictive models used in this app\n"
        "- Review race-engineer decisions and strategy options\n"
        "- Explain how to use the app tabs:\n"
        "  * Strategy Brain (offline strategy multiverse)\n"
        "  * Driver Insights (pace, sectors, degradation)\n"
        "  * Predictive Models (lap-time forecasting)\n"
        "  * Strategy Chat (this assistant)\n"
        "  * Live Race Copilot (live-state driven recommendations)\n"
        "  * Vision (computer vision / image-based insights)\n"
        "- Handle light smalltalk, but I’m specialised for racing and strategy."
    )


# ---------------------------------------------------------------------------
# Chat history handling
# ---------------------------------------------------------------------------


def _normalise_history(chat_history: Optional[List[Dict[str, str]]]) -> List[ChatMessage]:
    """Convert raw chat history list of dicts into ChatMessage objects."""
    if not chat_history:
        return []
    messages: List[ChatMessage] = []
    for raw in chat_history:
        try:
            messages.append(ChatMessage.from_dict(raw))
        except Exception:
            # Best effort; skip malformed entries
            continue
    return messages


def _pair_history(
    messages: List[ChatMessage],
    max_pairs: int = 4,
) -> List[Tuple[Optional[ChatMessage], Optional[ChatMessage]]]:
    """
    Group recent history into (user, assistant) pairs.

    Returns list ordered oldest -> newest.
    Each item is (user_msg, assistant_msg); either can be None.
    """
    if not messages:
        return []

    pairs: List[Tuple[Optional[ChatMessage], Optional[ChatMessage]]] = []
    current_user: Optional[ChatMessage] = None
    current_assistant: Optional[ChatMessage] = None

    # Iterate from newest to oldest, then reverse at the end
    for msg in reversed(messages):
        if msg.role == "assistant":
            if current_assistant is None:
                current_assistant = msg
            else:
                # If we already had an assistant response, push existing pair
                pairs.append((current_user, current_assistant))
                current_user = None
                current_assistant = msg
        elif msg.role == "user":
            if current_user is None:
                current_user = msg
            else:
                # Push older user-only if no assistant yet
                pairs.append((current_user, current_assistant))
                current_user = msg
                current_assistant = None

        if len(pairs) >= max_pairs:
            break

    # Push any remaining current pair
    if current_user is not None or current_assistant is not None:
        pairs.append((current_user, current_assistant))

    # We built from newest to oldest, reverse to get chronological order
    return list(reversed(pairs[-max_pairs:]))


def _truncate_text(s: str, max_chars: int = 350) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _summarise_history_for_prompt(chat_history: Optional[List[Dict[str, str]]]) -> str:
    """
    Turn the last few conversation turns into a compact text block for prompting.

    This is what actually makes the assistant aware of what has been said.
    """
    messages = _normalise_history(chat_history)
    pairs = _pair_history(messages, max_pairs=4)

    if not pairs:
        return "No prior conversation in this session."

    lines: List[str] = []
    for user_msg, assistant_msg in pairs:
        if user_msg is not None:
            lines.append("User: " + _truncate_text(user_msg.content))
        if assistant_msg is not None:
            lines.append("Engineer: " + _truncate_text(assistant_msg.content))
        lines.append("")  # blank line between pairs

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Local refinement fallbacks (when model is None)
# ---------------------------------------------------------------------------


def _shorten_text_local(answer: str, max_chars: int = 400) -> str:
    """Very naive local 'shortener' when Gemini is not available."""
    if not answer:
        return "No answer to shorten."
    # Heuristic: keep first 2 paragraphs or up to max_chars
    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
    if not paragraphs:
        return _truncate_text(answer, max_chars=max_chars)
    shortened = "\n\n".join(paragraphs[:2])
    return _truncate_text(shortened, max_chars=max_chars)


def _simplify_text_local(answer: str) -> str:
    """Very simple 'explain simply' fallback."""
    if not answer:
        return "No answer to simplify."
    first_para = answer.split("\n\n")[0].strip()
    return (
        "Here is a very short and simple version:\n\n"
        + _truncate_text(first_para, max_chars=400)
    )


# ---------------------------------------------------------------------------
# Main answer generation
# ---------------------------------------------------------------------------


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

    - Handles different modes (race_core, race_education, tool_help, smalltalk, offtopic).
    - For smalltalk: replies locally and does not call Gemini.
    - For other modes: uses Gemini if available; otherwise falls back to safe,
      generic guidance.
    """
    question = (question or "").strip()
    if not question:
        return "Please type a question for the race engineer copilot."

    mode = _classify_question(question)

    # --- Smalltalk: do it locally, cheap and fast ---
    if mode == "smalltalk":
        greeting = (
            "Hey! 👋 I’m your race engineer assistant.\n\n"
            + _summarise_scope()
            + "\n\nYou can ask things like:\n"
            "- \"If we box this lap, do we undercut the car ahead?\"\n"
            "- \"How does tyre degradation affect our strategy at Barber?\"\n"
            "- \"What does an undercut mean in racing?\""
        )
        return greeting

    # --- No model available ---
    if model is None:
        if mode == "race_core":
            return (
                "Gemini is not configured in this environment, so I cannot read live "
                "data or generate custom strategy text.\n\n"
                + _summarise_scope()
                + "\n\nGeneral race-strategy checklist:\n"
                "- Estimate your current stint age and tyre performance window.\n"
                "- Compare the time loss of a pit stop vs. the lap-time gain on fresh tyres.\n"
                "- Consider track position: pitting might drop you into traffic.\n"
                "- Weigh the probability of a Safety Car in the next 3–5 laps.\n"
                "- Use your telemetry and timing data to refine the call."
            )
        if mode == "race_education":
            return (
                "Gemini is offline here, but I can give you a quick concept.\n\n"
                "In general, many race strategy terms boil down to managing lap time,\n"
                "tyre life, and track position. For more depth, try enabling Gemini\n"
                "with a GEMINI_API_KEY and asking again."
            )
        if mode == "tool_help":
            return (
                "This app provides several tools to support race engineering:\n"
                "- Strategy Brain: analyse and compare pit strategies offline.\n"
                "- Driver Insights: telemetry-based pace and sector analysis.\n"
                "- Predictive Models: forecasts and risk assessments for lap times.\n"
                "- Strategy Chat: this assistant for natural-language questions.\n"
                "- Decision Reviewer: sanity-check proposed strategic calls.\n"
                "- Vision: computer vision features for track and car analysis.\n\n"
                "Currently, AI features are offline, so I cannot use Gemini-specific logic."
            )
        # offtopic
        return (
            "That question is mostly outside the scope of my race engineer role.\n\n"
            + _summarise_scope()
        )

    # --- Model is available: construct rich prompt ---

    history_text = _summarise_history_for_prompt(chat_history)
    live_text = _format_live_state_for_prompt(live_state)

    prompt = textwrap.dedent(
        f"""
        You are a highly experienced GT race engineer working with a Toyota GR86 Cup car
        and this specific Streamlit/telemetry app.

        ### Static race context
        {context}

        ### Current live snapshot (may be approximate or partially missing)
        {live_text}

        ### Recent conversation (oldest to newest)
        {history_text}

        ### Question mode
        The detected mode for this question is: {mode!r}
        Valid modes: race_core, race_education, tool_help, smalltalk, offtopic.

        ### Instructions by mode

        - For "race_core" (direct race engineering / strategy questions):
          - Provide a structured answer with the following headings:
            ### Recommendation
            ### Rationale
            ### Risks & what to watch
            ### Next checks
          - Use between 3 and 8 concise bullet points total.
          - Base reasoning only on the static context, live snapshot, and prior conversation.
          - If critical numbers (laps remaining, exact gaps, precise tyre age) are missing:
            - DO NOT invent them.
            - Under a heading **Missing info**, list up to 3 key data points that would change the decision.
            - Give conditional logic, for example:
              - "If we have <= 5 laps left, then ..."
              - "If the Safety Car probability is low, then ..."

        - For "race_education" (explain racing concepts):
          - Explain the concept clearly as if to a junior race engineer.
          - Use plain language first, then give a short more-technical note.
          - Provide 1–2 concrete examples from GT / sprint racing.
          - Do NOT fabricate specific lap times or car IDs.

        - For "tool_help" (questions about the app itself):
          - Focus on explaining how to use the relevant tabs:
            * Strategy Brain
            * Driver Insights
            * Predictive Models
            * Strategy Chat
            * Live Race Copilot
            * Vision
          - Clearly separate what the app can do vs. what it cannot.
          - Do not make up fake telemetry or secrets.
          - If context is insufficient (e.g., we don't know which tab is visible),
            ask the user to specify.

        - For "offtopic":
          - Politely say the question is not race-related.
          - Optionally provide a very short generic answer (1–2 sentences).
          - Then steer the user back to race-engineering, suggesting 2–3 example questions.

        - Always:
          - Be honest about uncertainty and missing information.
          - Avoid hallucinating precise numbers (lap times, gaps, fuel levels, tyre ages).
          - If clarification is needed, end the response with a section:
            ### Follow-up questions
            and list 1–3 concrete questions back to the engineer.

        ### Engineer's question
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


# ---------------------------------------------------------------------------
# Refinement of existing answers
# ---------------------------------------------------------------------------


def refine_answer(
    model: Optional[genai.GenerativeModel],
    base_answer: str,
    question: str,
    action: RefineAction,
    context: str,
    live_state: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Refine a previously generated answer according to the requested action.

    This is used by UI buttons like:
      - "Shorten"
      - "More detail"
      - "Visualize in plots"
      - "Explain simply"

    We treat the previous answer as a draft that we want to transform, not replace
    from scratch (so we include it in the prompt).
    """
    base_answer = (base_answer or "").strip()
    question = (question or "").strip()

    if not base_answer:
        return "There is no previous answer to refine."

    if model is None:
        # Fallbacks when Gemini is unavailable
        if action == "shorten":
            return _shorten_text_local(base_answer)
        if action == "simplify":
            return _simplify_text_local(base_answer)
        if action == "more_detail":
            return (
                "Gemini is not configured, so I cannot safely expand this answer.\n\n"
                "You can ask a more specific follow-up, for example:\n"
                "- \"Explain more about tyre degradation in this stint.\"\n"
                "- \"Walk me through how you would compare 1-stop vs 2-stop here.\""
            )
        if action == "visualize":
            return (
                "Gemini is not configured, but here are some plot ideas you can implement "
                "in your notebooks or app:\n\n"
                "- Lap time vs lap number for the current stint.\n"
                "- Gap to car ahead/behind vs lap number.\n"
                "- Tyre-related metrics (e.g. APS mean vs lap) to visualise degradation."
            )
        return "Unknown refinement action."

    history_text = _summarise_history_for_prompt(chat_history)
    live_text = _format_live_state_for_prompt(live_state)

    # Action-specific instruction snippet
    if action == "shorten":
        action_block = textwrap.dedent(
            """
            You will:
            - Keep the core recommendation and reasoning.
            - Rewrite the answer to be much shorter and punchier.
            - Target at most 3–6 short bullet points.
            - Keep the structure if useful (e.g., headings), but brevity is more important.
            """
        ).strip()
    elif action == "more_detail":
        action_block = textwrap.dedent(
            """
            You will:
            - Expand the existing answer with more detailed reasoning.
            - Make implicit logic explicit (e.g., how tyre deg and pit windows interact).
            - Use concrete but approximate language without inventing exact numbers.
            - Stay under about 350 words.
            """
        ).strip()
    elif action == "visualize":
        action_block = textwrap.dedent(
            """
            You will:
            - Propose 1–3 specific plots that the race engineer could create.
            - For each plot, provide:
              * A short title.
              * What goes on the x-axis and y-axis.
              * What pattern to look for.
            - Optionally include tiny pseudocode snippets in Python using pandas and matplotlib
              (no need to be fully runnable; clarity over perfection).
            - Do NOT claim that plots already exist; you are suggesting what to build.
            """
        ).strip()
    elif action == "simplify":
        action_block = textwrap.dedent(
            """
            You will:
            - Rewrite the answer so that a new race engineer or interested fan can understand it.
            - Explain technical terms like 'undercut', 'stint', 'deg' in plain language.
            - Preserve the main recommendation, but reduce jargon.
            - Keep the answer focused and under about 250 words.
            """
        ).strip()
    else:
        return "Unknown refinement action."

    prompt = textwrap.dedent(
        f"""
        You are refining a previous answer from a GT race engineer assistant.

        ### Static race context
        {context}

        ### Current live snapshot (may be approximate or partially missing)
        {live_text}

        ### Recent conversation (oldest to newest)
        {history_text}

        ### Original question
        {question}

        ### Original answer to refine
        {base_answer}

        ### Refinement action
        The requested refinement action is: {action!r}

        ### What you must do
        {action_block}

        ### General rules
        - Do not contradict the original answer unless it is clearly unsafe or illogical.
        - Do not invent precise lap times, gaps, fuel levels, or tyre ages.
        - If the original answer seems to lack critical information, you may add a short
          'Missing info' note.
        - Return ONLY the refined answer text, with no extra commentary about the process.
        """
    ).strip()

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            return "No refined answer returned by the model."
        return text
    except Exception as e:
        return f"(Gemini refine error: {e})"