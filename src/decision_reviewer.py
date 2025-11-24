# src/decision_reviewer.py
from __future__ import annotations
import json
from typing import Any, Dict
import google.generativeai as genai

def build_review_prompt(decision: Dict[str, Any]) -> str:
    return f"""
You are a senior GT race engineer reviewing a proposed decision.

Here is the structured context (JSON):

{json.dumps(decision, indent=2)}

Your job:
- Decide: **Agree** or **Disagree** with the proposed action.
- In 3–5 bullet points, justify the verdict.
- Explicitly mention tyre state, risk level, and impact of a possible caution.
Answer in this format:

Verdict: <AGREE or DISAGREE + short reason>
Bullets:
- ...
- ...
- ...
"""

def review_decision(model, decision: Dict[str, Any]) -> str:
    prompt = build_review_prompt(decision)
    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"(Decision review unavailable: {e})"