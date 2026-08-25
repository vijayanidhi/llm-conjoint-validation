"""
05_llm_pipeline.py

Calls Claude via the Anthropic API to act as a synthetic survey respondent
completing the same 16-task digital camera choice-based conjoint (CBC)
survey that a real bayesm respondent completed.

Design decisions (see README for rationale):
- ONE API call per synthetic respondent, containing all 16 tasks. This keeps
  a synthetic respondent's preferences internally consistent across tasks,
  mirroring how a real person's stable preferences show up across a survey.
- No persona/demographic backstory in the baseline version (kept simple to
  start -- persona variants are a natural robustness/follow-up test).
- Output is forced into strict JSON so responses are trivially parseable
  into the same {y: [...], X: [...]} format bayesm's `camera` data uses,
  making it a drop-in for the same R modeling scripts (02 and 03).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python 05_llm_pipeline.py --n_respondents 332 --start 1

Requires: pip install anthropic --break-system-packages
"""

import os
import json
import time
import argparse
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-5"   # swap as needed; keep consistent across the run for a clean comparison
TASKS_DIR = Path("../data/llm_tasks")
OUTPUT_DIR = Path("../data/llm_responses")

SYSTEM_PROMPT = """You are simulating a single consumer completing a market research \
survey about digital cameras. You will see 16 separate shopping scenarios. In each \
scenario you are shown 4 different digital cameras plus the option to buy none of them. \
Imagine you are an ordinary consumer in the market for a digital camera, with your own \
consistent (but unstated) preferences about brand, features, and price. Answer as that \
SAME person would across all 16 scenarios -- your preferences should be internally \
consistent (e.g. if you show a strong preference for low price in task 1, keep showing \
that in later tasks, don't flip preferences task-to-task).

For each of the 16 tasks, choose exactly one option: 1, 2, 3, or 4 (the camera in that \
position) or 5 (none of these / no purchase).

Respond with ONLY a JSON array of 16 integers, one per task, in task order. \
No explanation, no markdown, no extra text. Example format: [1, 3, 5, 2, ...]"""


def format_task_prompt(tasks: list) -> str:
    lines = []
    for t in tasks:
        lines.append(f"\nTask {t['task_number']}:")
        for i, alt in enumerate(t["alternatives"], start=1):
            lines.append(f"  Option {i}: {alt['description']}")
    return "\n".join(lines)


def get_synthetic_respondent_choices(client: anthropic.Anthropic, tasks: list, respondent_id: int) -> list:
    """Sends all 16 tasks in one call, returns a list of 16 chosen alt numbers (1-5)."""
    prompt = format_task_prompt(tasks)

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    # Find the actual text block -- the model may return a thinking block first
    text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise ValueError(f"Respondent {respondent_id}: no text block in response: {response.content!r}")
    raw_text = text_blocks[0].strip()

    # Defensive parsing: strip markdown fences if the model adds them despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        choices = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Respondent {respondent_id}: could not parse model output as JSON: {raw_text!r}") from e

    if len(choices) != 16 or not all(c in (1, 2, 3, 4, 5) for c in choices):
        raise ValueError(f"Respondent {respondent_id}: malformed choices: {choices}")

    return choices


def main(n_respondents: int, start: int):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment before running this script.")

    client = anthropic.Anthropic(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(start, start + n_respondents):
        out_path = OUTPUT_DIR / f"synthetic_respondent_{i:03d}.json"
        if out_path.exists():
            print(f"[{i}] already done, skipping")
            continue

        task_path = TASKS_DIR / f"respondent_{i:03d}_tasks.json"
        with open(task_path) as f:
            tasks = json.load(f)

        try:
            choices = get_synthetic_respondent_choices(client, tasks, i)
        except Exception as e:
            print(f"[{i}] FAILED: {e}")
            continue

        with open(out_path, "w") as f:
            json.dump({"respondent_id": i, "y": choices}, f, indent=2)

        print(f"[{i}] choices: {choices}")
        time.sleep(0.2)  # light rate-limit courtesy pause; tune based on your tier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_respondents", type=int, default=332)
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()
    main(args.n_respondents, args.start)
