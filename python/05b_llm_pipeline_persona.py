"""
05b_llm_pipeline_persona.py

Persona-conditioned version of the synthetic respondent pipeline. Identical
to 05_llm_pipeline.py except each API call is prefixed with a randomized
consumer persona (see 10_generate_personas.py), so the model role-plays as
that specific person rather than an unspecified "ordinary consumer."

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python 05b_llm_pipeline_persona.py --n_respondents 332 --start 1
"""

import os
import json
import time
import argparse
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-5"
TASKS_DIR = Path("../data/llm_tasks")
PERSONAS_PATH = Path("../data/personas.json")
OUTPUT_DIR = Path("../data/llm_responses_persona")

SYSTEM_PROMPT_TEMPLATE = """You are simulating a single consumer completing a market research \
survey about digital cameras. {persona_description}

You will see 16 separate shopping scenarios. In each scenario you are shown 4 different \
digital cameras plus the option to buy none of them. Answer as the person described above \
would, keeping your preferences internally consistent across all 16 scenarios based on your \
age, income, photography experience, tech-savviness, and budget mindset -- don't flip \
preferences task-to-task.

For each of the 16 tasks, choose exactly one option: 1, 2, 3, or 4 (the camera in that \
position) or 5 (none of these / no purchase).

Respond with ONLY a JSON array of 16 integers, one per task, in task order. \
No explanation, no markdown, no extra text. Example format: [1, 3, 5, 2, ...]"""


def format_task_prompt(tasks):
    lines = []
    for t in tasks:
        lines.append("\nTask " + str(t["task_number"]) + ":")
        for i, alt in enumerate(t["alternatives"], start=1):
            lines.append("  Option " + str(i) + ": " + alt["description"])
    return "\n".join(lines)


def get_synthetic_respondent_choices(client, tasks, persona_description, respondent_id):
    prompt = format_task_prompt(tasks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(persona_description=persona_description)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )

    text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise ValueError("Respondent " + str(respondent_id) + ": no text block in response")
    raw_text = text_blocks[0].strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        choices = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError("Respondent " + str(respondent_id) + ": could not parse JSON: " + raw_text) from e

    if len(choices) != 16 or not all(c in (1, 2, 3, 4, 5) for c in choices):
        raise ValueError("Respondent " + str(respondent_id) + ": malformed choices: " + str(choices))

    return choices


def main(n_respondents, start):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment before running this script.")

    client = anthropic.Anthropic(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PERSONAS_PATH) as f:
        personas = {p["respondent_id"]: p for p in json.load(f)}

    for i in range(start, start + n_respondents):
        out_path = OUTPUT_DIR / ("synthetic_respondent_persona_%03d.json" % i)
        if out_path.exists():
            print("[" + str(i) + "] already done, skipping")
            continue

        task_path = TASKS_DIR / ("respondent_%03d_tasks.json" % i)
        if not task_path.exists():
            print("[" + str(i) + "] no task file, stopping")
            break

        with open(task_path) as f:
            tasks = json.load(f)

        persona = personas.get(i)
        if persona is None:
            print("[" + str(i) + "] no persona found, skipping")
            continue

        try:
            choices = get_synthetic_respondent_choices(client, tasks, persona["description"], i)
        except Exception as e:
            print("[" + str(i) + "] FAILED: " + str(e))
            continue

        with open(out_path, "w") as f:
            json.dump({"respondent_id": i, "y": choices, "persona": persona}, f, indent=2)

        print("[" + str(i) + "] choices: " + str(choices))
        time.sleep(0.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_respondents", type=int, default=332)
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()
    main(args.n_respondents, args.start)
