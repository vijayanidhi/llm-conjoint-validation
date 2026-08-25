"""
10_generate_personas.py

Generates 332 randomized consumer personas for the persona-conditioned arm
of the LLM synthetic respondent pipeline. Traits are sampled independently
and randomly to mirror the variation of a real population, since the real
bayesm camera dataset contains no respondent demographics to match against.

Traits (chosen for plausible relevance to camera purchase decisions):
- age: broad adult range
- income_bracket: affects price sensitivity
- photography_experience: affects feature/brand knowledge and preferences
- tech_savviness: affects interest in wifi/video/swivel-type features
- budget_mindset: a direct behavioral framing, not just a demographic proxy

Output: data/personas.json -- one persona per respondent_id (1-332)
"""

import json
import random
from pathlib import Path

random.seed(42)  # reproducibility

AGE_RANGE = (18, 75)
INCOME_BRACKETS = [
    "under $30,000/year", "$30,000-$60,000/year",
    "$60,000-$100,000/year", "$100,000-$150,000/year", "over $150,000/year",
]
PHOTOGRAPHY_EXPERIENCE = [
    "a complete beginner who has never owned a dedicated camera",
    "a casual hobbyist who takes photos occasionally",
    "an enthusiast who photographs regularly as a hobby",
    "a semi-professional with significant photography experience",
]
TECH_SAVVINESS = [
    "not very tech-savvy, prefers simple and familiar products",
    "moderately comfortable with technology",
    "very tech-savvy, enjoys researching specs and new features",
]
BUDGET_MINDSET = [
    "very price-conscious, always looking for the best deal",
    "willing to pay a bit more for quality but still budget-aware",
    "not very price-sensitive, prioritizes features and brand over cost",
]

N_RESPONDENTS = 332

personas = []
for i in range(1, N_RESPONDENTS + 1):
    persona = {
        "respondent_id": i,
        "age": random.randint(*AGE_RANGE),
        "income_bracket": random.choice(INCOME_BRACKETS),
        "photography_experience": random.choice(PHOTOGRAPHY_EXPERIENCE),
        "tech_savviness": random.choice(TECH_SAVVINESS),
        "budget_mindset": random.choice(BUDGET_MINDSET),
    }
    persona["description"] = (
        f"You are {persona['age']} years old, with a household income of {persona['income_bracket']}. "
        f"You are {persona['photography_experience']}. You are {persona['tech_savviness']}. "
        f"When shopping, you are {persona['budget_mindset']}."
    )
    personas.append(persona)

out_path = Path("../data/personas.json")
with open(out_path, "w") as f:
    json.dump(personas, f, indent=2)

print(f"Generated {len(personas)} personas -> {out_path}")
print("\nExample persona (respondent 1):")
print(json.dumps(personas[0], indent=2))
