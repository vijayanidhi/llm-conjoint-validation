# Synthetic Choice Panel Toolkit

**Research question:** Does persona conditioning close the gap between LLM-generated and real human preferences in choice-based conjoint (CBC) analysis, or does substantial, systematic divergence persist even when synthetic respondents are given demographic backstories?

**Status:** Core investigation complete. Real-data baseline, no-persona LLM arm, and persona-conditioned LLM arm have all been fit and compared. Headline finding below. Outstanding work (mixed logit robustness check, prompt-variation sensitivity, formal writeup) is listed at the end as future work.

**Advisor:** Prof. Joseph Pancras (Associate Professor of Marketing, UConn)

---

## Summary of findings

Persona conditioning does **not** close the gap between synthetic and real respondents — it makes one dimension of the gap worse, and introduces a new statistical problem that did not exist in the no-persona arm.

| Metric | Real respondents | LLM (no persona) | LLM (persona-conditioned) |
|---|---|---|---|
| "None of these" (outside good) selection rate | 25.3% | 0.2% | ~0.0% |
| Brand coefficients | Well-identified, meaningfully differentiated | Inflated but estimable | Quasi-complete separation (uninterpretable) |
| Price / feature coefficients | Well-identified | Well-identified | Well-identified |
| Pooling with real data (LR test) | — | Rejected, p < 2.2e-16 | Rejected |

In short: giving synthetic respondents demographic backstories pushed them to commit even more strongly to *some* purchase, further suppressing realistic abstention behavior, and caused the brand-level coefficients to blow up (huge estimates, huge standard errors, p > 0.8) — while price and feature sensitivity remained stable and interpretable in both LLM arms. Persona conditioning amplified an existing bias rather than correcting it, and introduced a new identification failure on top of it.

---

## Dataset

**bayesm "camera" dataset** — 332 respondents, 16 choice tasks each, 5 alternatives per task (4 cameras + a "none/outside" option), 10 attributes: canon, sony, nikon, panasonic, pixels, zoom, video, swivel, wifi, price (in hundreds of USD).

Source: Allenby, Brazell, Howell & Rossi (2014), *Quantitative Marketing and Economics*, distributed via the `bayesm` R package on CRAN. Chosen for its real (not simulated) survey data and no known prior use in LLM-synthetic-respondent validation research. Approved by Prof. Pancras.

*Secondary/backup datasets considered:* UCL IoT DCE datasets (SmartTV, Wearables, Thermostats, Security Camera — real, PLOS ONE 2020) as secondary validation; `bayesm` bank dataset or Expedia hotel search as scale/robustness backups. The Chapman & Feit Sports Car dataset was excluded (simulated data, already used in a competing forthcoming Marketing Science paper).

---

## Setup

Requires R with the `bayesm` package (and compiled dependencies Rcpp, RcppArmadillo, system BLAS/LAPACK), and Python 3 with the `anthropic` package for the LLM pipeline.

\`\`\`bash
# R dependencies (Debian/Ubuntu)
sudo apt-get install -y r-base-core r-cran-rcpp r-cran-rcpparmadillo \
    liblapack-dev libblas-dev build-essential gfortran

# R package (build from CRAN GitHub mirror if CRAN is unreachable)
install.packages("bayesm")

# Python dependency for the LLM pipeline
pip install anthropic --break-system-packages
\`\`\`

The Python LLM scripts require `ANTHROPIC_API_KEY` set in your environment. Never commit the key — it is read from the environment only.

---

## Phase 1: Real-data baseline

Establishes ground-truth part-worths from real human choices before any synthetic respondent work begins.

**Scripts (run in order):**

| Script | Purpose |
|---|---|
| `R/01_load_camera.R` | Loads `bayesm::camera`, saves to `data/camera.rds` |
| `R/02_fit_pooled_mnl.R` | Fits a single pooled (homogeneous) MNL via `optim` (BFGS) |
| `R/03_fit_latent_class_mnl.R` | Fits finite-mixture latent-class MNL via EM for K = 2, 3, 4, vectorized, with multiple random restarts per K for stability |
| `R/03b_fit_k4_only.R` | Standalone K=4 fit (slower — run separately/in background) |

**Results (real data):**

| Model | LogLik | BIC |
|---|---|---|
| Pooled MNL | −6503.75 | 13065.5 |
| 2-class latent-class MNL | −5809.36 | 11740.6 |
| 3-class latent-class MNL | −5420.09 | 11026.0 |
| 4-class latent-class MNL (provisional) | −5269.41 | 10788.4 |

*Note on K=4:* this run hit the 60-iteration EM cap without confirmed convergence — treat as provisional. BIC is still improving with diminishing returns per added class, so K=3–4 is the practical elbow. A fully converged K=4/K=5 run is listed under Future Work.

**Pooled MNL part-worths** (all significant except the panasonic brand dummy): canon 0.465, sony 0.238, nikon 0.312, panasonic 0.023, pixels 0.758, zoom 0.819, video 0.628, swivel 0.367, wifi 0.578, price −1.486.

**3-class solution (illustrative):** one price-tolerant/brand-indifferent feature-driven segment, one strongly brand-loyal segment (brand coefficients ~4.5–4.8), and one price-sensitive/brand-averse segment. Both restarts converged to the same solution at each K, indicating stable fits.

**Output files:** `output/pooled_mnl_baseline.rds/.csv`, `output/latent_class_mnl.rds`, `output/latent_class_mnl_k4.rds`

---

## Phase 2: No-persona LLM synthetic respondent pipeline

**Design:** One Claude API call per synthetic respondent, containing all 16 choice tasks at once (not 16 separate calls). This keeps a synthetic respondent's answers internally consistent, the way one real person's stable preferences would be, and cuts API calls from 5,312 down to 332. Each synthetic respondent answers the *same* 16 task designs a specific real respondent saw, so real-vs-synthetic comparisons are apples-to-apples. No demographic/persona backstory is given in this arm — that conditioning is introduced separately in Phase 5.

**Scripts (run in order):**

| Script | Purpose |
|---|---|
| `R/04_export_tasks_for_llm.R` | Converts each real respondent's choice-task design into human-readable JSON |
| `python/05_llm_pipeline.py` | Calls the Claude API once per synthetic respondent, forces a strict JSON array of 16 choices back. Run: `python 05_llm_pipeline.py --n_respondents 332 --start 1` |
| `R/06_assemble_synthetic_camera.R` | Reassembles LLM JSON output into the same `{y, X}` structure `bayesm`'s camera data uses, producing `data/synthetic_camera.rds` — after which `02_fit_pooled_mnl.R` and `03_fit_latent_class_mnl.R` run unchanged against it |

**Pipeline execution notes:** Two issues were fixed during the full run: (1) Claude Sonnet used extended thinking by default, consuming most of the token budget before producing visible output — fixed by setting `thinking={"type": "disabled"}` and `max_tokens=1024`, which also cut per-respondent API cost by more than half; (2) four respondents (108, 110, 114, 137) initially failed with malformed/empty responses and were retried individually. **Final: all 332 synthetic respondents completed.**

**Key findings (full 332-respondent run):**

- **Outside-good gap confirmed at scale.** Real respondents chose "none of these" 25.3% of the time; synthetic respondents chose it 0.2% of the time. Consistent with the initial 10-respondent pilot (0%) — a robust, systematic finding, not sampling noise. This was deliberately left as-is rather than prompt-engineered away, since tuning the prompt to match known ground truth would make the validation circular.
- **Synthetic respondents barely differentiate between named brands.** Real brand part-worths show a clear ordering (canon 0.47 > nikon 0.31 > sony 0.24 > panasonic 0.02, relative to the omitted brand). Synthetic brand part-worths are all large and nearly equal (canon 4.89, sony 4.59, nikon 4.28, panasonic 4.56) — the LLM strongly rewards "having any named brand" over the omitted baseline, but does not capture the real, meaningful preference differences among named brands.
- **Low agreement on relative attribute importance.** Spearman rank correlation between real and synthetic part-worths across all 10 attributes: **−0.103** (essentially no agreement, slightly negative).
- **Scale difference (expected, less concerning on its own).** All synthetic coefficients are 2–10x larger in raw magnitude than real ones — consistent with synthetic respondents' choices being less noisy/more internally consistent, which mechanically inflates all coefficients together (MNL's coefficient scale isn't separately identified from response noise). Normalizing every coefficient relative to price does not resolve the brand-differentiation or rank-correlation findings above, so those are not simply scale artifacts.

**Formal pooling test (likelihood-ratio test):** H0: real and synthetic respondents share one common preference structure, vs. H1: separate part-worths.

- LL (real-only): −6503.75
- LL (synthetic-only): −3260.65
- LL (pooled, shared β): −11177.52
- **LR statistic: 2826.25 (df=10), p < 2.2e-16 — H0 decisively rejected.**

This is the core quantitative answer to the baseline research question: under this pipeline and prompt design, real and synthetic respondents have statistically significantly different preference structures.

*Caveat:* this test does not separate genuine taste differences from scale/noise differences between populations — lower response noise in the synthetic arm would alone cause rejection even under identical underlying tastes. A Swait–Louviere-style relative-scale test is the natural refinement, listed under Future Work.

**Output:** `output/real_vs_synthetic_comparison.csv`

---

## Phase 3: Persona-conditioned LLM arm

Following a literature review indicating the broad "can LLMs approximate real CBC preferences" question had already been partially addressed in recent (2024–2026) work, the research question was refined — with Prof. Pancras's approval — to the persona-conditioning question stated at the top of this README.

**Design:** Each of the 332 synthetic respondents is run a second time with a randomized persona (age, income bracket, photography experience, tech-savviness, budget mindset) prepended to the system prompt. Traits are sampled independently at random (seed = 42, for reproducibility), since the real `bayesm` camera dataset contains no respondent demographics to match against.

**Scripts:**

| Script | Purpose |
|---|---|
| `python/10_generate_personas.py` | Generates `data/personas.json` — 332 randomized personas, reproducible via fixed seed |
| `python/05b_llm_pipeline_persona.py` | Persona-conditioned version of the main pipeline; same task structure and output format as `05_llm_pipeline.py`, with `thinking` disabled and `max_tokens=1024` set from the start based on lessons from the no-persona run |

**Result: full 332-respondent persona run completed cleanly, zero failures.**

**Key findings:**

- **Outside-good avoidance got worse, not better.** Persona-conditioned synthetic respondents chose "none of these" ~0.0% of the time — even lower than the already-low 0.2% in the no-persona arm, and far from the real 25.3%. Adding a demographic backstory pushed synthetic respondents toward *more* commitment to purchase, not more realistic abstention.
- **Quasi-complete separation in brand coefficients.** Brand-level MNL estimates in the persona arm are extreme (point estimates ~14.5, standard errors ~68.6, p > 0.8) — a classic quasi-complete separation signature, meaning brand choice became so deterministic under certain personas that the model cannot estimate stable, interpretable brand effects at all.
- **Price and feature coefficients remained well-identified.** Despite the brand-level breakdown, price sensitivity and feature part-worths (pixels, zoom, video, swivel, wifi) stayed stable and interpretable in the persona arm — the divergence from real respondents is concentrated in brand-related preferences and outside-good behavior, not across the board.

**Takeaway:** persona conditioning is not a simple fix for the real-vs-synthetic gap. It can amplify an existing systematic bias (outside-good avoidance) rather than correct it, and it can introduce entirely new estimation problems (separation) in parts of the model that were previously well-behaved — while leaving other parts of the model (price, features) unaffected. This asymmetry — some coefficients robust, others broken, under the exact same conditioning treatment — is itself informative about *which* aspects of consumer choice LLMs can vs. cannot approximate with persona-based prompting.

---

## Repository structure

\`\`\`
├── R/
│   ├── 01_load_camera.R
│   ├── 02_fit_pooled_mnl.R
│   ├── 03_fit_latent_class_mnl.R
│   ├── 03b_fit_k4_only.R
│   ├── 04_export_tasks_for_llm.R
│   └── 06_assemble_synthetic_camera.R
├── python/
│   ├── 05_llm_pipeline.py
│   ├── 05b_llm_pipeline_persona.py
│   └── 10_generate_personas.py
├── data/
│   ├── camera.rds
│   ├── synthetic_camera.rds
│   ├── llm_tasks/
│   ├── llm_responses/
│   ├── llm_responses_persona/
│   └── personas.json
└── output/
    ├── pooled_mnl_baseline.rds / .csv
    ├── latent_class_mnl.rds
    ├── latent_class_mnl_k4.rds
    └── real_vs_synthetic_comparison.csv
\`\`\`

---

## Future work

The core research question has been answered with the real-data baseline and both LLM arms above. The following would extend the project's rigor but are not required to interpret the headline finding:

- **Mixed logit** on the real data as an alternative (continuous) heterogeneity model, to cross-check the latent-class segmentation story.
- **Converged K=4/K=5 latent-class re-run** — the current K=4 result is provisional (hit the EM iteration cap).
- **Swait–Louviere-style relative-scale test** to separate genuine taste divergence from response-noise/scale differences in the real-vs-synthetic pooling test.
- **Prompt-variation robustness check** — e.g., a prompt variant that explicitly reminds the model that real shoppers often decline to buy, run as a labeled robustness arm (not folded into the baseline, to avoid circularity).
- **Sample-size sensitivity check** on the LLM arms.
- **Formal bias/limitations writeup** synthesizing all of the above into a methods-paper-style discussion section.
- **Packaging** the pipeline as a reusable, documented toolkit for validating LLM synthetic respondents against other CBC datasets.

---

## References

Allenby, G. M., Brazell, J. D., Howell, J. R., & Rossi, P. E. (2014). Economic valuation of product features. *Quantitative Marketing and Economics*, 12(4), 421-456.
