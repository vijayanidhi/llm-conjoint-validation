# Synthetic Choice Panel Toolkit — Real-Data Baseline

This is the "ground truth" baseline stage of the project: pulling the real
human CBC dataset and fitting the models that later synthetic (LLM-generated)
respondent data will be validated against.

## Dataset

**bayesm "camera" dataset** — 332 respondents, 16 choice tasks each, 5
alternatives per task (4 cameras + a "none/outside" option), 10 attributes:
canon, sony, nikon, panasonic, pixels, zoom, video, swivel, wifi, price
(price is in hundreds of USD). Source: Allenby, Brazell, Howell & Rossi
(2014), distributed via the `bayesm` R package on CRAN.

## Setup

Requires R with the `bayesm` package (and its compiled dependencies
`Rcpp`, `RcppArmadillo`, plus system BLAS/LAPACK). On Debian/Ubuntu:

```bash
sudo apt-get install -y r-base-core r-cran-rcpp r-cran-rcpparmadillo \
    liblapack-dev libblas-dev build-essential gfortran
```

Then install bayesm itself (if not available via CRAN mirror, build from
the CRAN GitHub mirror `github.com/cran/bayesm`):

```r
install.packages("bayesm")   # or build from source if CRAN is unreachable
```

## Scripts (run in order)

- `R/01_load_camera.R` — loads `bayesm::camera`, prints dataset shape,
  saves to `data/camera.rds` for reuse by later scripts.
- `R/02_fit_pooled_mnl.R` — fits a single pooled (homogeneous) conditional
  logit / MNL via `optim` (BFGS) as the simplest baseline. Outputs
  coefficients, SEs, z-stats, p-values, log-likelihood, and BIC.
- `R/03_fit_latent_class_mnl.R` — fits finite-mixture latent-class MNL
  (Kamakura & Russell 1989 style) via EM for K = 2, 3, (4). Vectorized
  for speed (E-step and M-step gradient computed via matrix ops, not
  per-task R loops). Multiple random restarts per K to check stability.
  Reports class-specific part-worths, class shares, log-likelihood, BIC.
- `R/03b_fit_k4_only.R` — standalone K=4 fit (split out because it's
  slower — run separately / in background if needed).

## Results so far (real data, `bayesm` camera dataset)

| Model | LogLik | BIC |
|---|---|---|
| Pooled MNL | −6503.75 | 13065.5 |
| 2-class latent-class MNL | −5809.36 | 11740.6 |
| 3-class latent-class MNL | −5420.09 | 11026.0 |
| 4-class latent-class MNL | −5269.41 | 10788.4 |

**Caveat on K=4:** this run hit the 60-iteration EM cap without the
log-likelihood change dropping below the convergence tolerance — treat it as
provisional, not converged. Re-run with a higher `max_iter` before treating
it as final. BIC is still improving at each step (13065 → 11741 → 11026 →
10788) but with diminishing returns per class added, so the true "elbow" in
BIC vs. K is somewhere around here — worth a converged K=4 (and maybe K=5)
run to pin down definitively before picking a final segment count.

Pooled MNL part-worths (all significant except panasonic brand dummy):
canon 0.465, sony 0.238, nikon 0.312, panasonic 0.023, pixels 0.758,
zoom 0.819, video 0.628, swivel 0.367, wifi 0.578, price −1.486.

3-class solution (illustrative): one price-tolerant/brand-indifferent
feature-driven segment, one strongly brand-loyal segment (brand
coefficients ~4.5–4.8), and one price-sensitive/brand-averse segment.

## Output files

- `output/pooled_mnl_baseline.rds` / `.csv` — pooled MNL fit + coefficient table
- `output/latent_class_mnl.rds` — K=2,3 latent-class fits
- `output/latent_class_mnl_k4.rds` — K=4 fit (once background job completes)

## Phase 2: LLM synthetic respondent pipeline

**Design:** one Claude API call per synthetic respondent, containing all 16
choice tasks at once (not 16 separate calls). This keeps a synthetic
respondent's answers internally consistent, the way one real person's stable
preferences would be, and cuts API calls from 5,312 down to ~332. Synthetic
respondents are given no persona/demographic backstory in this baseline
version -- that's a natural robustness/follow-up variant to test later.
Each synthetic respondent answers the SAME 16 task designs (same
brand/feature/price combinations) that a specific real respondent saw, so
comparisons are apples-to-apples.

**Scripts (run in order):**

- `R/04_export_tasks_for_llm.R` — converts each real respondent's choice-task
  design (X matrix) into human-readable JSON (e.g. "Nikon camera, $79, with:
  higher zoom, swivel display"). Output: `data/llm_tasks/respondent_NNN_tasks.json`
  (one file per real respondent, 332 total).
- `python/05_llm_pipeline.py` — calls the Claude API once per synthetic
  respondent with all 16 tasks, forces a strict JSON array of 16 choices
  (1-4 = a camera, 5 = no purchase) back out. Saves to
  `data/llm_responses/synthetic_respondent_NNN.json`. Requires
  `ANTHROPIC_API_KEY` in your environment and `pip install anthropic`.
  Run with: `python 05_llm_pipeline.py --n_respondents 332 --start 1`
- `R/06_assemble_synthetic_camera.R` — reassembles the LLM JSON output back
  into the exact same `{y, X}` list-of-lists structure bayesm's `camera`
  data uses. Output: `data/synthetic_camera.rds`. Once this exists, you can
  run `02_fit_pooled_mnl.R` and `03_fit_latent_class_mnl.R` UNCHANGED against
  it (just point them at `synthetic_camera.rds` instead of `camera.rds`) to
  get synthetic part-worths directly comparable to the real ones already
  fitted.

**Not yet done / next steps:**
- Actually run `05_llm_pipeline.py` against the live API (needs your API key
  — I don't have one in this sandbox, so this hasn't been executed yet)
- Phase 4: compare real vs. synthetic part-worths (same coefficients side by
  side, plus formal tests like a likelihood-ratio test for whether pooling
  real+synthetic data is statistically justified)
- Consider a persona-based variant as a robustness check (Phase 5)
- Mixed logit still outstanding on the real-data side
- K=4 latent-class rerun to confirm convergence

## Early finding from the 10-respondent test batch

Real respondents chose "none of these" (the outside good) 25.3% of the time
across the full real dataset. The first 10 synthetic (Claude-generated)
respondents chose "none" 0% of the time (0/160 choices). This is being kept
as-is for the main baseline run rather than prompt-engineered away, since
tuning the prompt to match known ground truth would make the validation
circular. This gap will be formally reported in the real-vs-synthetic
comparison (Phase 4), and a prompt variant that explicitly reminds the model
that real shoppers often decline to buy is planned as a labeled robustness
check in Phase 5 -- not folded into the baseline.

## Phase 4: Real vs. Synthetic comparison (full 332-respondent run)

**Pipeline execution notes:** the full run required fixing two issues after
the initial attempt -- (1) Claude Sonnet 5 used extended thinking by default,
consuming most of the token budget before producing visible output (fixed by
setting `thinking={"type": "disabled"}` in the API call, which also cut
per-respondent cost by more than half); (2) a handful of respondents (108,
110, 114, 137) failed with malformed/empty responses and were retried
individually. Final dataset: all 332 synthetic respondents completed.

**Key findings:**

1. **Outside-good gap confirmed at full scale.** Real respondents chose
   "none of these" 25.3% of the time; synthetic respondents chose it 0.2% of
   the time (compare to 0% in the initial 10-respondent pilot). This is a
   robust, systematic finding, not sampling noise.

2. **Synthetic respondents barely differentiate between named brands.** Real
   brand part-worths show a clear ordering (canon 0.47 > nikon 0.31 > sony
   0.24 > panasonic 0.02, relative to the omitted 5th brand). Synthetic
   brand part-worths are all large and nearly equal (canon 4.89, sony 4.59,
   nikon 4.28, panasonic 4.56) -- suggesting the LLM strongly rewards "having
   *any* named brand" over the omitted baseline, but does not capture the
   real, meaningful preference differences *among* named brands.

3. **Low agreement on relative attribute importance.** Spearman rank
   correlation between real and synthetic part-worths across all 10
   attributes: -0.103 (essentially no agreement, slightly negative).

4. **Scale difference (less concerning, expected/mechanical).** All
   synthetic coefficients are 2-10x larger in raw magnitude than real ones.
   This is consistent with synthetic respondents' choices being less noisy
   (more internally consistent) than real humans' -- since MNL's coefficient
   scale is not separately identified from response noise, lower noise
   mechanically inflates all coefficients together. Normalizing every
   coefficient relative to price (which controls for this scale effect) does
   not resolve findings 2-3 above, so those are not simply scale artifacts.

See `output/real_vs_synthetic_comparison.csv` for the full numeric table.

## Formal pooling test (likelihood-ratio test)

Tested H0: real and synthetic respondents share one common preference
structure (pooled MNL), vs H1: they have separate part-worths.

- LL (real-only): -6503.75
- LL (synthetic-only): -3260.65
- LL (pooled, shared beta): -11177.52
- LR statistic: 2826.25 (df=10), p < 2.2e-16

**H0 is decisively rejected.** Real and synthetic respondents have
statistically significantly different preference structures under this
pipeline and prompt design. This is the core quantitative answer to the
project's research question so far.

Caveat: this test does not separate genuine taste differences from
scale/noise differences between populations (synthetic respondents are less
noisy/more internally consistent, which alone would cause rejection even
under identical tastes). A Swait-Louviere style test with an explicit
relative scale parameter is a natural refinement for the final writeup, to
isolate how much of this gap is "different preferences" vs "different
response consistency."

## Phase 5 (promoted to core): Persona-conditioned experimental arm

Following literature review, the research question was refined with
Prof. Pancras's approval to:

"Does persona conditioning close the gap between LLM-generated and real
human CBC preferences, or does a substantial, systematic divergence persist
even when synthetic respondents are given demographic backstories?"

**Design:** each of the 332 synthetic respondents is now also run a SECOND
time with a randomized persona (age, income bracket, photography experience,
tech-savviness, budget mindset) prepended to the system prompt. Traits are
sampled independently and randomly (see `python/10_generate_personas.py`),
since the real bayesm camera dataset contains no respondent demographics to
match against.

**Scripts:**
- `python/10_generate_personas.py` — generates `data/personas.json` (332
  randomized personas, reproducible via fixed random seed)
- `python/05b_llm_pipeline_persona.py` — persona-conditioned version of the
  main pipeline; identical task structure and output format to
  `05_llm_pipeline.py`, but with `thinking` already disabled and
  `max_tokens=1024` set from the start (lessons learned from the no-persona
  run's cost and truncation issues). Output: `data/llm_responses_persona/`.

**Not yet run:** this arm has not yet been executed against the live API.
Once run, a three-way comparison (real vs. no-persona-LLM vs.
persona-LLM) will be added via new R scripts extending 07/08/09.
