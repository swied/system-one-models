# Repository Guidelines

## Purpose & Project Structure

This repository explores typed decision models and compares Jev, Liquid D1,
and local Laya on shared labeled scenarios. Keep documentation focused on the
models, notebook workflows, and interpretation of benchmark results.

Reusable Python code lives in `src/system_one_models/`; import it as
`system_one_models`. The distribution name is `system-one-models`.
`decision_benchmark.py` provides `create_backend`, `load_prompts`,
`run_benchmark`, and `BenchmarkResult`, plus model adapters and scoring helpers.
`prompts.json` holds the benchmark scenarios. Preserve `py.typed`.

`notebooks/model-bakeoff.ipynb` runs the comparison and exports reports. The
quickstarts are `quickstart-jev.ipynb`, `quickstart-d1.ipynb`, and
`quickstart-laya.ipynb`. `gpu-test.ipynb` contains hardware exploration.
Planning documentation lives in `docs/model-bakeoff-learning-plan.md`. The product-facing
interpretation of the saved real-model bakeoff lives in
`docs/model-bakeoff-results.md`. Offline regression tests live in `tests/`.
Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync --locked`: install the locked environment and editable package.
- `uv run python -m ipykernel install --user --name system-one-models --display-name "Python (system-one-models)"`:
  register the notebook kernel; repeat after moving or recreating the environment.
- `uv run jupyter lab notebooks/`: start the notebook interface.
- `uv run python -c "import system_one_models; print(system_one_models.hello())"`:
  check the package import.
- `uv run python -m unittest discover -s tests`: run offline regression checks.
- `uv build`: create wheel and source distributions in `dist/`.

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>`
for development dependencies; commit both metadata and lockfile changes.
Avoid persistent dependency changes through notebook `%pip` or ad hoc pip
commands. Python development targets 3.13; package metadata requires >=3.13.

## Model & Benchmark Conventions

Supported backend names are `jev`, `d1`, and `laya`. Hosted adapters use the
TypeSafe SDK and typed `Choice`, `Score`, and `Noul` questions. Laya uses
`Router.predict` and unwraps its `answers` response. Follow the quickstarts and
installed dependency interfaces when changing model calls.

Module imports must not initialize clients, load secrets, or download weights.
Initialize adapters explicitly with `create_backend`. Close hosted clients when
finished. Keep reusable logic in the package and orchestration in notebooks;
do not restore the old `bake-off.py` command-line driver.

Scenarios contain `id`, `state`, `questions`, and `labels`. Choice labels name
options; noul labels are `yes` or `no`; score labels match ordered criteria.
Score positions start at zero. Preserve fractional expected scores for MAE and
rank correlation; score calibration uses the most probable level. Brier measures
predicted-class probability versus correctness. Request latency is repeated on
question records. Exclude failed answers from quality metrics and report their
error counts and details. Do not silently present unknown pricing as zero.
Cost estimates use caller-supplied input-token pricing per million tokens and
represent 1,000 scenario requests, excluding output and hardware costs.

Use `MOCK = True` for offline bakeoff validation; the notebook currently saves
`MOCK = False`, so inspect its configuration before executing it. Mock mode
requires no credentials, GPU, or weights and uses deterministic simulated
answers and latency. Mock results
validate the workflow and must not be described as real model measurements.
Reports default to `/tmp/system-one-models-results` and include `results.md`,
`results.csv`, and `results.json`. `run_benchmark` returns data without writing
files; `BenchmarkResult.write` performs exports.

## Results Documentation

Write results reports in plain English for product managers. Explain what the
numbers mean for customer experience, mistakes, human review, response time,
and operating costs. Define technical terms when needed, and connect model
comparisons to specific product decisions.

Ground measurements in the saved outputs of `notebooks/model-bakeoff.ipynb`
and question counts in `prompts.json`. Check whether the run used real models
or mocks before interpreting it. When updating `docs/model-bakeoff-results.md`,
recheck its numbers and conclusions against the run it describes; do not treat
the current ranking as a permanent property of the models.

The currently documented real run contains 18 scenarios and 33 questions:
13 choice, 14 yes/no, and six ratings. D1 led on choice and yes/no accuracy;
Jev had slightly lower rating error; Laya had the weakest answer quality but
the shortest typical response time and was the only locally run model.
Discuss Laya fine-tuning as an experiment that could improve results, not a
guaranteed outcome. Keep improving answer quality distinct from improving
the reliability of stated probabilities.

Support external factual claims with direct internet references, preferring
official documentation, model-maintainer sources, and original research.
Distinguish our measurements, vendor-reported results on other datasets, and
proposed product implications. Internet references do not independently verify
our notebook results. Validate local Markdown links before finishing doc edits.

Explain material limits: the small sample, related questions within scenarios,
subjective labels, hosted network time versus local calls, initialization outside
the timed benchmark, and missing model-version or hardware details. Report
confidence warnings without assuming which answers they affected. Unknown
costs remain unknown; local execution does not imply free operation or guarantee
privacy. Avoid claiming production readiness or measured tuning improvements
from this exploratory run.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, `snake_case` functions
and variables, `PascalCase` classes, and uppercase constants. Use descriptive
module names and type annotations for new public interfaces. No formatter or
linter is currently configured.

## Testing Guidelines

Tests use stdlib unittest; no coverage threshold is configured. Place regression
tests in `tests/test_*.py` and keep them offline using SDK responses and stub
routers. Run the import check and regression suite for relevant source changes.
For changes affecting notebook execution or benchmark reports, restart the kernel
and run the bakeoff cells in order with `MOCK = True`. The user also confirmed
that all bakeoff cells ran successfully in mock mode during initial validation.

Validate live inference separately when relevant credentials and hardware are
available. Record which backends were exercised, whether weights were loaded,
and any hardware requirements. Do not imply an offline test validated live API
behavior or model loading. Documentation-only edits need checks against current
code and notebook settings, rather than new implementation tests.

## Credentials & Generated Files

Read hosted credentials from `TYPESAFE_API_KEY` and `LIQUID_API_KEY`; Laya uses
`HF_TOKEN` when authenticated model access is needed. The notebooks use
python-dotenv to load these files outside the repository:

- `~/.config/typesafe/secrets.env`: `TYPESAFE_API_KEY`.
- `~/.config/liquid/secrets.env`: `LIQUID_API_KEY`.
- `~/.config/huggingface/secrets.env`: `HF_TOKEN`.

The bakeoff loads them only in real mode; adapters themselves read the environment
or explicit arguments and do not load files. Existing environment variables take
precedence over dotenv files. Restart the kernel after changing saved keys.
The Laya quickstart currently requires `HF_TOKEN`; the adapter permits it to be
absent when authentication is unnecessary.

Never commit secrets or sensitive notebook outputs, and never print credentials.
Keep `.venv/`, build output, notebook checkpoints, and generated reports out of
commits unless explicitly needed as reviewed artifacts. Use placeholder values
in credential documentation; do not inspect actual secret files to validate docs.

## Commit & Pull Request Guidelines

Use concise, imperative subjects such as `Add D1 benchmark scenarios`. Describe
the change, validation performed, and required API, model, or GPU setup in pull
requests; link relevant issues when available. Explain scoring changes and their
impact on comparisons. Keep README setup instructions aligned with the notebook
and module behavior.
