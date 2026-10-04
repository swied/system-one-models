# System One Models

Explore typed decision models from Jupyter notebooks and compare their answers
on a shared set of labeled scenarios. This repository provides working model
quickstarts, reusable Python adapters, and a bakeoff that measures accuracy,
score quality, calibration, and latency.

The questions cover tasks such as routing support tickets, identifying phishing,
assessing urgency, and deciding whether an action needs human review. Each model
receives the same state and question definitions so its results can be compared.

## Models and question types

| Backend | How it runs | Credentials | Quickstart |
| --- | --- | --- | --- |
| `jev` | Hosted TypeSafe API through `typesafe-sdk` | `TYPESAFE_API_KEY` | [quickstart-jev.ipynb](notebooks/quickstart-jev.ipynb) |
| `d1` | Hosted Liquid API through `typesafe-sdk`; adapter defaults to `d1:free` | `LIQUID_API_KEY` | [quickstart-d1.ipynb](notebooks/quickstart-d1.ipynb) |
| `laya` | Local inference through `laya.Router`; initialization may download weights | `HF_TOKEN` when needed for model access | [quickstart-laya.ipynb](notebooks/quickstart-laya.ipynb) |

All three adapters accept these question types:

- **Choice:** select a named option, such as billing, sales, or support.
- **Score:** rate a state against an ordered rubric. Returned numeric scores can
  fall between levels; rubric positions start at zero.
- **Noul:** estimate the probability of a yes answer or true statement.

Jev and D1 require network access for inference. Laya runs locally after its
weights are available; hardware and memory requirements depend on the router
configuration. Mock mode needs no API keys, weights, or GPU.

## Install and launch with uv

The project targets Python 3.13 and requires Python >=3.13. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run these
commands from the repository root:

```bash
uv sync --locked
uv run python -m ipykernel install --user --name system-one-models --display-name "Python (system-one-models)"
uv run jupyter lab notebooks/
```

`uv sync --locked` creates `.venv`, installs the locked runtime and development
dependencies, and installs this package in editable mode. JupyterLab, ipykernel,
and python-dotenv are included in the development dependencies. You do not need
to activate the environment or install the package separately.

Open [model-bakeoff.ipynb](notebooks/model-bakeoff.ipynb) and select **Python
(system-one-models)** as the kernel. For an offline first run, set
`MOCK = True` in the configuration cell and run all cells in order. This exercises
the complete comparison and report workflow with deterministic simulated answers.
Mock metrics do not measure real models. Check the saved setting before running:
the notebook currently has `MOCK = False`.

For VS Code, install its Python and Jupyter extensions and select
`.venv/bin/python` (Windows: `.venv\Scripts\python.exe`) in the notebook's
**Select Kernel → Python Environments** menu.

Kernel registration records this environment's absolute Python path. Register
it again after moving the repository or recreating `.venv`. To remove it:

```bash
uv run jupyter kernelspec uninstall system-one-models
```

## Save and load credentials

Obtain a TypeSafe API key for Jev, a Liquid API key for D1, and a Hugging Face
access token if your Laya configuration needs authenticated model downloads.
The notebooks read credentials from these files outside the repository:

| File | Contents |
| --- | --- |
| `~/.config/typesafe/secrets.env` | `TYPESAFE_API_KEY=your_typesafe_key` |
| `~/.config/liquid/secrets.env` | `LIQUID_API_KEY=your_liquid_key` |
| `~/.config/huggingface/secrets.env` | `HF_TOKEN=your_huggingface_token` |

On Linux or macOS, create the directories and files, then edit each file to add
its corresponding line from the table:

```bash
mkdir -p ~/.config/typesafe ~/.config/liquid ~/.config/huggingface
touch ~/.config/typesafe/secrets.env ~/.config/liquid/secrets.env ~/.config/huggingface/secrets.env
chmod 700 ~/.config/typesafe ~/.config/liquid ~/.config/huggingface
chmod 600 ~/.config/typesafe/secrets.env ~/.config/liquid/secrets.env ~/.config/huggingface/secrets.env
```

Use your actual keys in those files. Keep them out of notebooks, source code,
commits, and printed outputs. The commands above create empty files without
replacing existing contents.

The bakeoff loads all three files with `load_dotenv` when `MOCK = False`. Each
quickstart loads its own service's file. Existing environment variables take
precedence over file values. After changing a saved key, restart the kernel and
rerun the loading cells so an older environment value does not remain in use.

You can also supply these environment variables before starting Jupyter. When
using the Python adapters independently, load the relevant file yourself:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.home() / ".config" / "liquid" / "secrets.env")
```

Adapters read environment variables when they are created; importing the module
does not load credential files. Hosted adapters also accept an explicit
`api_key`, and Laya accepts an explicit `token`. The standalone Laya quickstart
currently requires `HF_TOKEN` to be set; the benchmark adapter permits it to be
absent when authentication is unnecessary.

## Run the bakeoff

In `model-bakeoff.ipynb`, choose the models and switch to real inference:

```python
MOCK = False
MODEL_NAMES = ["jev", "d1", "laya"]
```

Choose a subset such as `["d1"]` to run only that model. Restart the kernel and
run all cells so the adapters are recreated with the new configuration.
Hosted calls use your service account; Laya initialization may download weights.

The notebook loads [prompts.json](src/system_one_models/prompts.json), runs every
selected backend on every scenario, displays a Markdown report, and saves three
files to `OUTPUT_DIR` (default: `/tmp/system-one-models-results`):

- `results.md`: comparison table and calibration bins.
- `results.csv`: aggregate metrics for analysis in a spreadsheet.
- `results.json`: metrics and individual question records, including errors.

The reported metrics have these meanings:

| Metric | Interpretation |
| --- | --- |
| Choice / noul accuracy | Fraction of valid answers matching the labels; noul uses a 0.5 yes/no threshold. |
| Score MAE | Mean absolute difference between the numeric score and labeled rubric index; lower is better. |
| Score rho | Spearman rank correlation between scores and labeled indexes. |
| ECE | Difference between predicted probabilities and observed accuracy across bins; lower is better. |
| Brier | Mean squared error of predicted-class probability versus correctness; lower is better. |
| p50 / p95 latency | Request latency summarized over valid question records; each question in a scenario shares that request's latency. |
| Estimated cost | Input-token cost per 1,000 scenario requests, available only when pricing and token usage are supplied. |

Score calibration uses the most probable rubric level, while score MAE uses the
expected numeric score. The Brier calculation measures predicted-class
correctness rather than a full multiclass probability distribution. Failed
answers are excluded from quality metrics and counted as errors; inspect the
error counts and records alongside accuracy. Mock latency is simulated.

Pricing is not hard-coded. Pass `input_prices_per_million` to `run_benchmark`
with your applicable input-token prices to enable estimates. Missing costs
appear as `--`; estimates exclude output-token charges and local hardware costs.
These scenarios are a small exploratory benchmark, so conclusions apply to the
questions and labels you choose.

## Use a model from another notebook

Reusable code lives in
[decision_benchmark.py](src/system_one_models/decision_benchmark.py). The editable
installation lets notebooks import it without changing `sys.path`.

After loading the appropriate credentials, call a model directly:

```python
from system_one_models.decision_benchmark import create_backend, load_prompts

backend = create_backend("d1")  # Also accepts "jev" and "laya".
try:
    scenario = load_prompts()[0]
    answers = backend.predict(scenario["state"], scenario["questions"])
    print(answers)
finally:
    if hasattr(backend, "close"):
        backend.close()
```

`predict` returns answer dictionaries keyed by question name. Jev and D1 accept
`model` and `base_url` overrides; Laya accepts `preload` and additional Router
options. Close hosted clients when finished to release their HTTP connections.

You can also run a complete offline comparison from Python:

```python
from system_one_models.decision_benchmark import create_backend, run_benchmark

backends = [create_backend(name, mock=True) for name in ("jev", "d1", "laya")]
result = run_benchmark(backends)
print(result.markdown())
result.metrics  # Aggregate metrics by backend.
result.records  # Individual answers and errors.
result.write("/tmp/system-one-models-results")
```

To use your own scenarios, call `load_prompts("path/to/prompts.json")` and pass
its result as the second argument to `run_benchmark`. Each scenario contains
`id`, `state`, `questions`, and `labels`. Question and label names must match;
choice labels name an option, noul labels are `"yes"` or `"no"`, and score labels
match a rubric description in the ordered `criteria` list.

## Repository layout

```text
src/system_one_models/
    decision_benchmark.py       Model adapters, scoring, and report exports
    prompts.json                Shared labeled scenarios
    __init__.py                 Package entry point
    py.typed                    Typed-package marker
notebooks/
    model-bakeoff.ipynb          Model comparison and report workflow
    quickstart-jev.ipynb         Hosted Jev example
    quickstart-d1.ipynb        Hosted Liquid D1 example
    quickstart-laya.ipynb        Local Laya example
    gpu-test.ipynb              Hardware exploration
tests/                         Offline unittest regression checks
docs/model-bakeoff-learning-plan.md           Exploration plan
pyproject.toml                  Dependencies and package metadata
uv.lock                         Locked dependency versions
.python-version                 Development Python version
```

## Development and validation

Use uv project commands to keep dependency metadata, the lockfile, and the local
installation coordinated:

```bash
uv add <package>                  # Add a runtime dependency.
uv add --dev <package>            # Add a development/notebook dependency.
uv remove <package>              # Remove a runtime dependency.
uv remove --dev <package>        # Remove a development dependency.
uv lock --upgrade-package <package>  # Intentionally upgrade one dependency.
uv sync --locked
uv run python -c "import system_one_models; print(system_one_models.hello())"
uv run python -m unittest discover -s tests
uv build
```

The tests check SDK adapters, Laya response handling, scoring, failure records,
and reproducible report generation without service calls or weight downloads.
For notebook validation, restart the kernel and run all affected cells in order;
use mock mode for an offline bakeoff check. Live calls and local weight loading
need separate validation with credentials and suitable hardware.

Restart the kernel after dependency changes. For source edits, restart it or
use IPython autoreload:

```python
%load_ext autoreload
%autoreload 2
```

`.python-version` selects the development interpreter; `requires-python` declares
the supported range. The lockfile pins dependencies, not the interpreter patch
release. Use `uv python pin <version>` to change the selected Python version,
then run `uv sync` and register the kernel again. If changing the supported range,
update `pyproject.toml` and run `uv lock` followed by `uv sync`.

Commit dependency metadata and lockfile changes together. Keep `.venv`, build
output, notebook checkpoints, secrets, and generated reports out of commits
unless a generated report is deliberately included as a reviewed artifact.
Clear sensitive notebook outputs before committing.

See uv's [project guide](https://docs.astral.sh/uv/guides/projects/) and
[Jupyter guide](https://docs.astral.sh/uv/guides/integration/jupyter/) for more
background on the environment workflow.
