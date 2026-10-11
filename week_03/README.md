# Week 3: System One exploration

This folder is ready for the next week's exercises. Add the goals, notebooks,
scripts, data descriptions, and findings here as the work develops.

Use the shared environment and kernel described in the
[repository README](../README.md). From the repository root:

```bash
uv sync --locked
uv run jupyter lab .
```

Select **Python (system-one-models)** for new notebooks. Import reusable code
from `system_one_models`; shared scenarios are available through
`system_one_models.decision_benchmark.load_prompts()`.

Keep week-specific files here, reusable Python code in `src/system_one_models/`,
and offline regression tests in `tests/`. Resolve notebook data paths explicitly
against the intended week folder. For scripts launched from the repository root,
use `uv run python week_03/<script>.py` and week-prefixed file arguments.
