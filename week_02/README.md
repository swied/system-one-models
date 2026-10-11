# Week 2: Jev, Liquid D1, and Laya bakeoff

Compare typed model answers on shared labeled scenarios. Follow the
[repository README](../README.md) for installation, credentials, metric
interpretation, and the shared **Python (system-one-models)** kernel.

- [Model bakeoff notebook](notebooks/model-bakeoff.ipynb)
- [Jev quickstart](notebooks/quickstart-jev.ipynb)
- [Liquid D1 quickstart](notebooks/quickstart-d1.ipynb)
- [Laya quickstart](notebooks/quickstart-laya.ipynb)
- [GPU exploration](notebooks/gpu-test.ipynb)
- [Learning plan](docs/model-bakeoff-learning-plan.md)
- [Saved real-model results and product interpretation](docs/model-bakeoff-results.md)
- [Shared scenarios](../src/system_one_models/prompts.json)

From the repository root:

```bash
uv sync --locked
uv run jupyter lab week_02/notebooks/
```

The editable package provides `system_one_models` imports from the nested
notebook directory. `load_prompts()` locates the shared package data without
depending on the kernel's working directory. No `sys.path` changes are needed.

For offline validation, set `MOCK = True` in the bakeoff configuration, restart
the kernel, and run every cell in order. The saved notebook uses `MOCK = False`
and contains real-model outputs; keep those measurements distinct from mock
validation. Reports default to `/tmp/system-one-models-results`. Live inference
requires service credentials or local Laya weights and appropriate hardware.
