# Run Laya on a firewalled Vertex AI Workbench instance

Download the Hugging Face model cache on your connected work laptop, package it
as an archive, upload it to Google Cloud Storage (GCS), and extract it on your
Workbench instance. Set offline mode **before importing Laya or Hugging Face
libraries** so loading uses the transferred files without contacting Hugging Face.

These instructions cover [laya-finetuning.ipynb](laya-finetuning.ipynb) and your
`Router(preload=True, token=os.environ["HF_TOKEN"])` cell. The notebook currently
in this checkout defaults to `MOCK = True` and calls training/evaluation scripts;
it does not contain that Router cell. Instructions for both versions appear below.
The notebook is left unchanged.

## 1. Prepare the repository and Python environment

In a Workbench terminal, if GitHub access is allowed:

```bash
git clone https://github.com/swied/system-one-models.git
cd system-one-models
uv sync --locked
uv run python -m ipykernel install --user --name system-one-models --display-name "Python (system-one-models)"
```

Use the root environment and select **Python (system-one-models)** in JupyterLab.
The repository requires Python 3.13 or newer. Both machines should use the same
repository commit and locked dependencies. Run the download steps from the root
of a laptop clone after `uv sync --locked` there, too.

If GitHub is blocked on Workbench, clone on the laptop and transfer a source
archive through GCS using the same upload/download pattern as step 3:

```bash
git archive --format=tar.gz --output=../system-one-models-source.tar.gz HEAD
```

Extract it into a new `system-one-models` directory on Workbench. This contains
committed files only, without Git history, uncommitted edits, or ignored artifacts.
Existing ignored training datasets/checkpoints are not included; the notebook can
generate data and train new checkpoints.

**Model transfer does not install Python dependencies.** `uv sync --locked` needs
access to approved package sources and possibly a Python download. If those are
also blocked, provision Python and the locked environment through your company's
package mirror or a wheelhouse prepared for the Workbench Linux architecture,
Python version, and GPU runtime. A laptop `.venv` is not a portable substitute.
Resolve this prerequisite before loading models. The remaining instructions
assume `uv`, the locked environment, and the Google Cloud CLI are available.

## 2. Download the models on the connected laptop

Use a fresh, dedicated cache, outside the repository. This keeps your normal
Hugging Face login files out of the transfer. If access requires authentication,
make `HF_TOKEN` available through your approved local secret mechanism; do not
put its value in the script, notebook, archive, or bucket.

Run in a **Bash terminal** from the laptop repository root:

```bash
export LAYA_TRANSFER_DIR="$HOME/laya-offline-transfer"
mkdir -p "$LAYA_TRANSFER_DIR/hub"
uv run python - <<'PY'
import json
import os
import runpy
from pathlib import Path
from huggingface_hub import snapshot_download
from laya import Router

transfer = Path(os.environ["LAYA_TRANSFER_DIR"])
cache = transfer / "hub"
token = os.environ.get("HF_TOKEN")
manifest = []
snapshots = []

# Inspect this installed SDK's routing map without loading models.
# Laya 0.3.27 defaults to the English root plus multilingual and
# typed-decisions subfolders of convaiinnovations/laya.
router = Router(preload=False)
repos = set()
for spec in router.models.values():
    repos.add(spec[0] if isinstance(spec, (tuple, list)) else spec)

# Full snapshots retain tokenizer/config files and bundled checkpoints.
# Downloading main also records refs/main for unpinned Router loads.
for repo in sorted(repos):
    path = Path(snapshot_download(repo, cache_dir=cache, token=token))
    snapshots.append(path)
    manifest.append({"repo": repo, "revision": path.name})

# The week 3 scripts use a separate, pinned English revision.
revision = runpy.run_path("week_03/train.py")["MODEL_REVISION"]
path = Path(snapshot_download("convaiinnovations/laya", revision=revision,
                              cache_dir=cache, token=token))
snapshots.append(path)
manifest.append({"repo": "convaiinnovations/laya", "revision": path.name,
                 "purpose": "week_03 training and baseline evaluation"})

# Older checkpoints may omit a bundled encoder config or tokenizer.
# Cache referenced encoder repositories as well in those cases.
extra_repos = set()
for snapshot in snapshots:
    for config_file in snapshot.rglob("rl_agent_config.json"):
        cfg = json.loads(config_file.read_text())
        checkpoint = config_file.parent
        if (not (checkpoint / "encoder" / "config.json").is_file()
                or not (checkpoint / "tokenizer").is_dir()):
            encoder = cfg.get("encoder")
            if isinstance(encoder, str) and not Path(encoder).exists():
                extra_repos.add(encoder)
for repo in sorted(extra_repos):
    path = Path(snapshot_download(repo, cache_dir=cache, token=token))
    manifest.append({"repo": repo, "revision": path.name,
                     "purpose": "encoder/tokenizer fallback"})

(transfer / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Ready:", transfer)
print("Recorded revisions:", manifest)
PY
```

Full downloads can be large, especially the three-model bundle and extra encoder
repositories. Check disk space on both machines; Workbench needs space for both
the archive and extracted files. No GPU is needed for this download script.
It reads model configuration, not your secret files.

Hugging Face documents full-repository downloads and revision selection in its
[download guide](https://huggingface.co/docs/huggingface_hub/guides/download).
The cache retains revision references, snapshots, and shared blobs; see the
[cache layout](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

## 3. Archive and upload to GCS

Keep the entire `hub` directory together. Snapshots can contain relative symlinks
to blobs; transferring only a snapshot folder can break them. A tar archive
preserves this structure across GCS. Do not archive your ordinary `HF_HOME`
directory, which can contain authentication tokens.

On the laptop:

```bash
tar -czf "$LAYA_TRANSFER_DIR/laya-hf-cache.tar.gz" -C "$LAYA_TRANSFER_DIR" hub model-manifest.json
uv run python - <<'PY'
import hashlib
import os
from pathlib import Path
archive = Path(os.environ["LAYA_TRANSFER_DIR"]) / "laya-hf-cache.tar.gz"
with archive.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
archive.with_suffix(archive.suffix + ".sha256").write_text(
    digest + "  " + archive.name + "\n")
PY
export LAYA_GCS_PREFIX="gs://YOUR_BUCKET/laya-offline/v1"
gcloud storage cp "$LAYA_TRANSFER_DIR/laya-hf-cache.tar.gz" "$LAYA_GCS_PREFIX/"
gcloud storage cp "$LAYA_TRANSFER_DIR/laya-hf-cache.tar.gz.sha256" "$LAYA_GCS_PREFIX/"
```

Replace `YOUR_BUCKET` with an approved bucket. Use an authenticated laptop account
with permission to upload objects, and a Workbench identity with permission to
read them. The instance must be able to reach GCS through your permitted network
path. See Google's [gcloud storage cp reference](https://cloud.google.com/sdk/gcloud/reference/storage/cp).

## 4. Download, verify, and extract on Workbench

In the Workbench terminal, choose a new destination on persistent disk:

```bash
export LAYA_GCS_PREFIX="gs://YOUR_BUCKET/laya-offline/v1"
export LAYA_OFFLINE_DIR="$HOME/laya-offline-v1"
mkdir -p "$LAYA_OFFLINE_DIR"
gcloud storage cp "$LAYA_GCS_PREFIX/laya-hf-cache.tar.gz" "$LAYA_OFFLINE_DIR/"
gcloud storage cp "$LAYA_GCS_PREFIX/laya-hf-cache.tar.gz.sha256" "$LAYA_OFFLINE_DIR/"
cd "$LAYA_OFFLINE_DIR"
sha256sum -c laya-hf-cache.tar.gz.sha256
tar -xzf laya-hf-cache.tar.gz
```

Proceed only if the checksum says `OK`. The result should include
`$HOME/laya-offline-v1/hub/models--convaiinnovations--laya/` and
`model-manifest.json`. Keep this location outside the Git checkout.

## 5. Configure the notebook before any model imports

Restart the notebook kernel. Insert this as the **first code cell**, before
imports from `system_one_models`, `laya`, `transformers`, or `huggingface_hub`:

```python
import os
from pathlib import Path

OFFLINE_DIR = Path.home() / "laya-offline-v1"
os.environ["HF_HUB_CACHE"] = str(OFFLINE_DIR / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
assert (OFFLINE_DIR / "hub").is_dir(), "Extract the model cache first"
```

Setting these variables in a separate terminal does not change an already-running
Jupyter kernel. The first cell also ensures the notebook's Python subprocesses
inherit the offline configuration. Hugging Face reads its environment settings
at import time; `HF_HUB_OFFLINE=1` uses cached files and disables Hub HTTP calls.
See its [environment-variable reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
and [Transformers offline instructions](https://huggingface.co/docs/transformers/installation#offline-mode).

### If your notebook has the Router cell

Replace the failing cell with:

```python
from laya import Router

router = Router(preload=True)  # Loads all three from the transferred cache.
```

Remove any earlier cell that requires `HF_TOKEN` or raises an error when it is
missing. Offline loading of already-downloaded files does not require a token.
The original `os.environ["HF_TOKEN"]` expression itself raises `KeyError` if no
key exists. A token cannot solve blocked network access.

If you only need English and want to avoid keeping three models in memory:

```python
router = Router(preload=False)
router.preload(["english"])
```

This only changes preloading: subsequent automatic routing may still load another
checkpoint. Keep all downloaded models available if retaining automatic routing.
The constructor, model map, and preloading behavior above were checked against
the installed Laya **0.3.27** `router.py` and `agent.py`. If you update the SDK,
repeat the laptop download step against the new routing map.

### For the current week 3 training notebook

Set `MOCK = False` in the notebook's setup cell to train and evaluate real models.
Keep the other cells in order. The scripts can retain their default Hub model
names because the cache includes the pinned revision read from `train.py`.
Training and baseline evaluation load that revision; calibration and tuned-model
evaluation use checkpoints generated locally. `MOCK = True` exercises the
workflow without loading weights or training.

Before real training, check the GPU in the notebook:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

The default real workflow expects CUDA. Model downloads do not supply an NVIDIA
driver or guarantee GPU memory fit. Start with head-only training and consult the
[week 3 workflow guide](../README.md) for batch sizes and full-training limits.
Review the generated training spot-checks before proceeding with real training.
Existing run outputs are protected; choose a fresh run directory for a new run.

## 6. Check the pinned cache before training

After the offline setup cell, this checks cache availability without loading
weights or requiring a GPU:

```python
import runpy
from huggingface_hub import snapshot_download

# Locate the repository regardless of the notebook working directory.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "week_03" / "train.py").is_file())
revision = runpy.run_path(str(ROOT / "week_03" / "train.py"))["MODEL_REVISION"]
snapshot = Path(snapshot_download("convaiinnovations/laya", revision=revision,
                                  local_files_only=True))
assert (snapshot / "rl_agent_config.json").is_file()
assert (snapshot / "model.safetensors").is_file()
print("Pinned model cache found:", snapshot)
```

Then run the relevant real loading/training cells. This cache check alone does
not validate inference, GPU fit, or training.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `KeyError: 'HF_TOKEN'` | Remove the required-token expression/check in the offline notebook. |
| A Hugging Face connection is still attempted | Restart the kernel and run the offline cell first; verify the error's library and cache path. |
| `LocalEntryNotFoundError`, incomplete snapshot, or missing revision | Repeat the laptop download for the exact revision requested; transfer the complete cache again to a new directory. Check `model-manifest.json`. |
| Missing tokenizer or encoder configuration | Ensure the archive includes all files and fallback encoder repositories from step 2. A weights file alone is insufficient. |
| Broken snapshot symlinks | Extract the entire tar archive, including `blobs`, `refs`, and `snapshots`, together. |
| CUDA unavailable or GPU memory exhausted | Check the Workbench GPU, driver, and PyTorch setup. Preload only English for English work; lower training micro-batch size as described in the workflow guide. |
| GCS permission/network error | Check the bucket path, Workbench identity's object-read permission, and approved GCS connectivity. |
| `uv sync` fails on package downloads | Provision the locked Python environment through an approved mirror or compatible wheelhouse; the model archive does not contain packages. |

No model weights were downloaded or loaded while writing this guide, and these
steps have not been exercised on your Workbench instance. Commands and notebook
settings were checked against the repository and installed SDK.
