# Qwen Molecular Generation

A configurable pipeline for zero-shot conditional molecular generation with a Hugging Face causal/chat model. The model is prompted with one or more target molecular properties and asked to return a SMILES string. RDKit is used to validate generated molecules, calculate their properties, and evaluate generation quality.

> **Current experiment scope:** the checked-in configurations run **zero-shot** generation (`shots: 0`). Despite the repository name, the current scripts do not yet retrieve or insert in-context examples.

## Features

- Conditional generation for LogP, QED, TPSA, and synthetic accessibility (SAS).
- Single-property and multi-property target configurations.
- Reproducible target and job manifests with per-job seeds.
- Resumable generation: completed job IDs are skipped when an output JSONL already exists.
- RDKit-based SMILES validation and descriptor calculation.
- ChemFM-compatible evaluation of validity, uniqueness, novelty, and target-property errors.
- Optional Langfuse upload for generation traces, evaluation summaries, and logs.
- Model ID override at runtime, allowing compatible Hugging Face models to be tested without editing every config.

## Repository structure

```text
qwen-icl-molecular-generation/
├── configs/
│   ├── logp_zero_shot.json
│   ├── qed_zero_shot.json
│   ├── tpsa_zero_shot.json
│   ├── sas_zero_shot.json
│   ├── tpsa_sas_zero_shot.json
│   ├── logp_tpsa_zero_shot.json
│   ├── sas_logp_zero_shot.json
│   └── logp_sas_tpsa_zero_shot.json
├── src/
│   ├── common.py
│   ├── prepare_dataset.py
│   ├── build_manifest.py
│   ├── generate_qwen.py
│   ├── evaluate_chemfm_property.py
│   ├── evaluate_chemfm_multi.py
│   ├── aggregate_all_results.py
│   ├── save_all_logs.py
│   └── upload_to_langfuse.py
├── requirements.txt
├── requirements-langfuse.txt
└── .gitignore
```

Datasets, model checkpoints, manifests, generated molecules, evaluation outputs, logs, and credentials are intentionally excluded from Git.

## Requirements

- Python 3.11 is the environment used for development of this pipeline.
- A CUDA-enabled NVIDIA GPU for model generation.
- A compatible Hugging Face model and sufficient GPU memory for its weights and generation context.
- RDKit and the Python dependencies listed in `requirements.txt`.

Create an environment and install dependencies:

```bash
git clone https://github.com/gotnochill815-web/qwen-icl-molecular-generation.git
cd qwen-icl-molecular-generation

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Google Colab, select a GPU runtime before running generation. Colab may already provide a CUDA-enabled PyTorch installation. Check that `torch.cuda.is_available()` is `True` after installing dependencies; use a PyTorch build compatible with the runtime GPU rather than replacing it with a CPU-only build.

Verify the environment:

```bash
python - <<'PY'
import torch
import transformers
from rdkit import Chem

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("RDKit import: OK")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## Dataset preparation

The preprocessing script expects normalized GuacaMol `train.csv` and `val.csv` files. Each file must contain these columns:

- `smiles`
- normalized descriptor columns: `qed`, `logp`, `sas`, `tpsa`
- normalization metadata columns: `mean_qed`, `mean_logp`, `mean_sas`, `mean_tpsa`, `std_qed`, `std_logp`, `std_sas`, `std_tpsa`

The script converts normalized descriptor values back to their original scale using:

```text
original_value = normalized_value * std + mean
```

By default, place the input files here:

```text
data/guacamol_full/train.csv
data/guacamol_full/val.csv
```

Then run:

```bash
python src/prepare_dataset.py
```

Or pass a different input directory:

```bash
python src/prepare_dataset.py --data-dir /path/to/normalized_guacamol
```

The processed files are written to:

```text
outputs/processed/guacamol_full/train.csv
outputs/processed/guacamol_full/val.csv
outputs/processed/guacamol_full/metadata.json
```

The experiment configs use these processed files. The training split is also used as the reference set for novelty evaluation.

## Run an experiment

Each JSON file in `configs/` defines one experiment, including the target properties, target values or quantile selection, model settings, generation parameters, evaluation tolerances, and output paths.

The available experiments are:

| Config | Target properties |
|---|---|
| `logp_zero_shot.json` | LogP |
| `qed_zero_shot.json` | QED |
| `tpsa_zero_shot.json` | TPSA |
| `sas_zero_shot.json` | SAS |
| `tpsa_sas_zero_shot.json` | TPSA + SAS |
| `logp_tpsa_zero_shot.json` | LogP + TPSA |
| `sas_logp_zero_shot.json` | SAS + LogP |
| `logp_sas_tpsa_zero_shot.json` | LogP + SAS + TPSA |

### 1. Build target and job manifests

Run this once for each experiment config. For example:

```bash
python src/build_manifest.py --config configs/logp_zero_shot.json
```

Repeat for the other JSON configs. The script reads the validation split, selects target property values, and writes the target manifest and one JSONL job per requested molecule. The output paths are defined in each config under `paths`.

### 2. Generate molecules

Run generation on a GPU:

```bash
python src/generate_qwen.py --config configs/logp_zero_shot.json
```

The model ID is read from the config. To use another compatible Hugging Face model without editing the config:

```bash
python src/generate_qwen.py \
  --config configs/logp_zero_shot.json \
  --model-id YOUR_ORG/YOUR_MODEL
```

For a small smoke test, limit the run to a few jobs:

```bash
python src/generate_qwen.py \
  --config configs/logp_zero_shot.json \
  --limit 2
```

The default output is the JSONL path specified in the config. Each completed molecule is appended as a record. If the script is run again with the same config and output file, it skips job IDs already present and continues with pending jobs. Use a fresh output path (via `--output`) when starting an independent run.

Optional token-level debugging for the first generated sample:

```bash
python src/generate_qwen.py \
  --config configs/logp_zero_shot.json \
  --limit 1 \
  --debug-tokens
```

### 3. Evaluate generated molecules

Use `evaluate_chemfm_property.py` for a single-property experiment:

```bash
python src/evaluate_chemfm_property.py \
  --config configs/logp_zero_shot.json \
  --input outputs/raw/logp_zero_shot.jsonl \
  --tag logp_zero_shot_chemfm
```

Use `evaluate_chemfm_multi.py` for a two- or three-property experiment:

```bash
python src/evaluate_chemfm_multi.py \
  --config configs/tpsa_sas_zero_shot.json \
  --input outputs/raw/tpsa_sas_zero_shot.jsonl \
  --tag tpsa_sas_zero_shot_chemfm
```

Replace the config, input path, and tag for the experiment being evaluated.

Evaluation produces `molecules.csv`, `summary.csv`, and `summary.json` under:

```text
outputs/evaluation/<tag>/
```

The evaluation calculates validity, uniqueness, novelty relative to the training split, and absolute target-property errors. Novelty is calculated among unique valid canonical SMILES.

### 4. Aggregate all experiment results

After all eight experiment summaries exist, run:

```bash
python src/aggregate_all_results.py
```

Aggregated files are written to:

```text
outputs/evaluation/all_experiments/all_results.csv
outputs/evaluation/all_experiments/per_target_results.csv
outputs/evaluation/all_experiments/all_results.json
outputs/evaluation/all_experiments/missing_results.txt
```

### 5. Reconstruct experiment logs

After all eight generation JSONL files and evaluations are available:

```bash
python src/save_all_logs.py
```

This reconstructs generation logs from the raw records and reruns the evaluations while saving their console output under `outputs/logs/`.

## Prompt locations

Prompt text is currently defined in the Python source rather than separate files:

- `src/common.py` → `build_prompt()` creates the user prompt containing the requested property targets.
- `src/generate_qwen.py` → `format_chat_prompt()` selects the system instruction and applies the model tokenizer's chat template when available.

There is currently no `prompts/` directory. The checked-in experiment configs set `shots` to `0`, and the generation pipeline does not add in-context demonstration molecules. Some system-instruction text mentions examples; those references are not accompanied by examples in the current zero-shot input.

## Using a different model (including Colab)

The generator accepts `--model-id` and reads model-loading options from the config's `model` section:

- `model_id`
- `auto_model_class`
- `dtype`
- `device_map`
- `trust_remote_code`
- `local_files_only`

A replacement model must be compatible with the selected Transformers auto-model class and tokenizer/processor interface. Its chat template must also be compatible with the formatting path in `format_chat_prompt()`. Some model architectures require changes to the model-loading or prompt-formatting code; changing only `--model-id` does not guarantee compatibility. Check available GPU memory before loading large checkpoints.

## Optional Langfuse integration

Install the optional dependency set:

```bash
pip install -r requirements-langfuse.txt
```

Set these environment variables in your runtime. Do not put API keys in config files or commit them to Git:

```bash
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

The uploader expects all eight experiments to have their raw generation JSONL, evaluation summary, generation log, and evaluation log. It currently validates that each experiment contains 100 generation records.

Validate inputs without uploading:

```bash
python src/upload_to_langfuse.py --dry-run
```

Upload validated experiments:

```bash
python src/upload_to_langfuse.py
```

Force a re-upload even when the local import checkpoint indicates an experiment was already uploaded:

```bash
python src/upload_to_langfuse.py --force
```


