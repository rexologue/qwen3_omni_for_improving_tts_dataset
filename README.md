# Qwen3 Omni TTS Dataset Processing

This repository provides utilities to **restore** or **tag** TTS datasets using Qwen3 Omni via vLLM, with:

- A single entrypoint (`process_dataset.py`) driven by a `task` field in YAML.
- Accurate progress bars for long runs (total row count computed up front).
- A Docker launcher (`launcher.py`) that maps **host** paths to fixed **container** paths automatically.
- A `run_many.py` tool for batch processing multiple datasets.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Installation (local)](#installation-local)
- [Configuration (task-based)](#configuration-task-based)
- [Process a dataset (local)](#process-a-dataset-local)
- [Single inference](#single-inference)
- [Batch processing many datasets](#batch-processing-many-datasets)
- [Docker workflow](#docker-workflow)
- [Troubleshooting](#troubleshooting)

---

## Repository layout

- `process_dataset.py` — unified entrypoint for restore/tag tasks.
- `single_inference.py` — one-off inference with the same task config schema.
- `run_many.py` — runs the task for each dataset under a root folder.
- `dataset_utils.py` — dataset loader and CSV helpers (with total row counting).
- `launcher.py` — Docker launcher that maps host paths to container paths.
- `prompts.py` — prompt templates for restore/tag.
- `config.py` — configuration schemas and validation.
- `requirements.txt` — minimal Python dependencies.
- `Dockerfile` — base image + dependencies for Docker runs.

---

## Installation (local)

> **Note:** Qwen3 Omni / vLLM typically requires CUDA-capable hardware and compatible drivers.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using vLLM directly (outside Docker), install it following your environment constraints.

---

## Configuration (task-based)

All dataset processing uses a **single YAML schema** with a required `task` field:

- `task: restore` — restore / cleanup transcript text.
- `task: tag` — produce tag annotations (requires few-shot examples).

### Restore example

```yaml
task: restore
model: /path/to/your/model
dataset_dir: /path/to/dataset_root
out: /path/to/output/result.csv

# optional
batch_size: 8
max_seqs: 8
max_model_len: 4096
gpu_mem: 0.75
dtype: bfloat16
kv_cache_dtype: fp8
seed: 1234
temperature: 0.05
top_p: 0.9
top_k: 40
max_tokens: 2048
tp: 1
accent: true
```

### Tag example (few-shot required)

```yaml
task: tag
model: /path/to/your/model
few_shot_json: /path/to/examples.json
dataset_dir: /path/to/dataset_root
out: /path/to/output/result.csv
```

`few_shot_json` must be a JSON array with items like:

```json
[
  {
    "audio": "/path/to/example.wav",
    "text": "Original transcript",
    "response": "<tag>Example response</tag>"
  }
]
```

---

## Process a dataset (local)

```bash
python3 process_dataset.py --config restore_dataset.example.yaml
```

For tagging:

```bash
python3 process_dataset.py --config tag_dataset.example.yaml
```

The output is appended to `out`. If the CSV already exists, previously processed `audio_path` rows are skipped.

---

## Single inference

Single inference uses the same `task` model. Example YAML:

```yaml
task: restore
model: /path/to/your/model
audio: /path/to/audio.wav
transcript: "Ваш текст"
```

For tagging single-inference (few-shot required):

```yaml
task: tag
model: /path/to/your/model
audio: /path/to/audio.wav
transcript: "Ваш текст"
few_shot_json: /path/to/examples.json
```

Run:

```bash
python3 single_inference.py --config single_inference.example.yaml
```

---

## Batch processing many datasets

`run_many.py` processes every subfolder under `datasets_root` with a shared config section:

```yaml
datasets_root: /data/datasets
output_dir: /data/output
skip_datasets:
  - part-1
  - test
fail_fast: false

dataset:
  task: restore
  model: /path/to/your/model
  batch_size: 8
  max_seqs: 8
  max_model_len: 4096
```

Run:

```bash
python3 run_many.py --config run_many.example.yaml
```

If you want isolation per dataset, use:

```bash
python3 run_many.py --config run_many.example.yaml --use-subprocess
```

---

## Docker workflow

### 1) Build the image

```bash
docker build . -t my-qwen-omni-tool
```

### 2) Use the launcher (recommended)

The launcher accepts **host paths only** and automatically maps them into fixed container locations:

- Model → `/app/model`
- Dataset → `/app/data`
- Output dir → `/app/output`
- Config → `/app/config.yaml`
- Few-shot JSON → `/app/few_shot/examples.json`
- Few-shot audio directories → `/app/few_shot_audio/*`

Run:

```bash
python3 launcher.py --config restore_dataset.example.yaml --image my-qwen-omni-tool --gpu 0
```

For a dry-run (print the generated Docker command):

```bash
python3 launcher.py --config restore_dataset.example.yaml --dry-run
```

### 3) Quick-run script

`job.sh` is a convenience wrapper:

```bash
./job.sh restore_dataset.example.yaml
```

Override defaults:

```bash
IMAGE_NAME=my-qwen-omni-tool GPU_ID=1 ./job.sh tag_dataset.example.yaml
```

---

## Troubleshooting

**Q: The progress bar doesn't show a total.**  
A: `DatasetLoader` now computes total row count before streaming. Ensure `metadata.csv` is accessible and contains required columns:  
`audio_path`, `text`, `speaker_name`, `language` (and optionally `text_accent`).

**Q: Docker can’t find my model or dataset.**  
A: The launcher mounts **host paths** into fixed container paths. Ensure the paths in your YAML exist on the host.

**Q: Tagging fails with missing few-shot examples.**  
A: For `task: tag`, `few_shot_json` is required and every example must contain `audio`, `text`, and `response` fields.

---

## Examples included

- `restore_dataset.example.yaml`
- `tag_dataset.example.yaml`
- `run_many.example.yaml`
- `single_inference.example.yaml`

Use these as templates and replace host paths with your own.
