# Qwen3 Omni TTS Dataset Processing

This repository provides utilities to **restore** or **tag** TTS datasets using Qwen3 Omni via vLLM, with:

- A single entrypoint (`process_dataset.py`) driven by a `task` field in YAML.
- Accurate progress bars for long runs (total row count computed up front).
- A Docker launcher (`launcher.py`) that maps **host** paths to fixed **container** paths automatically.
- A `run_many.py` tool for batch processing multiple datasets.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Configuration (task-based)](#configuration-task-based)
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

## Configuration (task-based)

All dataset processing uses a **single YAML schema** with a required `task` field and a Docker section:

- `task: restore` — restore / cleanup transcript text.
- `task: tag` — produce tag annotations (requires few-shot examples).

### Restore example (launcher YAML)

```yaml
docker:
  image: my-qwen-omni-tool
  build: true
  context: .
  dockerfile: Dockerfile
  gpu: "0"
  shm_size: 16g
  container_name: null
  use_host_user: false
  dry_run: false

dataset:
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
docker:
  image: my-qwen-omni-tool
  build: true
  context: .
  dockerfile: Dockerfile
  gpu: "0"
  shm_size: 16g
  container_name: null
  use_host_user: false
  dry_run: false

dataset:
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

Docker section options:

- `image` — Docker image tag to build/run.
- `build` — build the image before running.
- `context` — build context directory.
- `dockerfile` — path to Dockerfile.
- `gpu` — GPU device id or `"all"`.
- `shm_size` — shared memory size for the container.
- `container_name` — optional container name.
- `use_host_user` — run container with host UID/GID.
- `dry_run` — print docker commands without executing.

---

## Docker workflow

### Use the launcher (recommended)

The launcher accepts **host paths only** and automatically maps them into fixed container locations:

- Model → `/app/model`
- Dataset → `/app/data`
- Output dir → `/app/output`
- Config → `/app/config.yaml`
- Few-shot JSON → `/app/few_shot/examples.json`
- Few-shot audio directories → `/app/few_shot_audio/*`

Run:

```bash
python3 launcher.py --config restore_dataset.example.yaml
```

For a dry-run (print the generated Docker command):

```bash
python3 launcher.py --config restore_dataset.example.yaml
```

Set `docker.dry_run: true` in the YAML to enable dry-run output.

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
