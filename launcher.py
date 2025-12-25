import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from config import ConfigError, ProcessDatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docker launcher with host-to-container path mapping")
    parser.add_argument("--config", type=Path, required=True, help="YAML конфиг для process_dataset.py")
    parser.add_argument("--image", default="my-qwen-omni-tool", help="Docker image name")
    parser.add_argument("--container-name", default=None, help="Docker container name")
    parser.add_argument("--gpu", default="all", help="GPU device id or 'all'")
    parser.add_argument("--shm-size", default="16g", help="Shared memory size for Docker")
    parser.add_argument("--use-host-user", action="store_true", help="Запускать контейнер от UID/GID хоста")
    parser.add_argument("--dry-run", action="store_true", help="Показать docker команду без запуска")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError("Корневой элемент YAML должен быть словарём")
    return data


def _map_few_shot_examples(
    examples: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[tuple[Path, Path, str]]]:
    mapped_examples: list[dict[str, str]] = []
    mounts: list[tuple[Path, Path, str]] = []
    parent_mapping: dict[Path, Path] = {}

    for ex in examples:
        audio_path = Path(ex["audio"]).resolve()
        parent = audio_path.parent
        if parent not in parent_mapping:
            container_dir = Path("/app/few_shot_audio") / f"set_{len(parent_mapping) + 1}"
            parent_mapping[parent] = container_dir
            mounts.append((parent, container_dir, "ro"))

        container_audio = parent_mapping[parent] / audio_path.name
        mapped = dict(ex)
        mapped["audio"] = str(container_audio)
        mapped_examples.append(mapped)

    return mapped_examples, mounts


def _build_docker_command(
    args: argparse.Namespace,
    mapped_config_path: Path,
    cfg: ProcessDatasetConfig,
    extra_mounts: list[tuple[Path, Path, str]],
    mapped_few_shot_path: Path | None,
) -> list[str]:
    host_model = Path(cfg.model).resolve()
    host_dataset = cfg.dataset_dir.resolve()
    host_out = cfg.out.resolve()
    host_out.parent.mkdir(parents=True, exist_ok=True)

    mounts = [
        (host_model, Path("/app/model"), "ro"),
        (host_dataset, Path("/app/data"), "ro"),
        (host_out.parent, Path("/app/output"), "rw"),
        (mapped_config_path, Path("/app/config.yaml"), "ro"),
    ]
    if mapped_few_shot_path is not None:
        mounts.append((mapped_few_shot_path, Path("/app/few_shot/examples.json"), "ro"))

    mounts.extend(extra_mounts)

    cmd = ["docker", "run", "--rm"]
    if args.gpu == "all":
        cmd += ["--gpus", "all"]
    else:
        cmd += ["--gpus", f"device={args.gpu}"]

    cmd += ["--shm-size", args.shm_size]

    if args.container_name:
        cmd += ["--name", args.container_name]

    if args.use_host_user:
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        cmd += ["-e", "USER=omni", "-e", "HOME=/home/omni"]

    for host_path, container_path, mode in mounts:
        mount_flag = f"{host_path}:{container_path}:{mode}"
        cmd += ["-v", mount_flag]

    cmd += [args.image, "python3", "process_dataset.py", "--config", "/app/config.yaml"]
    return cmd


def main() -> None:
    args = parse_args()
    data = _load_yaml(args.config)
    cfg = ProcessDatasetConfig.from_dict(data)

    mapped_examples: list[dict[str, str]] = []
    extra_mounts: list[tuple[Path, Path, str]] = []
    mapped_few_shot_path: Path | None = None

    if cfg.task == "tag":
        mapped_examples, extra_mounts = _map_few_shot_examples(cfg.examples)

    mapped_data = dict(data)
    mapped_data["model"] = "/app/model"
    mapped_data["dataset_dir"] = "/app/data"
    mapped_data["out"] = str(Path("/app/output") / cfg.out.name)

    if cfg.task == "tag":
        mapped_data["few_shot_json"] = "/app/few_shot/examples.json"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        mapped_config_path = tmp_root / "config.yaml"

        if cfg.task == "tag":
            mapped_few_shot_path = tmp_root / "examples.json"
            with mapped_few_shot_path.open("w", encoding="utf-8") as f:
                json.dump(mapped_examples, f, ensure_ascii=False, indent=2)

        with mapped_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(mapped_data, f, allow_unicode=True)

        cmd = _build_docker_command(
            args,
            mapped_config_path,
            cfg,
            extra_mounts,
            mapped_few_shot_path,
        )

        if args.dry_run:
            print(" ".join(cmd))
            return

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
