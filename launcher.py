import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config import ConfigError, ProcessDatasetConfig


@dataclass
class DockerRunConfig:
    image: str
    build: bool
    context: Path
    dockerfile: Path
    gpu: str
    shm_size: str
    container_name: str | None
    use_host_user: bool
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docker launcher with host-to-container path mapping")
    parser.add_argument("--config", type=Path, required=True, help="YAML конфиг для docker запуска")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError("Корневой элемент YAML должен быть словарём")
    return data


def _load_launcher_config(path: Path) -> tuple[DockerRunConfig, dict[str, Any]]:
    data = _load_yaml(path)
    if "dataset" not in data:
        raise ConfigError("В конфиге отсутствует секция dataset")
    if "docker" not in data:
        raise ConfigError("В конфиге отсутствует секция docker")

    dataset_section = data["dataset"]
    docker_section = data["docker"]

    if not isinstance(dataset_section, dict):
        raise ConfigError("Секция dataset должна быть словарём")
    if not isinstance(docker_section, dict):
        raise ConfigError("Секция docker должна быть словарём")

    allowed_docker_fields = {
        "image",
        "build",
        "context",
        "dockerfile",
        "gpu",
        "shm_size",
        "container_name",
        "use_host_user",
        "dry_run",
    }
    unknown = sorted(set(docker_section.keys()) - allowed_docker_fields)
    if unknown:
        raise ConfigError(f"Неизвестные поля в docker секции: {', '.join(unknown)}")

    docker_cfg = DockerRunConfig(
        image=str(docker_section.get("image", "my-qwen-omni-tool")),
        build=bool(docker_section.get("build", True)),
        context=Path(docker_section.get("context", ".")),
        dockerfile=Path(docker_section.get("dockerfile", "Dockerfile")),
        gpu=str(docker_section.get("gpu", "all")),
        shm_size=str(docker_section.get("shm_size", "16g")),
        container_name=docker_section.get("container_name"),
        use_host_user=bool(docker_section.get("use_host_user", False)),
        dry_run=bool(docker_section.get("dry_run", False)),
    )

    return docker_cfg, dataset_section


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
    docker_cfg: DockerRunConfig,
    mapped_config_path: Path,
    cfg: ProcessDatasetConfig,
    output_csv: Path,
    extra_mounts: list[tuple[Path, Path, str]],
    mapped_few_shot_path: Path | None,
) -> list[str]:
    host_model = Path(cfg.model).resolve()
    host_dataset = cfg.dataset_dir.resolve()
    host_out = output_csv.resolve()
    host_out.parent.mkdir(parents=True, exist_ok=True)
    host_out.touch(exist_ok=True)
    container_out = Path("/app/output/result.csv")

    mounts = [
        (host_model, Path("/app/model"), "ro"),
        (host_dataset, Path("/app/data"), "ro"),
        (host_out, container_out, "rw"),
        (mapped_config_path, Path("/app/config.yaml"), "ro"),
    ]
    if mapped_few_shot_path is not None:
        mounts.append((mapped_few_shot_path, Path("/app/few_shot/examples.json"), "ro"))

    mounts.extend(extra_mounts)

    cmd = ["docker", "run", "--rm"]
    if docker_cfg.gpu == "all":
        cmd += ["--gpus", "all"]
    else:
        cmd += ["--gpus", f"device={docker_cfg.gpu}"]

    cmd += ["--shm-size", docker_cfg.shm_size]

    if docker_cfg.container_name:
        cmd += ["--name", docker_cfg.container_name]

    if docker_cfg.use_host_user:
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        cmd += ["-e", "USER=omni", "-e", "HOME=/home/omni"]

    for host_path, container_path, mode in mounts:
        mount_flag = f"{host_path}:{container_path}:{mode}"
        cmd += ["-v", mount_flag]

    cmd += [docker_cfg.image, "python3", "process_dataset.py", "--config", "/app/config.yaml"]
    return cmd


def main() -> None:
    args = parse_args()
    docker_cfg, dataset_data = _load_launcher_config(args.config)
    output_csv_value = dataset_data.get("output_csv") or dataset_data.get("out")
    if not output_csv_value:
        raise ConfigError("В секции dataset не указан output_csv или out")
    output_csv = Path(output_csv_value).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.touch(exist_ok=True)
    cfg = ProcessDatasetConfig.from_dict(dataset_data)

    mapped_examples: list[dict[str, str]] = []
    extra_mounts: list[tuple[Path, Path, str]] = []
    mapped_few_shot_path: Path | None = None

    if cfg.task == "tag":
        mapped_examples, extra_mounts = _map_few_shot_examples(cfg.examples)

    mapped_data = dict(dataset_data)
    mapped_data["model"] = "/app/model"
    mapped_data["dataset_dir"] = "/app/data"
    container_out = Path("/app/output/result.csv")
    mapped_data["out"] = str(container_out)
    if "output_csv" in mapped_data:
        mapped_data["output_csv"] = str(container_out)

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
            docker_cfg,
            mapped_config_path,
            cfg,
            output_csv,
            extra_mounts,
            mapped_few_shot_path,
        )
        
        print(cmd)

        if docker_cfg.build:
            build_cmd = [
                "docker",
                "build",
                "-t",
                docker_cfg.image,
                "-f",
                str(docker_cfg.dockerfile),
                str(docker_cfg.context),
            ]
            if docker_cfg.dry_run:
                print(" ".join(build_cmd))
            else:
                subprocess.run(build_cmd, check=True)

        if docker_cfg.dry_run:
            print(" ".join(cmd))
            return

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
