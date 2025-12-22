import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from config import RunManyConfig
from vllm.qwen3_omni.restore_dataset import run_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск process_dataset для нескольких подпапок с YAML-конфигом")
    parser.add_argument("--config", type=Path, required=True, help="Путь до run_many YAML конфига")
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="Запускать process_dataset как отдельный процесс (изолирует память/состояние)",
    )
    return parser.parse_args()


def build_env(device: str | None) -> dict[str, str] | None:
    if device is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    return env


def run_subprocess(cfg: RunManyConfig, dataset_cfg_path: Path) -> None:
    env = build_env(cfg.device)
    subprocess.run([sys.executable, "process_dataset.py", "--config", str(dataset_cfg_path)], env=env, check=True)


def main() -> None:
    args = parse_args()
    config = RunManyConfig.from_yaml(args.config)

    datasets = [p for p in sorted(config.datasets_root.iterdir()) if p.is_dir()]

    for ds_dir in datasets:
        if ds_dir.name in config.skip_datasets:
            print(f"[RUN_MANY] Пропускаю {ds_dir.name} (skip list)")
            continue

        try:
            ds_cfg = config.make_dataset_config(ds_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[RUN_MANY] Ошибка подготовки конфига для {ds_dir.name}: {exc}")
            if config.fail_fast:
                raise
            continue

        print("\n\n\n**********************************\n")
        print(f"[RUN_MANY] Now running {ds_dir.name}")
        print("\n**********************************\n\n\n")

        if args.use_subprocess:
            ds_cfg_path = config.output_dir / f"{ds_dir.name}.yaml"
            with ds_cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {**config.dataset, "dataset_dir": str(ds_dir), "out": str(ds_cfg.out)},
                    f,
                    allow_unicode=True,
                )
            try:
                run_subprocess(config, ds_cfg_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[RUN_MANY] Ошибка при обработке {ds_dir.name}: {exc}")
                if config.fail_fast:
                    raise
            finally:
                ds_cfg_path.unlink(missing_ok=True)
        else:
            try:
                run_with_config(ds_cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[RUN_MANY] Ошибка при обработке {ds_dir.name}: {exc}")
                if config.fail_fast:
                    raise


if __name__ == "__main__":
    main()
