"""Shared configuration schemas and YAML loading helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

import json
import yaml


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError("Корневой элемент YAML должен быть словарём")
    return data


def _validate_required(data: Dict[str, Any], required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(data.keys()))
    if missing:
        raise ConfigError(f"В конфиге не хватает обязательных полей: {', '.join(missing)}")


def _check_unknown_fields(data: Dict[str, Any], allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    unknown = sorted(set(data.keys()) - allowed_set)
    if unknown:
        raise ConfigError(f"Неизвестные поля в конфиге: {', '.join(unknown)}")


@dataclass
class RestoreConfig:
    model: str
    dataset_dir: Path
    out: Path
    batch_size: int = 4
    max_seqs: int = 8
    max_model_len: int = 4096
    gpu_mem: float = 0.75
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "fp8"
    seed: int = 1234
    temperature: float = 0.05
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    tp: int = 1
    accent: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RestoreConfig":
        _validate_required(data, {"model", "dataset_dir", "out"})
        _check_unknown_fields(data, {f.name for f in cls.__dataclass_fields__.values()})

        cfg = cls(
            model=str(data["model"]),
            dataset_dir=Path(data["dataset_dir"]),
            out=Path(data["out"]),
            batch_size=int(data.get("batch_size", cls.batch_size)),
            max_seqs=int(data.get("max_seqs", cls.max_seqs)),
            max_model_len=int(data.get("max_model_len", cls.max_model_len)),
            gpu_mem=float(data.get("gpu_mem", cls.gpu_mem)),
            dtype=str(data.get("dtype", cls.dtype)),
            kv_cache_dtype=str(data.get("kv_cache_dtype", cls.kv_cache_dtype)),
            seed=int(data.get("seed", cls.seed)),
            temperature=float(data.get("temperature", cls.temperature)),
            top_p=float(data.get("top_p", cls.top_p)),
            top_k=int(data.get("top_k", cls.top_k)),
            max_tokens=int(data.get("max_tokens", cls.max_tokens)),
            tp=int(data.get("tp", cls.tp)),
            accent=bool(data.get("accent", cls.accent)),
        )
        cfg.validate_paths()
        return cfg

    @classmethod
    def from_yaml(cls, path: Path) -> "RestoreConfig":
        return cls.from_dict(_load_yaml(path))

    def validate_paths(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"dataset_dir не найден: {self.dataset_dir}")
        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"dataset_dir должен быть директорией: {self.dataset_dir}")
        self.out.parent.mkdir(parents=True, exist_ok=True)
        
@dataclass
class TagConfig:
    model: str
    examples: list[dict[str, str]]
    dataset_dir: Path
    out: Path
    batch_size: int = 4
    max_seqs: int = 8
    max_model_len: int = 4096
    gpu_mem: float = 0.75
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "fp8"
    seed: int = 1234
    temperature: float = 0.05
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    tp: int = 1
    accent: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TagConfig":
        # 1. Basic validation
        _validate_required(data, {"model", "dataset_dir", "out", "few_shot_json"})
        
        # 2. Load and validate examples
        examples_path = Path(data["few_shot_json"])
        if not examples_path.is_file():
            raise ConfigError(f"Incorrect path to few-shot JSON: {examples_path}")
        
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
            
        required_fields = {"audio", "text", "response"}
        for i, ex in enumerate(examples):
            if not required_fields.issubset(ex.keys()):
                raise ConfigError(f"Example {i} missing fields: {required_fields - set(ex.keys())}")
            
            # Audio path check (assuming paths in JSON are relative to the JSON file or absolute)
            audio_path = Path(ex["audio"])
            if not audio_path.exists():
                raise ConfigError(f"Audio file not found for example {i}: {audio_path}")

        # 3. Dynamic initialization
        # This automatically picks up defaults from the dataclass definition
        init_kwargs = {}
        for field in cls.__dataclass_fields__.values():
            if field.name == "examples":
                init_kwargs["examples"] = examples
            elif field.name in data:
                # Cast to the correct type based on the dataclass hint
                val = data[field.name]
                init_kwargs[field.name] = field.type(val) if not hasattr(field.type, "__origin__") else val
            elif field.name == "dataset_dir" or field.name == "out":
                init_kwargs[field.name] = Path(data[field.name])
            elif field.name == "model":
                init_kwargs[field.name] = str(data["model"])

        # Create the instance
        # Note: Using .get() logic is cleaner if you want to keep your manual mapping:
        cfg = cls(
            model=str(data["model"]),
            examples=examples,
            dataset_dir=Path(data["dataset_dir"]),
            out=Path(data["out"]),
            # For the rest, we loop through the remaining optional fields
            **{k: data[k] for k in data if k in cls.__dataclass_fields__ and k not in ["model", "examples", "dataset_dir", "out"]}
        )
        
        cfg.validate_paths()
        return cfg

    @classmethod
    def from_yaml(cls, path: Path) -> "TagConfig":
        return cls.from_dict(_load_yaml(path))

    def validate_paths(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"dataset_dir не найден: {self.dataset_dir}")
        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"dataset_dir должен быть директорией: {self.dataset_dir}")
        self.out.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class SingleInferenceConfig:
    model: str
    audio: Path
    transcript: str
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "fp8"
    gpu_mem: float = 0.75
    tp: int = 1
    max_seqs: int = 4
    max_model_len: int = 8192
    seed: int = 1234
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleInferenceConfig":
        _validate_required(data, {"model", "audio", "transcript"})
        _check_unknown_fields(data, {f.name for f in cls.__dataclass_fields__.values()})

        cfg = cls(
            model=str(data["model"]),
            audio=Path(data["audio"]),
            transcript=str(data["transcript"]),
            dtype=str(data.get("dtype", cls.dtype)),
            kv_cache_dtype=str(data.get("kv_cache_dtype", cls.kv_cache_dtype)),
            gpu_mem=float(data.get("gpu_mem", cls.gpu_mem)),
            tp=int(data.get("tp", cls.tp)),
            max_seqs=int(data.get("max_seqs", cls.max_seqs)),
            max_model_len=int(data.get("max_model_len", cls.max_model_len)),
            seed=int(data.get("seed", cls.seed)),
            temperature=float(data.get("temperature", cls.temperature)),
            top_p=float(data.get("top_p", cls.top_p)),
            top_k=int(data.get("top_k", cls.top_k)),
            max_tokens=int(data.get("max_tokens", cls.max_tokens)),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: Path) -> "SingleInferenceConfig":
        return cls.from_dict(_load_yaml(path))

    def validate(self) -> None:
        if not self.audio.exists():
            raise FileNotFoundError(f"audio не найден: {self.audio}")
        if not self.audio.is_file():
            raise ConfigError(f"audio должен быть файлом: {self.audio}")


@dataclass
class RunManyConfig:
    datasets_root: Path
    output_dir: Path
    dataset: Dict[str, Any]
    skip_datasets: list[str] = field(default_factory=list)
    device: str | None = None
    fail_fast: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManyConfig":
        _validate_required(data, {"datasets_root", "output_dir", "dataset"})
        _check_unknown_fields(data, {f.name for f in cls.__dataclass_fields__.values()})

        dataset_section = data.get("dataset")
        if not isinstance(dataset_section, dict):
            raise ConfigError("Поле dataset должно быть словарём с настройками для process_dataset")

        cfg = cls(
            datasets_root=Path(data["datasets_root"]),
            output_dir=Path(data["output_dir"]),
            dataset=dataset_section,
            skip_datasets=list(data.get("skip_datasets", [])),
            device=str(data["device"]) if data.get("device") is not None else None,
            fail_fast=bool(data.get("fail_fast", False)),
        )
        cfg.validate_paths()
        return cfg

    @classmethod
    def from_yaml(cls, path: Path) -> "RunManyConfig":
        return cls.from_dict(_load_yaml(path))

    def validate_paths(self) -> None:
        if not self.datasets_root.exists():
            raise FileNotFoundError(f"datasets_root не найден: {self.datasets_root}")
        if not self.datasets_root.is_dir():
            raise NotADirectoryError(f"datasets_root должен быть директорией: {self.datasets_root}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def make_dataset_config(self, dataset_dir: Path) -> RestoreConfig:
        dataset_name = dataset_dir.name
        dataset_cfg = {
            **self.dataset,
            "dataset_dir": dataset_dir,
            "out": self.output_dir / f"{dataset_name}.csv",
        }
        return RestoreConfig.from_dict(dataset_cfg)
