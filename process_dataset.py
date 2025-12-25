import argparse
import csv
import itertools
import re
from pathlib import Path
from typing import Any

import jiwer
from ruaccent import RUAccent
from tqdm import tqdm

from config import ProcessDatasetConfig
from dataset_utils import DatasetLoader, read_csv_header, read_done_audio_paths
from mllm_utils import build_few_shot_inputs, build_inputs, generate_texts, init_llm, init_sampling, load_processor
from prompts import get_prompt
from utils import ANY_TAG_PATTERN


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Обработка датасета с YAML-конфигом (task=restore/tag)")
    ap.add_argument("--config", type=Path, required=True, help="Путь до YAML конфига")
    return ap.parse_args()


def _init_accentizer(use_accent: bool) -> RUAccent | None:
    if not use_accent:
        return None
    accentizer = RUAccent()
    accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False, device="CUDA")
    print("Loaded accentizer!")
    return accentizer


def _process_restore_batch(
    batch: list[dict[str, Any]],
    config: ProcessDatasetConfig,
    processor,
    llm,
    sampling,
    accentizer: RUAccent | None,
    writer: csv.DictWriter,
    fieldnames: list[str],
) -> float:
    batch_cer = 0.0
    texts = [(p["text"] or "").strip() for p in batch]

    reqs = []
    for p, t in zip(batch, texts):
        prompt_text = get_prompt(t)
        reqs.append(build_inputs(p["abs_audio"], prompt_text, processor))

    gen_texts = generate_texts(llm, reqs, sampling)

    for gen_text, p, base_text in zip(gen_texts, batch, texts):
        rel_path = p["rel_audio"]

        if accentizer is not None and (p["language"] or "").lower() != "en":
            if p["text_accent"]:
                base_accent_text = p["text_accent"]
            else:
                base_accent_text = accentizer.process_all(base_text)

            out_accent_text = accentizer.process_all(gen_text)
        else:
            base_accent_text = base_text
            out_accent_text = gen_text

        cer = jiwer.cer(base_text, gen_text)
        batch_cer += cer

        row_full: dict[str, Any] = {
            "audio_path": rel_path,
            "base_text": base_text,
            "out_text": gen_text,
            "base_accent_text": base_accent_text,
            "out_accent_text": out_accent_text,
            "speaker_name": (p["speaker_name"] or ""),
            "language": p["language"],
            "cer": cer,
        }
        row = {k: row_full.get(k, "") for k in fieldnames}
        writer.writerow(row)

    if batch:
        batch_cer /= len(batch)

    return batch_cer


def _process_tag_batch(
    batch: list[dict[str, Any]],
    config: ProcessDatasetConfig,
    processor,
    llm,
    sampling,
    accentizer: RUAccent | None,
    writer: csv.DictWriter,
    fieldnames: list[str],
) -> None:
    texts = [(p["text"] or "").strip() for p in batch]

    reqs = []
    for p, t in zip(batch, texts):
        prompt = get_prompt(t, prompt_type="tags")
        reqs.append(build_few_shot_inputs(
            p["abs_audio"],
            prompt,
            t,
            processor,
            config.examples,
        ))

    gen_texts = generate_texts(llm, reqs, sampling)

    for gen_text, p, base_text in zip(gen_texts, batch, texts):
        rel_path = p["rel_audio"]

        if accentizer is not None and (p["language"] or "").lower() != "en":
            def process_match(match):
                tag_name = match.group(1)
                content = match.group(2)
                processed_content = accentizer.process_all(content)
                return f"<{tag_name}>{processed_content}</{tag_name}>"

            tag_accent_text = re.sub(ANY_TAG_PATTERN, process_match, gen_text)
        else:
            tag_accent_text = gen_text

        row_full: dict[str, Any] = {
            "audio_path": rel_path,
            "base_text": base_text,
            "tag_text": gen_text,
            "tag_accent_text": tag_accent_text,
            "speaker_name": (p["speaker_name"] or ""),
            "language": p["language"],
        }
        row = {k: row_full.get(k, "") for k in fieldnames}
        writer.writerow(row)


def run_with_config(config: ProcessDatasetConfig) -> None:
    output_csv = config.out
    dataset_root = config.dataset_dir.resolve()

    already_done = read_done_audio_paths(output_csv)

    dataset_loader = DatasetLoader(dataset_root, config.batch_size, already_done)
    dataset_iter = iter(dataset_loader)
    first_batch = next(dataset_iter, None)

    if first_batch is None:
        print(f"Нечего обрабатывать. Уже готово: {len(already_done)}")
        return

    existing_header = read_csv_header(output_csv)
    if existing_header is not None:
        fieldnames = existing_header
    elif config.task == "tag":
        fieldnames = [
            "audio_path",
            "base_text",
            "tag_text",
            "tag_accent_text",
            "speaker_name",
            "language",
        ]
    else:
        fieldnames = [
            "audio_path",
            "base_text",
            "out_text",
            "base_accent_text",
            "out_accent_text",
            "speaker_name",
            "language",
            "cer",
        ]

    accentizer = _init_accentizer(config.accent)

    llm = init_llm(
        config.model,
        dtype=config.dtype,
        kv_cache_dtype=config.kv_cache_dtype,
        gpu_memory_utilization=config.gpu_mem,
        tensor_parallel_size=config.tp,
        limit_mm_per_prompt={"audio": 6 if config.task == "tag" else 1, "image": 0, "video": 0},
        max_num_seqs=config.max_seqs,
        max_model_len=config.max_model_len,
        seed=config.seed,
        trust_remote_code=True,
    )
    sampling = init_sampling(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_tokens,
    )
    processor = load_processor(config.model)

    pbar = tqdm(total=dataset_loader.total_rows, initial=dataset_loader.skipped_rows, desc="rows")

    need_header = existing_header is None

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if need_header:
            writer.writeheader()

        for batch in itertools.chain([first_batch], dataset_iter):
            if config.task == "tag":
                _process_tag_batch(batch, config, processor, llm, sampling, accentizer, writer, fieldnames)
                pbar.update(len(batch))
                continue

            batch_cer = _process_restore_batch(
                batch,
                config,
                processor,
                llm,
                sampling,
                accentizer,
                writer,
                fieldnames,
            )
            pbar.update(len(batch))
            pbar.set_postfix_str(f"cer: {batch_cer:.4f}")

    print("Готово:", output_csv)
    print(f"Всего обработано: {dataset_loader.total_pairs}")


def main() -> None:
    args = parse_args()
    config = ProcessDatasetConfig.from_yaml(args.config)
    run_with_config(config)


if __name__ == "__main__":
    main()
