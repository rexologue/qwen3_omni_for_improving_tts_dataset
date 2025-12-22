import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

import jiwer
from tqdm import tqdm
from ruaccent import RUAccent

from config import RestoreConfig
from dataset_utils import DatasetLoader, read_csv_header, read_done_audio_paths
from prompts import get_prompt
from mllm_utils import init_llm, init_sampling, load_processor, build_inputs, generate_texts


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Обработка датасета с использованием RESTORE_PROMPT и YAML-конфига")
    ap.add_argument("--config", type=Path, required=True, help="Путь до YAML конфига")
    return ap.parse_args()


def run_with_config(config: RestoreConfig) -> None:
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

    accentizer = None
    if config.accent:
        accentizer = RUAccent()
        accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False, device="CUDA")
        print("Loaded accentizer!")

    llm = init_llm(
        config.model,
        dtype=config.dtype,
        kv_cache_dtype=config.kv_cache_dtype,
        gpu_memory_utilization=config.gpu_mem,
        tensor_parallel_size=config.tp,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
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

    pbar = tqdm(total=None, initial=len(already_done), desc="rows")

    need_header = existing_header is None

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if need_header:
            writer.writeheader()

        for batch in itertools.chain([first_batch], dataset_iter):
            batch_cer = 0
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

            if len(batch) > 0:
                batch_cer /= len(batch)

            pbar.update(len(batch))
            pbar.set_postfix_str(f"cer: {batch_cer:.4f}")

    print("Готово:", output_csv)
    print(f"Всего обработано: {dataset_loader.total_pairs}")


def main() -> None:
    args = parse_args()
    config = RestoreConfig.from_yaml(args.config)
    run_with_config(config)


if __name__ == "__main__":
    main()
