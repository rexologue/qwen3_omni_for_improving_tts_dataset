import argparse

import csv
import time
from typing import Any
from pathlib import Path

from tqdm import tqdm
from mllm_utils import init_llm, init_sampling, load_processor, build_inputs, generate_texts
from prompts import get_bench_prompt
from utils import iter_batches, PUNCTUATIONS, PUNCT_SLUG
from textproc import extract_words, list_accuracy, punct_precision_metrics

#########################
# BENCH / IO UTILITIES  #
#########################

def read_to_dict(csv_path: Path) -> dict[str, str]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        rows = [(r[0], "|".join(r[1:])) for r in reader if len(r) >= 2]
        return {k: v for k, v in rows}

def load_bench(bench_dir: Path) -> list[dict[str, str]]:
    metadata_path = bench_dir / "metadata.csv"
    wavs_dir = bench_dir / "wavs"
    if not metadata_path.exists():
        raise ValueError("Incorrect bench path! metadata.csv is not found!")
    if not wavs_dir.exists():
        raise ValueError("Incorrect bench path! wavs/ directory is not found!")

    md = read_to_dict(metadata_path)
    wavs: dict[str, Path] = {}
    for f in wavs_dir.iterdir():
        if f.suffix == ".wav":
            wavs[f.stem] = f.resolve().expanduser()

    bench: list[dict[str, str]] = []
    for k, v in md.items():  # детерминированный порядок как в metadata.csv
        if k in wavs:
            bench.append({"text": v, "path": str(wavs[k]), "key": k})
    return bench

#########################
# MAIN / ARGPARSE / RUN #
#########################

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/user5/.models/qwen3-omni-awq")
    ap.add_argument("--bench_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_seqs", type=int, default=8)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.75)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv_cache_dtype", default="fp8")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    return ap.parse_args()

def main():
    args = parse_args()
    output_csv = args.out
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    bench = load_bench(args.bench_dir)

    llm = init_llm(
        args.model,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_mem,
        tensor_parallel_size=args.tp,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
        max_num_seqs=args.max_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
        trust_remote_code=True,
    )
    sampling = init_sampling(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    processor = load_processor(args.model)

    total_batches = max(1, (len(bench) + args.batch_size - 1) // args.batch_size)
    pbar = tqdm(total=total_batches)

    need_header = not output_csv.exists() or output_csv.stat().st_size == 0
    is_first = need_header
    writer = None

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        for batch in iter_batches(bench, args.batch_size):
            reqs: list[dict[str, Any]] = []
            keys: list[str] = []
            ref_words_list: list[list[str]] = []
            ref_texts: list[str] = []

            # Формируем подсказки единообразно (build_inputs общий)
            for b in batch:
                ref_text = b["text"]
                ref_words = extract_words(ref_text)
                input_transcribe = " ".join(ref_words)
                prompt_text = get_bench_prompt(input_transcribe)

                reqs.append(build_inputs(b["path"], prompt_text, processor))
                keys.append(b["key"])
                ref_words_list.append(ref_words)
                ref_texts.append(ref_text)

            t0 = time.perf_counter()
            gen_texts = generate_texts(llm, reqs, sampling)
            t1 = time.perf_counter()
            batch_time = t1 - t0

            for key, ref_words, ref_text, gen_text in zip(keys, ref_words_list, ref_texts, gen_texts):
                gen_words = extract_words(gen_text)
                word_acc = list_accuracy(gen_words, ref_words)

                punct_overall_acc, punct_acc_per_type = punct_precision_metrics(ref_text, gen_text)

                res_dict = {
                    "key": key,
                    "res": gen_text,
                    "word_acc": word_acc,
                    "punct_acc_overall": punct_overall_acc,
                    "gen_time_sec": batch_time,
                }
                for p in PUNCTUATIONS:
                    slug = PUNCT_SLUG[p]
                    res_dict[f"punct_acc_{slug}"] = punct_acc_per_type[p]

                if is_first:
                    writer = csv.DictWriter(f, fieldnames=list(res_dict.keys()))
                    writer.writeheader()
                    is_first = False

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(res_dict.keys()))
                writer.writerow(res_dict)

            pbar.update(1)

    print("Готово:", args.out)

if __name__ == "__main__":
    main()
