import argparse

import csv
from pathlib import Path
from typing import Any

import jiwer
from tqdm import tqdm
from ruaccent import RUAccent

from utils import iter_batches
from prompts import get_prompt
from mllm_utils import init_llm, init_sampling, load_processor, build_inputs, generate_texts

EXPECTED_FIELDS = set(["audio_path", "text", "speaker_name", "language"])

##################
# Helper: resume #
##################

def read_done_audio_paths(out_csv: Path) -> set[str]:
    """Считывает существующий out CSV и возвращает множество уже обработанных audio_path."""
    done: set[str] = set()

    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return done
    
    with out_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return done
        
        for row in reader:
            ap = (row.get("audio_path") or "").strip()

            if ap:
                done.add(ap)

    return done


def read_csv_header(csv_path: Path) -> list[str] | None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)

        if header and any(h.strip() for h in header):
            return header
        
    return None


##############
# DATA IO    #
##############

def load_dataset(dataset_dir: Path, skip_rel_paths: set[str]) -> tuple[list[dict[str, Any]], int]:
    """
    Ожидается dataset_dir / metadata.csv с колонками:
      - audio_path (RELATIVE путь)
      - text
      - text_accent (опционально)
      - speaker_name
      - language

    skip_rel_paths — множество относительных путей (как в metadata.csv), которые нужно пропустить (resume).

    Возвращает:
      data: [{"speaker_name": <str|None>, "abs_audio": <abs str>, "rel_audio": <rel str>, "text": <str>, "text_accent": <str|None>, "language": <str>}, ...]
      overall_len: общее число НЕпропущенных пар
    """
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Не найден metadata.csv по пути: {meta_path}")

    data = []
    overall_len = 0

    with meta_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        fieldnames = reader.fieldnames or []

        if len(set(fieldnames) & EXPECTED_FIELDS) < len(EXPECTED_FIELDS):
            raise ValueError(f"metadata.csv должен содержать колонки: {EXPECTED_FIELDS}")

        for row in reader:
            rel_audio = row.get("audio_path")
            text = row.get("text")
            text_accent = row.get("text_accent")
            speaker_name = row.get("speaker_name")
            language = row.get("language")

            if not rel_audio or not text or not language:
                continue

            if rel_audio in skip_rel_paths:
                continue

            abs_audio = (dataset_dir / rel_audio).resolve()
            data.append({
                "abs_audio": str(abs_audio), "rel_audio": rel_audio, "text": text, "text_accent": text_accent,
                "speaker_name": speaker_name, "language": language
            })

            overall_len += 1

    return data, overall_len


#########################
# MAIN / ARGPARSE / RUN #
#########################

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="/home/user5/.models/qwen3-omni-awq")
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt_version", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_seqs", type=int, default=8)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.75)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv_cache_dtype", default="fp8")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--no_accent", action="store_true", help="Отключить акцентуацию (ruaccent).")

    return ap.parse_args()


def main():
    args = parse_args()

    output_csv = args.out
    dataset_root = args.dataset_dir.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # RESUME: читаем уже готовые пути из out CSV
    already_done = read_done_audio_paths(output_csv)

    # Грузим датасет, сразу выкидывая уже готовые элементы
    data, total_pairs = load_dataset(dataset_root, already_done)

    # Если нечего обрабатывать — выходим аккуратно
    if total_pairs == 0:
        print(f"Нечего обрабатывать. Уже готово: {len(already_done)}")
        return

    # Строим стабильный заголовок: если CSV уже есть — используем его, иначе берём супермножество
    existing_header = read_csv_header(output_csv)
    if existing_header is not None:
        fieldnames = existing_header
        
    else:
        fieldnames = ["audio_path", "base_text", "out_text", "base_accent_text", "out_accent_text", "speaker_name", "language", "cer"]

    # Инициализация ruaccent (основной)
    if not args.no_accent:
        accentizer = RUAccent()
        accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False, device="CUDA")
        print("Loaded accentizer!")

    # LLM / сэмплинг / процессор
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

    # Прогресс считаем в строках (а не батчах)
    already_len = len(already_done)
    total_rows_to_process = total_pairs
    pbar = tqdm(total=already_len + total_rows_to_process, initial=already_len, desc="rows")

    need_header = existing_header is None

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if need_header:
            writer.writeheader()

        for batch in iter_batches(data, args.batch_size):
            batch_cer = 0
            texts = [(p["text"] or "").strip() for p in batch]

            reqs = []
            for p, t in zip(batch, texts):
                prompt_text = get_prompt(t, args.prompt_version)
                reqs.append(build_inputs(p["abs_audio"], prompt_text, processor))

            gen_texts = generate_texts(llm, reqs, sampling)  # type: ignore

            for gen_text, p, base_text in zip(gen_texts, batch, texts):
                try:
                    rel_path = p["rel_audio"]

                    if not args.no_accent:
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

                    # Строим строку и фильтруем под fieldnames
                    row_full = {
                        "audio_path": rel_path,
                        "base_text": base_text,
                        "out_text": gen_text,
                        "base_accent_text": base_accent_text,
                        "out_accent_text": out_accent_text,
                        "speaker_name": (p["speaker_name"] or ""),
                        "language": p["language"],
                        "cer": cer
                    }
                    row = {k: row_full.get(k, "") for k in fieldnames}
                    writer.writerow(row)

                except Exception as e:
                    print(f"SKIP ROW DUE TO: {e}\n")

            batch_cer /= args.batch_size
            # апдейтим на количество обработанных СТРОК
            pbar.update(len(batch))
            pbar.set_postfix_str(f"cer: {batch_cer:.4f}")

    print("Готово:", args.out)

if __name__ == "__main__":
    main()
