import csv
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.csv as pa_csv

EXPECTED_FIELDS = {"audio_path", "text", "speaker_name", "language"}


def read_done_audio_paths(out_csv: Path) -> set[str]:
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


class DatasetLoader:
    def __init__(self, dataset_dir: Path, batch_size: int, skip_rel_paths: set[str]):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.skip_rel_paths = skip_rel_paths
        self.total_pairs = 0

        meta_path = self.dataset_dir / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Не найден metadata.csv по пути: {meta_path}")

        self._reader = pa_csv.open_csv(
            meta_path,
            read_options=pa_csv.ReadOptions(block_size=10_000_000),
            parse_options=pa_csv.ParseOptions(delimiter="|"),
        )

        schema_fields = set(self._reader.schema.names)
        if len(schema_fields & EXPECTED_FIELDS) < len(EXPECTED_FIELDS):
            raise ValueError(f"metadata.csv должен содержать колонки: {EXPECTED_FIELDS}")

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        buffer: list[dict[str, Any]] = []

        for record_batch in self._reader:
            columns = {name: record_batch.column(name) for name in record_batch.schema.names}
            num_rows = record_batch.num_rows

            text_accent_col = columns.get("text_accent")
            if text_accent_col is None:
                text_accent_col = pa.array([None] * num_rows)

            speaker_col = columns.get("speaker_name")

            for row_idx in range(num_rows):
                rel_audio = columns["audio_path"][row_idx].as_py()
                text = columns["text"][row_idx].as_py()
                text_accent = text_accent_col[row_idx].as_py()
                speaker_name = speaker_col[row_idx].as_py() if speaker_col is not None else None
                language = columns["language"][row_idx].as_py()

                if not rel_audio or not text or not language:
                    continue
                if rel_audio in self.skip_rel_paths:
                    continue

                abs_audio = (self.dataset_dir / rel_audio).resolve()
                buffer.append(
                    {
                        "abs_audio": str(abs_audio),
                        "rel_audio": rel_audio,
                        "text": text,
                        "text_accent": text_accent,
                        "speaker_name": speaker_name,
                        "language": language,
                    }
                )
                self.total_pairs += 1

                if len(buffer) >= self.batch_size:
                    yield buffer
                    buffer = []

        if buffer:
            yield buffer
