import os

import sys
import subprocess
from pathlib import Path

IN_PATH = Path("/home/user5/espeech4")
DEVICE = "4"
BATCH_SIZE = 1024

SKIP_DATASETS = ["part-2", "part-1", "part0", "part1", "part2", "part3", "part4", "part5", "part6"]

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = DEVICE

all_datasets_dirs = [p for p in IN_PATH.iterdir()]

for dataset_dir in all_datasets_dirs:
    if dataset_dir.name in SKIP_DATASETS or not dataset_dir.is_dir():
        continue
    
    print("\n\n\n**********************************\n")
    print(f"[RUN_MANY] Now running {dataset_dir.name}")
    print("\n**********************************\n\n\n")

    subprocess.run([
        sys.executable,
        "repunct.py",
        "--dataset_dir", str(dataset_dir),
        "--out", str(IN_PATH / f"{dataset_dir.name}.csv"),
        "--batch_size", str(BATCH_SIZE)
    ], env=env, check=True)
