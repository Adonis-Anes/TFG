import os
import re
from collections import defaultdict

BASE_DIR = os.path.join("data", "eeg")
RAW_DIR = "raw"
UNPROCESSED_DIR = "unprocessed"
OUTPUT_FILE = "task_run_summary.txt"

# Regex para extraer task y run del nombre del archivo
FILENAME_PATTERN = re.compile(r"sub-(\d+)_task-([a-zA-Z0-9\-]+)_run-(\d+)_")

def extraer_task_runs(base_dir):
    resumen = defaultdict(lambda: defaultdict(list))

    for subject in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject, RAW_DIR, UNPROCESSED_DIR)
        if not os.path.isdir(subject_path):
            continue

        for fname in os.listdir(subject_path):
            match = FILENAME_PATTERN.search(fname)
            if match:
                sub_id, task, run = match.groups()
                resumen[sub_id][task].append(run)

    return resumen

def escribir_resumen(resumen, output_file):
    with open(output_file, "w") as f:
        for sub_id in sorted(resumen.keys()):
            f.write(f"sub: {sub_id}\n")
            for task in sorted(resumen[sub_id].keys()):
                runs = sorted(set(resumen[sub_id][task]))
                runs_str = ",".join(runs)
                f.write(f"task: {task} | runs: {runs_str}\n")
            f.write("\n")

if __name__ == "__main__":
    resumen = extraer_task_runs(BASE_DIR)
    escribir_resumen(resumen, OUTPUT_FILE)
    print(f"\nResumen guardado en: {OUTPUT_FILE}")
