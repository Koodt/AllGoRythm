import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../algorithms"))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../datasets"))
DATASET_JSON = os.path.join(DATASET_DIR, "train.json")

dataset = []


def generate_tags(file_path: str) -> list[str]:
    rel_path = os.path.relpath(file_path, BASE_DIR)
    parts = rel_path.split(os.sep)
    tags = parts[:-1]
    return tags


for root, _, files in os.walk(BASE_DIR):
    for f in files:
        if f.endswith((".go", ".py", ".cpp")):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as code_file:
                code_str = code_file.read()
            label = os.path.basename(root).capitalize()
            dataset.append(
                {
                    "code": code_str,
                    "tags": generate_tags(path),
                }
            )

os.makedirs(DATASET_DIR, exist_ok=True)
if dataset:
    with open(DATASET_JSON, "w", encoding="utf-8") as out_file:
        json.dump(dataset, out_file, indent=2)

    print(f"Dataset generated with {len(dataset)} examples.")
else:
    print(f"Dataset no generated.")
