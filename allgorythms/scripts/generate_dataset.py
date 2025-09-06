import os
import re
import json

import chardet
from nltk.stem import SnowballStemmer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../algorithms"))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../datasets"))
DATASET_JSON = os.path.join(DATASET_DIR, "train.json")
STOP_WORDS = {"to", "from", "is"}
stemmer = SnowballStemmer("english")

dataset = []


def camel_case_split(identifier: str) -> list[str]:
    matches = re.finditer(
        r".+?(?:(?<=\p{Ll})(?=\p{Lu})|(?<=\p{Lu})(?=\p{Lu}\p{Ll})|$)", identifier
    )
    return [m.group(0) for m in matches]

def generate_tags(file_path: str) -> list[str]:
    rel_path = os.path.relpath(file_path, BASE_DIR)
    parts = rel_path.split(os.sep)

    lang = parts[0].lower()
    dirs = [p.replace("-", "_").lower() for p in parts[1:-1]]
    filename = os.path.splitext(parts[-1])[0].replace("-", "_")

    file_tags = []
    for part in re.split(r"[_]+", filename):
        if not part or part.lower() in STOP_WORDS:
            continue
        words = re.findall(r'[A-Z][a-z]*|[a-z]+|\d+', part)
        stemmed = [stemmer.stem(w.lower()) for w in words if w.lower() not in STOP_WORDS]
        if stemmed:
            file_tags.append(" ".join(stemmed))

    all_tags = dirs + file_tags
    filtered_tags = []
    seen = set()
    for t in all_tags:
        if t and t not in seen:
            filtered_tags.append(t)
            seen.add(t)

    return [lang] + filtered_tags


for root, _, files in os.walk(BASE_DIR):
    for f in files:
        if f.endswith((".go", ".py", ".cpp")):
            path = os.path.join(root, f)
            try:
                raw = open(path, "rb").read()
                encoding = chardet.detect(raw)["encoding"] or "utf-8"
                code_str = raw.decode(encoding, errors="ignore")
            except Exception as e:
                print(f"Failed to read {path}: {e}")
                continue
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
