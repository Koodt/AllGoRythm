import logging
import os
import re
import sys

import ast
import pickle
import requests
from textwrap import indent

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(THIS_DIR, "../model/algo_recognizer.pkl")

def github_url_validate(url: str) -> str:
    logger.info(f"Fetching {url} ...")
    if not ("github.com" in url or "/blob/" in url):
        msg = "Not a valid GitHub URL"
        logger.error(msg)
        raise ValueError(msg)
    return url.replace(
        "github.com", "raw.githubusercontent.com"
    ).replace("/blob/", "/")


def extract_functions_python(code: str) -> list[str]:
    funcs =[]
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno
            end_line = max(
                getattr(n, "lineno", start_line) for n in ast.walk(node)
            )
            func_code = "\n".join(code.splitlines()[start_line - 1:end_line])
            funcs.append((node.name, start_line, end_line, func_code))

    return funcs


def extract_functions_regex(code: str) -> list[str]:
    funcs = []
    pattern = r"(?:func\s+|void\s+|int\s+|float\s+|double\s+|char\s+)(\w+)\s*\([^)]*\)\s*{"

    for match in re.finditer(pattern, code):
        name = match.group(1)
        start = code[: match.start()].count("\n") + 1
        snippet = "\n".join(code.splitlines()[start - 1:start + 20])
        funcs.append((name, start, start + 20, snippet))

    return funcs


def main():
    if len(sys.argv) < 2:
        logger.info("Usage: python find_algos.py <github_file_url>")
        sys.exit(1)

    url = github_url_validate(sys.argv[1])
    logger.info(f"GET request {url} ...")
    resp = requests.get(url)
    if resp.status_code != 200:
        logger.error(f"Failed to fetch file: {resp.status_code}")
        sys.exit(1)

    code = resp.text

    logger.info(f"Opening model {MODEL_PKL} ...")
    with open(MODEL_PKL, "rb") as f:
        model_data = pickle.load(f)

    vectorizer = model_data["vectorizer"]
    clf_lang = model_data["clf_lang"]
    clf_tags = model_data["clf_tags"]
    mlb = model_data["mlb"]

    if url.endswith(".py"):
        logger.info(f"Extract python code ...")
        funcs = extract_functions_python(code)
    else:
        logger.info(f"Extract GO or CPP code ...")
        funcs = extract_functions_regex(code)

    if not funcs:
        logger.info("No functions detected.")

    for name, start, end, func_code in funcs:
        X = vectorizer.transform([func_code])
        pred_lang = clf_lang.predict(X)[0]
        pred_tags = mlb.inverse_transform(clf_tags.predict(X))[0]

        if pred_tags:
            print("=" * 60)
            print(f"Function: {name}")
            print(f"Lines: {start}-{end}")
            print(f"Predicted language: {pred_lang}")
            print(f"Predicted tags: {pred_tags}")
            print("\n" + indent(func_code, "    "))

if __name__ == "__main__":
    logger.info("Programm started.")
    main()
    logger.info("Programm end.")
