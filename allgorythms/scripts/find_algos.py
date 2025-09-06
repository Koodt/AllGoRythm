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


def extract_cpp_functions_with_classes(code: str):
    funcs = []
    class_stack = []
    brace_stack = []

    lines = code.splitlines()
    for lineno, line in enumerate(lines):
        line_strip = line.strip()

        class_match = re.match(r'\b(class|struct)\s+(\w+)', line_strip)
        if class_match:
            class_stack.append(class_match.group(2))

        if '{' in line_strip:
            brace_stack.append('{')

        if '}' in line_strip and brace_stack:
            brace_stack.pop()
            if class_stack and not brace_stack:
                class_stack.pop()

        func_match = re.match(
            r'\b(?:void|int|float|double|char|bool|string)\s+(\w+)\s*\(([^)]*)\)\s*\{',
            line_strip
        )
        if func_match:
            name = func_match.group(1)
            params = func_match.group(2)
            start_line = lineno + 1
            snippet_lines = lines[start_line-1:start_line+20]
            snippet = "\n".join(snippet_lines)
            full_name = '.'.join(class_stack + [name]) if class_stack else name
            funcs.append((full_name, params, start_line, snippet))

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
    elif url.endswith(".cpp"):
        logger.info(f"Extract CPP code ...")
        funcs = extract_cpp_functions_with_classes(code)
    else:
        logger.info(f"Extract GO code ...")
        funcs = extract_functions_regex(code)

    if not funcs:
        logger.info("No functions detected.")

    for name, start, end, func_code in funcs:
        X = vectorizer.transform([func_code])
        pred_lang = clf_lang.predict(X)[0]
        y_pred = clf_tags.predict(X)
        tags_list = mlb.inverse_transform(y_pred)

        pred_tags = list(tags_list[0]) if tags_list else []

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
