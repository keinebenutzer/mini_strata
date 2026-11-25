#!/usr/bin/env python3
"""Read strata_src/terminal/setup.py and print the CTk options (KV_WORDS and --kv choices) as JSON.

Used by the native host to expose the CTk options to the web UI without hardcoding them.
Reads KV_WORDS (a dict literal) and the ``--kv`` add_argument choices list from setup.py
via the ast module, then prints {"kv_words": {...}, "kv_choices": [...]}.
"""
import ast
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"kv_words": {}, "kv_choices": []}))
        return
    repoPath = Path(sys.argv[1])
    setupPath = None
    for candidate in (repoPath / "setup.py", repoPath / "terminal" / "setup.py"):
        if candidate.exists():
            setupPath = candidate
            break
    if setupPath is None:
        print(json.dumps({"error": f"setup.py not found under {repoPath}"}))
        return
    try:
        with open(setupPath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(setupPath))
    except Exception as e:  # noqa: BLE001 - surface any read/parse error as JSON
        print(json.dumps({"error": str(e)}))
        return

    kv_words = {}
    kv_choices = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "KV_WORDS":
                    try:
                        kv_words = ast.literal_eval(node.value)
                    except Exception:  # noqa: BLE001
                        pass
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "add_argument":
                # The first positional arg is the flag string, e.g. "--kv".
                flag = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                if flag == "--kv":
                    kw = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                    if kw.get("choices") is not None:
                        try:
                            kv_choices = ast.literal_eval(kw["choices"])
                        except Exception:  # noqa: BLE001
                            pass

    print(json.dumps({"kv_words": kv_words, "kv_choices": kv_choices}))


if __name__ == "__main__":
    main()
