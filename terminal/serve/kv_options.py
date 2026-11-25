"""The CTk possibilities (--kv) that the Monitor and the Terminal show.

terminal/setup.py's menu lists the MAIN tree's whole set.  A family runner's own parser takes far
fewer: the q36 runner (arch "qwen35moe", the engine of the small machines) answers anything else
with "strata-q36: --kv takes fp32, fp16, fp8, fp4 or turbo3" and ends with 2.  A Monitor that
lists the whole menu therefore offers widths the running engine refuses - so the menu is read here
and narrowed to the family that is actually running.

serve/test_kv_options_engine_filter.py checks ENGINE_KV_TAKEN against the engine source's own
--kv parser, so the table cannot drift away from the engine.
"""
from __future__ import annotations

import ast
from pathlib import Path

# arch -> the --kv words this family's runner takes, in the spelling setup.py's menu uses.
# An arch that is not listed is not narrowed: its runner takes the menu's whole set.
ENGINE_KV_TAKEN = {
    # engine/src/qwen35/q36_engine.cu: fp32|f32, fp16|f16|half, fp8|int8|q8|q8_0, fp4|q4|q4_0,
    # turbo3.  The menu's turbo2, k3v8, k8v4 and k8v8 are main-tree widths this runner does not have;
    # fp32 is the engine's own default and is not a word of the menu, so it never appears in the row.
    "qwen35moe": ("fp32", "int8", "q4_0", "fp16", "fp8", "turbo3"),
}

SETUP = Path(__file__).resolve().parent.parent / "setup.py"


def setup_menu() -> tuple[dict, list]:
    """KV_WORDS and the --kv choices read out of terminal/setup.py - not a hardcoded copy."""
    tree = ast.parse(SETUP.read_text(encoding="utf-8"), filename=str(SETUP))
    words, choices = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KV_WORDS" for t in node.targets):
            try:
                words = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001 - a dict that cannot be read leaves the row out, not broken
                pass
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == "add_argument" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--kv":
                for kw in node.keywords:
                    if kw.arg == "choices":
                        try:
                            choices = ast.literal_eval(kw.value)
                        except Exception:  # noqa: BLE001
                            pass
    return words, choices


def options_for(arch: str | None) -> dict | None:
    """The menu's words and choices, kept to the ones the engine of this arch accepts."""
    words, choices = setup_menu()
    taken = ENGINE_KV_TAKEN.get(arch) if arch else None
    if taken:
        choices = [c for c in choices if c in taken]
        words = {k: v for k, v in words.items() if k in taken}
    return {"words": words, "choices": choices, "arch": arch} if (words or choices) else None


def configured_kv(engine) -> str | None:
    """The --kv the run config handed the engine, for a runner that reports none in its INFO lines
    (the q36 engine prints only batch_slots, so the Monitor's KV row would otherwise stay empty)."""
    args = (getattr(engine, "spawn", None) or [None, None])[1] or []
    for i, a in enumerate(args):
        a = str(a)
        if a == "--kv" and i + 1 < len(args):
            return str(args[i + 1])
        if a.startswith("--kv="):
            return a.split("=", 1)[1]
    return None
