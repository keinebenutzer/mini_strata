"""The Monitor's CTk row must list only the --kv widths the running engine's own parser takes.

The menu in terminal/setup.py is the MAIN tree's whole set; the q36 runner of the small machines
refuses four of them.  This test reads the engine's --kv block and compares it with the table in
serve/kv_options.py, so a new engine width or a renamed word shows up here instead of as a row in
the WebGUI that the engine answers with exit 2.

    python -m pytest serve/test_kv_options_engine_filter.py
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from serve import kv_options

Q36 = Path(__file__).resolve().parents[2] / "engine" / "src" / "qwen35" / "q36_engine.cu"


def parser_branches() -> list[tuple[str, list[str]]]:
    """(the kv_mode, the words that set it) of the engine's own --kv block."""
    src = Q36.read_text(encoding="utf-8")
    start = src.index('a == "--kv"')
    end = src.index("(not %s)", start)
    block = src[start:end]
    out = []
    for line in block.splitlines():
        m = re.search(r"kv_mode = (KV_\w+)", line)
        if m:
            out.append((m.group(1), re.findall(r'v == "([^"]+)"', line)))
    return out


def shown_for(arch: str | None) -> list:
    opts = kv_options.options_for(arch)
    assert opts is not None, f"the menu could not be read for arch {arch!r}"
    return opts["choices"]


class KvOptionsEngineFilter(unittest.TestCase):
    def test_table_covers_every_branch_of_the_engine_parser(self):
        branches = parser_branches()
        self.assertGreaterEqual(len(branches), 5, "the --kv block was not found in the engine source")
        taken = kv_options.ENGINE_KV_TAKEN["qwen35moe"]
        for mode, words in branches:
            self.assertTrue(any(w in taken for w in words),
                            f"the engine sets {mode} from {words}, none of them in the table {taken}")

    def test_table_names_nothing_the_engine_refuses(self):
        accepted = {w for _, words in parser_branches() for w in words}
        for w in kv_options.ENGINE_KV_TAKEN["qwen35moe"]:
            self.assertIn(w, accepted, f"{w} is offered but the q36 parser does not take it")

    def test_menu_words_the_engine_refuses_are_dropped(self):
        _, choices = kv_options.setup_menu()
        shown = shown_for("qwen35moe")
        refused = set(choices) - set(kv_options.ENGINE_KV_TAKEN["qwen35moe"])
        self.assertTrue(refused, "the menu and the engine agree; if that ever changes, look here")
        for w in refused:
            self.assertNotIn(w, shown, f"the row still offers {w}, which the engine refuses")
        self.assertEqual(set(shown), set(choices) & set(kv_options.ENGINE_KV_TAKEN["qwen35moe"]))

    def test_an_arch_without_a_table_keeps_the_whole_menu(self):
        _, choices = kv_options.setup_menu()
        self.assertEqual(shown_for("qwen4exp"), choices)
        self.assertEqual(shown_for(None), choices)

    def test_configured_kv_is_read_from_the_engine_start_line(self):
        class Eng:
            spawn = ("strata", ["--native", "m.gguf", "--kv", "turbo3", "--max-context", "51536"], None)

        class Eq:
            spawn = ("strata", ["--native", "m.gguf", "--kv=q4_0"], None)

        class None_:
            spawn = ("strata", ["--native", "m.gguf"], None)

        self.assertEqual(kv_options.configured_kv(Eng()), "turbo3")
        self.assertEqual(kv_options.configured_kv(Eq()), "q4_0")
        self.assertIsNone(kv_options.configured_kv(None_()))


if __name__ == "__main__":
    unittest.main()
