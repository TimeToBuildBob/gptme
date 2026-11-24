"""Edge-case repros for gptme/context/ (adaptive_compressor, compress, task_analyzer, selector, scout)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from gptme.codeblock import Codeblock
from gptme.context.adaptive_compressor import AdaptiveCompressor, extractive_compress
from gptme.context.compress import strip_reasoning
from gptme.context.selector.base import ContextItem
from gptme.context.selector.config import ContextSelectorConfig
from gptme.context.selector.rule_based import RuleBasedSelector
from gptme.context.task_analyzer import (
    _extract_prompt_signals,
    classify_task,
    extract_features,
)
from gptme.message import Message


# --- A: greedy selection `break` drops everything after one oversized top sentence
def test_extractive_compress_long_top_sentence_drops_all_prose():
    first = "ERROR: " + "the build failed because of a missing dependency " * 20 + "."
    rest = " ".join(f"Short note number {i} here." for i in range(10))
    content = first + " " + rest
    out = extractive_compress(content, target_ratio=0.5)
    # Budget is ~50% of the input; plenty of the short sentences fit.
    assert len(out) > 0.2 * len(content), f"kept {len(out)}/{len(content)} chars: {out!r}"


# --- B: code block fence glued onto the previous sentence (not at line start)
def test_extractive_compress_keeps_fence_at_line_start():
    prose = " ".join(f"This is filler sentence {i} about nothing." for i in range(12))
    content = prose + "\n```python\nprint('hi')\n```\nDone."
    assert len(Codeblock.iter_from_markdown(content)) == 1
    out = extractive_compress(content, target_ratio=0.5)
    assert "print('hi')" in out
    blocks = Codeblock.iter_from_markdown(out)
    assert len(blocks) == 1, f"code block no longer parseable: {out[-80:]!r}"


# --- C: nested fences (4-backtick outer, 3-backtick inner) corrupt the code
def test_extractive_compress_preserves_nested_fence_body():
    prose = " ".join(f"This is filler sentence {i} about nothing." for i in range(12))
    block = (
        "````markdown\n"
        "# Usage\n"
        "```python\n"
        "# Load the config. Validate it. Apply defaults. Then run.\n"
        "run()\n"
        "```\n"
        "````"
    )
    content = prose + "\n" + block + "\n" + prose
    out = extractive_compress(content, target_ratio=0.3)
    assert block in out, f"code block body corrupted; got tail: {out[-300:]!r}"


# --- D: AdaptiveCompressor crashes on a content string > NAME_MAX
def test_adaptive_compressor_long_content_string_does_not_crash():
    c = AdaptiveCompressor(enable_logging=False)
    content = "x" * 300 + ". More text. Even more. And more. Last one."
    # docstring: context_files may be "file paths or content strings"
    result = c.compress("Fix the bug", context_files=[content])
    assert result.original_content == content


# --- E: strip_reasoning mutates fenced code (removes tags inside code, collapses blank lines)
def test_strip_reasoning_leaves_code_blocks_intact():
    code = (
        "```python\n"
        "PATTERN = r'<think>.*?</think>'\n"
        "\n"
        "\n"
        "\n"
        "def f():\n"
        "    return PATTERN\n"
        "```"
    )
    content = "<think>let me plan</think>\nHere is the code:\n" + code
    stripped, saved = strip_reasoning(content)
    assert saved > 0
    assert code in stripped, f"code block altered: {stripped!r}"


# --- F: classification confidence is always 1.0 after max-normalisation
def test_classify_task_no_evidence_is_not_full_confidence():
    features = extract_features(prompt="")
    cls = classify_task(features)
    assert cls.confidence < 1.0, (
        f"empty prompt, no files -> {cls.primary_type} with confidence {cls.confidence}"
    )


# --- G: substring keyword matching gives false task signals
def test_prompt_signals_no_substring_false_positives():
    signals = _extract_prompt_signals("Please debug the padding")
    # "bug" is a substring of "debug", "add" of "padding"
    assert signals["fix_score"] == 0.0 and signals["implementation_score"] == 0.0, signals


@dataclass
class _Item(ContextItem):
    _id: str
    _meta: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self._id

    @property
    def metadata(self) -> dict[str, Any]:
        return self._meta

    @property
    def identifier(self) -> str:
        return self._id


# --- H: `tools` as a string is iterated character-by-character
def test_rule_selector_tools_string_not_iterated_by_char():
    sel = RuleBasedSelector(ContextSelectorConfig())
    item = _Item("lesson-shell", {"tools": "shell"})
    # query mentions a different tool; no "shell" anywhere
    out = sel.select("tool:python please help", [item])
    assert out == [], "lesson with tools='shell' matched a python query via single chars"


# --- I: non-hashable priority crashes the rule selector
def test_rule_selector_list_priority_does_not_crash():
    sel = RuleBasedSelector(ContextSelectorConfig())
    item = _Item("lesson-a", {"keywords": ["deploy"], "priority": ["high"]})
    out = sel.select("deploy the app", [item])
    assert out == [item]


# --- J: scout duplicates/junk lines consume the _MAX_SCOUT_FILES cap
def test_scout_duplicate_lines_do_not_consume_cap(tmp_path: Path):
    from gptme.context import scout

    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 2\n")
    response = Message("assistant", "\n".join(["a.py"] * 20 + ["b.py"]))
    with (
        patch.object(scout, "_build_file_tree", return_value="a.py\nb.py"),
        patch("gptme.llm.reply", return_value=response),
    ):
        files = scout.scout_files("x", tmp_path, "test/model")
    names = sorted(f.path.name for f in files)
    assert names == ["a.py", "b.py"], names
