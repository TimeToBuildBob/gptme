"""Model metadata / resolution defects (models lens).

Run from the checkout root:
  .venv/bin/python -m pytest /tmp/bob-scratch/9cd9/models/test_models_lens.py -q -p no:cacheprovider
"""

from collections.abc import Iterator

import pytest

from gptme.config import Config, UserConfig, get_config, set_config
from gptme.llm.models import get_model
from gptme.util.context_budget import get_context_budget


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in (
        "GPTME_MAX_TOKENS",
        "GPTME_CONTEXT_BUDGET",
        "GPTME_THINKING_EFFORT",
        "GPTME_CONTEXT_LENGTH",
    ):
        monkeypatch.delenv(var, raising=False)
    previous = get_config()
    set_config(Config(user=UserConfig()))
    try:
        yield
    finally:
        set_config(previous)


def _autocompact_budget(full: str) -> int:
    """Same call autocompact/decision.py and hook.py make for the default model."""
    m = get_model(full)
    return get_context_budget(
        m.context,
        max_output=m.max_output or 8192,
        model_id=m.full,
        model_context_budget=m.context_budget,
    )


# --- 1. OpenAI dated snapshot ids inherit GPT-6 metadata ---------------------


@pytest.mark.parametrize(
    ("snapshot", "base"),
    [
        ("gpt-4o-2024-11-20", "gpt-4o"),
        ("gpt-4.1-2025-04-14", "gpt-4.1"),
        ("gpt-5-mini-2025-08-07", "gpt-5-mini"),
    ],
)
def test_openai_dated_snapshot_resolves_to_its_base_model(snapshot, base):
    snap = get_model(f"openai/{snapshot}")
    ref = get_model(f"openai/{base}")
    assert (
        snap.context,
        snap.max_output,
        snap.supports_reasoning,
        snap.supports_responses_api,
        snap.price_input,
        snap.price_output,
    ) == (
        ref.context,
        ref.max_output,
        ref.supports_reasoning,
        ref.supports_responses_api,
        ref.price_input,
        ref.price_output,
    )


def test_gpt4o_snapshot_compaction_budget_fits_its_128k_window():
    # gpt-4o-2024-11-20 is a 128k model; a budget above the window means the
    # API rejects the request before compaction ever fires.
    assert _autocompact_budget("openai/gpt-4o-2024-11-20") < 128_000


def test_gpt4o_snapshot_gets_no_reasoning_effort(monkeypatch):
    from gptme.llm.llm_openai import _resolve_reasoning_effort

    monkeypatch.setenv("GPTME_THINKING_EFFORT", "high")
    meta = get_model("openai/gpt-4o-2024-11-20")
    # gpt-4o rejects reasoning.effort with HTTP 400.
    assert _resolve_reasoning_effort("openai", meta) is None


# --- 2. max_output == context collapses the compaction budget to 1000 --------


@pytest.mark.parametrize(
    "full",
    [
        "moonshot/kimi-k3",
        "moonshot/kimi-k2.6",
        "xai/grok-4",
        "xai/grok-3",
        "openai-subscription/gpt-5.3-codex-spark",
    ],
)
def test_window_sized_max_output_does_not_collapse_compaction_budget(full):
    # These providers get no max_tokens on the wire (_resolve_max_tokens -> None),
    # so the advertised max_output is not a real reservation. A 1000-token
    # budget is smaller than the system prompt: compaction fires every turn.
    m = get_model(full)
    assert _autocompact_budget(full) >= m.context // 4


# --- 3. Bare model names pick a subscription/gateway over the vendor API -----


@pytest.mark.parametrize(
    ("bare", "expected_provider"),
    [
        ("grok-4.6", "xai"),
        ("deepseek-v4-pro", "deepseek"),
        # control: openai already wins over openai-subscription by dict order
        ("gpt-5.5", "openai"),
    ],
)
def test_bare_model_name_resolves_to_vendor_api(bare, expected_provider):
    assert get_model(bare).provider == expected_provider


# --- 4. Static data (needs doc check) -----------------------------------------


def test_gemini_25_pro_long_context_price_tier():
    # needs doc check: https://ai.google.dev/gemini-api/docs/pricing lists
    # Gemini 2.5 Pro at $1.25/$10 for prompts <=200k and $2.50/$15 above.
    # data.py even notes ">200k context price is 2x/1.5x" but sets no tier.
    m = get_model("gemini/gemini-2.5-pro")
    assert m.effective_prices(300_000)[:2] == (2.5, 15)


def test_gpt4o_price_matches_openai_pricing_page():
    # needs doc check: https://platform.openai.com/docs/pricing lists gpt-4o
    # (alias of gpt-4o-2024-08-06) at $2.50 in / $10.00 out per 1M.
    # $5/$15 is the gpt-4o-2024-05-13 snapshot price.
    m = get_model("openai/gpt-4o")
    assert (m.price_input, m.price_output) == (2.5, 10)
