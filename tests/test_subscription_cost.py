"""A subscription login bills nothing: its calls are priced at list and
shown as an estimate, `(sub)`, beside what a search was actually billed."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.harness.budget import format_cost, journal_spend, on_subscription
from hillclimb.harness.candidate import AgentInfo, Candidate
from hillclimb.harness.journal import Journal
from hillclimb.harness.pricing import model_prices
from hillclimb.tui.watch import _agent_cost

CATALOGUE = {
    "data": [
        {"id": "openai/gpt-5.1-codex", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        {"id": "anthropic/claude-sonnet-4.5", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
    ]
}


@pytest.mark.parametrize(
    ("total", "sub", "shown"),
    [
        (0.0, 0.0, "-"),
        (None, 0.0, "-"),
        (1.6, 0.0, "$1.60"),
        (1.6, 1.6, "$1.60 (sub)"),
        (1.6, 1.2, "$0.40 + $1.20 (sub)"),
    ],
)
def test_format_cost(total, sub, shown):
    assert format_cost(total, sub) == shown


def test_journal_spend_splits_the_estimate_from_the_bill(tmp_path: Path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.candidate_result(Candidate(candidate_id="c001", operator="draft", agent=AgentInfo(cost_usd=1.2, subscription=True)))
    journal.candidate_result(Candidate(candidate_id="c002", operator="draft", agent=AgentInfo(cost_usd=0.4, subscription=False)))
    # a journal written before the flag: counted, never called an estimate
    journal.candidate_result(Candidate(candidate_id="c003", operator="draft", agent=AgentInfo(cost_usd=0.1)))

    spend = journal_spend(journal)
    assert spend.cost_usd == pytest.approx(1.7)
    assert spend.subscription_cost_usd == pytest.approx(1.2)
    assert format_cost(spend.cost_usd, spend.subscription_cost_usd) == "$0.50 + $1.20 (sub)"


def test_on_subscription_reads_the_agent_login():
    class Agent:
        auth = "subscription"

    assert on_subscription(Agent())
    Agent.auth = "api-key"
    assert not on_subscription(Agent())
    assert not on_subscription(object())  # dummy, toy, a registered agent without a login


def test_agent_cost_cell():
    assert _agent_cost(AgentInfo(cost_usd=0.42, subscription=True)) == "$0.42 (sub)"
    assert _agent_cost(AgentInfo(cost_usd=0.42, subscription=False)) == "$0.42"
    assert _agent_cost(AgentInfo()) == "-"


@pytest.mark.parametrize(
    ("model", "prompt"),
    [
        ("openai/gpt-5.1-codex", 1e-6),  # an OpenRouter id, exactly
        ("gpt-5.1-codex", 1e-6),  # what a ChatGPT login runs
        ("claude-sonnet-4-5-20250929", 3e-6),  # a provider id: dated, dashes for dots
        ("gpt-4o", None),  # not in the catalogue: no invented number
    ],
)
def test_model_prices_match_provider_ids(model, prompt):
    prices = model_prices(model, catalogue=CATALOGUE)
    assert (prices.prompt if prices else None) == prompt


def test_an_ambiguous_bare_id_is_not_priced():
    catalogue = {"data": [*CATALOGUE["data"], {"id": "azure/gpt-5.1-codex", "pricing": {"prompt": "0.000009"}}]}
    assert model_prices("gpt-5.1-codex", catalogue=catalogue) is None
