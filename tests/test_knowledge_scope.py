"""Whose knowledge reaches a search: `learning.scope`, scripted searches that
teach nothing, and `hillclimb knowledge prune` for what an older search left."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hillclimb.api import scripted_only
from hillclimb.cli import common
from hillclimb.config import RouteConfig
from hillclimb.harness.candidate import AgentInfo, Candidate
from hillclimb.harness.journal import Journal
from hillclimb.modules.memory.credit import CreditEvent, credit_event_path, write_credit_event
from hillclimb.modules.memory.knowledge import (
    KnowledgeCard,
    claim_in_scope,
    distill_card,
    iter_cards,
    load_cards,
    load_live_cards,
    write_card,
    write_live_card,
)
from hillclimb.modules.memory.skills import SKILL_CODE_FILENAME, SKILL_META_FILENAME, load_skills
from tests.factories import trial as mk_trial


def card(problem_id: str, run_ref: str, *, family: str = "suite", agents=("claude-code",), key: str = "") -> KnowledgeCard:
    return KnowledgeCard(
        problem_id=problem_id, family=family, run_ref=run_ref, agents=list(agents), problem_key=key,
        selected_operator="improve", n_ok=1,
    )


def seed(knowledge_dir: Path) -> None:
    write_card(knowledge_dir, card("suite-solar", "r1/solar"))
    write_card(knowledge_dir, card("suite-wind", "r1/wind"))
    write_card(knowledge_dir, card("suite-solar", "r2/smoke", agents=("dummy",)))


class TestScope:
    @pytest.mark.parametrize(
        ("scope", "expected"),
        [
            ("all", {"r1/solar", "r1/wind"}),
            ("family", {"r1/solar", "r1/wind"}),
            ("problem", {"r1/solar"}),
        ],
    )
    def test_cards(self, tmp_path, scope, expected):
        seed(tmp_path)
        cards = load_cards(tmp_path, problem_id="suite-solar", family="suite", scope=scope)
        # the dummy search's card never informs anyone, whatever the scope
        assert {c.run_ref for c in cards} == expected

    def test_problem_key_decides_when_both_cards_carry_one(self, tmp_path):
        write_card(tmp_path, card("solar", "r1/a", key="emflow://suite:solar"))
        write_card(tmp_path, card("solar", "r1/b", key="emflow://other:solar"))
        cards = load_cards(
            tmp_path, problem_id="solar", family="suite", scope="problem", problem_key="emflow://suite:solar",
        )
        assert [c.run_ref for c in cards] == ["r1/a"]

    def test_live_cards(self, tmp_path):
        write_live_card(tmp_path, card("suite-wind", "r/wind"), "wind")
        write_live_card(tmp_path, card("other", "r/other", family="other"), "other")
        write_live_card(tmp_path, card("suite-solar", "r/smoke", agents=("toy",)), "smoke")

        def seen(scope):
            cards = load_live_cards(tmp_path, family="suite", scope=scope, problem_id="suite-solar")
            return {c.problem_id for c in cards}

        assert seen("all") == {"suite-wind", "other"}
        assert seen("family") == {"suite-wind"}
        assert seen("problem") == set()

    @pytest.mark.parametrize(
        ("where", "scope", "admitted"),
        [
            ({"problem_id": "suite-solar", "family": "suite"}, "problem", True),
            ({"problem_id": "suite-wind", "family": "suite"}, "problem", False),
            ({"problem_id": "suite-wind", "family": "suite"}, "family", True),
            ({"problem_id": "x", "family": "other"}, "family", False),
            ({}, "family", False),  # a paper's claim names no problem
            ({}, "all", True),
        ],
    )
    def test_claims(self, where, scope, admitted):
        node = SimpleNamespace(data={"scope": where})
        assert claim_in_scope(node, scope=scope, problem_id="suite-solar", family="suite") is admitted


class TestScripted:
    def test_scripted_only(self, config):
        config.agent = "dummy"
        assert scripted_only(config)
        config.routing = {"improve": RouteConfig(agent="toy"), "distill": RouteConfig(agent="claude-code")}
        assert scripted_only(config)  # a memory pass's route does not count
        config.routing["draft"] = RouteConfig(agent="claude-code")
        assert not scripted_only(config)
        config.routing = {}
        config.agent = "claude-code"
        assert not scripted_only(config)

    def test_card_names_its_agents_and_problem(self, tmp_path):
        journal = Journal(tmp_path / "journal.jsonl")
        for cid, name in (("c001", "claude-code"), ("c002", "codex")):
            journal.candidate_result(Candidate(
                candidate_id=cid, operator="draft", status="passing", agent=AgentInfo(name=name),
                trials=[mk_trial(val_score=0.5)],
            ))
        problem = SimpleNamespace(
            problem_id="p", metric_name="m", higher_is_better=True, problem_key="emflow://suite:p",
        )
        made = distill_card(journal, problem=problem, run_ref="r/p")
        assert made.agents == ["claude-code", "codex"]
        assert made.problem_key == "emflow://suite:p"

    def test_a_dummy_search_writes_no_knowledge(self, task, config, tmp_path, monkeypatch):
        from hillclimb.agents.fake import FakeAgent
        from hillclimb.api import create_search, execute_search
        from hillclimb.harness.budget import BudgetManager
        from hillclimb.harness.dirs import create_run_dir
        from tests.conftest import ok_script

        knowledge_dir = tmp_path / "knowledge"
        config.learning.dir = knowledge_dir
        config.agent = "dummy"
        config.paths.runtime_python = Path(sys.executable)
        config.budget.stop_margin_s = 1
        config.budget.max_evaluations = 1
        agent = FakeAgent()
        agent.queue(script=ok_script(0.7), notes="approach\n")
        monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
        logged: list[str] = []

        run_dir = create_run_dir(config.paths.runs_dir, "smoke")
        search_dir = create_search(config, task, run_dir, "smoke", 3600)
        outcome = execute_search(
            config, task, search_dir, BudgetManager(3600, stop_margin_s=1), log=logged.append,
        )

        assert outcome.state == "done"
        assert any("scripted coding agents only" in line for line in logged)
        assert list(iter_cards(knowledge_dir)) == []
        assert load_skills(knowledge_dir) == []


class TestPrune:
    @pytest.fixture
    def knowledge(self, config, tmp_path, monkeypatch) -> Path:
        knowledge_dir = tmp_path / "knowledge"
        config.learning.dir = knowledge_dir
        monkeypatch.setattr(common, "load_config", lambda: config)
        seed(knowledge_dir)
        # an older card: no agents on it, no search left in runs/
        write_card(knowledge_dir, card("suite-solar", "r0/gone", agents=()))
        skill_dir = knowledge_dir / "skills" / "suite--r2-smoke"
        skill_dir.mkdir(parents=True)
        (skill_dir / SKILL_CODE_FILENAME).write_text("print('canned')\n")
        (skill_dir / SKILL_META_FILENAME).write_text(
            "schema_version: 1\nproblem_id: suite-solar\nfamily: suite\nrun_ref: r2/smoke\nscore: 1.0\n"
        )
        write_credit_event(knowledge_dir, CreditEvent(
            run_ref="r2/smoke", problem_id="suite-solar", family="suite", claim_ids=["x"], reward=1.0, basis="b",
        ))
        return knowledge_dir

    def prune(self, *args):
        from hillclimb import cli

        return CliRunner().invoke(cli.app, ["knowledge", "prune", *args])

    def refs(self, knowledge_dir):
        return {c.run_ref for _, c in iter_cards(knowledge_dir)}

    def test_by_agent(self, knowledge):
        result = self.prune("--agent", "dummy", "--agent", "toy")
        assert result.exit_code == 0, result.output
        assert self.refs(knowledge) == {"r1/solar", "r1/wind", "r0/gone"}
        assert load_skills(knowledge) == []
        assert not credit_event_path(knowledge, "r2/smoke").exists()
        assert "kept 1 card(s) whose coding agent is unknown" in result.output

    def test_by_problem_and_search(self, knowledge):
        assert self.prune("--problem", "suite-wind").exit_code == 0
        assert self.refs(knowledge) == {"r1/solar", "r2/smoke", "r0/gone"}
        assert self.prune("--search", "r0/gone").exit_code == 0
        assert self.refs(knowledge) == {"r1/solar", "r2/smoke"}

    def test_filters_combine(self, knowledge):
        # dummy AND suite-wind: nothing is both
        assert self.prune("--agent", "dummy", "--problem", "suite-wind").exit_code == 0
        assert len(self.refs(knowledge)) == 4

    def test_dry_run_deletes_nothing(self, knowledge):
        result = self.prune("--agent", "dummy", "--dry-run")
        assert result.exit_code == 0
        assert "would remove 1 card(s)" in result.output
        assert len(self.refs(knowledge)) == 4
        assert load_skills(knowledge)

    def test_needs_a_filter(self, knowledge):
        assert self.prune().exit_code == 2
