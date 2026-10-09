"""Procedural memory: winning solutions harvested as reference scaffolds.

Claims say what helped; skills ARE what helped. After a search whose selected
candidate is a real winner (non-baseline, scored), its `solution.py` is
copied verbatim into `knowledge/skills/<family>--<run-ref>/` with a metadata
card — no model calls. The next search on the SAME problem, measured on the
same metric in the same direction, gets the best skill copied into its FIRST
draft's candidate dir as `reference_solution.py`, with a prompt cue to adapt
rather than resubmit. Another problem's code is never handed over this way: a
sibling's winner optimizes a different score (what transfers across problems
are claims, in text). Later drafts stay reference-free so the search still
explores. A search only scripted agents (dummy, toy) ran is never harvested.

Kept deliberately small: at most SKILLS_PER_PROBLEM skills per problem and
metric, best-score-first (direction-aware) — a library of proven scaffolds,
not an archive.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from hillclimb.harness.candidate import utcnow
from hillclimb.harness.direction import better
from hillclimb.harness.direction import legacy_direction_key
from hillclimb.modules.memory.knowledge import extract_libraries, same_problem, scripted

SKILLS_DIRNAME = "skills"
SKILLS_PER_PROBLEM = 2
SKILL_CODE_FILENAME = "skill.py"
SKILL_META_FILENAME = "skill.yaml"
SKILL_SCHEMA_VERSION = 1


class Skill(BaseModel):
    schema_version: int = SKILL_SCHEMA_VERSION
    problem_id: str
    family: str
    # as on the card it was harvested with; empty on older skills
    problem_key: str = ""
    agents: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    metric: str = ""
    higher_is_better: bool = True

    @model_validator(mode="before")
    @classmethod
    def _legacy_direction_key(cls, data):
        return legacy_direction_key(data)
    score: float | None = None
    holdout: float | None = None
    libraries: list[str] = Field(default_factory=list)
    summary: str = ""
    run_ref: str = ""
    created_at: str = Field(default_factory=utcnow)


def _skill_dir(knowledge_dir: Path, family: str, run_ref: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_ref).strip("-") or "run"
    return knowledge_dir / SKILLS_DIRNAME / f"{family}--{slug}"


def load_skills(knowledge_dir: Path, *, family: str = "") -> list[tuple[Skill, Path]]:
    directory = knowledge_dir / SKILLS_DIRNAME
    if not directory.exists():
        return []
    skills: list[tuple[Skill, Path]] = []
    for skill_dir in sorted(directory.iterdir()):
        meta = skill_dir / SKILL_META_FILENAME
        if not meta.is_file() or not (skill_dir / SKILL_CODE_FILENAME).is_file():
            continue
        try:
            data = yaml.safe_load(meta.read_text()) or {}
            if data.get("schema_version") != SKILL_SCHEMA_VERSION:
                continue
            skill = Skill.model_validate(data)
        except Exception:  # noqa: BLE001 — a corrupt skill must never block anything
            continue
        if family and skill.family != family:
            continue
        skills.append((skill, skill_dir))
    return skills


def harvest_skill(
    journal,
    *,
    problem,
    card,
    knowledge_dir: Path,
    selection: str,
    log,
) -> Path | None:
    """Post-search harvest (mechanical). Quality gate: only a scored,
    non-baseline winner of a search a model worked on enters the library;
    per problem and metric it keeps the SKILLS_PER_PROBLEM best by validation
    score."""
    if scripted(card.agents):
        return None
    selected = journal.selected_candidate(problem.higher_is_better, selection)
    if selected is None or selected.operator == "baseline" or selected.val_score is None:
        return None
    solution = Path(selected.candidate_dir) / "solution.py"
    if not solution.exists():
        return None
    existing = load_skills(knowledge_dir, family=card.family)
    scored = [
        (s, d) for s, d in existing
        if s.score is not None and comparable(
            s, problem_id=card.problem_id, problem_key=card.problem_key,
            metric=card.metric, higher_is_better=card.higher_is_better,
        )
    ]
    if len(scored) >= SKILLS_PER_PROBLEM:
        # direction-normalized score: smaller is better, so the worst is max
        worst_skill, worst_dir = max(
            scored,
            key=lambda pair: (-1 if problem.higher_is_better else 1) * pair[0].score,
        )
        if not better(selected.val_score, worst_skill.score, problem.higher_is_better):
            return None
        shutil.rmtree(worst_dir, ignore_errors=True)
    from hillclimb.modules.memory.claims import problem_concepts

    kind = "emflow" if card.target.startswith("emflow://") else "csv"
    skill = Skill(
        problem_id=card.problem_id,
        family=card.family,
        problem_key=card.problem_key,
        agents=card.agents,
        concepts=problem_concepts(kind, card.metric),
        metric=card.metric,
        higher_is_better=card.higher_is_better,
        score=selected.val_score,
        holdout=selected.holdout_score,
        libraries=extract_libraries(solution),
        summary=(selected.summary or "").strip()[:240],
        run_ref=card.run_ref,
    )
    skill_dir = _skill_dir(knowledge_dir, card.family, card.run_ref)
    skill_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(solution, skill_dir / SKILL_CODE_FILENAME)
    (skill_dir / SKILL_META_FILENAME).write_text(
        yaml.safe_dump(skill.model_dump(exclude_none=True), sort_keys=False)
    )
    return skill_dir


def comparable(
    skill: Skill, *, problem_id: str, problem_key: str, metric: str, higher_is_better: bool
) -> bool:
    """A skill whose score means the same as this search's: the same problem,
    the same metric (where both name one), the same direction."""
    return (
        same_problem(problem_id, problem_key, skill.problem_id, skill.problem_key)
        and (not metric or not skill.metric or metric == skill.metric)
        and skill.higher_is_better == higher_is_better
    )


def select_skill(
    knowledge_dir: Path,
    *,
    family: str,
    problem_id: str,
    problem_key: str = "",
    metric: str = "",
    higher_is_better: bool,
) -> tuple[Skill, Path] | None:
    """The best-scoring skill harvested on THIS problem and metric, or None.
    A family sibling's or a concept neighbour's winner is never handed over:
    its score measures something else, and a first draft scaffolded on it
    climbs the wrong hill."""
    same = [
        (s, d) for s, d in load_skills(knowledge_dir, family=family)
        if s.score is not None
        and not scripted(s.agents)
        and comparable(
            s, problem_id=problem_id, problem_key=problem_key,
            metric=metric, higher_is_better=higher_is_better,
        )
    ]
    if not same:
        return None
    return min(same, key=lambda pair: (-1 if higher_is_better else 1) * pair[0].score)
