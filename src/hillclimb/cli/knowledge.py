"""`hillclimb knowledge …`, `paper …` and the `graph` viewer: the memory side."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import typer

from hillclimb.cli import common
from hillclimb.cli.common import _m, fail, say, warn
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.harness.run import search_ref, search_selection
from hillclimb.harness.store import latest_search, open_store
from hillclimb.problem import load_problem

knowledge_app = typer.Typer(cls=HillclimbGroup, help="Cross-search learning: cards distilled from finished searches")


app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("backfill")
def knowledge_backfill():
    """Distill cards from every finished search that lacks one.

    Walks runs/ and bootstraps learning from pre-existing history.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.knowledge import distill_card, write_card

    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)
    written = 0
    store = open_store(config)
    for record in store.searches():
            meta, search_dir = record.meta, record.search_dir
            # any finished search teaches something — parked and stopped
            # searches included; "unknown" covers pre-upgrade status files
            if record.state == "running":
                continue
            journal = Journal(store.journal(record.key))
            if not journal.scored_candidates():
                continue
            problem = SimpleNamespace(
                problem_id=meta.problem_id,
                metric_name=meta.metric,
                higher_is_better=meta.higher_is_better,
            )
            target = meta.problem if meta.problem.startswith("emflow://") else ""
            card = distill_card(
                journal, problem=problem, run_ref=search_ref(search_dir),
                target=target, budget_s=meta.budget_s,
                selection=search_selection(meta),
            )
            path = write_card(knowledge_dir, card)
            written += 1
            say(f"  [path]{_m(search_ref(search_dir))}[/] -> [path]{_m(path.relative_to(knowledge_dir))}[/]")
    say(f"[head]{written} knowledge card(s) written[/] to [path]{_m(knowledge_dir)}[/]")


@knowledge_app.command("live")
def knowledge_live(run: str = typer.Argument("latest", help="Run id, or `latest`")):
    """Show the live cards concurrent searches in a run are sharing.

    The discoveries a sibling's next operator would receive.
    """
    from hillclimb.modules.memory.knowledge import (
        load_live_cards,
        render_live_experience,
    )

    config = common.load_config()
    runs_dir = config.paths.runs_dir
    store = open_store(config)
    if run == "latest":
        latest = latest_search(store)
        if latest is None:
            fail(f"No searches found in [path]{_m(runs_dir)}[/]")
            raise typer.Exit(1)
        run_dir = latest.search_dir.parents[1]
    else:
        run_dir = runs_dir / run
        if not any(r.run_id == run for r in store.runs()):
            raise typer.BadParameter(f"No run named {run!r} in {runs_dir}")
    cards = load_live_cards(run_dir)
    if not cards:
        say(f"no live cards under [path]{_m(run_dir / 'knowledge')}[/]")
        raise typer.Exit(0)
    say(f"[head]Run {_m(run_dir.name)}[/] — {len(cards)} live card(s)")
    for card in cards:
        val = f"{card.selected_val:.5g}" if card.selected_val is not None else "-"
        say(
            f"  [path]{_m(card.run_ref)}[/]: {card.n_ok} passing / {card.n_failing} failing / "
            f"{card.n_buggy} buggy of "
            f"{card.n_candidates}, best {_m(card.metric or 'score')} [head]{val}[/]"
        )
    say()
    typer.echo(render_live_experience(cards, max_cards=len(cards)))


@knowledge_app.command("show")
def knowledge_show(target: str = typer.Argument(..., help="Problem target, e.g. circle-packing")):
    """Render the prior experience a new search would receive.

    Scoped to this target.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.knowledge import (
        load_cards,
        problem_family,
        render_prior_experience,
    )

    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)
    problem = load_problem(target, config)
    cards = load_cards(
        knowledge_dir, problem_id=problem.problem_id,
        family=problem_family(problem.problem_id, str(target)),
        scope=config.learning.scope, problem_key=problem.problem_key,
    )
    if not cards:
        say(f"no knowledge cards for [path]{_m(problem.problem_id)}[/] in [path]{_m(knowledge_dir)}[/]")
        raise typer.Exit(0)
    from hillclimb.modules.memory.files import FilesMemory

    # as many cards as a search's prompts would show (the folder's memory_params)
    max_cards = config.climber_params("memory_params").get("max_cards", FilesMemory.DEFAULTS["max_cards"])
    typer.echo(render_prior_experience(cards, max_cards=int(max_cards)))


@knowledge_app.command("prune")
def knowledge_prune(
    agent: list[str] = typer.Option(
        [], "--agent", help="Searches run only on these coding agents, e.g. --agent dummy --agent toy"
    ),
    problem: list[str] = typer.Option([], "--problem", help="Searches on this problem (its id or problem key)"),
    search: list[str] = typer.Option([], "--search", help="One search, as <run-id>/<search-id>"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would go; delete nothing"),
):
    """Remove what some searches taught: their cards, claims, skills and credit.

    A card goes when it matches every filter given. An older card does not
    name its coding agents; --agent then reads the search's record in runs/,
    and a card whose search is gone is kept. The graph is
    rebuilt afterwards; playbooks built from the removed claims stay until
    the next `hillclimb knowledge consolidate`.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.credit import credit_event_path
    from hillclimb.modules.memory.graph import rebuild_graph
    from hillclimb.modules.memory.knowledge import iter_cards
    from hillclimb.modules.memory.skills import load_skills

    if not (agent or problem or search):
        fail("name what to prune: [cmd]--agent[/], [cmd]--problem[/] or [cmd]--search[/]")
        raise typer.Exit(2)
    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)
    store = open_store(config) if agent else None
    unknown = 0

    def agents_of(card) -> set[str] | None:
        if card.agents:
            return set(card.agents)
        run_id, _, search_id = card.run_ref.partition("/")
        record = store.search((run_id, search_id)) if store is not None and search_id else None
        return {record.meta.agent} if record is not None and record.meta.agent else None

    doomed = []
    for path, card in iter_cards(knowledge_dir):
        if problem and card.problem_id not in problem and card.problem_key not in problem:
            continue
        if search and card.run_ref not in search:
            continue
        if agent:
            ran_on = agents_of(card)
            if ran_on is None:
                unknown += 1
                continue
            if not ran_on <= set(agent):
                continue
        doomed.append((path, card))
    refs = {card.run_ref for _, card in doomed}
    skills = [skill_dir for skill, skill_dir in load_skills(knowledge_dir) if skill.run_ref in refs]
    credits = [path for path in (credit_event_path(knowledge_dir, ref) for ref in sorted(refs)) if path.exists()]
    verb = "would remove" if dry_run else "removed"
    for path, card in doomed:
        say(f"[note]{verb}[/] [path]{_m(path.relative_to(knowledge_dir))}[/] [note]({_m(card.run_ref)})[/]")
    for path in [*skills, *credits]:
        say(f"[note]{verb}[/] [path]{_m(path.relative_to(knowledge_dir))}[/]")
    if not dry_run:
        import shutil

        for path, _ in doomed:
            path.unlink()
        for skill_dir in skills:
            shutil.rmtree(skill_dir, ignore_errors=True)
        for path in credits:
            path.unlink()
        if doomed:
            rebuild_graph(knowledge_dir, module=_graph_module(config))
    say(
        f"[head]{verb} {len(doomed)} card(s)[/], {len(skills)} skill(s), {len(credits)} credit record(s)"
        + (" [note](graph rebuilt)[/]" if doomed and not dry_run else "")
    )
    if unknown:
        warn(f"kept {unknown} card(s) whose coding agent is unknown (no agents on the card, no search in runs/)")


@knowledge_app.command("distill")
def knowledge_distill(
    search: str = typer.Argument(
        "latest", help="Search ref (run-id/search-id), or `latest`"
    ),
    backfill: bool = typer.Option(
        False, "--backfill", help="Extract claims for every knowledge card that has none"
    ),
):
    """Run the LLM claims pass on a finished search.

    Distills typed claims (entities, concepts) — or, with `--backfill`,
    extracts them across existing cards.
    """
    import yaml as _yaml

    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.claims import distill_claims, distill_claims_from_card
    from hillclimb.modules.memory.knowledge import (
        SCHEMA_VERSION,
        KnowledgeCard,
        distill_card,
        write_card,
    )

    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)

    if backfill:
        distilled = 0
        for path in sorted(knowledge_dir.glob("*/*.yaml")):
            try:
                data = _yaml.safe_load(path.read_text()) or {}
                if data.get("schema_version") != SCHEMA_VERSION or data.get("claims"):
                    continue
                card = KnowledgeCard.model_validate(data)
            except Exception:  # noqa: BLE001
                continue
            work_dir = knowledge_dir / ".distill" / path.stem
            card.claims = distill_claims_from_card(
                card, work_dir=work_dir, knowledge_dir=knowledge_dir,
                config=config, log=_log,
            )
            if card.claims:
                write_card(knowledge_dir, card)
                distilled += 1
                say(f"  [path]{_m(path.relative_to(knowledge_dir))}[/]: {len(card.claims)} claim(s)")
        say(f"[head]{distilled} card(s) backfilled with claims[/]")
        return

    store = open_store(config)
    if search == "latest":
        record = latest_search(store)
        if record is None:
            fail(f"No searches found in [path]{_m(config.paths.runs_dir)}[/]")
            raise typer.Exit(1)
    else:
        run_id, _, search_id = search.partition("/")
        record = store.search((run_id, search_id))
        if record is None:
            raise typer.BadParameter(f"No search at {store.search_dir((run_id, search_id))}")
    meta, search_dir = record.meta, record.search_dir
    journal = Journal(store.journal(record.key))
    if not journal.scored_candidates():
        fail("search has no scored candidates — nothing to distill")
        raise typer.Exit(1)
    problem = SimpleNamespace(
        problem_id=meta.problem_id,
        metric_name=meta.metric,
        higher_is_better=meta.higher_is_better,
    )
    target = meta.problem if meta.problem.startswith("emflow://") else ""
    card = distill_card(
        journal, problem=problem, run_ref=search_ref(search_dir),
        target=target, budget_s=meta.budget_s,
        selection=search_selection(meta),
    )
    card.claims = distill_claims(
        journal, problem=problem, card=card, search_dir=search_dir,
        knowledge_dir=knowledge_dir, config=config, log=_log,
    )
    path = write_card(knowledge_dir, card)
    say(f"[head]{len(card.claims)} claim(s)[/] -> [path]{_m(path)}[/]")


def _log(message: str) -> None:
    """A library's progress line, in the CLI's voice (escaped, never markup)."""
    say(_m(message))


def _graph_module(config):
    """The graph module these commands read and rebuild through: the user's
    `climber.graph`, else the climber's `graph:` (the built-in when the
    climber itself will not load). A bad `climber.graph` is exit 2."""
    from hillclimb.harness.glue import build_graph_module

    try:
        return build_graph_module(config, log=lambda message: warn(_m(message)))
    except ValueError as exc:
        fail(_m(exc))
        raise typer.Exit(2) from exc


@knowledge_app.command("query")
def knowledge_query(
    terms: str = typer.Argument(..., help="Keywords, e.g. 'gradient boosting' or a technique slug"),
    family: str = typer.Option("", "--family", help="Restrict claims to one problem family"),
    limit: int = typer.Option(5, "--limit"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Read-only memory lookup (no model calls).

    Also advertised to coding agents so they can consult accumulated
    knowledge mid-search.
    """
    import json as _json

    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.graph import load_or_build_graph, render_query_hits

    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)
    module = _graph_module(config)
    hits = module.query(load_or_build_graph(knowledge_dir, module=module), terms, family=family, limit=limit)
    if as_json:
        typer.echo(_json.dumps(hits, indent=1))
    else:
        typer.echo(render_query_hits(hits))


@knowledge_app.command("consolidate")
def knowledge_consolidate(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would generalize / which playbooks would rewrite; no writes, no coding agent calls"
    ),
):
    """The sleep phase: generalize claims, rewrite playbooks.

    Lifts multi-family claims up the concept hierarchy (mechanical) and
    rewrites per-concept playbooks (one coding agent call per qualifying concept,
    routing key `consolidate`). Playbook rewrites land as reviewable git
    diffs.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.consolidate import consolidate

    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)
    summary = consolidate(knowledge_dir, config, _log, dry_run=dry_run)
    verb = "would generalize" if dry_run else "generalized"
    say(f"[head]{verb} {len(summary['generalized'])} claim(s)[/]")
    if dry_run:
        say(
            "playbook candidates: "
            + (", ".join(f"[path]{_m(c)}[/]" for c in summary["playbook_concepts"]) or "[note](none)[/]")
        )
    else:
        say(f"[head]{len(summary['playbooks_written'])} playbook(s) written[/]")


@knowledge_app.command("rebuild")
def knowledge_rebuild():
    """Force-rebuild knowledge/graph.json from cards and registries.

    The graph is a derived index — always safe to rebuild, never hand-edit.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.graph import graph_path, rebuild_graph

    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)
    module = _graph_module(config)
    graph = rebuild_graph(knowledge_dir, module=module)
    say(f"[head]rebuilt[/] [path]{_m(graph_path(knowledge_dir))}[/] [note]({_m(module.name)})[/]")
    say_graph_stats(graph, knowledge_dir)


def say_graph_stats(graph, knowledge_dir) -> None:
    """The knowledge graph at a glance, in the CLI's voice: what it holds,
    then what kinds of node and edge, largest first, side by side."""
    from collections import Counter
    from datetime import datetime

    nodes = Counter(n.type for n in graph.nodes)
    edges = Counter(e.type for e in graph.edges)
    try:
        built = datetime.fromisoformat(graph.built_at).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        built = str(graph.built_at)
    searches = len(graph.events)
    say(
        f"[head]knowledge graph[/]  [path]{_m(common.short_path(knowledge_dir))}[/]  [note]built {_m(built)}[/]"
    )
    say(
        f"  [head]{len(graph.nodes)}[/] nodes  [note]·[/]  [head]{len(graph.edges)}[/] edges  [note]·[/]  "
        f"[head]{searches}[/] search{'' if searches == 1 else 'es'} learned from"
    )
    node_rows = nodes.most_common()
    edge_rows = edges.most_common()
    rows = []
    for index in range(max(len(node_rows), len(edge_rows))):
        node = node_rows[index] if index < len(node_rows) else ("", "")
        edge = edge_rows[index] if index < len(edge_rows) else ("", "")
        rows.append((_m(node[0]), f"[head]{node[1]}[/]", "", _m(edge[0]), f"[head]{edge[1]}[/]"))
    say()
    common.table(
        [("node", "path"), ("", None, "right"), ("", None), ("edge", "path"), ("", None, "right")], rows
    )
    superseded = sum(1 for n in graph.nodes if n.superseded_at is not None)
    tracked = [n for n in graph.nodes if n.type == "claim" and "track" in n.data]
    if superseded or tracked:
        say()
    if superseded:
        say(f"  [note]{superseded} superseded claim{'' if superseded == 1 else 's'}[/]")
    if tracked:
        retired = sum(1 for n in tracked if n.data.get("retired"))
        say(f"  [note]{len(tracked)} claim(s) with a track record, {retired} retired by record[/]")


paper_app = typer.Typer(
    cls=HillclimbGroup,
    help="Distill PDF papers into knowledge claims that seed future searches",
)


app.add_typer(paper_app, name="paper")


def _paper_knowledge_dir() -> tuple[Config, Path]:
    from hillclimb.api import resolve_knowledge_dir

    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)
    return config, knowledge_dir


@paper_app.command("add")
def paper_add(
    pdfs: list[Path] = typer.Argument(..., help="PDF paper(s) to distill into claims"),
    problem: str = typer.Option(
        None, "--problem", help="Scope the claims: a problem id (or a provider target). "
        "Omitted, claims are global and reach searches through concept overlap only"
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-distill even when this exact PDF was already ingested"
    ),
):
    """Distill papers into typed claims and rebuild the knowledge graph.

    One coding agent pass per paper (routing key `paper`, default model sonnet)
    writes knowledge/papers/<slug>.yaml; the claims then ride the normal
    retrieval and credit paths — inspect the wiring with `hillclimb graph`
    before starting a run.
    """
    from hillclimb.modules.memory.graph import rebuild_graph
    from hillclimb.modules.memory.papers import distill_paper

    config, knowledge_dir = _paper_knowledge_dir()
    common.ensure_agents_ready(config)
    ingested = 0
    for pdf in pdfs:
        if not pdf.exists():
            fail(f"no such file: [path]{_m(pdf)}[/]")
            raise typer.Exit(1)
        record = distill_paper(
            config, knowledge_dir, pdf, problem=problem, force=force, log=_log
        )
        if record is not None:
            ingested += 1
    if ingested:
        rebuild_graph(knowledge_dir, module=_graph_module(config))
        say("[head]knowledge graph rebuilt[/]")
    if ingested < len(pdfs):
        raise typer.Exit(1)


@paper_app.command("list")
def paper_list():
    """Ingested papers: slug, scope, claim count, and ingestion date."""
    from hillclimb.modules.memory.papers import load_papers

    _config, knowledge_dir = _paper_knowledge_dir()
    papers = load_papers(knowledge_dir)
    if not papers:
        say("no papers ingested yet — add one with [cmd]hillclimb paper add <pdf>[/]")
        return
    for paper in papers:
        scope = paper.problem_id or paper.family or "global"
        title = f"  {paper.title!r}" if paper.title else ""
        say(
            f"[path]{_m(paper.slug)}[/]  [note]{_m(f'[{scope}]')}[/]  {len(paper.claims)} claim(s)  "
            f"added {_m(paper.added_at[:10])}[note]{_m(title)}[/]"
        )


@knowledge_app.command("graph")
def knowledge_graph(
    stats: bool = typer.Option(False, "--stats", help="Print index stats instead of the TUI"),
):
    """Explore the knowledge graph.

    Default: the interactive TUI screen (zoom/pan/click, time scrubber);
    `--stats` prints a text summary.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.graph import load_or_build_graph

    config = common.load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        fail("learning is disabled or no [path]knowledge/[/] dir resolvable")
        raise typer.Exit(1)
    if stats:
        module = _graph_module(config)
        say_graph_stats(load_or_build_graph(knowledge_dir, module=module), knowledge_dir)
        return
    try:
        from hillclimb.tui.graphview import GraphApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb knowledge graph` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc
    GraphApp(config).run()
