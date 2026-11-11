from __future__ import annotations

from functools import lru_cache
import json
import logging
import os
from pathlib import Path

from ...environment.schema import EditDefinition
from .prompts import EDIT_CONTEXT
from .state import StrategyState


DEFAULT_EDIT_REGISTRY = Path(__file__).resolve().parents[2] / "configs/environment/edits.jsonl"


@lru_cache(maxsize=8)
def load_edit_registry(path: str) -> dict[str, EditDefinition]:
    try:
        edits = [EditDefinition.model_validate(json.loads(line))
                 for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len({edit.edit_id for edit in edits}) != len(edits):
            raise ValueError("Duplicate edit IDs")
        return {edit.edit_id: edit for edit in edits}
    except (OSError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning("Cannot load edit context from %s: %s", path, exc)
        return {}


def _edit(edit_code: str) -> EditDefinition | None:
    path = Path(os.environ.get("IRIS_EDIT_REGISTRY") or DEFAULT_EDIT_REGISTRY).expanduser()
    return load_edit_registry(str(path.resolve())).get(edit_code)


def get_edit_definition(edit_code: str) -> str:
    edit = _edit(edit_code)
    return edit.description if edit else ""


def render_edit_context(edit_code: str) -> str:
    return EDIT_CONTEXT.get(edit_code, "")


def strategy_ref(state: StrategyState) -> str:
    return f"strategy@{state.version}"


def render_directives(state: StrategyState) -> list[str]:
    out: list[str] = []
    if state.rendered_prompt:
        out.append(state.rendered_prompt)
    else:
        out.extend(
            line for line in (p.render() for p in state.active_policies()) if line
        )
    if state.diversity_categories:
        counts = ", ".join(
            f"{name}: {count}"
            for name, count in sorted(
                state.diversity_categories.items(), key=lambda kv: -kv[1]
            )
        )
        out.append(f"Current run counts: {counts}.")
    return out


def render_memory(state: StrategyState) -> str:
    out = [
        f"# strategy@{state.version}  ({state.edit_code}, from batch "
        f"{state.observed_batch})"
    ]
    if state.diagnosis:
        out.append(f"\n## diagnosis\n{state.diagnosis}")
    if state.rationale:
        out.append(f"\n## rationale\n{state.rationale}")

    active = state.active_policies()
    out.append(f"\n## active policies ({len(active)})")
    for policy in active:
        out.append(
            f"\n### {policy.policy_id} (v{policy.version})\n"
            f"- in force since {policy.deployed_at_batch}, "
            f"from claim `{policy.source_hypothesis}`\n"
            f"- when: {policy.trigger}\n"
            f"- do: {policy.procedure}\n"
            f"- except: {policy.boundary or '(none stated)'}"
        )

    dropped = [p for p in state.policies if not p.active]
    if dropped:
        out.append(f"\n## withdrawn policies ({len(dropped)})")
        for policy in dropped:
            out.append(
                f"- {policy.policy_id}: {policy.removed_reason or 'never promoted'}"
            )

    out.append(f"\n## claims ({len(state.hypotheses)})")
    for hypothesis in state.hypotheses:
        line = (
            f"- [{hypothesis.status}] {hypothesis.hypothesis_id}: "
            f"supported in {hypothesis.support}, contradicted in "
            f"{hypothesis.oppose}, untested streak {hypothesis.untested_streak}"
        )
        if hypothesis.retired_reason:
            line += f" — retired: {hypothesis.retired_reason}"
        out.append(line)
        out.append(f"  {hypothesis.statement}")
    return "\n".join(out)
