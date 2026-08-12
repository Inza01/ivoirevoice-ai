"""Validate the repository knowledge system and agent-facing invariants."""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "ivoirevoice"
MAX_AGENT_GUIDE_LINES = 120
MAX_PYTHON_LINES = 1_500

REQUIRED_PATHS = (
    "AGENTS.md",
    "docs/index.md",
    "docs/knowledge-map.yaml",
    "docs/design-docs/core-beliefs.md",
    "docs/product-specs/mvp.md",
    "docs/exec-plans/README.md",
    "docs/exec-plans/template.md",
    "docs/QUALITY.md",
    "docs/RELIABILITY.md",
    "docs/SECURITY.md",
    "docs/technical-debt.md",
    ".github/workflows/verify.yml",
    ".github/pull_request_template.md",
)

PLAN_HEADINGS = (
    "## Status",
    "## Objective",
    "## Scope",
    "## Acceptance criteria",
    "## Progress",
    "## Decisions",
    "## Validation",
)

ALLOWED_DEPENDENCIES: Mapping[str, frozenset[str]] = {
    "api": frozenset({"config", "exceptions", "models", "services"}),
    "config": frozenset({"exceptions"}),
    "data": frozenset({"exceptions"}),
    "evaluation": frozenset({"data", "exceptions", "models"}),
    "exceptions": frozenset(),
    "logging_config": frozenset(),
    "models": frozenset({"exceptions"}),
    "services": frozenset({"data", "evaluation", "exceptions", "models"}),
    "training": frozenset({"data", "evaluation", "exceptions", "models"}),
    "ui": frozenset({"exceptions", "services"}),
}

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
MAKE_TARGET = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_-]*):(?:\s|$)", re.MULTILINE)


def _candidate_files(pattern: str) -> tuple[Path, ...]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            pattern,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    relative_paths = result.stdout.decode("utf-8").split("\0")
    return tuple(REPOSITORY_ROOT / path for path in relative_paths if path)


def _mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return value
    errors.append(f"{label} doit être un mapping YAML.")
    return {}


def _load_knowledge_map(path: Path, errors: list[str]) -> Mapping[str, Any]:
    try:
        raw_value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        errors.append(f"Impossible de lire docs/knowledge-map.yaml : {exc}")
        return {}
    return _mapping(raw_value, "knowledge-map", errors)


def check_required_paths(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Return missing harness paths."""

    return [
        f"Fichier harness requis absent : {relative_path}"
        for relative_path in REQUIRED_PATHS
        if not (root / relative_path).exists()
    ]


def check_agent_guide(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Keep AGENTS.md concise and useful as a navigation map."""

    path = root / "AGENTS.md"
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"Impossible de lire AGENTS.md : {exc}"]
    errors: list[str] = []
    line_count = len(text.splitlines())
    if line_count > MAX_AGENT_GUIDE_LINES:
        errors.append(
            f"AGENTS.md contient {line_count} lignes; maximum {MAX_AGENT_GUIDE_LINES}. "
            "Déplacer le détail dans docs/."
        )
    for required_reference in ("docs/index.md", "make verify-fast", "make verify"):
        if required_reference not in text:
            errors.append(f"AGENTS.md doit référencer {required_reference}.")
    return errors


def check_markdown_links(paths: Iterable[Path], root: Path = REPOSITORY_ROOT) -> list[str]:
    """Validate repository-local links without making network requests."""

    errors: list[str] = []
    root = root.resolve()
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"Impossible de lire {path.relative_to(root)} : {exc}")
            continue
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or target.startswith("#"):
                continue
            relative_target = unquote(target.split("#", maxsplit=1)[0])
            if not relative_target:
                continue
            resolved = (path.parent / relative_target).resolve()
            label = path.relative_to(root)
            if not resolved.is_relative_to(root):
                errors.append(f"{label}: lien sortant du dépôt interdit : {target}")
            elif not resolved.exists():
                errors.append(f"{label}: lien local absent : {target}")
    return errors


def check_execution_plans(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Require resumable structure for active and completed plans."""

    errors: list[str] = []
    plan_root = root / "docs" / "exec-plans"
    plan_paths = tuple((plan_root / "active").glob("*.md")) + tuple(
        (plan_root / "completed").glob("*.md")
    )
    for path in plan_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"Impossible de lire {path.relative_to(root)} : {exc}")
            continue
        for heading in PLAN_HEADINGS:
            if heading not in text:
                errors.append(f"{path.relative_to(root)}: section requise absente : {heading}")
    return errors


def _internal_target(module_name: str | None) -> str | None:
    if not module_name or not module_name.startswith("ivoirevoice."):
        return None
    parts = module_name.split(".")
    return parts[1] if len(parts) > 1 else None


def _import_targets(tree: ast.AST, relative_path: Path) -> Iterable[str]:
    source_module = ("ivoirevoice", *relative_path.with_suffix("").parts)
    source_package = source_module[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _internal_target(alias.name)
                if target:
                    yield target
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            if node.level:
                parent_count = node.level - 1
                if parent_count > len(source_package):
                    continue
                prefix = source_package[: len(source_package) - parent_count]
                suffix = tuple(module_name.split(".")) if module_name else ()
                module_name = ".".join((*prefix, *suffix))
            target = _internal_target(module_name)
            if target:
                yield target


def check_architecture(source_root: Path = SOURCE_ROOT) -> list[str]:
    """Enforce existing package dependency directions."""

    errors: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        relative_path = path.relative_to(source_root)
        source_area = relative_path.parts[0].removesuffix(".py")
        if source_area == "__init__":
            continue
        allowed = ALLOWED_DEPENDENCIES.get(source_area)
        if allowed is None:
            errors.append(
                f"{relative_path}: domaine inconnu; documenter sa frontière dans check_harness.py."
            )
            continue
        try:
            source_text = path.read_text(encoding="utf-8")
            tree = ast.parse(source_text, filename=str(relative_path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append(f"{relative_path}: impossible d'analyser les imports : {exc}")
            continue
        line_count = len(source_text.splitlines())
        if line_count > MAX_PYTHON_LINES:
            errors.append(
                f"{relative_path}: {line_count} lignes; maximum {MAX_PYTHON_LINES}. "
                "Créer un plan de découpage."
            )
        for target_area in sorted(set(_import_targets(tree, relative_path))):
            if target_area != source_area and target_area not in allowed:
                errors.append(
                    f"{relative_path}: dépendance interdite {source_area} -> {target_area}. "
                    "Consulter docs/architecture.md."
                )
    return errors


def _make_targets(root: Path, errors: list[str]) -> frozenset[str]:
    try:
        makefile = (root / "Makefile").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"Impossible de lire Makefile : {exc}")
        return frozenset()
    return frozenset(MAKE_TARGET.findall(makefile))


def _check_path_value(root: Path, label: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} doit contenir un chemin non vide.")
    elif not (root / value).exists():
        errors.append(f"{label} référence un chemin absent : {value}")


def check_knowledge_map(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Validate source/docs/tests routing and executable quality gates."""

    errors: list[str] = []
    knowledge = _load_knowledge_map(root / "docs" / "knowledge-map.yaml", errors)
    if knowledge.get("version") != 1:
        errors.append("knowledge-map.version doit valoir 1.")

    entrypoints = _mapping(knowledge.get("entrypoints"), "entrypoints", errors)
    for name, path in entrypoints.items():
        _check_path_value(root, f"entrypoints.{name}", path, errors)

    domains = _mapping(knowledge.get("domains"), "domains", errors)
    for domain_name, raw_domain in domains.items():
        domain = _mapping(raw_domain, f"domains.{domain_name}", errors)
        for field in ("source", "tests", "docs"):
            _check_path_value(
                root,
                f"domains.{domain_name}.{field}",
                domain.get(field),
                errors,
            )

    index_path = root / "docs" / "index.md"
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"Impossible de lire docs/index.md : {exc}")
        index_text = ""
    indexed_paths = {
        (index_path.parent / unquote(target.split("#", maxsplit=1)[0])).resolve()
        for target in MARKDOWN_LINK.findall(index_text)
        if target and not urlparse(target).scheme and not target.startswith("#")
    }
    documented_paths = {
        (root / value).resolve()
        for value in entrypoints.values()
        if isinstance(value, str) and value.startswith("docs/") and value != "docs/index.md"
    }
    documented_paths.update(
        (root / domain["docs"]).resolve()
        for raw_domain in domains.values()
        if isinstance(raw_domain, dict)
        for domain in (raw_domain,)
        if isinstance(domain.get("docs"), str)
    )
    for missing_path in sorted(documented_paths - indexed_paths):
        errors.append(
            f"docs/index.md doit référencer {missing_path.relative_to(root).as_posix()}."
        )

    targets = _make_targets(root, errors)
    quality_gates = _mapping(knowledge.get("quality_gates"), "quality_gates", errors)
    runtime_entrypoints = _mapping(
        knowledge.get("runtime_entrypoints"), "runtime_entrypoints", errors
    )
    for group_name, commands in (
        ("quality_gates", quality_gates),
        ("runtime_entrypoints", runtime_entrypoints),
    ):
        for name, command in commands.items():
            if not isinstance(command, str) or not command.startswith("make "):
                errors.append(f"{group_name}.{name} doit être une commande make.")
                continue
            target = command.split(maxsplit=1)[1]
            if target not in targets:
                errors.append(f"{group_name}.{name} référence une cible absente : {target}")
    return errors


def collect_errors(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Run every harness check."""

    errors = check_required_paths(root)
    errors.extend(check_agent_guide(root))
    errors.extend(check_markdown_links(_candidate_files("*.md"), root))
    errors.extend(check_execution_plans(root))
    errors.extend(check_architecture(root / "src" / "ivoirevoice"))
    errors.extend(check_knowledge_map(root))
    return errors


def main() -> int:
    """CLI entry point with remediation-oriented output."""

    errors = collect_errors()
    if errors:
        print("Échec du harness engineering :")
        for error in errors:
            print(f"- {error}")
        return 1
    markdown_count = len(_candidate_files("*.md"))
    python_count = len(tuple(SOURCE_ROOT.rglob("*.py")))
    print(f"Documents contrôlés : {markdown_count}")
    print(f"Modules Python contrôlés : {python_count}")
    print("Harness engineering : valide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
