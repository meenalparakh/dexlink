from __future__ import annotations

from pathlib import Path
import re

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPOSITORY_ROOT / "agent-skills"
EXPECTED_SKILLS = {
    "dexlink-datasets",
    "dexlink-generate-data",
    "dexlink-render-data",
    "dexlink-multitask",
    "dexlink-submit-cluster",
    "dexlink-troubleshoot",

    "dexlink-create-task",
    "dexlink-evaluate-policy",
    "dexlink-install",
    "dexlink-run-task",
}


def _frontmatter(skill_path: Path) -> dict[str, object]:
    text = skill_path.read_text(encoding="utf-8")
    marker = "\n---\n"
    if not text.startswith("---\n") or marker not in text[4:]:
        raise AssertionError(f"Invalid frontmatter delimiters in {skill_path}.")
    frontmatter_text = text[4:].split(marker, 1)[0]
    parsed = yaml.safe_load(frontmatter_text)
    assert isinstance(parsed, dict), skill_path
    return parsed


def test_public_agent_skill_packages_are_complete() -> None:
    skill_dirs = {
        path.name
        for path in SKILLS_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert skill_dirs == EXPECTED_SKILLS

    for skill_name in sorted(EXPECTED_SKILLS):
        skill_dir = SKILLS_ROOT / skill_name
        frontmatter = _frontmatter(skill_dir / "SKILL.md")
        assert frontmatter["name"] == skill_name
        assert isinstance(frontmatter["description"], str)
        assert frontmatter["description"].strip()

        metadata = yaml.safe_load(
            (skill_dir / "agents/openai.yaml").read_text(encoding="utf-8")
        )
        assert isinstance(metadata, dict)
        default_prompt = metadata["interface"]["default_prompt"]
        assert f"${skill_name}" in default_prompt


def test_agent_skills_reference_public_entry_points() -> None:
    pyproject_text = (REPOSITORY_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    scripts_section = pyproject_text.split("[project.scripts]", 1)[1].split(
        "\n[", 1
    )[0]
    entry_points = set(
        re.findall(r'^([a-z0-9-]+)\s*=\s*"[^\n]+"$', scripts_section, re.MULTILINE)
    )
    required_entry_points = {
        "dexlink-eval",
        "dexlink-eval-metrics",
        "dexlink-reference",
        "dexlink-train",
        "dexlink-validate-task",
    }
    assert required_entry_points <= entry_points

    skill_text = "\n".join(
        (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
        for name in sorted(EXPECTED_SKILLS)
    )
    for entry_point in required_entry_points:
        assert entry_point in skill_text


def test_agent_skills_are_portable() -> None:
    forbidden = (
        "/" + "home/",
        "/" + "mnt/",
        "/" + "n/fs/",
        "mp" + "8772",
    )
    for path in SKILLS_ROOT.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, (path, token)
