#!/usr/bin/env python3
"""Tests for the Container Config Linter."""

import os
import tempfile
import pytest
from pathlib import Path
from container_linter import lint_workspace, Severity, lint_content_quality, LintReport


@pytest.fixture
def valid_workspace(tmp_path):
    """Create a minimal valid workspace."""
    (tmp_path / "SOUL.md").write_text(
        "# My Soul\n\n"
        "## Core Truths\nBe helpful.\n\n"
        "## Boundaries\nStay safe.\n\n"
        "## Vibe\nChill.\n\n"
        "## Continuity\nRead your files.\n"
    )
    (tmp_path / "AGENTS.md").write_text(
        "# AGENTS.md\n\n"
        "## Memory\nUse files.\n\n"
        "## Safety\nDon't break things.\n\n"
        "## Tools\nCheck skills.\n\n"
        "## Heartbeats\nCheck in.\n"
        "More lines.\n" * 10
    )
    (tmp_path / "IDENTITY.md").write_text(
        "# IDENTITY.md\n\n"
        "- **Name:** TestBot\n"
        "- **Vibe:** Chill\n"
        "- **Emoji:** 🤖\n"
    )
    (tmp_path / "MEMORY.md").write_text("# Memory\n\nSome memories here.\n")
    (tmp_path / "USER.md").write_text("# User\n\nSome notes.\n\n## Context\nStuff.\n")

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "2026-02-13.md").write_text("# 2026-02-13\n\nToday.\n")

    return tmp_path


@pytest.fixture
def empty_workspace(tmp_path):
    """Create an empty workspace."""
    return tmp_path


@pytest.fixture
def partial_workspace(tmp_path):
    """Workspace missing core files."""
    (tmp_path / "SOUL.md").write_text("# Soul\n\nJust a soul.\n" * 5)
    return tmp_path


def test_valid_workspace_no_errors(valid_workspace):
    report = lint_workspace(str(valid_workspace))
    errors = [r for r in report.results if r.severity == Severity.ERROR]
    assert len(errors) == 0, f"Unexpected errors: {[str(e) for e in errors]}"


def test_empty_workspace_reports_missing_core(empty_workspace):
    report = lint_workspace(str(empty_workspace))
    error_rules = [r.rule for r in report.errors]
    assert error_rules.count("core-file-exists") == 3  # SOUL, AGENTS, IDENTITY


def test_partial_workspace_missing_files(partial_workspace):
    report = lint_workspace(str(partial_workspace))
    missing = [r for r in report.errors if r.rule == "core-file-exists"]
    missing_files = [r.file for r in missing]
    assert "AGENTS.md" in missing_files
    assert "IDENTITY.md" in missing_files
    assert "SOUL.md" not in missing_files


def test_identity_missing_fields(tmp_path):
    (tmp_path / "IDENTITY.md").write_text("# Identity\n\nJust a heading.\n")
    (tmp_path / "SOUL.md").write_text("# Soul\n" + "content\n" * 12)
    (tmp_path / "AGENTS.md").write_text("# Agents\n" + "content\n" * 20)
    report = lint_workspace(str(tmp_path))
    field_errors = [r for r in report.errors if r.rule == "required-field"]
    field_names = [r.message for r in field_errors]
    assert any("Name" in m for m in field_names)
    assert any("Emoji" in m for m in field_names)


def test_secret_detection(tmp_path):
    (tmp_path / "SOUL.md").write_text(
        "# Soul\n\n"
        "My key is sk-ant-oat01-ABCDEFGHIJ1234567890\n"
        "More content\n" * 10
    )
    (tmp_path / "AGENTS.md").write_text("# Agents\n" + "content\n" * 20)
    (tmp_path / "IDENTITY.md").write_text(
        "# ID\n- **Name:** X\n- **Vibe:** Y\n- **Emoji:** Z\n"
    )
    report = lint_workspace(str(tmp_path))
    secret_errors = [r for r in report.errors if r.rule == "no-secrets"]
    assert len(secret_errors) >= 1


def test_empty_file_detected(tmp_path):
    (tmp_path / "SOUL.md").write_text("")
    (tmp_path / "AGENTS.md").write_text("# Agents\n" + "content\n" * 20)
    (tmp_path / "IDENTITY.md").write_text(
        "# ID\n- **Name:** X\n- **Vibe:** Y\n- **Emoji:** Z\n"
    )
    report = lint_workspace(str(tmp_path))
    empty_errors = [r for r in report.errors if r.rule == "non-empty"]
    assert len(empty_errors) == 1


def test_memory_dir_missing(tmp_path):
    (tmp_path / "SOUL.md").write_text("# Soul\n" + "content\n" * 12)
    (tmp_path / "AGENTS.md").write_text("# Agents\n" + "content\n" * 20)
    (tmp_path / "IDENTITY.md").write_text(
        "# ID\n- **Name:** X\n- **Vibe:** Y\n- **Emoji:** Z\n"
    )
    report = lint_workspace(str(tmp_path))
    mem_warnings = [r for r in report.warnings if r.rule == "memory-dir-exists"]
    assert len(mem_warnings) == 1


def test_duplicate_headings(tmp_path):
    (tmp_path / "SOUL.md").write_text(
        "# Soul\n\n## Core\nStuff.\n\n## Core\nDuplicate.\n" + "more\n" * 8
    )
    (tmp_path / "AGENTS.md").write_text("# Agents\n" + "content\n" * 20)
    (tmp_path / "IDENTITY.md").write_text(
        "# ID\n- **Name:** X\n- **Vibe:** Y\n- **Emoji:** Z\n"
    )
    report = lint_workspace(str(tmp_path))
    dup_warnings = [r for r in report.warnings if r.rule == "duplicate-heading"]
    assert len(dup_warnings) >= 1


def test_nonexistent_workspace():
    report = lint_workspace("/nonexistent/path/does/not/exist")
    assert len(report.errors) == 1
    assert report.errors[0].rule == "workspace-exists"


def test_broken_link_detection(tmp_path):
    (tmp_path / "SOUL.md").write_text(
        "# Soul\n\n[click here]()\n" + "more\n" * 10
    )
    (tmp_path / "AGENTS.md").write_text("# Agents\n" + "content\n" * 20)
    (tmp_path / "IDENTITY.md").write_text(
        "# ID\n- **Name:** X\n- **Vibe:** Y\n- **Emoji:** Z\n"
    )
    report = lint_workspace(str(tmp_path))
    link_warnings = [r for r in report.warnings if r.rule == "broken-link"]
    assert len(link_warnings) >= 1


def test_lint_real_spencer_workspace():
    """Integration test against Spencer's actual workspace."""
    ws = os.path.expanduser("~/spencer")
    if not Path(ws).exists():
        pytest.skip("Spencer workspace not found")
    report = lint_workspace(ws)
    # Should lint without crashing
    assert report.files_checked >= 3
    # No secrets should be in workspace files
    secret_errors = [r for r in report.errors if r.rule == "no-secrets"]
    assert len(secret_errors) == 0, f"Secrets found: {[str(e) for e in secret_errors]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
