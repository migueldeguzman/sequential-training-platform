#!/usr/bin/env python3
"""
Container Config Linter — validates SOUL.md, AGENTS.md, IDENTITY.md, MEMORY.md
for structural consistency and common issues in agent workspaces.

Part of the SDB platform tooling for the AI Lab Containers research pillar.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class LintResult:
    file: str
    line: int
    severity: Severity
    message: str
    rule: str

    def __str__(self):
        icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}[self.severity.value]
        return f"{icon} {self.file}:{self.line} [{self.rule}] {self.message}"


@dataclass
class LintReport:
    results: list[LintResult] = field(default_factory=list)
    files_checked: int = 0
    files_found: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)

    def add(self, result: LintResult):
        self.results.append(result)

    @property
    def errors(self) -> list[LintResult]:
        return [r for r in self.results if r.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[LintResult]:
        return [r for r in self.results if r.severity == Severity.WARNING]

    @property
    def infos(self) -> list[LintResult]:
        return [r for r in self.results if r.severity == Severity.INFO]

    def summary(self) -> str:
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Container Lint Report")
        lines.append(f"{'='*60}")
        lines.append(f"Files checked: {self.files_checked}")
        lines.append(f"Files found:   {', '.join(self.files_found) or 'none'}")
        if self.files_missing:
            lines.append(f"Files missing: {', '.join(self.files_missing)}")
        lines.append(f"")
        lines.append(f"Errors:   {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        lines.append(f"Info:     {len(self.infos)}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ── File schemas: required/optional sections per container file ──────────

SCHEMAS = {
    "SOUL.md": {
        "description": "Agent soul/personality definition",
        "required_patterns": [
            (r"^#\s+", "Top-level heading (# Title)"),
        ],
        "recommended_sections": [
            "Core",
            "Boundar",  # Boundaries
            "Vibe",
            "Continuit",  # Continuity
        ],
        "min_lines": 10,
        "max_lines": 500,
    },
    "AGENTS.md": {
        "description": "Agent behavior rules and workspace conventions",
        "required_patterns": [
            (r"^#\s+", "Top-level heading (# Title)"),
        ],
        "recommended_sections": [
            "Memory",
            "Safety",
            "Tool",
            "Heartbeat",
        ],
        "min_lines": 15,
        "max_lines": 1000,
    },
    "IDENTITY.md": {
        "description": "Agent identity card",
        "required_patterns": [
            (r"^#\s+", "Top-level heading (# Title)"),
        ],
        "required_fields": [
            ("Name", r"\*\*Name:?\*\*"),
            ("Vibe", r"\*\*Vibe:?\*\*"),
            ("Emoji", r"\*\*Emoji:?\*\*"),
        ],
        "min_lines": 3,
        "max_lines": 100,
    },
    "MEMORY.md": {
        "description": "Agent long-term memory",
        "required_patterns": [
            (r"^#\s+", "Top-level heading (# Title)"),
        ],
        "recommended_sections": [],
        "min_lines": 1,
        "max_lines": 2000,
    },
    "USER.md": {
        "description": "Human context file",
        "required_patterns": [
            (r"^#\s+", "Top-level heading (# Title)"),
        ],
        "recommended_sections": [],
        "min_lines": 3,
        "max_lines": 500,
    },
    "HEARTBEAT.md": {
        "description": "Heartbeat configuration",
        "required_patterns": [],
        "recommended_sections": [],
        "min_lines": 1,
        "max_lines": 300,
    },
    "TOOLS.md": {
        "description": "Local tool notes",
        "required_patterns": [],
        "recommended_sections": [],
        "min_lines": 1,
        "max_lines": 500,
    },
}

# Core files that should exist in every agent workspace
CORE_FILES = ["SOUL.md", "AGENTS.md", "IDENTITY.md"]
OPTIONAL_FILES = ["MEMORY.md", "USER.md", "HEARTBEAT.md", "TOOLS.md"]


def read_file(path: Path) -> Optional[tuple[str, list[str]]]:
    """Read file, return (content, lines) or None if not found."""
    try:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        return content, lines
    except (FileNotFoundError, PermissionError):
        return None


def lint_structure(filename: str, lines: list[str], schema: dict, report: LintReport):
    """Check file against its schema."""

    # Check minimum lines
    min_lines = schema.get("min_lines", 1)
    if len(lines) < min_lines:
        report.add(LintResult(
            file=filename, line=1, severity=Severity.WARNING,
            message=f"File has only {len(lines)} lines (minimum recommended: {min_lines})",
            rule="min-length"
        ))

    # Check maximum lines
    max_lines = schema.get("max_lines", 2000)
    if len(lines) > max_lines:
        report.add(LintResult(
            file=filename, line=len(lines), severity=Severity.WARNING,
            message=f"File has {len(lines)} lines (maximum recommended: {max_lines}). Consider trimming.",
            rule="max-length"
        ))

    # Check required patterns
    for pattern, description in schema.get("required_patterns", []):
        found = False
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                found = True
                break
        if not found:
            report.add(LintResult(
                file=filename, line=1, severity=Severity.ERROR,
                message=f"Missing required pattern: {description}",
                rule="required-pattern"
            ))

    # Check recommended sections (by heading content)
    for section_fragment in schema.get("recommended_sections", []):
        found = False
        for i, line in enumerate(lines, 1):
            if re.match(r"^#{1,3}\s+", line) and section_fragment.lower() in line.lower():
                found = True
                break
        if not found:
            report.add(LintResult(
                file=filename, line=1, severity=Severity.INFO,
                message=f"Recommended section not found: '{section_fragment}*'",
                rule="recommended-section"
            ))

    # Check required fields (for IDENTITY.md etc.)
    for field_name, field_pattern in schema.get("required_fields", []):
        found = False
        for i, line in enumerate(lines, 1):
            if re.search(field_pattern, line):
                found = True
                # Check if field has a value
                after_field = re.split(field_pattern, line, maxsplit=1)
                if len(after_field) > 1:
                    value = after_field[-1].strip().strip("*").strip()
                    if not value or value in ("(optional)", "(tbd)", ""):
                        report.add(LintResult(
                            file=filename, line=i, severity=Severity.WARNING,
                            message=f"Field '{field_name}' is empty or placeholder",
                            rule="empty-field"
                        ))
                break
        if not found:
            report.add(LintResult(
                file=filename, line=1, severity=Severity.ERROR,
                message=f"Missing required field: {field_name}",
                rule="required-field"
            ))


def lint_content_quality(filename: str, content: str, lines: list[str], report: LintReport):
    """Check for common content issues across all files."""

    # Check for secrets/tokens
    secret_patterns = [
        (r"sk-ant-[a-zA-Z0-9\-]{10,}", "Anthropic API key"),
        (r"sk-[a-zA-Z0-9]{32,}", "OpenAI API key"),
        (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
        (r"xoxb-[0-9]{10,}", "Slack bot token"),
        (r"xoxp-[0-9]{10,}", "Slack user token"),
        (r"AKIA[0-9A-Z]{16}", "AWS access key"),
        (r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----", "Private key"),
        (r"password\s*[:=]\s*['\"][^'\"]{4,}['\"]", "Hardcoded password"),
    ]

    for i, line in enumerate(lines, 1):
        for pattern, description in secret_patterns:
            if re.search(pattern, line):
                report.add(LintResult(
                    file=filename, line=i, severity=Severity.ERROR,
                    message=f"Potential secret detected: {description}",
                    rule="no-secrets"
                ))

    # Check for trailing whitespace (more than just style — can cause parse issues)
    trailing_ws_count = 0
    for i, line in enumerate(lines, 1):
        if line != line.rstrip() and line.strip():  # Non-empty lines with trailing whitespace
            trailing_ws_count += 1
    if trailing_ws_count > 10:
        report.add(LintResult(
            file=filename, line=1, severity=Severity.INFO,
            message=f"{trailing_ws_count} lines have trailing whitespace",
            rule="trailing-whitespace"
        ))

    # Check for broken markdown links
    for i, line in enumerate(lines, 1):
        # [text](url) where url is empty
        if re.search(r"\[.+?\]\(\s*\)", line):
            report.add(LintResult(
                file=filename, line=i, severity=Severity.WARNING,
                message="Empty markdown link target",
                rule="broken-link"
            ))

    # Check for very long lines (potential copy-paste issues)
    for i, line in enumerate(lines, 1):
        if len(line) > 500:
            report.add(LintResult(
                file=filename, line=i, severity=Severity.INFO,
                message=f"Very long line ({len(line)} chars) — may indicate formatting issues",
                rule="long-line"
            ))

    # Check encoding issues (null bytes, control chars)
    for i, line in enumerate(lines, 1):
        if "\x00" in line:
            report.add(LintResult(
                file=filename, line=i, severity=Severity.ERROR,
                message="Null byte detected — file may be corrupted",
                rule="encoding"
            ))

    # Check for duplicate headings at the same level
    headings: dict[str, list[int]] = {}
    for i, line in enumerate(lines, 1):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            key = f"{match.group(1)}|{match.group(2).strip().lower()}"
            headings.setdefault(key, []).append(i)
    for key, line_numbers in headings.items():
        if len(line_numbers) > 1:
            report.add(LintResult(
                file=filename, line=line_numbers[1], severity=Severity.WARNING,
                message=f"Duplicate heading '{key.split('|')[1]}' (also at line {line_numbers[0]})",
                rule="duplicate-heading"
            ))


def lint_memory_dir(workspace: Path, report: LintReport):
    """Check the memory/ directory for structural issues."""
    memory_dir = workspace / "memory"

    if not memory_dir.exists():
        report.add(LintResult(
            file="memory/", line=0, severity=Severity.WARNING,
            message="No memory/ directory found — agent has no daily memory files",
            rule="memory-dir-exists"
        ))
        return

    # Check for daily files
    daily_files = sorted(memory_dir.glob("????-??-??.md"))
    if not daily_files:
        report.add(LintResult(
            file="memory/", line=0, severity=Severity.INFO,
            message="No daily memory files (YYYY-MM-DD.md) found in memory/",
            rule="daily-files"
        ))

    # Check daily file naming
    for f in memory_dir.glob("*.md"):
        name = f.stem
        # Skip non-daily files
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", name):
            continue
        # Validate date format
        parts = name.split("-")
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if not (2024 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31):
                report.add(LintResult(
                    file=f"memory/{f.name}", line=0, severity=Severity.WARNING,
                    message=f"Daily file has unusual date: {name}",
                    rule="date-format"
                ))
        except ValueError:
            report.add(LintResult(
                file=f"memory/{f.name}", line=0, severity=Severity.ERROR,
                message=f"Invalid date in filename: {name}",
                rule="date-format"
            ))

    # Check for oversized daily files
    for f in daily_files:
        size = f.stat().st_size
        if size > 50_000:  # 50KB
            report.add(LintResult(
                file=f"memory/{f.name}", line=0, severity=Severity.WARNING,
                message=f"Daily file is {size // 1024}KB — consider archiving older entries",
                rule="file-size"
            ))


def lint_workspace(workspace_path: str, strict: bool = False) -> LintReport:
    """
    Lint an entire agent workspace directory.

    Args:
        workspace_path: Path to the agent workspace root
        strict: If True, treat warnings as errors

    Returns:
        LintReport with all findings
    """
    workspace = Path(workspace_path).resolve()
    report = LintReport()

    if not workspace.is_dir():
        report.add(LintResult(
            file=str(workspace), line=0, severity=Severity.ERROR,
            message=f"Workspace directory not found: {workspace}",
            rule="workspace-exists"
        ))
        return report

    # Check for core files
    for filename in CORE_FILES:
        filepath = workspace / filename
        if filepath.exists():
            report.files_found.append(filename)
        else:
            report.files_missing.append(filename)
            report.add(LintResult(
                file=filename, line=0, severity=Severity.ERROR,
                message=f"Core container file missing: {filename}",
                rule="core-file-exists"
            ))

    # Check optional files
    for filename in OPTIONAL_FILES:
        filepath = workspace / filename
        if filepath.exists():
            report.files_found.append(filename)
        # Optional files don't generate errors when missing

    # Lint each found file
    all_files = CORE_FILES + OPTIONAL_FILES
    for filename in all_files:
        filepath = workspace / filename
        result = read_file(filepath)
        if result is None:
            continue

        report.files_checked += 1
        content, lines = result

        # Empty file check
        if not content.strip():
            report.add(LintResult(
                file=filename, line=1, severity=Severity.ERROR,
                message=f"File is empty",
                rule="non-empty"
            ))
            continue

        # Schema validation
        if filename in SCHEMAS:
            lint_structure(filename, lines, SCHEMAS[filename], report)

        # Content quality
        lint_content_quality(filename, content, lines, report)

    # Memory directory checks
    lint_memory_dir(workspace, report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Container Config Linter — validate agent workspace files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/spencer                    # Lint Spencer's workspace
  %(prog)s ~/giles --strict             # Strict mode (warnings = errors)
  %(prog)s ~/spencer ~/giles ~/mia      # Lint multiple workspaces
  %(prog)s . --json                     # JSON output for CI
        """
    )
    parser.add_argument(
        "workspaces", nargs="+", metavar="DIR",
        help="Agent workspace directory/directories to lint"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors (non-zero exit)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Only show errors and warnings, suppress info"
    )

    args = parser.parse_args()

    total_errors = 0
    total_warnings = 0

    for ws_path in args.workspaces:
        ws_name = Path(ws_path).name
        print(f"\n🔍 Linting workspace: {ws_name} ({Path(ws_path).resolve()})")
        print("-" * 60)

        report = lint_workspace(ws_path, strict=args.strict)

        # Print results
        for result in sorted(report.results, key=lambda r: (r.file, r.line)):
            if args.quiet and result.severity == Severity.INFO:
                continue
            print(f"  {result}")

        print(report.summary())

        total_errors += len(report.errors)
        total_warnings += len(report.warnings)

        if args.json:
            import json
            data = {
                "workspace": str(Path(ws_path).resolve()),
                "files_checked": report.files_checked,
                "files_found": report.files_found,
                "files_missing": report.files_missing,
                "results": [
                    {
                        "file": r.file,
                        "line": r.line,
                        "severity": r.severity.value,
                        "message": r.message,
                        "rule": r.rule,
                    }
                    for r in report.results
                ]
            }
            print(json.dumps(data, indent=2))

    # Exit code
    if total_errors > 0:
        sys.exit(1)
    elif args.strict and total_warnings > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
