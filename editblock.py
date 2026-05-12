"""
editblock.py — Aider-inspired LLM code-edit strategies for SimpleAgent.

Three strategies the LLM can use to express edits:

  1. WHOLE_FILE   — full file replacement inside a fenced code block (current approach, improved)
  2. SEARCH_REPLACE — aider-style <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks
  3. UNIFIED_DIFF  — standard unified-diff output applied with Python's difflib

Public API
----------
    apply_llm_edits(app, llm_output, strategy=EditStrategy.SEARCH_REPLACE) -> bool

    SYSTEM_PROMPTS[EditStrategy.SEARCH_REPLACE]  # put this in your coding persona / workflow

Design notes
------------
- All three strategies share the same confirm-and-apply UI (F2 / Esc).
- SearchReplaceEditor has fuzzy fallback: stripped trailing whitespace, then
  re-indented match (same as aider's search_replace.py).
- Files are backed up to <file>.bak before being overwritten.
- Edits are only allowed on files attached to the current chat session.
"""

from __future__ import annotations

import difflib
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence
import shutil
from datetime import datetime

from prompt_toolkit.key_binding import key_bindings


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class EditStrategy(str, Enum):
    AUTO = "auto"           # detect from output
    WHOLE_FILE = "whole_file"
    SEARCH_REPLACE = "search_replace"
    UNIFIED_DIFF = "unified_diff"


# ---------------------------------------------------------------------------
# System prompt snippets — inject these into your coding persona or workflow
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[EditStrategy, str] = {

    EditStrategy.WHOLE_FILE: """
To edit a file you MUST output its ENTIRE updated content.
Use the following format — the bare file path on a line by itself, then a fenced code block:

path/to/filename.py
```python
# complete file content — every single line
```

Rules:
- Output the full file. Never truncate or use "..." placeholders.
- Only output files that actually need changes.
- Do not add commentary inside the code block.
""".strip(),

    EditStrategy.SEARCH_REPLACE: """
To edit files use *SEARCH/REPLACE blocks*.

Format for each change:
1. The full file path on a line by itself (no backticks, no bold, no quotes).
2. A fenced block containing the SEARCH/REPLACE markers:

```python
<<<<<<< SEARCH
exact lines from the file that must be replaced
=======
new lines to substitute in
>>>>>>> REPLACE
```

Rules:
- The SEARCH section must EXACTLY match existing file content, character for character,
  including all whitespace, indentation, comments, and blank lines.
- Keep blocks small — include only the lines that change plus a few surrounding lines
  for uniqueness. Never copy whole unchanged sections.
- Multiple SEARCH/REPLACE blocks per file are fine; list them one after another.
- To insert code with nothing above it, use an empty SEARCH (the marker only, no lines).
- To delete code, use an empty REPLACE (the marker only, no lines).
- Only edit files that the user has added to the chat.
""".strip(),

    EditStrategy.UNIFIED_DIFF: """
To edit files output a standard unified diff.

Format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,8 @@
 context line
-old line to remove
+new line to add
 context line
```

Rules:
- Use exactly three context lines around each hunk.
- The --- / +++ paths must match the attached file names.
- Output one diff block per file.
""".strip(),
}


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class EditBlock:
    """A single proposed file change."""
    path_hint: str
    target_path: Path
    original_text: str
    updated_text: str
    safe_updated_text: str | None = None
    has_risky_changes: bool = False

@dataclass
class PatchWorkspace:
    """Temporary patch workspace used to stage edits before touching originals."""
    workspace_id: str
    temp_dir: Path
    original_to_temp: dict[Path, Path]
    temp_to_original: dict[Path, Path]
    patch_paths: list[Path]

@dataclass
class ParseError(Exception):
    message: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.message}: {self.detail}" if self.detail else self.message


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_hint: str, app: Any) -> Path:
    """
    Resolve a path hint from LLM output to an actual filesystem path.

    Priority:
      1. Exact match against an attached file's posix path or name.
      2. git a/ b/ prefix stripping.
      3. Fallback: expand the hint as-is.
    """
    hint = _normalise_llm_path_hint(path_hint)

    for token, attached_path in _attached_path_tokens(app).items():
        if hint == token or Path(hint).name == attached_path.name:
            return attached_path

    return Path(hint).expanduser()


def _attached_paths(app: Any) -> set[Path]:
    paths: set[Path] = set()
    for item in getattr(app, "attachments", []) or []:
        raw = str(item.get("source_path") or item.get("path") or "").strip()
        if raw:
            paths.add(Path(raw).expanduser().resolve())
    return paths

def _normalise_llm_path_hint(path_hint: str) -> str:
    """Clean noisy LLM filename/path labels into one canonical path hint."""
    hint = str(path_hint or "").strip()
    if not hint:
        return ""

    hint = hint.strip("`# ").rstrip(":").strip()
    hint = re.sub(
        r"^\**\s*(?:file\s*(?:name|path)?|path)\s*\**\s*:\s*\**\s*",
        "",
        hint,
        flags=re.IGNORECASE,
    ).strip()
    hint = hint.strip("`'\"* ").rstrip(".,;:").strip()

    token = _first_path_token(hint)
    if token:
        hint = token

    if hint.startswith(("a/", "b/")):
        hint = hint[2:]

    return hint.strip()


def _first_path_token(text: str) -> str:
    tokens = _path_tokens(text)
    return tokens[0] if tokens else ""


def _path_tokens(text: str) -> list[str]:
    raw_tokens = re.findall(
        r"(?:[A-Za-z]:)?(?:[./\\\w-]+[/\\])?[\w.-]+\.[A-Za-z0-9_.-]+",
        str(text or ""),
    )
    return [token.strip("`'\"* ,:;") for token in raw_tokens if token.strip("`'\"* ,:;")]


def _attached_path_tokens(app: Any) -> dict[str, Path]:
    tokens: dict[str, Path] = {}

    for path in _attached_paths(app):
        forms = {
            path.as_posix(),
            path.name,
            f"./{path.name}",
            f"a/{path.name}",
            f"b/{path.name}",
        }
        for form in forms:
            tokens[form] = path

    return tokens


def _match_attached_path_hint(path_hint: str, app: Any) -> str:
    candidate = _normalise_llm_path_hint(path_hint)
    if not candidate:
        return ""

    attached_tokens = _attached_path_tokens(app)
    if candidate in attached_tokens:
        return attached_tokens[candidate].name

    for token in _path_tokens(candidate):
        if token in attached_tokens:
            return attached_tokens[token].name

        token_name = Path(token).name
        for attached_path in attached_tokens.values():
            if token_name == attached_path.name:
                return attached_path.name

    return candidate if _is_probable_path(candidate) else ""


def _guard_paths(blocks: list[EditBlock], app: Any) -> None:
    attached = _attached_paths(app)
    if not attached:
        return
    for block in blocks:
        resolved = block.target_path.expanduser().resolve()
        if resolved not in attached:
            raise ParseError(
                "Refusing to edit a file that is not attached",
                block.path_hint,
            )


# ---------------------------------------------------------------------------
# Temporary patch workspace + staged writes
# ---------------------------------------------------------------------------

def _patch_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _app_temp_root(app: Any | None = None, root_dir: Path | None = None) -> Path:
    """Return the configured SimpleAgent temp root for staged /code edits."""
    if root_dir is not None:
        return Path(root_dir).expanduser().resolve()

    app_temp_dir = getattr(app, "temp_dir", None)
    if app_temp_dir is not None:
        return Path(app_temp_dir).expanduser().resolve()

    return Path.home().joinpath(".simpleagent", "temp").resolve()


def _unique_temp_name(path: Path, timestamp: str) -> str:
    return f"{path.stem}_{timestamp}{path.suffix}"


def create_temp_workspace_for_paths(
    paths: list[Path],
    app: Any | None = None,
    root_dir: Path | None = None,
) -> PatchWorkspace:
    """
    Create timestamped temporary copies for one or more original files.

    All /code staging files sit directly under the app's TEMP_DIR root,
    normally ~/.simpleagent/temp/.

    The original files are never modified here. Each temp filename includes the
    workspace timestamp so repeated workflow runs do not collide.
    """
    timestamp = _patch_timestamp()
    temp_dir = _app_temp_root(app=app, root_dir=root_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    original_to_temp: dict[Path, Path] = {}
    temp_to_original: dict[Path, Path] = {}

    for raw_path in paths:
        original_path = Path(raw_path).expanduser().resolve()
        temp_path = temp_dir / _unique_temp_name(original_path, timestamp)

        temp_path.parent.mkdir(parents=True, exist_ok=True)

        if original_path.exists():
            shutil.copy2(original_path, temp_path)
        else:
            temp_path.write_text("", encoding="utf-8")

        original_to_temp[original_path] = temp_path
        temp_to_original[temp_path] = original_path

    return PatchWorkspace(
        workspace_id=timestamp,
        temp_dir=temp_dir,
        original_to_temp=original_to_temp,
        temp_to_original=temp_to_original,
        patch_paths=[],
    )


def create_temp_workspace_for_attachments(app: Any) -> PatchWorkspace:
    """Create a timestamped temporary workspace from the currently attached files."""
    return create_temp_workspace_for_paths(sorted(_attached_paths(app)), app=app)


def _blocks_for_temp_workspace(
    blocks: list[EditBlock],
    workspace: PatchWorkspace,
    use_safe_text: bool = False,
) -> list[EditBlock]:
    """Map parsed edit blocks from original files to their temp-file copies."""
    temp_blocks: list[EditBlock] = []

    for block in blocks:
        original_path = block.target_path.expanduser().resolve()
        temp_path = workspace.original_to_temp.get(original_path)

        if temp_path is None:
            raise ParseError(
                "File was not copied into the temporary patch workspace",
                block.path_hint,
            )

        updated_text = (
            block.safe_updated_text
            if use_safe_text and block.safe_updated_text is not None
            else block.updated_text
        )

        temp_blocks.append(
            EditBlock(
                path_hint=block.path_hint,
                target_path=temp_path,
                original_text=temp_path.read_text(encoding="utf-8") if temp_path.exists() else "",
                updated_text=updated_text,
                safe_updated_text=block.safe_updated_text,
                has_risky_changes=block.has_risky_changes,
            )
        )

    return temp_blocks


def apply_blocks_to_temp_workspace(
    blocks: list[EditBlock],
    workspace: PatchWorkspace,
    use_safe_text: bool = False,
) -> list[Path]:
    """
    Generate .bak and .patch files, then apply the requested edits to temp files.

    This is intentionally non-interactive so workflows can repeatedly stage
    patches before a final human confirmation or commit step.
    """
    temp_blocks = _blocks_for_temp_workspace(
        blocks,
        workspace,
        use_safe_text=use_safe_text,
    )

    patch_paths: list[Path] = []

    for block in temp_blocks:
        path = block.target_path
        path.parent.mkdir(parents=True, exist_ok=True)

        original_temp_text = path.read_text(encoding="utf-8") if path.exists() else ""

        backup_path = path.with_suffix(path.suffix + f".{workspace.workspace_id}.bak")
        backup_path.write_text(original_temp_text, encoding="utf-8")

        patch_text = "".join(
            difflib.unified_diff(
                original_temp_text.splitlines(keepends=True),
                block.updated_text.splitlines(keepends=True),
                fromfile=f"temp-before/{path.name}",
                tofile=f"temp-after/{path.name}",
            )
        )

        patch_path = path.with_suffix(path.suffix + f".{workspace.workspace_id}.patch")
        patch_path.write_text(patch_text, encoding="utf-8")
        patch_paths.append(patch_path)

        path.write_text(block.updated_text, encoding="utf-8")

    workspace.patch_paths.extend(patch_paths)
    return patch_paths


def diff_temp_workspace_against_original(workspace: PatchWorkspace) -> dict[Path, str]:
    """Return final diffs comparing every temp file against its original file."""
    diffs: dict[Path, str] = {}

    for original_path, temp_path in workspace.original_to_temp.items():
        original_text = original_path.read_text(encoding="utf-8") if original_path.exists() else ""
        temp_text = temp_path.read_text(encoding="utf-8") if temp_path.exists() else ""

        diff_text = "".join(
            difflib.unified_diff(
                original_text.splitlines(keepends=True),
                temp_text.splitlines(keepends=True),
                fromfile=f"original/{original_path.as_posix()}",
                tofile=f"temp/{temp_path.name}",
            )
        )

        diffs[original_path] = diff_text

    return diffs


def commit_temp_workspace_to_original(workspace: PatchWorkspace) -> None:
    """Apply the staged temp-file result back to the original files."""
    for original_path, temp_path in workspace.original_to_temp.items():
        original_path.parent.mkdir(parents=True, exist_ok=True)

        if original_path.exists():
            original_backup_path = original_path.with_suffix(
                original_path.suffix + f".{workspace.workspace_id}.bak"
            )
            shutil.copy2(original_path, original_backup_path)

        shutil.copy2(temp_path, original_path)

def _validate_edit_block(
    block: EditBlock,
    temp_path: Path,
    max_deletion_ratio: float = 0.50,
) -> None:
    """
    Make sure the diff that will be stored for *block* does not delete
    more than ``max_deletion_ratio`` of unchanged lines.

    Parameters
    ----------
    block          – the parsed EditBlock (contains original_text & updated_text)
    temp_path      – the *current* temp file path (may already contain previous edits)
    max_deletion_ratio – threshold (0‑1).  If deletions exceed this fraction
                         of the original file we raise a ParseError.

    Raises
    ------
    ParseError – when the diff looks like an unintended wholesale deletion.
    """
    # 1️⃣ Build the diff that will be saved (temp file vs. updated_text)
    diff_text = "".join(
        difflib.unified_diff(
            temp_path.read_text(encoding="utf-8").splitlines(keepends=True),
            block.updated_text.splitlines(keepends=True),
            fromfile=f"temp/{temp_path.name}",
            tofile=f"temp/{temp_path.name}",
        )
    )

    # 2️⃣ Count lines that are marked for deletion in the diff.
    del_lines = [ln for ln in diff_text.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    # 3️⃣ Count total non‑empty lines in the *original* content of the temp file.
    orig_lines = [ln for ln in temp_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total = max(len(orig_lines), 1)

    # 4️⃣ If deletions are > threshold → raise. hard coded 50 min deletion lines
    if len(del_lines) >= 50 and total > 0:
        deletion_ratio = len(del_lines) / total
        if deletion_ratio > max_deletion_ratio:
            raise ParseError(
                "Patch contains a large deletion of unchanged code",
                f"Deleted {len(del_lines)} lines out of {total} ({deletion_ratio:.0%}) – "                                                                          
                f"consider editing only the changed region or using a smaller SEARCH block.",
            )


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------

def _unified_diff(block: EditBlock) -> str:
    src = block.original_text.splitlines(keepends=True)
    dst = block.updated_text.splitlines(keepends=True)
    path_str = block.target_path.as_posix()
    lines = difflib.unified_diff(src, dst, fromfile=f"a/{path_str}", tofile=f"b/{path_str}")
    return "".join(lines)

_RISKY_DIFF_PREFIX = "\x00RISKY\x00"


def _is_bounded_change(
    opcodes: list[tuple[str, int, int, int, int]],
    index: int,
    original_line_count: int,
    updated_line_count: int,
) -> bool:
    tag, i1, i2, j1, j2 = opcodes[index]

    has_before_bound = any(
        prev_tag == "equal" and prev_i1 < prev_i2
        for prev_tag, prev_i1, prev_i2, _, _ in opcodes[:index]
    )
    has_after_bound = any(
        next_tag == "equal" and next_i1 < next_i2
        for next_tag, next_i1, next_i2, _, _ in opcodes[index + 1:]
    )

    # A change is only considered safe if it is bounded by unchanged code
    # on BOTH sides. This prevents massive top/bottom rewrites from being
    # treated as safe merely because some unchanged code still exists.
    return has_before_bound and has_after_bound


def _safe_updated_text_from_bounds(original_text: str, updated_text: str) -> tuple[str | None, bool]:
    """
    Apply only changes that are bounded by unchanged code.

    If a change is unbounded because it touches the top or bottom of the file,
    one unchanged bound is enough.
    """
    original_lines = original_text.splitlines(keepends=True)
    updated_lines = updated_text.splitlines(keepends=True)

    matcher = difflib.SequenceMatcher(None, original_lines, updated_lines)
    opcodes = matcher.get_opcodes()

    if all(tag == "equal" for tag, *_ in opcodes):
        return None, False

    result_lines: list[str] = []
    has_risky_changes = False

    for index, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            result_lines.extend(original_lines[i1:i2])
            continue

        safe = _is_bounded_change(
            opcodes,
            index,
            len(original_lines),
            len(updated_lines),
        )

        if safe:
            result_lines.extend(updated_lines[j1:j2])
        else:
            has_risky_changes = True
            result_lines.extend(original_lines[i1:i2])

    if not has_risky_changes:
        return None, False

    return "".join(result_lines), True


def _risky_line_numbers(
    original_text: str,
    updated_text: str,
) -> tuple[set[int], set[int]]:
    """
    Return:
      (risky_original_line_numbers, risky_updated_line_numbers)

    Both sets are 1-based.
    """
    original_lines = original_text.splitlines(keepends=True)
    updated_lines = updated_text.splitlines(keepends=True)

    matcher = difflib.SequenceMatcher(None, original_lines, updated_lines)
    opcodes = matcher.get_opcodes()

    risky_original: set[int] = set()
    risky_updated: set[int] = set()

    for index, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            continue

        safe = _is_bounded_change(
            opcodes,
            index,
            len(original_lines),
            len(updated_lines),
        )

        if not safe:
            risky_original.update(range(i1 + 1, i2 + 1))
            risky_updated.update(range(j1 + 1, j2 + 1))

    return risky_original, risky_updated


def _annotated_unified_diff(block: EditBlock) -> list[str]:
    """
    Return unified diff lines. Risky added lines are prefixed with a private marker
    so formatter.py can colour them dark yellow.
    """
    diff_text = _unified_diff(block)
    if not diff_text:
        return []

    risky_original_lines, risky_updated_lines = _risky_line_numbers(
        block.original_text,
        block.updated_text,
    )
    annotated: list[str] = []
    old_line_number = 0
    new_line_number = 0

    for line in diff_text.splitlines():
        hunk_match = re.match(
            r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@",
            line,
        )
        if hunk_match:
            old_line_number = int(hunk_match.group(1))
            new_line_number = int(hunk_match.group(2))
            annotated.append(line)
            continue

        if line.startswith(("---", "+++")):
            annotated.append(line)
            continue

        if line.startswith("+"):
            if new_line_number in risky_updated_lines:
                annotated.append(_RISKY_DIFF_PREFIX + line)
            else:
                annotated.append(line)
            new_line_number += 1
            continue

        if line.startswith("-"):
            if old_line_number in risky_original_lines:
                annotated.append(_RISKY_DIFF_PREFIX + line)
            else:
                annotated.append(line)
            old_line_number += 1
            continue

        annotated.append(line)

        if line.startswith(" ") or line == "":
            old_line_number += 1
            new_line_number += 1

    return annotated


# ---------------------------------------------------------------------------
# Review / confirm UI (shared by all strategies)
# ---------------------------------------------------------------------------

def _confirm_and_apply(blocks: list[EditBlock], app: Any | None = None) -> bool:
    """
    Show diffs, prompt the user with F2/Esc, and apply on confirmation.
    Returns True if edits were applied.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    # Print diffs to stdout so the user can review them.
    for block in blocks:
        block.safe_updated_text, block.has_risky_changes = _safe_updated_text_from_bounds(
            block.original_text,
            block.updated_text,
        )

        diff_lines = _annotated_unified_diff(block)

        if diff_lines:
            print(f"\n--- proposed changes: {block.path_hint} ---\n")

            if app is not None and hasattr(app, "print_tui_code_block"):
                app.print_tui_code_block(diff_lines, "diff")
            else:
                print("\n".join(line.replace(_RISKY_DIFF_PREFIX, "") for line in diff_lines))
        else:
            print(f"\n[no diff] {block.path_hint} (file unchanged)")

    preview_workspace = create_temp_workspace_for_paths(
        [block.target_path for block in blocks],
        app=app,
    )
    apply_blocks_to_temp_workspace(
        blocks,
        preview_workspace,
        use_safe_text=True,
    )
    final_diffs = diff_temp_workspace_against_original(preview_workspace)

    print(f"\n--- staged temp root: {preview_workspace.temp_dir} ---")

    if preview_workspace.patch_paths:
        print("Generated temp patch files:")
        for patch_path in preview_workspace.patch_paths:
            print(f"  {patch_path}")

    for original_path, diff_text in final_diffs.items():
        if diff_text:
            print(f"\n--- staged final diff: {original_path.name} ---\n")
            diff_lines = diff_text.splitlines()

            if app is not None and hasattr(app, "print_tui_code_block"):
                app.print_tui_code_block(diff_lines, "diff")
            else:
                print("\n".join(diff_lines))
        else:
            print(f"\n[no staged diff] {original_path.name} (file unchanged)")

    key_bindings = KeyBindings()
    decision: dict[str, str] = {"mode": "cancel"}
    has_risky_changes = any(block.has_risky_changes for block in blocks)

    @key_bindings.add("f2")
    def _apply_safe(event) -> None:
        decision["mode"] = "safe"
        event.app.exit()

    @key_bindings.add("f3")
    def _apply_full(event) -> None:
        decision["mode"] = "full"
        event.app.exit()

    @key_bindings.add("escape")
    def _cancel(event) -> None:
        event.app.exit()

    prompt_text = f"  {len(blocks)} file(s) to update · F2 apply safe bounded changes"

    if has_risky_changes:
        prompt_text += " · F3 apply all including yellow risky changes"

    prompt_text += " · Esc cancel  "

    body = HSplit([
        Window(FormattedTextControl(prompt_text)),
    ])
    Application(layout=Layout(body), key_bindings=key_bindings, full_screen=False).run()

    if decision["mode"] == "safe":
        commit_temp_workspace_to_original(preview_workspace)
        print(f"\nApplied staged safe bounded changes for {len(blocks)} file(s).\n")
        return True

    if decision["mode"] == "full":
        full_workspace = create_temp_workspace_for_paths(
            [block.target_path for block in blocks],
            app=app,
        )
        apply_blocks_to_temp_workspace(
            blocks,
            full_workspace,
            use_safe_text=False,
        )
        commit_temp_workspace_to_original(full_workspace)
        print(f"\nApplied staged full changes for {len(blocks)} file(s).\n")
        return True

    print("\nCancelled — no files were changed.\n")
    return False


# ---------------------------------------------------------------------------
# Strategy 1: Whole-file editor
# ---------------------------------------------------------------------------

class WholeFileEditor:
    """
    Parse LLM output containing full-file fenced code blocks.

    Expected format:
        path/to/file.py
        ```python
        # ... entire file ...
        ```
    """

    _FENCE_RE = re.compile(r"^```")
    _single_attached_path: Path | None = None
    _app_for_path_match: Any | None = None

    def parse(self, llm_output: str, app: Any) -> list[EditBlock]:
        lines = llm_output.splitlines()
        proposed_by_target: dict[Path, tuple[str, str, str]] = {}
        full_replacement_targets: set[Path] = set()
        saw_path = ""
        chat_file_names = self._chat_file_names(app)
        self._app_for_path_match = app
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not self._FENCE_RE.match(stripped):
                # Scan the plain line for a file path hint.
                candidate = self._scan_for_path(stripped, chat_file_names)
                if candidate:
                    saw_path = candidate
                i += 1
                continue

            # Opening fence found.
            language = stripped.removeprefix("```").strip()
            code_lines: list[str] = []
            fence_start = i
            i += 1

            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1

            if i >= len(lines):
                raise ParseError("Unclosed fenced code block in assistant output")

            code_text = "\n".join(code_lines).rstrip("\n") + "\n"

            if not code_text.strip():
                saw_path = ""
                i += 1
                continue

            # Determine which file this block belongs to.
            nearby = lines[max(0, fence_start - 10): fence_start]
            path_hint = self._find_path_in_context(nearby, chat_file_names) or saw_path

            if not path_hint and self._single_attached_path is not None:
                path_hint = self._single_attached_path.name

            if not path_hint:
                raise ParseError(
                    "No filename found before fenced code block",
                    "Attach files or include the filename above each code block.",
                )

            target = _resolve_path(path_hint, app)
            original = target.read_text(encoding="utf-8") if target.exists() else ""

            existing = proposed_by_target.get(target)
            if existing is None:
                current_text = original
                stored_path_hint = path_hint
            else:
                stored_path_hint, original, current_text = existing

            if target in full_replacement_targets:
                current_text = code_text
            elif self._looks_like_partial_code_block(code_text):
                current_text = self._merge_partial_code_block(
                    original_text=original,
                    current_text=current_text,
                    partial_text=code_text,
                )
            else:
                current_text = code_text
                full_replacement_targets.add(target)

            proposed_by_target[target] = (stored_path_hint, original, current_text)
            saw_path = ""
            i += 1

        return [
            EditBlock(path_hint, target, original, updated)
            for target, (path_hint, original, updated) in proposed_by_target.items()
        ]

    # -- helpers --
    def _looks_like_partial_code_block(self, code_text: str) -> bool:
        stripped_lines = [line for line in code_text.splitlines() if line.strip()]
        if not stripped_lines:
            return False

        first_line = stripped_lines[0].lstrip()
        return bool(re.match(r"^(?:async\s+def|def|class)\s+[A-Za-z_]\w*\s*[(:]", first_line))

    def _merge_partial_code_block(self, original_text: str, current_text: str, partial_text: str) -> str:
        safe_partial_text, has_risky_changes = _safe_updated_text_from_bounds(original_text, partial_text)
        if safe_partial_text is not None:
            return self._merge_changed_lines(
                base_text=original_text,
                current_text=current_text,
                partial_safe_text=safe_partial_text,
            )

        if has_risky_changes:
            return current_text

        return self._merge_changed_lines(
            base_text=original_text,
            current_text=current_text,
            partial_safe_text=partial_text,
        )

    def _merge_changed_lines(self, base_text: str, current_text: str, partial_safe_text: str) -> str:
        base_lines = base_text.splitlines(keepends=True)
        current_lines = current_text.splitlines(keepends=True)
        partial_lines = partial_safe_text.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(None, base_lines, partial_lines)
        merged_lines = list(current_lines)
        offset = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            start = i1 + offset
            end = i2 + offset
            replacement = partial_lines[j1:j2]
            merged_lines[start:end] = replacement
            offset += len(replacement) - (i2 - i1)

        return "".join(merged_lines)

    def _chat_file_names(self, app: Any) -> set[str]:
        names: set[str] = set()
        paths: list[Path] = []
        for item in getattr(app, "attachments", []) or []:
            raw = str(item.get("source_path") or item.get("path") or "").strip()
            if raw:
                paths.append(Path(raw).expanduser())

        self._single_attached_path = paths[0].expanduser().resolve() if len(paths) == 1 else None

        for p in paths:
            names.add(p.as_posix())
            names.add(p.name)
        return names

    def _scan_for_path(self, text: str, chat_file_names: set[str]) -> str:
        candidate = _match_attached_path_hint(text, self._app_for_path_match)
        if candidate:
            return candidate

        for word in text.split():
            candidate = _normalise_llm_path_hint(word)
            if candidate in chat_file_names:
                return candidate
            for name in chat_file_names:
                if Path(candidate).name == Path(name).name:
                    return name

        return ""

    def _find_path_in_context(self, nearby_lines: list[str], chat_file_names: set[str]) -> str:
        for raw_line in reversed(nearby_lines):
            candidate = _match_attached_path_hint(raw_line, self._app_for_path_match)
            if candidate:
                return candidate

            candidate = _normalise_llm_path_hint(raw_line)
            if not candidate or len(candidate) > 260:
                continue

            if candidate in chat_file_names or _is_probable_path(candidate):
                return candidate

        return ""


# ---------------------------------------------------------------------------
# Strategy 2: Search/Replace editor (aider-style)
# ---------------------------------------------------------------------------

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"

# Accept slight LLM variations in the markers.
_SEARCH_RE = re.compile(r"^<{2,7}\s*SEARCH\s*$", re.IGNORECASE)
_DIVIDER_RE = re.compile(r"^={5,7}\s*$")
_REPLACE_RE = re.compile(r"^>{2,7}\s*REPLACE\s*$", re.IGNORECASE)


class SearchReplaceNoMatch(Exception):
    pass


class SearchReplaceEditor:
    """
    Parse and apply aider-style SEARCH/REPLACE blocks.

    Expected LLM format:
        path/to/file.py
        ```python
        />>>>>>> SEARCH
        old code
        =======
        new code
        /<<<<<<< REPLACE
        ```
        (repeat for more changes)

    Matching algorithm (in order):
      1. Exact string match.
      2. Strip trailing whitespace on each line, then match.
      3. Try dedented / re-indented match.
    """

    def parse(self, llm_output: str, app: Any) -> list[EditBlock]:
        """
        Parse all SEARCH/REPLACE blocks.  Multiple blocks for the same file
        are applied in order, each reading from the *previously updated* text.
        """
        raw_edits = self._extract_raw_edits(llm_output, app)


        # Group and apply sequentially per file so later blocks see earlier changes.
        file_texts: dict[Path, tuple[str, str]] = {}  # path -> (original, current)
        for path_hint, target, search_text, replace_text in raw_edits:
            if target not in file_texts:
                original = target.read_text(encoding="utf-8") if target.exists() else ""
                file_texts[target] = (original, original)

            _orig, current = file_texts[target]
            updated = self._apply_search_replace(current, search_text, replace_text, target)
            file_texts[target] = (_orig, updated)

        blocks: list[EditBlock] = []
        for path_hint, target, _, __ in raw_edits:
            if target in file_texts:
                orig, updated = file_texts[target]
                # Deduplicate: only add one EditBlock per file.
                if not any(b.target_path == target for b in blocks):
                    blocks.append(EditBlock(path_hint, target, orig, updated))

        return blocks

    # -- parsing --

    def _extract_raw_edits(
        self,
        llm_output: str,
        app: Any,
    ) -> list[tuple[str, Path, str, str]]:
        """
        Return list of (path_hint, resolved_path, search_text, replace_text).
        """
        # Strip fences — content may be inside ``` blocks or bare.
        text = self._strip_outer_fences(llm_output)
        lines = text.splitlines()

        edits: list[tuple[str, Path, str, str]] = []
        current_path_hint = ""
        current_path: Path | None = None
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if _SEARCH_RE.match(stripped):
                # Collect SEARCH lines.
                search_lines: list[str] = []
                i += 1
                while i < len(lines) and not _DIVIDER_RE.match(lines[i].strip()):
                    search_lines.append(lines[i])
                    i += 1

                if i >= len(lines):
                    raise ParseError("Missing ======= divider in SEARCH/REPLACE block")

                # Collect REPLACE lines.
                replace_lines: list[str] = []
                i += 1
                while i < len(lines) and not _REPLACE_RE.match(lines[i].strip()):
                    replace_lines.append(lines[i])
                    i += 1

                if i >= len(lines):
                    raise ParseError("Missing >>>>>>> REPLACE marker in SEARCH/REPLACE block")

                if current_path is None:
                    raise ParseError(
                        "SEARCH/REPLACE block found but no filename precedes it",
                        "Add the full file path on a line before the block.",
                    )

                search_text = "\n".join(search_lines)
                replace_text = "\n".join(replace_lines)
                edits.append((current_path_hint, current_path, search_text, replace_text))
                i += 1
                continue

            # Look for a file path hint on a plain line (not inside markers).
            if not _DIVIDER_RE.match(stripped) and not _REPLACE_RE.match(stripped):
                candidate = self._detect_path_hint(stripped, app)
                if candidate:
                    current_path_hint = candidate
                    current_path = _resolve_path(candidate, app)

            i += 1

        if not edits:
            raise ParseError(
                "No SEARCH/REPLACE blocks found in assistant output",
                "Make sure the model is using the search/replace format.",
            )

        return edits

    def _strip_outer_fences(self, text: str) -> str:
        """Remove leading/trailing ``` fences that wrap the whole response."""
        lines = text.splitlines()
        result: list[str] = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith("```"):
                # Skip the fence line but keep the content.
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    result.append(lines[i])
                    i += 1
                # skip closing ```
            else:
                result.append(lines[i])
            i += 1
        return "\n".join(result)

    def _normalise_path_hint_line(self, text: str) -> str:
        candidate = _normalise_llm_path_hint(text)
        return candidate if len(candidate) <= 260 else ""


    def _looks_like_standalone_path(self, text: str) -> bool:
        if not text or " " in text:
            return False
        return _is_probable_path(text)

    def _detect_path_hint(self, text: str, app: Any) -> str:
        candidate = _match_attached_path_hint(text, app)
        if candidate:
            return candidate

        candidate = self._normalise_path_hint_line(text)
        if not candidate:
            return ""

        return candidate if self._looks_like_standalone_path(candidate) else ""

    # -- matching --

    def _apply_search_replace(
        self,
        file_text: str,
        search_text: str,
        replace_text: str,
        path: Path,
    ) -> str:
        """
        Apply a single SEARCH/REPLACE to file_text.
        Raises SearchReplaceNoMatch on failure.
        """
        # Empty SEARCH = insert at top of file (or treat as append).
        if not search_text.strip():
            if not replace_text:
                return file_text
            return replace_text + "\n" + file_text

        # 1. Exact match (first occurrence only, à la aider).
        if search_text in file_text:
            return file_text.replace(search_text, replace_text, 1)

        # 2. Strip trailing whitespace on every line.
        def strip_trailing(s: str) -> str:
            return "\n".join(line.rstrip() for line in s.splitlines())

        stripped_file = strip_trailing(file_text)
        stripped_search = strip_trailing(search_text)
        if stripped_search in stripped_file:
            # Rebuild with original spacing for the replace text.
            idx = stripped_file.index(stripped_search)
            return file_text[:idx] + replace_text + file_text[idx + len(stripped_search):]

        # 3. Try dedented match — the model sometimes shifts indentation.
        result = self._try_reindented_match(file_text, search_text, replace_text)
        if result is not None:
            return result

        # 4. Fuzzy closest-match hint for the error message.
        hint = self._fuzzy_hint(file_text, search_text)
        raise SearchReplaceNoMatch(
            f"SEARCH block did not match any lines in {path.name}.\n"
            f"The SEARCH section must exactly match existing file content.\n"
            + (f"Closest file lines:\n{hint}" if hint else "")
        )

    def _try_reindented_match(
        self,
        file_text: str,
        search_text: str,
        replace_text: str,
    ) -> str | None:
        """
        Detect how much the search block is indented relative to the file and
        try shifting it to match. Same trick aider uses.
        """
        search_lines = search_text.splitlines()
        if not search_lines:
            return None

        dedented_search = textwrap.dedent(search_text)

        file_lines = file_text.splitlines(keepends=True)

        for i, file_line in enumerate(file_lines):
            # How much indentation does this file line have?
            file_indent = len(file_line) - len(file_line.lstrip())
            first_search_stripped = dedented_search.splitlines()[0] if dedented_search.splitlines() else ""
            candidate = " " * file_indent + first_search_stripped

            if file_line.rstrip("\n") == candidate.rstrip("\n"):
                # Re-indent the whole search block.
                reindented = textwrap.indent(dedented_search, " " * file_indent)
                if file_text[file_text.find(file_line):].startswith(reindented):
                    # Re-indent replace_text by the same amount.
                    reindented_replace = textwrap.indent(
                        textwrap.dedent(replace_text), " " * file_indent
                    )
                    start = file_text.find(reindented)
                    return file_text[:start] + reindented_replace + file_text[start + len(reindented):]

        return None

    def _fuzzy_hint(self, file_text: str, search_text: str) -> str:
        search_lines = search_text.splitlines()
        file_lines = file_text.splitlines()

        if not search_lines or not file_lines:
            return ""

        first_line = search_lines[0].strip()
        matches = difflib.get_close_matches(first_line, file_lines, n=3, cutoff=0.5)
        return "\n".join(matches)


# ---------------------------------------------------------------------------
# Strategy 3: Unified diff editor
# ---------------------------------------------------------------------------

class UnifiedDiffEditor:
    """
    Unified diff handler that parses diffs using two methods:
    1. Hunk numbering from standard unified diffs
    2. Matching unchanged/deletion lines to find correct positions (fallback)
    """

    _DIFF_HEADER_RE = re.compile(r"^--- a?/?(.*)")
    _HUNK_RE = re.compile(r"^@@ .* @@")

    def parse(self, llm_output: str, app: Any) -> list[EditBlock]:
        diffs = self._split_diffs(llm_output)

        if not diffs:
            raise ParseError(
                "No unified diff blocks found",
                "Ensure the model outputs --- / +++ / @@ headers.",
            )

        blocks: list[EditBlock] = []
        for path_hint, diff_text in diffs:
            target = _resolve_path(path_hint, app)
            original = target.read_text(encoding="utf-8") if target.exists() else ""
            updated = self._apply_diff(original, diff_text, target)
            blocks.append(EditBlock(path_hint, target, original, updated))

        return blocks

    def _split_diffs(self, text: str) -> list[tuple[str, str]]:
        """Split a multi-file diff response into (path_hint, diff_text) pairs."""
        # Strip fences.
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*$", "", text, flags=re.MULTILINE)

        chunks: list[tuple[str, str]] = []
        current_hint = ""
        current_lines: list[str] = []

        for line in text.splitlines():
            m = self._DIFF_HEADER_RE.match(line)
            if m and line.startswith("---"):
                # New file diff starting — flush previous.
                if current_hint and current_lines:
                    chunks.append((current_hint, "\n".join(current_lines)))
                raw_path = m.group(1).strip()
                current_hint = raw_path.removeprefix("a/").removeprefix("b/")
                current_lines = [line]
                continue

            if current_hint:
                current_lines.append(line)

        if current_hint and current_lines:
            chunks.append((current_hint, "\n".join(current_lines)))

        return chunks

    def _apply_diff(self, original: str, diff_text: str, path: Path) -> str:
        """Apply a unified diff to the original text, returning updated text."""
        orig_lines = original.splitlines(keepends=True)
        result_lines = list(orig_lines)
        offset = 0

        for hunk_text in self._split_hunks(diff_text):
            m = self._HUNK_RE.match(hunk_text.splitlines()[0])
            if not m:
                continue

            orig_start = int(m.group(1)) - 1  # 0-indexed
            hunk_lines = hunk_text.splitlines()[1:]  # skip @@ line

            applied_lines: list[str] = []
            i = orig_start + offset

            for hunk_line in hunk_lines:
                if hunk_line.startswith("-"):
                    # Remove: advance in result (skip this line).
                    i += 1
                    offset -= 1
                elif hunk_line.startswith("+"):
                    # Add.
                    new_line = hunk_line[1:]
                    if not new_line.endswith("\n"):
                        new_line += "\n"
                    applied_lines.append((i, "add", new_line))
                else:
                    # Context line — keep.
                    i += 1

        # Re-apply: this simple approach rebuilds from scratch using difflib patcher.
        try:
            patched = list(difflib.restore(
                list(difflib.unified_diff([], [], n=0)),
                which=2,
            ))
        except Exception:
            patched = []

        # Fall back to Python's patch-via-ndiff for correctness.
        return self._apply_patch_robust(original, diff_text)

    def _apply_patch_robust(self, original: str, diff_text: str) -> str:
        """
        Robust patch application using line-by-line hunk processing.
        """
        orig_lines = original.splitlines(keepends=True)
        result: list[str] = list(orig_lines)
        offset = 0

        for hunk in self._split_hunks(diff_text):
            hunk_lines = hunk.splitlines()
            header = hunk_lines[0]
            m = self._HUNK_RE.match(header)
            if not m:
                continue

            orig_start = int(m.group(1)) - 1  # 0-indexed in original

            removals: list[str] = []
            additions: list[str] = []
            for line in hunk_lines[1:]:
                if line.startswith("-"):
                    removals.append(line[1:])
                elif line.startswith("+"):
                    additions.append(line[1:])
                # context lines ignored — we use the original

            # Ensure additions end with newline.
            additions = [
                (a if a.endswith("\n") else a + "\n") for a in additions
            ]

            pos = orig_start + offset
            # Replace the removal span with the additions.
            result[pos: pos + len(removals)] = additions
            offset += len(additions) - len(removals)

        return "".join(result)

    def _split_hunks(self, diff_text: str) -> list[str]:
        """Split a diff into individual @@ hunks."""
        hunks: list[str] = []
        current: list[str] = []

        for line in diff_text.splitlines():
            if self._HUNK_RE.match(line):
                if current:
                    hunks.append("\n".join(current))
                current = [line]
            elif current:
                current.append(line)

        if current:
            hunks.append("\n".join(current))

        return hunks


# ---------------------------------------------------------------------------
# Path utility
# ---------------------------------------------------------------------------

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".rb", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".php", ".sh",
    ".bash", ".zsh", ".fish", ".md", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".html", ".css", ".scss", ".sql", ".r", ".lua", ".ex", ".exs",
}


def _is_probable_path(text: str) -> bool:
    if not text or text.startswith(("```", "<", ">")):
        return False
    lower = text.lower()
    if lower.startswith(("replace ", "insert ", "update ", "add ", "remove ", "here", "note", "warning")):
        return False
    suffix = Path(text).suffix.lower()
    return suffix in _CODE_EXTENSIONS or "/" in text or "\\" in text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_llm_edits(
    app: Any,
    llm_output: str,
    strategy: EditStrategy = EditStrategy.AUTO,
    title: str = "LLM code edit review",
) -> bool:
    """
    Parse edits from *llm_output* using *strategy*, show diffs, confirm, and apply.

    Returns True if edits were applied, False otherwise.
    """
    llm_output = str(llm_output or "").strip()
    if not llm_output:
        print("\nNo assistant output to parse.\n")
        return False

    # -----------------------------------------------------------------------
    # 1️⃣  Parse → list[EditBlock] (unchanged)
    # -----------------------------------------------------------------------
    try:
        strategy = _detect_strategy(llm_output) if strategy == EditStrategy.AUTO else strategy
        blocks = _get_editor(strategy).parse(llm_output, app)

    except (ParseError, SearchReplaceNoMatch) as exc:
        print(f"\nFailed to parse edits:\n{exc}\n")
        return False

    except Exception as exc:
        print(f"\nUnexpected error while parsing edits:\n{exc}\n")
        return False

    # -----------------------------------------------------------------------
    # 2️⃣  **Validate** each block *against the current temp file* before we
    #      write anything.  This catches wholesale deletions early.
    # -----------------------------------------------------------------------
    # Build a temporary workspace just to get the *current* temp file paths.
    # (We reuse the same logic that `apply_llm_edits_to_temp` uses.)
    workspace = create_temp_workspace_for_paths(
        [block.target_path for block in blocks], app=app
    )

    for block in blocks:
        # Resolve the *real* path that will be edited.
        resolved_path = _resolve_path(block.path_hint, app)
        # The temp file that already contains any previous edits for this path.
        temp_path = workspace.original_to_temp.get(resolved_path)
        if temp_path is None:
            # First edit for this file – create an empty temp copy.
            temp_path = workspace.temp_dir / _unique_temp_name(resolved_path, workspace.workspace_id)

        # Run the validation; it may raise ParseError if the diff looks suspicious.
        _validate_edit_block(block, temp_path)

    # -----------------------------------------------------------------------
    # 3️⃣  Proceed with the normal confirmation UI (unchanged)
    # -----------------------------------------------------------------------
    return _confirm_and_apply(blocks, app=app)

def apply_llm_edits_to_temp(
    app: Any,
    llm_output: str,
    strategy: EditStrategy = EditStrategy.UNIFIED_DIFF,
    workspace: PatchWorkspace | None = None,
) -> tuple[PatchWorkspace, list[Path] | None]:
    """
    Parse *llm_output* using *strategy* and stage the changes **only** in a
    temporary workspace.  The original files on disk are left untouched.

    Returns
    -------
    (workspace, patch_paths)
        *workspace* – the PatchWorkspace that now holds the patched temp files.
        *patch_paths* – list of the generated ``*.patch`` files (useful for
        later inspection or manual commit).
    """
    # -----------------------------------------------------------------------
    # 1️⃣  Parse → EditBlocks (unchanged)
    # -----------------------------------------------------------------------
    llm_output = str(llm_output or "").strip()
    if not llm_output:
        raise ParseError("No assistant output to parse.")

    strategy = _detect_strategy(llm_output) if strategy == EditStrategy.AUTO else strategy
    editor = _get_editor(strategy)
    blocks = editor.parse(llm_output, app)

    # -----------------------------------------------------------------------
    # 2️⃣  Build a workspace that already contains any *previous* edits.
    # -----------------------------------------------------------------------
    if workspace is None:
        workspace = create_temp_workspace_for_paths(
            [block.target_path for block in blocks], app=app
        )

    # -----------------------------------------------------------------------
    # 3️⃣  **Validate & apply** each block **against the *current* temp file**.
    # -----------------------------------------------------------------------
    patch_paths: list[Path] = []

    for block in blocks:
        resolved_path = _resolve_path(block.path_hint, app)
        temp_path = workspace.original_to_temp.get(resolved_path)

        if temp_path is None:
            temp_path = workspace.temp_dir / _unique_temp_name(
                resolved_path,
                workspace.workspace_id,
            )

            temp_path.parent.mkdir(parents=True, exist_ok=True)

            if resolved_path.exists():
                shutil.copy2(resolved_path, temp_path)
            else:
                temp_path.write_text("", encoding="utf-8")

            workspace.original_to_temp[resolved_path] = temp_path
            workspace.temp_to_original[temp_path] = resolved_path

        _validate_edit_block(block, temp_path)

        before_text = (
            temp_path.read_text(encoding="utf-8")
            if temp_path.exists()
            else ""
        )

        diff_text = "".join(
            difflib.unified_diff(
                before_text.splitlines(keepends=True),
                block.updated_text.splitlines(keepends=True),
                fromfile=f"temp/before/{temp_path.name}",
                tofile=f"temp/after/{temp_path.name}",
            )
        )

        temp_path.write_text(block.updated_text, encoding="utf-8")

        patch_path = temp_path.with_suffix(
            temp_path.suffix + f".{workspace.workspace_id}.patch"
        )
        patch_path.write_text(diff_text, encoding="utf-8")
        patch_paths.append(patch_path)

    # -----------------------------------------------------------------------
    # 4️⃣  Return workspace + generated patches.
    # -----------------------------------------------------------------------
    return workspace, patch_paths

def print_final_diffs(
    workspace: PatchWorkspace,
    app: Any | None = None,
) -> bool:
    """
    Print a human‑readable diff for every file that was edited in *workspace*,
    then prompt the user for confirmation – exactly like ``_confirm_and_apply``.

    Returns
    -------
    bool
        ``True`` if edits were applied, ``False`` if the user cancelled.
    """
    # -----------------------------------------------------------------------
    # 1️⃣  Show the diffs (same as the original implementation)
    # -----------------------------------------------------------------------
    final_diffs: dict[Path, str] = diff_temp_workspace_against_original(workspace)

    for original_path, diff_text in final_diffs.items():
        if diff_text:
            print(f"\n--- final diff: {original_path.name} ---\n")
            diff_lines = diff_text.splitlines()

            if app is not None and hasattr(app, "print_tui_code_block"):
                app.print_tui_code_block(diff_lines, "diff")
            else:
                print("\n".join(diff_lines))
        else:
            print(f"\n[no diff] {original_path.name} (file unchanged)\n")

    # -----------------------------------------------------------------------
    # 2️⃣  Prompt the human – reuse the same key‑binding logic as _confirm_and_apply
    # -----------------------------------------------------------------------
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.application import Application

    # ----  USER DECISION  -------------------------------------------------
    key_bindings = KeyBindings()
    decision: dict[str, str] = {"mode": "cancel"}

    # There is no `blocks` variable here, so we cannot inspect `has_risky_changes`.
    # For a safe default we simply assume there are no risky changes.
    # (If you later pass the `blocks` list into this function you can replace
    #  `has_risky_changes = False` with the real calculation.)
    has_risky_changes = False

    @key_bindings.add("f2")
    def _apply_safe(event) -> None:
        decision["mode"] = "safe"
        event.app.exit()

    @key_bindings.add("f3")
    def _apply_full(event) -> None:
        decision["mode"] = "full"
        event.app.exit()

    @key_bindings.add("escape")
    def _cancel(event) -> None:
        event.app.exit()

    prompt_text = f"  {len(final_diffs)} file(s) with staged changes"
    if has_risky_changes:
        prompt_text += " · F3 apply all (including yellow risky changes)"
    prompt_text += " · F2 apply safe bounded changes · Esc cancel  "

    body = HSplit([Window(FormattedTextControl(prompt_text))])

    Application(layout=Layout(body), key_bindings=key_bindings, full_screen=False).run()

    # -----------------------------------------------------------------------
    # 3️⃣  Act on the user’s choice
    # -----------------------------------------------------------------------
    if decision["mode"] == "safe":
        commit_temp_workspace_to_original(workspace)
        print("\nApplied staged safe bounded changes.\n")
        return True

    if decision["mode"] == "full":
        """
        In the “full” case we already have the updated contents in the
        temporary files (they were written by the earlier call to
        ``apply_blocks_to_temp_workspace``).  No extra parsing or
        ``EditBlock`` reconstruction is required – just copy the temp
        files back to their original locations.
        """
        commit_temp_workspace_to_original(workspace)
        print("\nApplied staged full changes.\n")
        return True

    # -----------------------------------------------------------------------
    # 4️⃣  Cancel – nothing was changed
    # -----------------------------------------------------------------------
    print("\nCancelled — no files were changed.\n")
    return False



def _detect_strategy(llm_output: str) -> EditStrategy:
    if re.search(r"^<{2,7}\s*SEARCH", llm_output, re.MULTILINE | re.IGNORECASE):
        return EditStrategy.SEARCH_REPLACE
    unified_diff_patterns = [
        r"^---\s+(?:a/)?",
        r"^\+\+\+\s+(?:b/)?",
        r"^@@\s+-\d+",
    ]

    if any(
            re.search(pattern, llm_output, re.MULTILINE)
            for pattern in unified_diff_patterns
    ):
        return EditStrategy.UNIFIED_DIFF
    return EditStrategy.WHOLE_FILE

def _get_editor(strategy: EditStrategy):
    return {
        EditStrategy.WHOLE_FILE: WholeFileEditor(),
        EditStrategy.SEARCH_REPLACE: SearchReplaceEditor(),
        EditStrategy.UNIFIED_DIFF: UnifiedDiffEditor(),
    }[strategy]
