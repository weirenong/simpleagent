

# workflows.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re
import shutil
import utils


DEFAULT_WORKFLOW_MARKDOWN = """
# Default SimpleAgent workflows

prompt_start: "default"
add_persona_context
add_recent_messages
add_attachment_context
add_web_context
add_original_user_prompt
prompt: "output"
prompt_end
""".strip()


@dataclass(slots=True)
class WorkflowCommand:
    name: str
    argument: str = ""
    raw: str = ""
    line_number: int = 0


@dataclass(slots=True)
class WorkflowPromptBlock:
    name: str
    commands: list[WorkflowCommand] = field(default_factory=list)
    output_name: str | None = None


@dataclass(slots=True)
class WorkflowDefinition:
    prompts: list[WorkflowPromptBlock] = field(default_factory=list)
    source_path: str | None = None


@dataclass(slots=True)
class WorkflowPromptResult:
    name: str
    messages: list[dict[str, str]]
    raw_output: str = ""
    visible_output: str = ""


@dataclass(slots=True)
class WorkflowResult:
    messages: list[dict[str, str]]
    prompt_results: dict[str, WorkflowPromptResult]
    visible_output: str = ""


class WorkflowParseError(ValueError):
    pass


class WorkflowExecutionError(RuntimeError):
    pass


class WorkflowParser:
    """
    Parse markdown workflows scripts into prompt blocks.

    Design goal:
    Users may decorate workflows files with Markdown headings, bullets, code fences,
    quotes, horizontal rules, or visual separators. The parser first strips common
    Markdown formatting and then only keeps recognised workflows command lines.

    Supported command examples:
    - prompt_start: "custom_name"
    - start_prompt: "custom_name"
    - add_persona_context
    - add_memory_context
    - add_recent_messages
    - add_attachment_context
    - add_web_context
    - add_system_context: "custom context"
    - add_original_user_prompt
    - add_to_original_user_prompt: "additional instructions"
    - add_prompt_output: "previous_prompt_name"
    - add_user_prompt: "new user-style instruction"
    - print: "message to show in the interface"
    - prompt: "output"
    - prompt_end
    """

    COMMAND_NAMES = {
        "prompt_start",
        "start_prompt",
        "prompt_end",
        "end_prompt",
        "add_persona_context",
        "add_memory_context",
        "add_recent_messages",
        "add_attachment_context",
        "add_web_context",
        "add_system_context",
        "add_original_user_prompt",
        "add_to_original_user_prompt",
        "add_prompt_output",
        "add_user_prompt",
        "print",
        "prompt",
    }

    def parse_file(self, path: str | Path) -> WorkflowDefinition:
        workflow_path = Path(path).expanduser()
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

        workflow = self.parse_text(workflow_path.read_text(encoding="utf-8"))
        workflow.source_path = str(workflow_path)
        return workflow

    def parse_text(self, markdown: str) -> WorkflowDefinition:
        prompts: list[WorkflowPromptBlock] = []
        current_prompt: WorkflowPromptBlock | None = None

        workflow_lines = collect_multiline_command_lines(markdown)

        for line_number, raw_line in workflow_lines:
            stripped_line = strip_markdown_formatting(raw_line)
            if not stripped_line:
                continue

            command = parse_command_line(stripped_line, line_number=line_number)
            if command is None:
                continue

            if command.name in {"prompt_start", "start_prompt"}:
                if current_prompt is not None:
                    raise WorkflowParseError(
                        f"Nested prompt block at line {line_number}: {raw_line}"
                    )

                prompt_name = command.argument or f"prompt_{len(prompts) + 1}"
                current_prompt = WorkflowPromptBlock(name=normalise_prompt_name(prompt_name))
                continue

            if command.name in {"prompt_end", "end_prompt"}:
                if current_prompt is None:
                    raise WorkflowParseError(f"prompt_end without prompt_start at line {line_number}")
                prompts.append(current_prompt)
                current_prompt = None
                continue

            if current_prompt is None:
                continue

            if command.name == "prompt":
                current_prompt.output_name = command.argument or "output"
                current_prompt.commands.append(command)
                continue

            current_prompt.commands.append(command)

        if current_prompt is not None:
            raise WorkflowParseError(f"Prompt block was not closed: {current_prompt.name}")

        if not prompts:
            raise WorkflowParseError("Workflow does not contain any prompt_start/prompt_end blocks.")

        return WorkflowDefinition(prompts=prompts)


class WorkflowRunner:
    """
    Execute workflows prompt blocks against a SimpleAgent app instance.

    Prompt construction assumptions:
    - Each prompt block builds a fresh message list.
    - Context instructions add system/context messages to that prompt only.
    - `add_original_user_prompt` inserts the original user prompt as a user message.
    - `add_to_original_user_prompt` appends instructions to the original user prompt.
    - `add_user_prompt` creates a new user message inside the current prompt.
    - `add_prompt_output` inserts a previous prompt's raw model output as context.
    - `prompt: "output"` marks that prompt's model output as visible to the user.
    """

    def __init__(self, app: Any, workflow: WorkflowDefinition) -> None:
        self.app = app
        self.workflow = workflow

    @classmethod
    def from_markdown(cls, app: Any, markdown: str) -> WorkflowRunner:
        return cls(app=app, workflow=WorkflowParser().parse_text(markdown))

    @classmethod
    def from_file(cls, app: Any, path: str | Path) -> WorkflowRunner:
        return cls(app=app, workflow=WorkflowParser().parse_file(path))

    @classmethod
    def default(cls, app: Any) -> WorkflowRunner:
        return cls.from_markdown(app=app, markdown=DEFAULT_WORKFLOW_MARKDOWN)

    def build_messages(self, original_user_prompt: str) -> list[dict[str, str]]:
        first_prompt = self.workflow.prompts[0]
        return self.build_prompt_messages(
            prompt_block=first_prompt,
            original_user_prompt=original_user_prompt,
            prompt_results={},
        )

    def run(
            self,
            original_user_prompt: str,
            execute_model: bool = False,
    ) -> WorkflowResult:
        prompt_results: dict[str, WorkflowPromptResult] = {}
        visible_output = ""
        last_messages: list[dict[str, str]] = []

        for prompt_block in self.workflow.prompts:
            if execute_model:
                self.print_workflow_messages(prompt_block)

            messages = self.build_prompt_messages(
                prompt_block=prompt_block,
                original_user_prompt=original_user_prompt,
                prompt_results=prompt_results,
            )
            last_messages = messages

            if execute_model:
                raw_output = self.app.run_chat_model_for_workflow(messages)
                visible_output_block = strip_thinking_blocks(raw_output)
            else:
                raw_output = ""
                visible_output_block = ""

            prompt_result = WorkflowPromptResult(
                name=prompt_block.name,
                messages=messages,
                raw_output=raw_output,
                visible_output=visible_output_block if prompt_block.output_name else "",
            )
            prompt_results[prompt_block.name] = prompt_result

            if prompt_block.output_name:
                visible_output = prompt_result.visible_output

            if not execute_model:
                break

        return WorkflowResult(
            messages=last_messages,
            prompt_results=prompt_results,
            visible_output=visible_output,
        )

    def print_workflow_messages(self, prompt_block: WorkflowPromptBlock) -> None:
        for command in prompt_block.commands:
            if command.name != "print" or not command.argument:
                continue

            print()
            if hasattr(self.app, "print_tui_markdown"):
                self.app.print_tui_markdown(command.argument)
            else:
                print(command.argument)
            print()

    def build_prompt_messages(
        self,
        prompt_block: WorkflowPromptBlock,
        original_user_prompt: str,
        prompt_results: dict[str, WorkflowPromptResult],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        original_prompt_additions: list[str] = []

        for command in prompt_block.commands:
            if command.name == "add_persona_context":
                self.add_system_message(messages, self.app.build_system_prompt())
                continue

            if command.name == "add_memory_context":
                self.add_memory_context(messages, original_user_prompt)
                continue

            if command.name == "add_recent_messages":
                messages.extend(self.get_recent_messages())
                continue

            if command.name == "add_attachment_context":
                self.add_attachment_context(messages, original_user_prompt)
                continue

            if command.name == "add_web_context":
                self.add_web_context(messages, original_user_prompt)
                continue

            if command.name == "add_system_context":
                if command.argument:
                    self.add_system_message(messages, command.argument)
                continue

            if command.name == "add_original_user_prompt":
                messages.append(
                    {
                        "role": "user",
                        "content": build_original_user_prompt(
                            original_user_prompt,
                            original_prompt_additions,
                        ),
                    }
                )
                continue

            if command.name == "add_to_original_user_prompt":
                if command.argument:
                    original_prompt_additions.append(command.argument)
                continue

            if command.name == "add_prompt_output":
                self.add_prompt_output_context(messages, command.argument, prompt_results)
                continue

            if command.name == "add_user_prompt":
                if command.argument:
                    messages.append({"role": "user", "content": command.argument})
                continue

            if command.name in {"prompt", "print"}:
                continue

            raise WorkflowExecutionError(
                f"Unsupported workflow command at line {command.line_number}: {command.raw}"
            )

        return normalise_messages(messages)

    def add_system_message(self, messages: list[dict[str, str]], content: str) -> None:
        content = str(content or "").strip()
        if content:
            messages.append({"role": "system", "content": content})

    def add_memory_context(self, messages: list[dict[str, str]], original_user_prompt: str) -> None:
        memory_context = self.app.get_relevant_memory_context(original_user_prompt)
        if not memory_context:
            return

        messages.append(
            {
                "role": "system",
                "content": (
                    "Compacted older conversation context selected by embeddings. "
                    "Treat this as background memory. The latest user message below is higher priority.\n\n"
                    f"{memory_context}"
                ),
            }
        )

    def get_recent_messages(self) -> list[dict[str, str]]:
        max_recent_messages = int(getattr(self.app, "max_recent_messages", 0) or 2)
        recent_messages = list(getattr(self.app, "messages", []))[-max_recent_messages:]
        return normalise_messages([dict(message) for message in recent_messages])

    def add_attachment_context(self, messages: list[dict[str, str]], original_user_prompt: str) -> None:
        attachment_context = self.app.get_attachment_context(original_user_prompt)
        if not attachment_context:
            return

        messages.append(
            {
                "role": "system",
                "content": (
                    "Attachment context. Text/table/document attachments were selected by embeddings. "
                    "Text-like attachments include both full source text and ranked embedding chunks. "
                    "Image attachments are included in full because visual descriptions should not be ranked away. "
                    "The latest user message is still the main task.\n\n"
                    f"Attachment context:\n{attachment_context}"
                ),
            }
        )

    def add_web_context(self, messages: list[dict[str, str]], original_user_prompt: str) -> None:
        web_context = self.app.get_relevant_web_context(original_user_prompt)
        if not web_context:
            return

        messages.append(
            {
                "role": "system",
                "content": (
                    "Web context selected by embeddings from pages loaded with /web. "
                    "Use this only when relevant to the latest user message. "
                    "The latest user message is still the main task.\n\n"
                    f"Web context:\n{web_context}"
                ),
            }
        )

    def add_prompt_output_context(
        self,
        messages: list[dict[str, str]],
        prompt_name: str,
        prompt_results: dict[str, WorkflowPromptResult],
    ) -> None:
        normalised_prompt_name = normalise_prompt_name(prompt_name)
        prompt_result = prompt_results.get(normalised_prompt_name)

        output_text = ""
        if prompt_result is not None:
            output_text = prompt_result.visible_output or strip_thinking_blocks(prompt_result.raw_output)

        if not output_text.strip():
            return

        messages.append(
            {
                "role": "system",
                "content": (
                    f"Output from prompt `{normalised_prompt_name}`:\n\n"
                    f"{output_text.strip()}"
                ),
            }
        )




# -----------------------------
# Public helpers
# -----------------------------


def strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL).strip()


def load_workflow(path: str | Path | None = None) -> WorkflowDefinition:
    if path is None:
        return WorkflowParser().parse_text(DEFAULT_WORKFLOW_MARKDOWN)
    return WorkflowParser().parse_file(path)


def build_workflow_messages(
    app: Any,
    original_user_prompt: str,
    workflow_path: str | Path | None = None,
) -> list[dict[str, str]]:
    workflow = load_workflow(workflow_path)
    return WorkflowRunner(app=app, workflow=workflow).build_messages(
        original_user_prompt=original_user_prompt,
    )


def run_workflow(
    app: Any,
    original_user_prompt: str,
    workflow_path: str | Path | None = None,
    execute_model: bool = False,
) -> WorkflowResult:
    workflow = load_workflow(workflow_path)
    return WorkflowRunner(app=app, workflow=workflow).run(
        original_user_prompt=original_user_prompt,
        execute_model=execute_model,
    )


def ensure_default_workflow_file(path: str | Path = "workflows/default.md") -> Path:
    workflow_path = Path(path)
    workflow_path.parent.mkdir(parents=True, exist_ok=True)

    if not workflow_path.exists():
        workflow_path.write_text(DEFAULT_WORKFLOW_MARKDOWN + "\n", encoding="utf-8")

    return workflow_path

def install_workflow_file(source_path: str | Path, destination_dir: str | Path) -> str:
    """
    Install a workflow markdown file into the user's workflow directory.

    Returns the installed workflow name, which is the copied file's stem.
    The copied file is parsed after copying so invalid workflow files fail early.
    """
    source = Path(source_path).expanduser().resolve()

    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Workflow file not found: {source}")

    if source.suffix.lower() != ".md":
        raise WorkflowParseError(f"Workflow file must be a .md file: {source}")

    workflow_name = normalise_workflow_file_name(source.stem)
    destination = Path(destination_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    target_path = destination / f"{workflow_name}.md"
    shutil.copy2(source, target_path)

    # Validate after copying.
    WorkflowParser().parse_file(target_path)

    return workflow_name


def collect_multiline_command_lines(markdown: str) -> list[tuple[int, str]]:
    """
    Collapse workflow command lines with multi-line quoted arguments.

    This allows workflow entries such as:

        add_user_prompt: "Line one.
        Line two.
        Line three."

    The parser still receives one logical command line, with embedded newlines
    preserved inside the argument.
    """
    logical_lines: list[tuple[int, str]] = []
    pending_line_number = 0
    pending_lines: list[str] = []
    pending_quote = ""

    for line_number, raw_line in enumerate(markdown.splitlines(), start=1):
        if pending_lines:
            pending_lines.append(raw_line)
            if closes_multiline_argument(raw_line, pending_quote):
                logical_lines.append((pending_line_number, "\n".join(pending_lines)))
                pending_line_number = 0
                pending_lines = []
                pending_quote = ""
            continue

        stripped_line = strip_markdown_formatting(raw_line)
        if starts_multiline_argument(stripped_line):
            pending_line_number = line_number
            pending_lines = [raw_line]
            pending_quote = get_argument_opening_quote(stripped_line)
            continue

        logical_lines.append((line_number, raw_line))

    if pending_lines:
        logical_lines.append((pending_line_number, "\n".join(pending_lines)))

    return logical_lines


def starts_multiline_argument(line: str) -> bool:
    if ":" not in line:
        return False

    raw_name, raw_argument = line.split(":", 1)
    name = normalise_command_name(raw_name)
    if name not in WorkflowParser.COMMAND_NAMES:
        return False

    argument = raw_argument.lstrip()
    if not argument or argument[0] not in {'"', "'"}:
        return False

    return not quoted_argument_is_closed(argument)


def get_argument_opening_quote(line: str) -> str:
    if ":" not in line:
        return ""

    argument = line.split(":", 1)[1].lstrip()
    if argument and argument[0] in {'"', "'"}:
        return argument[0]

    return ""


def closes_multiline_argument(line: str, quote: str) -> bool:
    if not quote:
        return False

    stripped = line.rstrip()
    if not stripped:
        return False

    backslash_count = 0
    for character in reversed(stripped[:-1]):
        if character == "\\":
            backslash_count += 1
        else:
            break

    return stripped.endswith(quote) and backslash_count % 2 == 0


def quoted_argument_is_closed(argument: str) -> bool:
    if not argument:
        return False

    quote = argument[0]
    if quote not in {'"', "'"}:
        return False

    escaped = False

    for index, character in enumerate(argument[1:], start=1):
        if escaped:
            escaped = False
            continue

        if character == "\\":
            escaped = True
            continue

        if character == quote:
            trailing = argument[index + 1 :].strip()
            return trailing == ""

    return False


# -----------------------------
# Parsing helpers
# -----------------------------


def strip_markdown_formatting(raw_line: str) -> str:
    line = raw_line.strip()

    if not line:
        return ""

    if line.startswith("```") or line.startswith("~~~"):
        return ""

    line = re.sub(r"^#{1,6}\s+", "", line)
    line = re.sub(r"^>+\s*", "", line)
    line = re.sub(r"^[-*+]\s+", "", line)
    line = re.sub(r"^\d+[.)]\s+", "", line)

    if line in {"---", "***", "___"}:
        return ""

    line = line.strip()

    # Strip inline code wrapping when users decorate commands as `command: value`.
    if len(line) >= 2 and line[0] == "`" and line[-1] == "`":
        line = line[1:-1].strip()

    line = line.strip("*_ ")
    return line.strip()


def parse_command_line(line: str, line_number: int) -> WorkflowCommand | None:
    command_text = line.strip()
    if not command_text:
        return None

    if ":" in command_text:
        raw_name, raw_argument = command_text.split(":", 1)
        name = normalise_command_name(raw_name)
        argument = strip_optional_quotes(raw_argument.strip())
    else:
        name = normalise_command_name(command_text)
        argument = ""

    if name not in WorkflowParser.COMMAND_NAMES:
        return None

    return WorkflowCommand(
        name=name,
        argument=argument,
        raw=line,
        line_number=line_number,
    )


def normalise_command_name(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    aliases = {
        "start_prompt": "prompt_start",
        "end_prompt": "prompt_end",
        "persona_context": "add_persona_context",
        "add_persona": "add_persona_context",
        "recent_messages": "add_recent_messages",
        "memory_context": "add_memory_context",
        "relevant_memory_context": "add_memory_context",
        "add_recent_memory": "add_recent_messages",
        "attachment_context": "add_attachment_context",
        "web_context": "add_web_context",
        "system_context": "add_system_context",
        "add_context": "add_system_context",
        "context": "add_system_context",
        "original_user_prompt": "add_original_user_prompt",
        "user_prompt": "add_user_prompt",
        "echo": "print",
        "append_original_user_prompt": "add_to_original_user_prompt",
        "append_to_original_user_prompt": "add_to_original_user_prompt",
        "prompt_output": "add_prompt_output",
        "use_prompt_output": "add_prompt_output",
        "output_prompt": "prompt",
    }

    return aliases.get(cleaned, cleaned)


def strip_optional_quotes(value: str) -> str:
    value = str(value or "")
    stripped = value.strip()

    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]

    return stripped


def normalise_prompt_name(value: str) -> str:
    cleaned = strip_optional_quotes(str(value or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9_\- ]+", "", cleaned).strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "prompt"

def normalise_workflow_file_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\- ]+", "", str(value or "").strip()).lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("_-")
    return cleaned or "workflow"

# -----------------------------
# Runtime helpers
# -----------------------------


def build_original_user_prompt(original_user_prompt: str, additions: list[str]) -> str:
    prompt = original_user_prompt.strip()
    clean_additions = [addition.strip() for addition in additions if addition.strip()]

    if not clean_additions:
        return prompt

    return f"{prompt}\n\n" + "\n\n".join(clean_additions)


def normalise_messages(value: Any) -> list[dict[str, str]]:
    if value is None or value == "":
        return []

    if isinstance(value, dict):
        role = str(value.get("role") or "system")
        content = str(value.get("content") or "")
        return [{"role": role, "content": content}] if content else []

    if isinstance(value, list):
        messages: list[dict[str, str]] = []
        for item in value:
            messages.extend(normalise_messages(item))
        return messages

    return [{"role": "system", "content": str(value)}]