# main.py
from __future__ import annotations

import json
import mimetypes
import platform
import shlex
import subprocess
import os
import random
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path

import workflows.workflow as workflows

import utils
from formatter import TuiFormatter

import editblock
from editblock import EditStrategy

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.application import Application
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import CheckboxList, RadioList, Dialog, Button, Label, Box
from prompt_toolkit.styles import Style

from ollama import OllamaClient, OllamaConfig
from utils import VectorMemoryIndex, vectorise_text
from pollinations import PollinationsClient, PollinationsConfig

from PIL import Image, ImageGrab

VERSION = "0.2.7"
APP_NAME = "SimpleAgent"
DEFAULT_MODEL = "pollinations/qwen-coder"
DEFAULT_EMBEDDING_MODEL = "pollinations/openai-3-small"
DEFAULT_VISION_MODEL = "pollinations/mistral"
MAX_RECENT_MESSAGES = 2
MAX_MEMORY_TEXT_LENGTH = 900
MAX_RELEVANT_MEMORY_ITEMS = 8
MAX_RELEVANT_ATTACHMENT_ITEMS = 8
MAX_RELEVANT_WEB_ITEMS = 8
MAX_ATTACH_COMPLETION_CANDIDATES = 80
MAX_ATTACHMENT_CONTEXT_TOKEN_RATIO = 0.40
MIN_ATTACHMENT_CONTEXT_TOKENS = 1_500
MAX_ATTACHMENT_CONTEXT_TOKENS = 6_000
RESERVED_RESPONSE_TOKENS = 1_500

CONFIG_DIR = Path.home() / ".simpleagent"
CONFIG_FILE = CONFIG_DIR / "config.json"
USER_WORKFLOWS_DIR = CONFIG_DIR / "workflows"
PROJECT_WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"
DEFAULT_WORKFLOW_PATH = PROJECT_WORKFLOWS_DIR / "default.md"

DEFAULT_WORKFLOW_NAME = "default"
DEFAULT_PERSONA_WORKFLOWS = {
    "Default": "default",
    "Coding": "coding",
}

TEMP_DIR = CONFIG_DIR / "temp"

SUPPORTED_ATTACHMENT_EXTENSIONS = (
    utils.TEXT_ATTACHMENT_EXTENSIONS
    | utils.IMAGE_ATTACHMENT_EXTENSIONS
    | utils.EXCEL_ATTACHMENT_EXTENSIONS
    | utils.PDF_ATTACHMENT_EXTENSIONS
    | utils.DOCX_ATTACHMENT_EXTENSIONS
)

IMAGE_ATTACHMENT_EXTENSIONS = utils.IMAGE_ATTACHMENT_EXTENSIONS

COMMANDS = {
    "/attach": "Attach supported files by path",
    "/web": "Load a URL or search and store ranked web context",
    "/paste": "Paste text or image from the clipboard",
    "/clear": "Clear current session history",
    "/code": "Stage, review, and apply code edits from the last assistant reply",
    "/workspace": "Reset workspace to terminal folder, or change it by path",
    "/model-chat": "Show or change the Ollama chat model",
    "/model-embedding": "Show or change the Ollama embeddings model",
    "/model-vision": "Show or change the Ollama vision model",
    "/models": "List installed Ollama models",
    "/persona": "Open persona manager for system prompt profiles",
    "/workflow": "Help text for assigning workflows",
    "/workflow-install": "Install a workflow .md file and create a persona for it",
    "/workflow-debug": "Print the full prompt messages sent during last run",
    "/response-debug": "Show raw unprocessed messages from all models since user prompt",
    "/markup": "Toggle markdown-style formatting for agent replies",
    "/history": "Show current session history",
    "/about": "Show app info",
    "/version": "Show version",
    "/help": "Show this help menu",
    "/api-pollinations": "Authenticate with Pollinations API using Bring Your Own Pollen",
    "/exit": "Exit app",
    "/quit": "Exit app",
    "/q": "Exit app",
}


COMMAND_USAGE = {
    "/code": "/code",
    "/model-chat": "/model-chat <name>",
    "/model-embedding": "/model-embedding <name>",
    "/model-vision": "/model-vision <name>",
    "/attach": "/attach <path...>",
    "/paste": "/paste",
    "/web": "/web <url or search query>",
    "/persona": "/persona",
    "/workflow": "/workflow",
    "/workflow-install": "/workflow-install <path-to-workflow.md>",
    "/workflow-debug": "/workflow-debug",
    "/response-debug": "/response-debug",
    "/markup": "/markup",
    "/workspace": "/workspace <path...>",
    "/api-pollinations": "/api-pollinations",
}

THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
STREAM_THINK_START = "<think>"
STREAM_THINK_END = "</think>"

LOADING_FRAMES = [".", "..", "..."]

LOADING_MESSAGES = [
    "tiny brain doing its best",
    "thinking with budget neurons",
    "borrowing wisdom from the void",
    "trying not to hallucinate",
    "counting tokens on fingers",
    "summoning tiny reasoning goblins",
    "spinning the hamster wheel",
    "pretending to be Claude Code",
    "doing mental push-ups",
    "upgrading from potato mode",
    "duct-taping logic together",
    "asking the small brain council",
    "adding extra cope tokens",
    "summoning one more brain cell",
    "running inference on vibes",
    "trying to sound expensive",
    "doing PhD work",
    "manifesting intelligence",
    "budget brain entering flow state",
    "aura farming",
    "thinking very hard, please clap",
    "counting tokens on fingers",
    "running on compact wisdom",
    "squinting at the prompt",
    "work work ship ship",
]

def command_preview(command: str, description: str) -> str:
    usage = COMMAND_USAGE.get(command, command)
    return f"{usage} — {description}"

BOLD = "\033[1m"
RESET = "\033[0m"

MASCOT_LINES = [
    f"  ▗▄▄▄▄▄▄▖    {BOLD}{APP_NAME}{RESET} v{VERSION}",
    " ▐▌> ██ <▐▌   Tiny local AI for tiny machines",
    " ▐▌   ▾  ▐▌   Work work · ship ship",
    "  ▝▀▛██▜▀▘    by Weiren.Ong, 2026",
    "    ▘  ▝      ",
]


def build_help_text() -> str:
    lines = ["", "Available commands:", ""]

    for command, description in COMMANDS.items():
        usage = COMMAND_USAGE.get(command, command)
        lines.append(f"  {usage:<22} {description}")

    lines.extend([
        "",
        "Normal usage:",
        "  Type anything and press Enter to chat with the model.",
        "  Press Shift+Enter to add a new line where supported.",
        "  Fallback: press Esc then Enter to add a new line.",
        "  Type / to open command suggestions.",
    ])

    return "\n".join(lines)

# -----------------------------
# Slash command completion
# -----------------------------

class SlashCommandCompleter(Completer):
    def __init__(self, get_workspace_dir) -> None:
        if callable(get_workspace_dir):
            self.get_workspace_dir = get_workspace_dir
        else:
            self.get_workspace_dir = lambda: Path(get_workspace_dir)

    def get_completions(self, document, complete_event: CompleteEvent):
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        if text.startswith("/attach "):
            yield from self.get_attach_path_completions(text)
            return

        current = text.split(" ", 1)[0]

        for command, description in COMMANDS.items():
            if current == "/" or command.startswith(current):
                yield Completion(
                    text=command,
                    start_position=-len(current),
                    display=command,
                    display_meta=command_preview(command, description),
                )

    def get_attach_path_completions(self, text: str):
        argument_text = text[len("/attach "):]
        fragment = self.current_attach_fragment(argument_text)
        project_dir = self.get_workspace_dir().resolve()

        for path in list_attach_completion_candidates(project_dir, fragment):
            try:
                display_path = path.relative_to(project_dir).as_posix()
            except ValueError:
                display_path = str(path)

            completion_text = shlex.quote(display_path)

            yield Completion(
                text=completion_text,
                start_position=-len(fragment),
                display=display_path,
                display_meta="directory" if path.is_dir() else "file",
            )

    def current_attach_fragment(self, argument_text: str) -> str:
        if not argument_text:
            return ""

        parts = re.split(r"\s+", argument_text)
        return parts[-1] if parts else ""

def list_attach_completion_candidates(project_dir: Path, fragment: str) -> list[Path]:
    cleaned_fragment = fragment.strip().strip("'\"")
    project_dir = project_dir.resolve()

    if cleaned_fragment:
        fragment_path = Path(cleaned_fragment).expanduser()
    else:
        fragment_path = Path("")

    if fragment_path.is_absolute():
        search_dir = fragment_path if fragment_path.is_dir() else fragment_path.parent
        prefix = "" if fragment_path.is_dir() else fragment_path.name
    else:
        relative_parent = fragment_path.parent if str(fragment_path.parent) != "." else Path("")
        search_dir = (project_dir / relative_parent).resolve()
        prefix = "" if cleaned_fragment.endswith(("/", os.sep)) else fragment_path.name

    try:
        search_dir.relative_to(project_dir)
    except ValueError:
        if not fragment_path.is_absolute():
            return []

    if not search_dir.exists() or not search_dir.is_dir():
        return []

    candidates: list[Path] = []

    try:
        children = sorted(search_dir.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return []

    for path in children:
        if len(candidates) >= MAX_ATTACH_COMPLETION_CANDIDATES:
            break

        if path.name.startswith("."):
            continue

        if prefix and not path.name.startswith(prefix):
            continue

        if should_skip_attach_completion_candidate(project_dir, path):
            continue

        if path.is_file() and path.suffix.lower() not in SUPPORTED_ATTACHMENT_EXTENSIONS:
            continue

        if not path.is_dir() and not path.is_file():
            continue

        candidates.append(path)

    return sorted(
        candidates,
        key=lambda path: (
            not path.is_dir(),
            path.name.lower(),
        ),
    )


def should_skip_attach_completion_candidate(project_dir: Path, path: Path) -> bool:
    try:
        relative_path = path.relative_to(project_dir)
    except ValueError:
        return True

    blocked_parts = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".pytest_cache",
        ".mypy_cache",
    }

    if set(relative_path.parts) & blocked_parts:
        return True

    return is_gitignored_for_attach_completion(project_dir, relative_path.as_posix())


def is_gitignored_for_attach_completion(project_dir: Path, relative_path: str) -> bool:
    gitignore_path = project_dir / ".gitignore"
    if not gitignore_path.exists():
        return False

    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative_path],
            cwd=project_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return matches_simple_gitignore_rule_for_attach_completion(gitignore_path, relative_path)


def matches_simple_gitignore_rule_for_attach_completion(gitignore_path: Path, relative_path: str) -> bool:
    try:
        lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    path = relative_path.strip("/")
    ignored = False

    for raw_line in lines:
        rule = raw_line.strip()
        if not rule or rule.startswith("#"):
            continue

        negated = rule.startswith("!")
        if negated:
            rule = rule[1:].strip()

        if not rule:
            continue

        directory_rule = rule.endswith("/")
        rule = rule.strip("/")
        matched = False

        if directory_rule:
            matched = path.startswith(rule + "/")
        elif "*" in rule:
            pattern = re.escape(rule).replace("\\*", ".*")
            matched = re.fullmatch(pattern, path) is not None
        else:
            matched = path == rule or path.startswith(rule + "/") or path.endswith("/" + rule)

        if matched:
            ignored = not negated

    return ignored

BUILT_IN_PERSONA_NAMES = {"Default", "Coding"}


DEFAULT_PERSONA_PROMPT = """
You are SimpleAgent, an agent built for fast, practical work.
Limit your internal reasoning to under 2000 words.

Operating rules:
- Be concise, high-signal, and action-oriented.
- Prefer tables, short sections, and concrete next steps when useful.
- Use available context first: current chat, memory, attachments, web context, and tool output.
- Do not invent tool access, file contents, web facts, or execution results.
- If context is missing, ask one focused question or state the assumption and proceed.
- For multi-step tasks, plan briefly, execute directly, then self-check the result.
- Avoid long theory unless the user asks for it or the task requires it.
- Preserve the user's intent, tone, and constraints.
""".strip()

DEFAULT_CODING_PERSONA_PROMPT = """
You are SimpleAgent, a precise coding assistant. Follow instructions exactly. Never add extra text.
Limit your internal reasoning to under 2000 words.
""".strip()

DEFAULT_PERSONAS = {
    "Default": DEFAULT_PERSONA_PROMPT,
    "Coding": DEFAULT_CODING_PERSONA_PROMPT,
}


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


class SimpleAgentTUI(TuiFormatter):
    def __init__(self) -> None:
        self.config = load_config()
        self.workspace_dir = self.load_workspace_dir()
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.model = self.config.get("model", DEFAULT_MODEL)
        self.host = self.config.get("host", "http://localhost:11434")
        self.embedding_model = self.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self.vision_model = self.config.get("vision_model", DEFAULT_VISION_MODEL)
        
        # Check if default models are Pollinations models and adjust accordingly
        if self.model.startswith("pollinations/"):
            # For Pollinations models, we don't need to initialize Ollama client for these
            # The Ollama client will be used only for non-Pollinations models
            pass
        else:
            # For Ollama models, initialize the Ollama client
            pass
        legacy_system_prompt = self.config.get("system_prompt")
        configured_personas = self.config.get("personas")

        if not isinstance(configured_personas, dict) or not configured_personas:
            configured_personas = {
                "Default": legacy_system_prompt or DEFAULT_PERSONA_PROMPT,
            }

        for persona_name, persona_prompt in DEFAULT_PERSONAS.items():
            configured_personas[persona_name] = persona_prompt

        self.personas: dict[str, str] = {
            str(name): str(prompt)
            for name, prompt in configured_personas.items()
            if str(name).strip() and str(prompt).strip()
        }

        if not self.personas:
            self.personas = {"Default": DEFAULT_PERSONA_PROMPT}

        for persona_name, persona_prompt in DEFAULT_PERSONAS.items():
            self.personas[persona_name] = persona_prompt

        configured_active_persona = str(self.config.get("active_persona") or "Default")
        self.active_persona = (
            configured_active_persona
            if configured_active_persona in self.personas
            else "Default"
            if "Default" in self.personas
            else next(iter(self.personas))
        )
        self.system_prompt = self.personas[self.active_persona]
        configured_persona_workflows = self.config.get("persona_workflows")
        if not isinstance(configured_persona_workflows, dict):
            configured_persona_workflows = {}
        self.persona_workflows: dict[str, str] = {
            str(name): str(workflow_name)
            for name, workflow_name in configured_persona_workflows.items()
            if str(name).strip() and str(workflow_name).strip()
        }
        for persona_name in self.personas:
            self.persona_workflows.setdefault(
                persona_name,
                DEFAULT_PERSONA_WORKFLOWS.get(persona_name, DEFAULT_WORKFLOW_NAME),
            )

        for persona_name, workflow_name in DEFAULT_PERSONA_WORKFLOWS.items():
            if persona_name in self.personas:
                self.persona_workflows[persona_name] = workflow_name
        self.save_persona_config()
        self.messages: list[dict[str, str]] = []
        self.max_recent_messages = MAX_RECENT_MESSAGES
        self.workflow_runner = self.build_workflow_runner_for_persona(self.active_persona)
        self.last_workflow_messages: list[dict[str, str]] = []
        self.memory_items: list[dict] = []
        self.memory_index = VectorMemoryIndex(embedding_key="embedding")
        self.attachments: list[dict[str, str]] = []
        self.attachment_context_items: list[dict] = []
        self.image_attachment_context_items: list[dict] = []
        self.text_attachment_full_context_items: list[dict] = []
        self.attachment_index = VectorMemoryIndex(embedding_key="embedding")
        self.web_context_items: list[dict] = []
        self.web_sources: list[dict[str, str]] = []
        self.web_index = VectorMemoryIndex(embedding_key="embedding")
        self.next_input_prefill: str = ""
        self.last_thinking: str = ""
        self.last_visible_reply: str = ""
        self.format_agent_replies: bool = bool(self.config.get("format_agent_replies", True))
        self.is_streaming_response: bool = False
        self.loading_active: bool = False
        self.loading_thread: threading.Thread | None = None
        self.loading_lock = threading.Lock()
        self.raw_model_responses: list[dict] = []  # Track raw responses for debugging

        self.model_num_context: int | None = None
        self.embedding_model_num_context: int | None = None
        self.vision_model_num_context: int | None = None

        self.key_bindings = KeyBindings()

        @self.key_bindings.add("/")
        def _(event):
            buffer = event.current_buffer
            buffer.insert_text("/")
            buffer.start_completion(select_first=True)
            event.app.invalidate()

        @self.key_bindings.add("enter")
        def _(event):
            event.current_buffer.validate_and_handle()

        # Some modern terminals encode Shift+Enter using CSI-u style escape
        # sequences. prompt_toolkit does not expose a portable "s-enter" key name,
        # so bind the common raw sequences directly.
        @self.key_bindings.add("escape", "[", "1", "3", ";", "2", "u", eager=True)
        def _(event):
            event.current_buffer.insert_text("\n")

        @self.key_bindings.add("escape", "[", "2", "7", ";", "2", ";", "1", "3", "~", eager=True)
        def _(event):
            event.current_buffer.insert_text("\n")

        # Fallback for terminals that do not send a distinct Shift+Enter.
        @self.key_bindings.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        @self.key_bindings.add("escape")
        def clear_slash_command(event) -> None:
            buffer = event.app.current_buffer
            if buffer.text.lstrip().startswith("/"):
                buffer.text = ""
                buffer.cursor_position = 0

        self.session = PromptSession(
            completer=SlashCommandCompleter(lambda: self.workspace_dir),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            editing_mode=EditingMode.EMACS,
            key_bindings=self.key_bindings,
            multiline=True,
            reserve_space_for_menu=3,
            bottom_toolbar=self.get_bottom_toolbar,
        )

        # Initialize Pollinations client
        pollinations_api_key = os.getenv("POLLINATIONS_API_KEY") or self.config.get("pollinations_api_key")
        self.pollinations_client = PollinationsClient(
            PollinationsConfig(api_key=pollinations_api_key)
        )
        
        # Only initialize Ollama client if we're not using Pollinations models for chat
        if not self.model.startswith("pollinations/"):
            self.client = OllamaClient(
                OllamaConfig(
                    model=self.model,
                    host=self.host,
                    temperature=0.7,
                    top_p=0.9,
                    timeout=180,
                )
            )
        else:
            # For Pollinations models, we'll use the Pollinations client for chat
            # but still initialize Ollama client for embedding/vision operations if needed
            self.client = OllamaClient(
                OllamaConfig(
                    model=self.model,  # This will be overridden when needed
                    host=self.host,
                    temperature=0.7,
                    top_p=0.9,
                    timeout=180,
                )
            )

        self.install_resize_handler()

    def redraw_after_workspace_change(self, message: str) -> None:
        print()
        self.clear_screen()
        self.show_landing_page()
        
        # Check connectivity
        pollinations_configured = bool(os.getenv("POLLINATIONS_API_KEY") or self.config.get("pollinations_api_key"))
        ollama_available = self.client.is_available()
        
        self.print_info(f"Workspace: {self.workspace_dir}")
        if pollinations_configured:
            self.print_info("Pollinations: Ready")
        else:
            self.print_info("Pollinations: Not Configured")
        if ollama_available:
            self.print_info("Ollama: Ready")
        else:
            self.print_info("Ollama: Unavailable")
        self.print_info(f"Model: {self.model}{self.format_num_context(self.model_num_context)}")
        self.print_info(f"Embedding: {self.embedding_model}{self.format_num_context(self.embedding_model_num_context)}")
        self.print_info(f"Vision: {self.vision_model}{self.format_num_context(self.vision_model_num_context)}")
        self.print_dim("Type /help for commands. F1 to collapse/expand thinking. Type /exit to quit.\n")
        self.print_info(message)
        self.print_dim("Saved to config.json.")
        self.print_dim("Session history, memory, attachments, web context, and screen cleared.")
        print()

    def load_workspace_dir(self) -> Path:
        configured_workspace = str(self.config.get("workspace") or "").strip()

        if configured_workspace:
            workspace_path = Path(configured_workspace).expanduser()
            if workspace_path.exists() and workspace_path.is_dir():
                return workspace_path.resolve()

        return Path.cwd().resolve()

    def set_workspace_dir(self, workspace_path_text: str) -> bool:
        workspace_path_text = workspace_path_text.strip()

        if not workspace_path_text:
            self.workspace_dir = Path.cwd().resolve()
            success_message = f"Workspace reset to terminal folder: {self.workspace_dir}"
        else:
            workspace_path = Path(workspace_path_text).expanduser()

            if not workspace_path.exists():
                print()
                self.print_error(f"Workspace path does not exist: {workspace_path}")
                print()
                return True

            if not workspace_path.is_dir():
                print()
                self.print_error(f"Workspace path is not a directory: {workspace_path}")
                print()
                return True

            self.workspace_dir = workspace_path.resolve()
            success_message = f"Workspace changed to: {self.workspace_dir}"

        self.config["workspace"] = str(self.workspace_dir)
        save_config(self.config)
        self.clear_session_state()
        self.redraw_after_workspace_change(success_message)
        return True

    def review_last_code_blocks(self, strategy: "EditStrategy | None" = None) -> None:
        if not getattr(self, "last_visible_reply", ""):
            print()
            self.print_dim("No assistant reply available yet. Send a prompt first, then run /code.")
            print()
            return

        if strategy is None:
            # Auto-detect: if the reply contains SEARCH markers use search/replace,
            # else fall back to whole-file.
            reply = self.last_visible_reply
            if "<<<<<<< SEARCH" in reply or "<<<<<<< search" in reply.lower():
                strategy = EditStrategy.SEARCH_REPLACE
            elif "--- a/" in reply or "--- a\\" in reply:
                strategy = EditStrategy.UNIFIED_DIFF
            else:
                strategy = EditStrategy.WHOLE_FILE

        editblock.apply_llm_edits(
            app=self,
            llm_output=self.last_visible_reply,
            strategy=strategy,
            title="Code edit review from last assistant reply",
        )

    def install_resize_handler(self) -> None:
        if not hasattr(signal, "SIGWINCH"):
            return

        def _handle_resize(_signum, _frame) -> None:
            run_in_terminal(self.handle_resize)

        signal.signal(signal.SIGWINCH, _handle_resize)

    def toggle_agent_reply_markup(self) -> None:
        self.format_agent_replies = not self.format_agent_replies
        self.config["format_agent_replies"] = self.format_agent_replies
        save_config(self.config)

        status = "on" if self.format_agent_replies else "off"
        print()
        self.print_info(f"Agent reply markup formatting: {status}")
        if not self.format_agent_replies:
            self.print_dim("Raw replies are useful for /code because SEARCH/REPLACE markers stay untouched.")
        print()

    def save_persona_config(self) -> None:
        self.config["personas"] = dict(self.personas)
        self.config["active_persona"] = self.active_persona
        self.config["persona_workflows"] = dict(self.persona_workflows)
        self.config.pop("system_prompt", None)
        save_config(self.config)

    def set_active_persona(self, persona_name: str) -> None:
        if persona_name not in self.personas:
            raise ValueError(f"Persona not found: {persona_name}")

        self.active_persona = persona_name
        self.system_prompt = self.personas[persona_name]
        self.persona_workflows.setdefault(
            persona_name,
            DEFAULT_PERSONA_WORKFLOWS.get(persona_name, DEFAULT_WORKFLOW_NAME),
        )
        self.workflow_runner = self.build_workflow_runner_for_persona(persona_name)
        self.save_persona_config()

    def build_workflow_runner_for_persona(self, persona_name: str) -> workflows.WorkflowRunner:
        workflow_name = self.persona_workflows.get(persona_name, DEFAULT_WORKFLOW_NAME)
        workflow_path = self.resolve_workflow_path(workflow_name)

        if workflow_path is None:
            return workflows.WorkflowRunner.default(app=self)

        try:
            return workflows.WorkflowRunner.from_file(app=self, path=workflow_path)
        except Exception as error:
            self.print_error(f"Could not load workflow '{workflow_name}' for persona '{persona_name}': {error}")
            self.print_dim("Falling back to built-in default workflow.")
            return workflows.WorkflowRunner.default(app=self)

    def resolve_workflow_path(self, workflow_name: str) -> Path | None:
        workflow_name = (workflow_name or DEFAULT_WORKFLOW_NAME).strip()

        if workflow_name == DEFAULT_WORKFLOW_NAME:
            if DEFAULT_WORKFLOW_PATH.exists():
                return DEFAULT_WORKFLOW_PATH
            return None

        candidate = Path(workflow_name).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate

        if not workflow_name.endswith(".md"):
            workflow_name = f"{workflow_name}.md"

        user_workflow_path = USER_WORKFLOWS_DIR / workflow_name
        if user_workflow_path.exists():
            return user_workflow_path

        project_workflow_path = PROJECT_WORKFLOWS_DIR / workflow_name
        if project_workflow_path.exists():
            return project_workflow_path

        return None

    def list_available_workflows(self) -> list[tuple[str, str]]:
        workflows: list[tuple[str, str]] = [(DEFAULT_WORKFLOW_NAME, "default")]

        seen_names = {DEFAULT_WORKFLOW_NAME}

        for folder, source_label in (
                (PROJECT_WORKFLOWS_DIR, "project"),
                (USER_WORKFLOWS_DIR, "user"),
        ):
            if not folder.exists():
                continue

            for path in sorted(folder.glob("*.md")):
                workflow_name = path.stem
                if workflow_name in seen_names:
                    continue

                seen_names.add(workflow_name)
                workflows.append((workflow_name, f"{workflow_name} ({source_label})"))

        return workflows

    def select_workflow_name(
            self,
            title: str,
            current_workflow: str = DEFAULT_WORKFLOW_NAME,
    ) -> str:
        workflows = self.list_available_workflows()
        workflow_names = [name for name, _label in workflows]
        selected_workflow = current_workflow if current_workflow in workflow_names else DEFAULT_WORKFLOW_NAME
        confirmed = False

        radio_list = RadioList(
            values=[(name, label) for name, label in workflows],
            default=selected_workflow,
        )

        def confirm() -> None:
            nonlocal selected_workflow, confirmed
            selected_workflow = radio_list.current_value
            confirmed = True
            application.exit()

        def cancel() -> None:
            nonlocal confirmed
            confirmed = False
            application.exit()

        dialog = Dialog(
            title=title,
            body=HSplit([
                Label(text="Use ↑/↓ to choose the workflow for this persona."),
                Box(body=radio_list, padding=1),
            ]),
            buttons=[
                Button(text="Select", handler=confirm),
                Button(text="Cancel", handler=cancel),
            ],
            with_background=False,
        )

        application = Application(
            layout=Layout(dialog),
            full_screen=False,
            style=self.build_web_dialog_style(),
        )

        print()
        with patch_stdout(raw=True):
            application.run()
        print()

        if not confirmed:
            return current_workflow or DEFAULT_WORKFLOW_NAME

        return selected_workflow

    def install_workflow_and_create_persona(self, workflow_path_text: str) -> None:
        workflow_path_text = workflow_path_text.strip()

        if not workflow_path_text:
            print()
            self.print_error("Usage: /workflow-install <path-to-workflow.md>")
            print()
            return

        workflow_path = Path(workflow_path_text).expanduser()

        try:
            workflow_name = workflows.install_workflow_file(
                source_path=workflow_path,
                destination_dir=USER_WORKFLOWS_DIR,
            )
        except Exception as error:
            print()
            self.print_error(f"Could not install workflow: {error}")
            print()
            return

        print()
        self.print_info(f"Installed workflow: {workflow_name}")
        self.print_dim(f"Saved to: {USER_WORKFLOWS_DIR / f'{workflow_name}.md'}")
        print()

        persona_name = self.session.prompt("New persona name for this workflow: ").strip()

        if not persona_name:
            self.print_dim("Persona creation cancelled. Workflow was installed but not assigned.")
            print()
            return

        if persona_name in self.personas:
            self.print_error(f"Persona already exists: {persona_name}")
            self.print_dim("Workflow was installed but not assigned.")
            print()
            return

        persona_prompt = self.prompt_persona_text(
            prompt_title=f"New persona prompt for workflow: {workflow_name}",
        )

        if not persona_prompt:
            self.print_dim("Persona creation cancelled. Workflow was installed but not assigned.")
            print()
            return

        self.personas[persona_name] = persona_prompt
        self.persona_workflows[persona_name] = workflow_name
        self.set_active_persona(persona_name)

        self.print_info(f"Added and activated persona: {persona_name}")
        self.print_info(f"Assigned workflow: {workflow_name}")
        print()

    # -----------------------------
    # App lifecycle
    # -----------------------------

    def run(self) -> None:
        self.clear_screen()
        self.show_landing_page()

        # Check connectivity
        pollinations_configured = bool(os.getenv("POLLINATIONS_API_KEY") or self.config.get("pollinations_api_key"))
        ollama_available = self.client.is_available()
        
        self.refresh_model_context_lengths()
        self.print_info(f"Workspace: {self.workspace_dir}")
        if pollinations_configured:
            self.print_info("Pollinations: Ready")
        else:
            self.print_info("Pollinations: Not Configured")
        if ollama_available:
            self.print_info("Ollama: Ready")
        else:
            self.print_info("Ollama: Unavailable")
        self.print_info(f"Model: {self.model}{self.format_num_context(self.model_num_context)}")
        self.print_info(f"Embedding: {self.embedding_model}{self.format_num_context(self.embedding_model_num_context)}")
        self.print_info(f"Vision: {self.vision_model}{self.format_num_context(self.vision_model_num_context)}")
        self.print_dim("Type /help for commands. Type /exit to quit.\n")

        while True:
            try:
                user_input = self.read_user_input()
            except KeyboardInterrupt:
                print()
                self.print_dim("Use /exit to quit.")
                continue
            except EOFError:
                print()
                break

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                should_continue = self.handle_command(user_input)
                if not should_continue:
                    break
                continue

            with patch_stdout(raw=True):
                self.refresh_changed_attachments_before_prompt()
                self.chat(user_input)

        self.print_dim("\nGoodbye. Keep building.\n")

    # -----------------------------
    # Chat
    # -----------------------------

    def run_chat_model_for_workflow(self, chat_messages: list[dict[str, str]]) -> str:
        """
        Run one workflow model prompt and return the full raw reply.

        Multi-prompt workflows call this once per prompt. Each call starts its own
        loading toolbar because stream_chat_reply() stops the toolbar when that
        individual model response completes.
        """
        self.start_loading_toolbar()
        return self.stream_chat_reply(chat_messages)

    def chat(self, user_input: str) -> None:
        # Build prompt messages before adding the latest user prompt to history.
        # The active workflow .md controls whether and where the original user
        # prompt is inserted via `add_original_user_prompt`.
        self.compact_messages()

        started_at = time.perf_counter()

        print()
        self.print_agent_header()
        self.start_loading_toolbar()

        try:
            # Collect all workflow results
            workflow_results = []
            for workflow_result in self.workflow_runner.run(
                original_user_prompt=user_input,
                execute_model=True,
            ):
                workflow_results.append(workflow_result)
            
            # Process the final result
            if workflow_results:
                final_workflow_result = workflow_results[-1]  # Get the last result
                elapsed_seconds = time.perf_counter() - started_at

                self.last_workflow_messages = self.flatten_workflow_debug_messages(final_workflow_result)

                raw_reply = self.get_final_workflow_reply(final_workflow_result)

                tokens_in = self.estimate_chat_tokens(final_workflow_result.messages)
                tokens_out = self.estimate_text_tokens(raw_reply)
                assistant_reply = self.extract_and_store_thinking(raw_reply)
                self.last_visible_reply = assistant_reply

                self.messages.append({"role": "user", "content": user_input})
                self.messages.append(
                    {"role": "assistant", "content": assistant_reply}
                )
                self.compact_messages()
                self.print_response_stats(elapsed_seconds, tokens_in, tokens_out)

        except Exception as exc:
            self.is_streaming_response = False
            self.stop_loading_toolbar()
            error_message = str(exc).strip()
            active_model = str(self.model or "")
            pollinations_model_name = active_model.removeprefix("pollinations/")
            is_pollinations_model = (
                    active_model.startswith("pollinations/")
                    or pollinations_model_name in self.pollinations_client.list_models_whitelisted()
            )

            if is_pollinations_model:
                pollinations_configured = bool(
                    os.getenv("POLLINATIONS_API_KEY")
                    or self.config.get("pollinations_api_key")
                )
                if not pollinations_configured:
                    self.print_error(
                        "Pollinations API key not configured. Please run /api-pollinations to authenticate."
                    )
                    return
                self.print_error(f"Pollinations model call failed: {error_message}")
                return
            if (
                    "Could not connect to Ollama" in error_message
                    or "ollama serve" in error_message
                    or "Connection refused" in error_message
            ):
                self.print_error(
                    "Could not connect to Ollama. Make sure Ollama is running with: ollama serve"
                )
                return
            self.print_error(f"Model call failed: {error_message}")


    def get_final_workflow_reply(self, workflow_result) -> str:
        """
        Return the workflow output that should be stored as the final reply.

        Workflow prompts may stream directly to the terminal. This method selects
        the final text for `/code`, memory, and stats without forcing `chat()` to
        print the same text again.
        """
        prompt_results = getattr(workflow_result, "prompt_results", {}) or {}

        if isinstance(prompt_results, dict) and prompt_results:
            ordered_results = list(prompt_results.items())
            preferred_results: list[tuple[str, object]] = []

            if "output" in prompt_results:
                preferred_results.append(("output", prompt_results["output"]))

            preferred_results.extend(
                (name, result)
                for name, result in reversed(ordered_results)
                if name != "output"
            )

            for _name, prompt_result in preferred_results:
                raw_output = str(getattr(prompt_result, "raw_output", "") or "").strip()
                if self.has_visible_model_output(raw_output):
                    return raw_output

        visible_output = str(getattr(workflow_result, "visible_output", "") or "").strip()
        if self.has_visible_model_output(visible_output):
            return visible_output

        return ""

    def has_visible_model_output(self, raw_output: str) -> bool:
        visible_output = THINK_BLOCK_PATTERN.sub("", raw_output or "").strip()
        return bool(visible_output)


    def flatten_workflow_debug_messages(self, workflow_result) -> list[dict[str, str]]:
        def normalise_debug_message(message: object) -> dict[str, str]:
            if not isinstance(message, dict):
                return {"role": "system", "content": str(message)}

            role = str(message.get("role") or "system")
            content = str(message.get("content") or "")
            return {"role": role, "content": content}

        def normalise_debug_messages(messages: object) -> list[dict[str, str]]:
            if not isinstance(messages, list):
                return []

            return [normalise_debug_message(message) for message in messages]

        debug_messages: list[dict[str, str]] = []
        prompt_results = getattr(workflow_result, "prompt_results", {}) or {}

        if not isinstance(prompt_results, dict) or not prompt_results:
            return normalise_debug_messages(getattr(workflow_result, "messages", []))

        for prompt_name, prompt_result in prompt_results.items():
            debug_messages.append(
                {
                    "role": "system",
                    "content": f"--- workflow prompt: {prompt_name} ---",
                }
            )
            debug_messages.extend(
                normalise_debug_messages(getattr(prompt_result, "messages", []))
            )

        return debug_messages

    def build_chat_messages(self, user_input: str) -> list[dict[str, str]]:
        chat_messages = self.workflow_runner.build_messages(
            original_user_prompt=user_input,
        )
        self.last_workflow_messages = [dict(message) for message in chat_messages]
        return chat_messages

    def compact_messages(self) -> None:
        if len(self.messages) <= MAX_RECENT_MESSAGES:
            return

        overflow_count = len(self.messages) - MAX_RECENT_MESSAGES
        messages_to_archive = self.messages[:overflow_count]
        self.messages = self.messages[overflow_count:]

        self.archive_memory_messages(messages_to_archive)

    def archive_memory_messages(self, messages: list[dict[str, str]]) -> None:
        if not messages:
            return

        for memory_item in self.build_memory_items_from_messages(messages):
            self.embed_and_store_memory_item(memory_item)

    def build_memory_items_from_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        memory_items = []
        index = 0

        while index < len(messages):
            message = messages[index]
            role = message.get("role", "unknown")
            content = message.get("content", "").strip()

            if not content:
                index += 1
                continue

            if role == "user" and index + 1 < len(messages):
                next_message = messages[index + 1]
                next_role = next_message.get("role", "unknown")
                next_content = next_message.get("content", "").strip()

                if next_role == "assistant" and next_content:
                    memory_items.append(
                        {
                            "role": "turn",
                            "content": self.format_memory_turn(content, next_content),
                        }
                    )
                    index += 2
                    continue

            memory_items.append(
                {
                    "role": role,
                    "content": self.truncate_memory_text(content),
                }
            )
            index += 1

        return memory_items

    def format_memory_turn(self, user_content: str, assistant_content: str) -> str:
        user_text = self.truncate_memory_text(user_content, limit=350)
        assistant_text = self.truncate_memory_text(assistant_content, limit=550)
        return f"User asked: {user_text}\nAssistant answered: {assistant_text}"

    def truncate_memory_text(self, text: str, limit: int = MAX_MEMORY_TEXT_LENGTH) -> str:
        compact = re.sub(r"\s+", " ", text).strip()

        if len(compact) <= limit:
            return compact

        return compact[: max(0, limit - 3)].rstrip() + "..."

    def embed_and_store_memory_item(self, memory_item: dict[str, str]) -> None:
        content = memory_item.get("content", "").strip()
        role = memory_item.get("role", "unknown")

        if not content:
            return

        memory_text = f"{role}: {content}"

        # Check if this is a Pollinations embedding model
        # Handle both "pollinations/openai-3-small" and "openai-3-small" formats
        embedding_model_name = self.embedding_model
        if embedding_model_name.startswith("pollinations/"):
            embedding_model_name = embedding_model_name.split("/", 1)[1]
        
        if embedding_model_name in self.pollinations_client.list_models_whitelisted():
            # Use Pollinations client for embeddings
            try:
                embedding = self.pollinations_client.create_embeddings(memory_text, model=embedding_model_name)
            except Exception:
                embedding = []
        else:
            # Use Ollama client for embeddings
            try:
                embedding = vectorise_text(self.client, memory_text, self.embedding_model)
            except Exception:
                embedding = []

        if not utils.is_embedding_vector(embedding):
            embedding = []

        stored_item = {
            "role": role,
            "content": content,
            "source_type": "conversation_memory",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "embedding": embedding,
        }

        self.memory_items.append(stored_item)
        self.memory_index.add_item(stored_item)

    def get_relevant_web_context(self, query_text: str) -> str:
        if not self.web_context_items:
            return ""

        # Check if this is a Pollinations embedding model
        # Handle both "pollinations/openai-3-small" and "openai-3-small" formats
        embedding_model_name = self.embedding_model
        if embedding_model_name.startswith("pollinations/"):
            embedding_model_name = embedding_model_name.split("/", 1)[1]
        
        if embedding_model_name in self.pollinations_client.list_models_whitelisted():
            # Use Pollinations client for embeddings
            try:
                query_embedding = self.pollinations_client.create_embeddings(query_text, model=embedding_model_name)
            except Exception:
                return ""
        else:
            # Use Ollama client for embeddings
            try:
                query_embedding = vectorise_text(self.client, query_text, embedding_model_name)
            except Exception:
                return ""

        if not utils.is_embedding_vector(query_embedding):
            return ""

        selected_items = self.web_index.search(
            query_embedding=query_embedding,
            top_k=MAX_RELEVANT_WEB_ITEMS,
            min_score=0.05,
        )

        if not selected_items:
            return ""

        lines: list[str] = []

        for rank, item in enumerate(selected_items, start=1):
            title = item.get("title") or item.get("source_path") or "web page"
            source_path = item.get("source_path") or "unknown"
            chunk_index = item.get("chunk_index", 0)
            score = float(item.get("similarity_score", 0.0))
            content = item.get("content", "").strip()

            if not content:
                continue

            if len(content) > 1_200:
                content = content[:1_197].rstrip() + "..."

            chunk_kind = item.get("chunk_kind") or "web"

            lines.append(
                f"--- rank {rank} | score {score:.3f} | {title} | "
                f"kind {chunk_kind} | chunk {chunk_index} ---\n"
                f"Source: {source_path}\n{content}"
            )

        return "\n\n".join(lines)

    def get_relevant_memory_context(self, query_text: str) -> str:
        if not self.memory_items:
            return ""

        # Check if this is a Pollinations embedding model
        # Handle both "pollinations/openai-3-small" and "openai-3-small" formats
        embedding_model_name = self.embedding_model
        if embedding_model_name.startswith("pollinations/"):
            embedding_model_name = embedding_model_name.split("/", 1)[1]
        
        if embedding_model_name in self.pollinations_client.list_models_whitelisted():
            # Use Pollinations client for embeddings
            try:
                query_embedding = self.pollinations_client.create_embeddings(query_text, model=embedding_model_name)
            except Exception:
                return ""
        else:
            # Use Ollama client for embeddings
            try:
                query_embedding = vectorise_text(self.client, query_text, embedding_model_name)
            except Exception:
                return ""

        if not utils.is_embedding_vector(query_embedding):
            return ""

        selected_items = self.memory_index.search(
            query_embedding=query_embedding,
            top_k=MAX_RELEVANT_MEMORY_ITEMS,
            min_score=0.05,
        )

        if not selected_items:
            return ""

        lines = []
        for rank, item in enumerate(selected_items, start=1):
            role = item.get("role", "unknown")
            score = float(item.get("similarity_score", 0.0))
            content = item.get("content", "").strip()

            if len(content) > 700:
                content = content[:697].rstrip() + "..."

            lines.append(f"- rank {rank} | score {score:.3f} | {role}: {content}")

        return "\n".join(lines)

    def get_relevant_attachment_context(self, query_text: str) -> str:
        if not self.attachment_context_items:
            return ""

        # Check if this is a Pollinations embedding model
        # Handle both "pollinations/openai-3-small" and "openai-3-small" formats
        embedding_model_name = self.embedding_model
        if embedding_model_name.startswith("pollinations/"):
            embedding_model_name = embedding_model_name.split("/", 1)[1]
        
        if embedding_model_name in self.pollinations_client.list_models_whitelisted():
            # Use Pollinations client for embeddings
            try:
                query_embedding = self.pollinations_client.create_embeddings(query_text, model=embedding_model_name)
            except Exception:
                return ""
        else:
            # Use Ollama client for embeddings
            try:
                query_embedding = vectorise_text(self.client, query_text, embedding_model_name)
            except Exception:
                return ""

        if not utils.is_embedding_vector(query_embedding):
            return ""

        selected_items = self.attachment_index.search(
            query_embedding=query_embedding,
            top_k=MAX_RELEVANT_ATTACHMENT_ITEMS,
            min_score=0.05,
        )

        if not selected_items:
            return ""

        lines: list[str] = []
        for rank, item in enumerate(selected_items, start=1):
            title = item.get("title") or item.get("filename") or "attachment"
            chunk_index = item.get("chunk_index", 0)
            score = float(item.get("similarity_score", 0.0))
            content = item.get("content", "").strip()

            if not content:
                continue

            if len(content) > 1_200:
                content = content[:1_197].rstrip() + "..."

            extension = item.get("extension") or "unknown"
            mime_type = item.get("mime_type") or "unknown"
            chunk_kind = item.get("chunk_kind") or "unknown"

            lines.append(
                f"--- rank {rank} | score {score:.3f} | {title} | "
                f"type {extension} | mime {mime_type} | kind {chunk_kind} | chunk {chunk_index} ---\n{content}"
            )

        return "\n\n".join(lines)

    def get_attachment_context(self, query_text: str) -> str:
        """
        Return attachment context for the current prompt.

        Text/table/document attachments are ranked by embedding relevance.
        Image attachments are always included in full while attached.
        """
        sections: list[str] = []

        image_context = self.get_full_image_attachment_context()
        if image_context:
            sections.append(image_context)

        full_text_context = self.get_full_text_attachment_context()
        if full_text_context:
            sections.append(full_text_context)

        ranked_context = self.get_relevant_attachment_context(query_text)
        if ranked_context:
            sections.append(ranked_context)

        return self.limit_attachment_context("\n\n".join(sections))

    def get_full_image_attachment_context(self) -> str:
        """
        Return full context for image attachments.

        Image attachments are not ranked because visual descriptions can lose meaning
        when partial chunks are selected. While an image is attached, its full metadata
        and vision description are included in attachment context.
        """
        if not self.image_attachment_context_items:
            return ""

        lines: list[str] = []

        for item in self.image_attachment_context_items:
            title = item.get("title") or item.get("filename") or "image attachment"
            content = item.get("content", "").strip()

            if not content:
                continue

            extension = item.get("extension") or "unknown"
            mime_type = item.get("mime_type") or "unknown"
            lines.append(
                f"--- {title} | type {extension} | mime {mime_type} | image full context ---\n{content}"
            )

        return "\n\n".join(lines)

    def get_full_text_attachment_context(self) -> str:
        """
        Return full source text context for attached text-like files.

        Ranked embedding chunks may duplicate parts of this content; that is intentional.
        The full file gives complete source-of-truth context, while ranked chunks tell the
        model which areas are likely most relevant to the current prompt.
        """
        if not self.text_attachment_full_context_items:
            return ""

        lines: list[str] = [
            "Full text attachment context:",
            "The full text below is provided as source-of-truth context. Ranked chunks later may duplicate relevant excerpts.",
        ]

        for item in self.text_attachment_full_context_items:
            title = item.get("title") or item.get("filename") or "text attachment"
            extension = item.get("extension") or "unknown"
            mime_type = item.get("mime_type") or "unknown"
            content = item.get("content", "").strip()

            if not content:
                continue

            lines.append(
                f"--- {title} | type {extension} | mime {mime_type} | full text ---\n"
                f"```\n{content}\n```"
            )

        return "\n\n".join(lines)

    def get_attachment_context_token_budget(self) -> int:
        """
        Calculate a safe attachment-context budget from the active chat model context length.
        """
        if not self.model_num_context:
            return MIN_ATTACHMENT_CONTEXT_TOKENS

        available_after_response = max(
            0,
            self.model_num_context - RESERVED_RESPONSE_TOKENS,
        )

        budget = int(available_after_response * MAX_ATTACHMENT_CONTEXT_TOKEN_RATIO)

        if self.text_attachment_full_context_items:
            return max(MIN_ATTACHMENT_CONTEXT_TOKENS, available_after_response)

        return max(
            MIN_ATTACHMENT_CONTEXT_TOKENS,
            min(budget, MAX_ATTACHMENT_CONTEXT_TOKENS),
        )

    def limit_attachment_context(self, context: str) -> str:
        """
        Keep injected attachment context inside a safe token budget.
        """
        token_budget = self.get_attachment_context_token_budget()

        if self.estimate_text_tokens(context) <= token_budget:
            return context

        lines = context.splitlines()
        kept_lines: list[str] = []
        used_tokens = 0

        for line in lines:
            line_tokens = self.estimate_text_tokens(line) + 1

            if used_tokens + line_tokens > token_budget:
                break

            kept_lines.append(line)
            used_tokens += line_tokens

        return (
                "\n".join(kept_lines).rstrip()
                + "\n\n[Attachment context truncated to fit token budget.]"
        )

    def open_persona_manager(self) -> None:
        while True:
            action = self.select_persona_action()

            if action == "close":
                return

            if action == "select":
                persona_name = self.select_persona_name("Select persona")
                if persona_name:
                    self.set_active_persona(persona_name)
                    self.print_info(f"Active persona set to: {persona_name}")
                    print()
                return

            if action == "edit":
                persona_name = self.select_persona_name("Edit persona")
                if persona_name:
                    self.edit_persona(persona_name)
                return

            if action == "add":
                self.add_persona()
                return

            if action == "delete":
                persona_name = self.select_persona_name("Delete persona")
                if persona_name:
                    self.delete_persona(persona_name)
                return

    def select_persona_action(self) -> str:
        actions = [
            ("select", "Select"),
            ("edit", "Edit"),
            ("add", "Add"),
            ("delete", "Delete"),
            ("close", "Close"),
        ]

        selected_action = "close"

        def choose(action: str) -> None:
            nonlocal selected_action
            selected_action = action
            application.exit()

        dialog = Dialog(
            title="Persona Manager",
            body=HSplit(
                [
                    Label(text="Choose what you want to do with your SimpleAgent personas."),
                    Label(text=f"Active persona: {self.active_persona}"),
                ]
            ),
            buttons=[
                Button(text=label, handler=lambda action=action: choose(action))
                for action, label in actions
            ],
            with_background=False,
        )

        application = Application(
            layout=Layout(dialog),
            full_screen=False,
            style=self.build_web_dialog_style(),
        )

        print()
        with patch_stdout(raw=True):
            application.run()
        print()

        return selected_action

    def select_persona_name(self, title: str) -> str | None:
        if not self.personas:
            return None

        persona_names = list(self.personas)
        selected_name = self.active_persona if self.active_persona in self.personas else persona_names[0]
        confirmed = False

        values = [
            (
                name,
                f"{name}"
                f"{'  [built-in]' if name in BUILT_IN_PERSONA_NAMES else ''}"
                f"{'  [active]' if name == self.active_persona else ''}"
                f"  [workflow: {self.persona_workflows.get(name, DEFAULT_WORKFLOW_NAME)}]",
            )
            for name in persona_names
        ]

        radio_list = RadioList(
            values=values,
            default=selected_name,
        )

        def confirm() -> None:
            nonlocal selected_name, confirmed
            selected_name = radio_list.current_value
            confirmed = True
            application.exit()

        def cancel() -> None:
            nonlocal confirmed
            confirmed = False
            application.exit()

        dialog = Dialog(
            title=title,
            body=HSplit([
                Label(text="Use ↑/↓ to move, Tab to focus buttons, Enter to confirm."),
                Box(body=radio_list, padding=1),
            ]),
            buttons=[
                Button(text="Select", handler=confirm),
                Button(text="Cancel", handler=cancel),
            ],
            with_background=False,
        )

        application = Application(
            layout=Layout(dialog),
            full_screen=False,
            style=self.build_web_dialog_style(),
        )

        print()
        with patch_stdout(raw=True):
            application.run()
        print()

        if not confirmed:
            return None

        return selected_name

    def prompt_persona_text(self, prompt_title: str, default_text: str = "") -> str:
        print()
        self.print_dim(prompt_title)
        self.print_dim(
            "Enter text. Finish with /done. Use /default or /default-coding to restore built-in prompts. Cancel with /cancel."
        )

        if default_text:
            self.print_dim("Current text:")
            print(default_text)
            self.print_dim("---")

        lines: list[str] = []

        while True:
            line = self.session.prompt("persona> ")

            command = line.strip()

            if command == "/cancel":
                print()
                return ""

            if command == "/default":
                print()
                return DEFAULT_PERSONA_PROMPT.strip()

            if command == "/default-coding":
                print()
                return DEFAULT_CODING_PERSONA_PROMPT.strip()

            if command == "/done":
                print()
                return "\n".join(lines).strip()

            lines.append(line)

    def add_persona(self) -> None:
        print()
        persona_name = self.session.prompt("New persona name: ").strip()

        if not persona_name:
            self.print_dim("Persona creation cancelled.")
            print()
            return

        if persona_name in self.personas:
            self.print_error(f"Persona already exists: {persona_name}")
            print()
            return

        persona_prompt = self.prompt_persona_text("New persona prompt")

        if not persona_prompt:
            self.print_dim("Persona creation cancelled.")
            print()
            return

        workflow_name = self.select_workflow_name(
            title=f"Workflow for persona: {persona_name}",
            current_workflow=DEFAULT_WORKFLOW_NAME,
        )

        self.personas[persona_name] = persona_prompt
        self.persona_workflows[persona_name] = workflow_name
        self.set_active_persona(persona_name)
        self.print_info(f"Added and activated persona: {persona_name}")
        print()

    def edit_persona(self, persona_name: str) -> None:
        existing_prompt = self.personas.get(persona_name, "")
        if persona_name in BUILT_IN_PERSONA_NAMES:
            self.print_dim(f"Existing prompt: {existing_prompt}")
            self.print_error(f"Built-in persona cannot be edited: {persona_name}")
            self.print_dim("Create a custom persona if you want to modify it.")
            print()
            return

        updated_prompt = self.prompt_persona_text(
            prompt_title=f"Editing persona: {persona_name}",
            default_text=existing_prompt,
        )

        if not updated_prompt:
            self.print_dim("Persona edit cancelled.")
            print()
            return

        workflow_name = self.select_workflow_name(
            title=f"Workflow for persona: {persona_name}",
            current_workflow=self.persona_workflows.get(persona_name, DEFAULT_WORKFLOW_NAME),
        )

        self.personas[persona_name] = updated_prompt
        self.persona_workflows[persona_name] = workflow_name

        if persona_name == self.active_persona:
            self.system_prompt = updated_prompt
            self.workflow_runner = self.build_workflow_runner_for_persona(persona_name)

        self.save_persona_config()
        self.print_info(f"Updated persona: {persona_name}")
        print()

    def delete_persona(self, persona_name: str) -> None:
        if persona_name in BUILT_IN_PERSONA_NAMES:
            self.print_error(f"Built-in persona cannot be deleted: {persona_name}")
            print()
            return

        if persona_name not in self.personas:
            self.print_error(f"Persona not found: {persona_name}")
            print()
            return

        if len(self.personas) <= 1:
            self.print_error("Cannot delete the only persona.")
            print()
            return

        confirmed = self.confirm_persona_delete(persona_name)

        if not confirmed:
            self.print_dim("Persona deletion cancelled.")
            print()
            return

        del self.personas[persona_name]
        self.persona_workflows.pop(persona_name, None)

        if "Default" not in self.personas:
            self.personas["Default"] = DEFAULT_PERSONA_PROMPT

        if self.active_persona == persona_name:
            self.active_persona = (
                "Default"
                if "Default" in self.personas
                else next(iter(self.personas))
            )
            self.system_prompt = self.personas[self.active_persona]

        self.save_persona_config()
        self.print_info(f"Deleted persona: {persona_name}")
        self.print_info(f"Active persona: {self.active_persona}")
        print()

    def confirm_persona_delete(self, persona_name: str) -> bool:
        confirmed = False

        def confirm() -> None:
            nonlocal confirmed
            confirmed = True
            application.exit()

        def cancel() -> None:
            nonlocal confirmed
            confirmed = False
            application.exit()

        dialog = Dialog(
            title="Delete persona",
            body=Label(text=f"Delete persona '{persona_name}'? This cannot be undone."),
            buttons=[
                Button(text="Delete", handler=confirm),
                Button(text="Cancel", handler=cancel),
            ],
            with_background=False,
        )

        application = Application(
            layout=Layout(dialog),
            full_screen=False,
            style=self.build_web_dialog_style(),
        )

        print()
        with patch_stdout(raw=True):
            application.run()
        print()

        return confirmed

    def build_web_dialog_style(self) -> Style:
        """
        Return prompt_toolkit styling for the web-result selector dialog.
        """
        return Style.from_dict(
            {
                "dialog": "bg:#101827 #d7e7ff",
                "dialog frame.label": "bg:#101827 #60a5fa bold",
                "dialog.body": "bg:#101827 #d7e7ff",
                "dialog shadow": "bg:#050814",

                "checkbox": "bg:#101827 #d7e7ff",
                "checkbox-selected": "bg:#101827 #60a5fa bold",
                "checkbox-checked": "bg:#101827 #22c55e bold",
                "checkbox-checked-selected": "bg:#1d4ed8 #ffffff bold",

                "button": "bg:#1e293b #d7e7ff",
                "button.focused": "bg:#2563eb #ffffff bold",
                "label": "bg:#101827 #93c5fd",
            }
        )

    def select_web_search_results(self, results: list[dict]) -> list[dict]:
        if not results:
            return []

        selected_indexes = list(range(len(results)))
        confirmed = False

        def build_label(index: int, result: dict) -> str:
            title = str(result.get("title") or result.get("label") or "Untitled result").strip()
            url = str(result.get("url") or result.get("href") or "").strip()

            if hasattr(self, "clip_text"):
                title = self.clip_text(title, 72)
                url = self.clip_text(url, 92)

            return f"{index + 1:02d}. {title}\n    {url}"

        checkbox = CheckboxList(
            values=[
                (index, build_label(index, result))
                for index, result in enumerate(results)
            ],
            default_values=selected_indexes,
        )

        def confirm() -> None:
            nonlocal selected_indexes, confirmed
            selected_indexes = list(checkbox.current_values)
            confirmed = True
            application.exit()

        def cancel() -> None:
            nonlocal selected_indexes, confirmed
            selected_indexes = []
            confirmed = False
            application.exit()

        dialog = Dialog(
            title="Select web results to scrape",
            body=HSplit(
                [
                    Label(text="Use Space to toggle results, Tab to move, Enter to confirm."),
                    Box(body=checkbox, padding=1),
                ]
            ),
            buttons=[
                Button(text="Confirm", handler=confirm),
                Button(text="Cancel", handler=cancel),
            ],
            with_background=False,
        )

        application = Application(
            layout=Layout(dialog),
            full_screen=False,
            style=self.build_web_dialog_style(),
        )

        print()
        with patch_stdout(raw=True):
            application.run()
        print()

        if not confirmed:
            return []

        selected_index_set = set(selected_indexes)
        return [
            result
            for index, result in enumerate(results)
            if index in selected_index_set
        ]

    def add_web_context(self, target: str) -> bool:
        """
        Load a URL or search query into embedded web context.
        """
        target = target.strip()
        if not target:
            return False

        if self.is_probable_url(target):
            url = self.normalise_web_url(target)
            print()
            self.print_dim(f"Loading webpage: {url}")

            try:
                page_text = utils.scrape_url_to_string(url)
            except Exception as error:
                self.print_error(f"Could not load webpage {url}: {error}")
                print()
                return False

            print()

            stored = self.store_web_text(
                text=page_text,
                source_path=url,
                title=url,
                source_type="webpage",
            )

            if stored:
                self.add_web_source(label=url, source_type="url")

            return stored

        print()
        self.print_dim(f"Searching DuckDuckGo: {target}")

        try:
            search_results = utils.duckduckgo_search_results(
                query=target,
                max_results=10,
            )
        except Exception as error:
            self.print_error(f"Could not search web for '{target}': {error}")
            print()
            return False

        if not search_results:
            self.print_error(f"No DuckDuckGo results found for: {target}")
            print()
            return False

        search_results = self.select_web_search_results(search_results)

        if not search_results:
            self.print_dim("No web results selected.")
            print()
            return False

        self.print_info(f"Scraping {len(search_results)} selected web result(s)...")

        try:
            # Handle both "pollinations/openai-3-small" and "openai-3-small" formats
            embedding_model_name = self.embedding_model
            if embedding_model_name.startswith("pollinations/"):
                embedding_model_name = embedding_model_name.split("/", 1)[1]

            if self.embedding_model.startswith("pollinations/"):
                embedding_client = self.pollinations_client
            else:
                embedding_client = self.client

            embedded_items = utils.duckduckgo_search_results_to_embedded_context_items(
                client=embedding_client,
                query=target,
                model=embedding_model_name,
                search_results=search_results,
            )
        except Exception as error:
            self.print_error(f"Could not scrape selected web results for '{target}': {error}")
            print()
            return False

        stored = self.store_web_context_items(
            embedded_items=embedded_items,
            title=f"DuckDuckGo search: {target}",
        )

        if stored:
            self.add_web_source(label=target, source_type="search")

        return stored

    def store_web_text(
            self,
            text: str,
            source_path: str,
            title: str,
            source_type: str,
    ) -> bool:
        """
        Chunk, embed, and store web text for later prompt retrieval.
        """
        text = (text or "").strip()

        if not text:
            print()
            self.print_error(f"No readable web content found: {title}")
            print()
            return False

        try:
            context_items = utils.build_context_items_from_text(
                text=text,
                source_type=source_type,
                source_path=source_path,
                title=title,
                metadata={
                    "extension": ".html",
                    "mime_type": "text/html",
                },
            )

            # Check if this is a Pollinations embedding model
            # Handle both "pollinations/openai-3-small" and "openai-3-small" formats
            embedding_model_name = self.embedding_model
            if embedding_model_name.startswith("pollinations/"):
                embedding_model_name = embedding_model_name.split("/", 1)[1]
            
            if embedding_model_name in self.pollinations_client.list_models_whitelisted():
                # Use Pollinations client for embeddings
                embedded_items = utils.vectorise_context_items(
                    client=self.pollinations_client,
                    context_items=context_items,
                    model=embedding_model_name,
                )
            else:
                # Use Ollama client for embeddings
                embedded_items = utils.vectorise_context_items(
                    client=self.client,
                    context_items=context_items,
                    model=embedding_model_name,
                )
        except Exception as error:
            print()
            self.print_error(f"Could not embed web context {title}: {error}")
            print()
            return False

        added_count = 0

        for item in embedded_items:
            stored_item = dict(item)
            self.web_context_items.append(stored_item)

            if self.web_index.add_item(stored_item):
                added_count += 1

        if added_count <= 0:
            print()
            self.print_error(f"No embeddable web chunks found: {title}")
            print()
            return False

        self.print_dim(f"Stored {added_count} web context chunk(s): {title}")
        self.print_loaded_webpages(embedded_items)
        print()
        return True

    def store_web_context_items(self, embedded_items: list[dict], title: str) -> bool:
        """
        Store pre-embedded web context items for later prompt retrieval.
        """
        added_count = 0

        for item in embedded_items:
            stored_item = dict(item)
            self.web_context_items.append(stored_item)

            if self.web_index.add_item(stored_item):
                added_count += 1

        if added_count <= 0:
            print()
            self.print_error(f"No embeddable web chunks found: {title}")
            print()
            return False

        self.print_dim(f"Stored {added_count} web context chunk(s): {title}")
        self.print_loaded_webpages(embedded_items)
        print()
        return True

    def print_loaded_webpages(self, embedded_items: list[dict]) -> None:
        """
        Print the unique webpages loaded by a /web search.
        """
        loaded_pages: list[tuple[str, str]] = []
        seen_urls: set[str] = set()

        for item in embedded_items:
            url = str(item.get("url") or item.get("source_path") or "").strip()
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            page_title = str(item.get("title") or url).strip()
            loaded_pages.append((page_title, url))

        if not loaded_pages:
            return

        self.print_dim("Loaded webpages:")

        for index, (page_title, url) in enumerate(loaded_pages, start=1):
            clipped_title = self.clip_text(page_title, 72)
            clipped_url = self.clip_text(url, 96)

            self.print_dim(f"  {index}. {clipped_title}")
            self.print_dim(f"     {clipped_url}")

    def add_web_source(self, label: str, source_type: str) -> None:
        """
        Track loaded web URLs/searches for the bottom toolbar.
        """
        label = label.strip()

        if not label:
            return

        if any(
                item.get("label") == label and item.get("source_type") == source_type
                for item in self.web_sources
        ):
            return

        self.web_sources.append(
            {
                "label": label,
                "source_type": source_type,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def is_probable_url(self, value: str) -> bool:
        """
        Return True when input looks like a URL/domain rather than a search query.
        """
        value = value.strip()

        if not value or " " in value:
            return False

        parsed = urlparse(value if "://" in value else f"https://{value}")
        return bool(parsed.netloc and "." in parsed.netloc)

    def normalise_web_url(self, value: str) -> str:
        value = value.strip()

        if "://" in value:
            return value

        return f"https://{value}"

    def estimate_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        # Ollama's Python streaming wrapper currently gives us text chunks here,
        # not the final prompt_eval_count/eval_count metadata. This is a lightweight
        # TUI-side approximation so every response still gets useful stats.
        total = 0
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            total += self.estimate_text_tokens(role)
            total += self.estimate_text_tokens(content)
            total += 4  # small chat-template overhead per message
        return total

    def estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 0

        # Rough cross-model estimate: words, numbers, punctuation, CJK characters,
        # and emoji-ish symbols. Not exact BPE tokens, but stable enough for TUI stats.
        token_like_chunks = re.findall(
            r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s\w]",
            text,
            flags=re.UNICODE,
        )
        return len(token_like_chunks)

    def print_response_stats(self, elapsed_seconds: float, tokens_in: int, tokens_out: int) -> None:
        total_tokens = tokens_in + tokens_out
        tokens_per_second = tokens_out / elapsed_seconds if elapsed_seconds > 0 else 0.0
        elapsed_text = self.format_elapsed_time(elapsed_seconds)
        stats = (
            f"stats: {elapsed_text} elapsed · "
            f"~{tokens_in:,} in · ~{tokens_out:,} out · "
            f"~{total_tokens:,} total · ~{tokens_per_second:.1f} tok/s"
        )
        print(self.dim(stats))
        print()

    def format_elapsed_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

    def start_loading_toolbar(self) -> None:
        with self.loading_lock:
            if self.loading_active:
                return

            self.loading_active = True
            self.loading_thread = threading.Thread(
                target=self.run_loading_toolbar,
                daemon=True,
            )
            self.loading_thread.start()

    def stop_loading_toolbar(self) -> None:
        with self.loading_lock:
            self.loading_active = False

        thread = self.loading_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.3)

        self.clear_loading_toolbar()

    def run_loading_toolbar(self) -> None:
        messages = LOADING_MESSAGES.copy()
        random.shuffle(messages)

        message_index = 0
        frame_index = 0
        frame_loop_count = 0
        frame_loops_per_message = 2

        while True:
            with self.loading_lock:
                if not self.loading_active:
                    break

            if message_index >= len(messages):
                messages = LOADING_MESSAGES.copy()
                random.shuffle(messages)
                message_index = 0

            message = messages[message_index]
            frame = LOADING_FRAMES[frame_index]
            self.render_loading_toolbar(f"{message}{frame}")

            frame_index += 1
            if frame_index >= len(LOADING_FRAMES):
                frame_index = 0
                frame_loop_count += 1

                if frame_loop_count >= frame_loops_per_message:
                    frame_loop_count = 0
                    message_index += 1

            time.sleep(0.5)

    def render_loading_toolbar(self, text: str) -> None:
        try:
            size = os.get_terminal_size()
            row = size.lines
            width = max(1, size.columns)
        except OSError:
            return

        toolbar_text = f"■ {text}"
        clipped = self.clip_text(toolbar_text, width - 1)

        sys.stdout.write("\0337")
        sys.stdout.write(f"\033[{row};1H")
        sys.stdout.write("\033[2K")
        sys.stdout.write(self.blue(clipped))
        sys.stdout.write("\0338")
        sys.stdout.flush()

    def clear_loading_toolbar(self) -> None:
        try:
            row = os.get_terminal_size().lines
        except OSError:
            return

        sys.stdout.write("\0337")
        sys.stdout.write(f"\033[{row};1H")
        sys.stdout.write("\033[2K")
        sys.stdout.write("\0338")
        sys.stdout.flush()

    def get_visual_line_count(self, text: str) -> int:
        width = os.get_terminal_size().columns or 88
        lines = text.splitlines() or [""]
        return sum(max(1, (len(line) + width - 1) // width) for line in lines)

    def reset_streaming_reply_buffer(self) -> None:
        self.streaming_reply_buffer = ""

    def print_streaming_reply_text(self, text: str) -> None:
        if not text:
            return

        self.streaming_reply_buffer += text

    def flush_streaming_reply_buffer(self) -> None:
        reply = getattr(self, "streaming_reply_buffer", "")

        if not reply:
            return

        self.print_tui_markdown(reply)
        self.streaming_reply_buffer = ""

    def stream_chat_reply(self, chat_messages: list[dict[str, str]], print_stream: bool = True) -> str:
        thinking_text = ""
        reply_text = ""
        self.reset_streaming_reply_buffer()
        self.streaming_thinking_line_count = 0
        self.streaming_thinking_last_block = ""
        self.streaming_thinking_closed = False

        pending = ""
        in_thinking = False
        thinking_started = False
        maybe_thinking_prefix = False

        # Track raw responses for debugging
        raw_responses = []
        
        # Check if this is a Pollinations model
        if self.model in self.pollinations_client.list_models_whitelisted():
            # Check if Pollinations API key is configured
            pollinations_configured = bool(os.getenv("POLLINATIONS_API_KEY") or self.config.get("pollinations_api_key"))
            
            if not pollinations_configured:
                self.print_error("Pollinations API key not configured. Please run /api-pollinations to authenticate.")
                raise Exception("Pollinations API key not configured")
            
            try:
                # Capture the raw stream from Pollinations
                raw_response_stream = self.pollinations_client.chat_completions(
                    messages=chat_messages,
                    model=self.model,
                    stream=True,
                    thinking=True,  # Enable thinking for Pollinations models
                )
            except Exception as e:
                # If Pollinations API fails, we should not fall back to Ollama for Pollinations models
                # Instead, we should report the error properly
                self.print_error(f"Pollinations API error: {e}")
                raise  # Re-raise the exception instead of falling back to Ollama
        else:
            # For Ollama models, use the Ollama client
            response_stream = self.client.chat(
                chat_messages,
                stream=True,
                model=self.model,
            )

        # Determine which client to use based on model type
        if self.model in self.pollinations_client.list_models_whitelisted():
            # For Pollinations models, we need to handle the JSON response format properly
            def normalize_pollinations_stream(stream):
                full_response = ""
                for chunk in stream:
                    # Store raw chunk for debugging BEFORE any processing
                    raw_responses.append(chunk)
                    
                    # Pollinations returns JSON chunks
                    if isinstance(chunk, dict):
                        # Handle the Pollinations response format
                        choices = chunk.get("choices", [])
                        if choices:
                            choice = choices[0]
                            message = choice.get("message", {})
                            content = message.get("content", "")
                            if content:
                                full_response += content
                                yield content
                    elif isinstance(chunk, str):
                        # Handle string chunks that might be content
                        full_response += chunk
                        yield chunk
            
            normalized_response_stream = normalize_pollinations_stream(raw_response_stream)
        else:
            # For Ollama, the response stream yields dictionaries with message content
            # But sometimes it can yield raw strings, so we need to handle both cases
            def normalize_ollama_stream(stream):
                full_response = ""
                for chunk in stream:
                    # Store raw chunk for debugging BEFORE any processing
                    raw_responses.append(chunk)
                    
                    if isinstance(chunk, dict):
                        content = chunk.get("message", {}).get("content", "")
                        full_response += content
                        yield content
                    else:
                        # Handle case where chunk is already a string
                        full_response += chunk
                        yield chunk
            
            normalized_response_stream = normalize_ollama_stream(response_stream)
        
        self.is_streaming_response = True

        for chunk in normalized_response_stream:
            if not chunk:
                continue

            pending += chunk

            while pending:
                lower_pending = pending.lower()

                if in_thinking:
                    end_index = lower_pending.find(STREAM_THINK_END)

                    if end_index == -1:
                        thinking_text += pending
                        pending = ""
                        break

                    thinking_fragment = pending[:end_index]
                    thinking_text += thinking_fragment
                    pending = pending[end_index + len(STREAM_THINK_END):]
                    in_thinking = False
                    continue

                start_index = lower_pending.find(STREAM_THINK_START)

                if start_index == -1:
                    if not thinking_started and not maybe_thinking_prefix:
                        stripped_pending = pending.lstrip()
                        leading_whitespace_length = len(pending) - len(stripped_pending)

                        if not stripped_pending:
                            break

                        if STREAM_THINK_START.startswith(stripped_pending.lower()):
                            maybe_thinking_prefix = True
                            break

                        visible_text = pending
                        self.print_streaming_reply_text(visible_text)
                        reply_text += visible_text
                        pending = ""
                        break

                    keep_length = len(STREAM_THINK_START) - 1

                    if len(pending) <= keep_length:
                        break

                    visible_text = pending[:-keep_length]
                    self.print_streaming_reply_text(visible_text)
                    reply_text += visible_text
                    pending = pending[-keep_length:]
                    break

                visible_text = pending[:start_index]

                if visible_text:
                    self.print_streaming_reply_text(visible_text)
                    reply_text += visible_text

                if not thinking_started:
                    thinking_started = True

                pending = pending[start_index + len(STREAM_THINK_START):]
                maybe_thinking_prefix = False
                in_thinking = True

        if pending:
            if in_thinking:
                thinking_text += pending
            else:
                self.print_streaming_reply_text(pending)
                reply_text += pending
                pending = ""

        thinking = self.normalise_thinking_text(thinking_text)
        visible_reply = reply_text.strip()

        if thinking_started and not self.streaming_thinking_closed:
            self.finish_streaming_thinking_display(thinking_text)

        if thinking:
            self.last_thinking = thinking
            self.last_visible_reply = visible_reply
        else:
            self.last_thinking = ""
            self.last_visible_reply = visible_reply

        if thinking_started and not self.streaming_thinking_closed:
            self.streaming_thinking_line_count = 0
            self.streaming_thinking_last_block = ""
            self.streaming_thinking_closed = True

        self.stop_loading_toolbar()
        self.flush_streaming_reply_buffer()
        self.is_streaming_response = False
        print()
        print()

        # Store raw responses for debugging - this now contains the complete response
        self.raw_model_responses = raw_responses
        
        # Return the complete response
        return f"{STREAM_THINK_START}{thinking_text}{STREAM_THINK_END}{reply_text}".strip()


    def compact_thinking_text(self, thinking: str) -> str:
        compact = re.sub(r"\s+", " ", thinking).strip()

        if not compact:
            return ""

        compact = re.sub(r"\s+([.,!?;:])", r"\1", compact)
        compact = re.sub(r"([([{])\s+", r"\1", compact)
        compact = re.sub(r"\s+([])}])", r"\1", compact)
        compact = re.sub(r"(?<=\w)\s+(?='\w)", "", compact)
        compact = re.sub(r"(?<=\d)\s+(?=\d)", "", compact)
        compact = re.sub(r"\s*([-–—])\s*", r" \1 ", compact)
        compact = re.sub(r"\s+", " ", compact).strip()
        return compact

    def render_streaming_thinking(self, previous_thinking: str, thinking_text: str) -> None:
        current_block = self.build_streaming_thinking_block(thinking_text)

        if not current_block:
            return

        sys.stdout.write(self.dim(current_block))
        sys.stdout.write("\n")
        sys.stdout.flush()


    def finish_streaming_thinking_display(self, thinking_text: str) -> None:
        if getattr(self, "streaming_thinking_closed", False):
            return

        current_block = self.build_streaming_thinking_block(thinking_text)

        if current_block and current_block != self.streaming_thinking_last_block:
            sys.stdout.write(self.dim(current_block))
            sys.stdout.write("\n")
            self.streaming_thinking_last_block = current_block

        sys.stdout.flush()
        self.streaming_thinking_closed = True

    def build_thinking_display_text(self, thinking: str) -> str:
        thinking = thinking.strip()

        if not thinking:
            return ""

        return thinking

    def build_streaming_thinking_block(self, thinking_text: str) -> str:
        thinking = self.normalise_thinking_text(thinking_text)
        return self.build_thinking_display_text(thinking)


    def extract_and_store_thinking(self, raw_reply: str) -> str:
        thinking_matches = [
            match.group(1)
            for match in THINK_BLOCK_PATTERN.finditer(raw_reply)
            if match.group(1).strip()
        ]

        thinking_text = self.normalise_thinking_text(" ".join(thinking_matches))
        visible_reply = THINK_BLOCK_PATTERN.sub("", raw_reply).strip()

        if thinking_text:
            self.last_thinking = thinking_text
        else:
            self.last_thinking = ""

        return visible_reply

    def normalise_thinking_text(self, thinking: str) -> str:
        compact = self.compact_thinking_text(thinking)

        if not compact:
            return ""

        return compact

    def print_model_reply(self, assistant_reply: str) -> None:
        if self.last_thinking:
            self.print_thinking_block()

        if assistant_reply:
            self.print_tui_markdown(assistant_reply)
        else:
            print()
            self.print_dim("No visible response returned.")
            print()

    def print_thinking_block(self) -> None:
        thinking = self.last_thinking.strip()
        display_text = self.build_thinking_display_text(thinking)

        if display_text:
            print(self.dim(display_text))


    def build_system_prompt(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{self.system_prompt}\n\nCurrent local datetime: {now}"

    # -----------------------------
    # Commands
    # -----------------------------

    def handle_command(self, raw: str) -> bool:
        command, arg = self.parse_command(raw)

        if command in {"/exit", "/quit", "/q"}:
            return False

        if command == "/code":
            with patch_stdout(raw=True):
                self.review_last_code_blocks()
            return True

        if command == "/help":
            with patch_stdout(raw=True):
                print(build_help_text())
            return True

        if command == "/about":
            with patch_stdout(raw=True):
                self.show_about()
            return True

        if command == "/version":
            with patch_stdout(raw=True):
                print()
                self.print_info(f"Current version: {VERSION}")
                print()
            return True

        if command == "/workspace":
            return self.set_workspace_dir(arg)

        if command == "/model-chat":
            if not arg:
                if self.model_num_context is None:
                    # Check if this is a Pollinations model
                    if self.model in self.pollinations_client.list_models_whitelisted():
                        self.model_num_context = None  # Pollinations models don't have context length info
                    else:
                        self.model_num_context = self.get_ollama_model_num_context(self.model)
                print()
                self.print_info(f"Current model: {self.model}{self.format_num_context(self.model_num_context)}")
                print()
                return True

            # Handle model selection with prefix
            if arg.startswith("ollama/") or arg.startswith("pollinations/"):
                model_name = arg.split("/", 1)[1] if "/" in arg else arg
            else:
                model_name = arg
            
            self.set_model(model_name, persist=True)
            return True

        if command == "/model-embedding":
            if not arg:
                if self.embedding_model_num_context is None:
                    # Check if this is a Pollinations model
                    if self.embedding_model in self.pollinations_client.list_models_whitelisted():
                        self.embedding_model_num_context = None  # Pollinations models don't have context length info
                    else:
                        self.embedding_model_num_context = self.get_ollama_model_num_context(self.embedding_model)
                print()
                self.print_info(
                    f"Current embeddings model: {self.embedding_model}{self.format_num_context(self.embedding_model_num_context)}"
                )
                print()
                return True

            # Handle model selection with prefix
            if arg.startswith("ollama/") or arg.startswith("pollinations/"):
                model_name = arg.split("/", 1)[1] if "/" in arg else arg
            else:
                model_name = arg
            
            self.set_embedding_model(model_name, persist=True)
            return True

        if command == "/model-vision":
            if not arg:
                if self.vision_model_num_context is None:
                    # Check if this is a Pollinations model
                    if self.vision_model in self.pollinations_client.list_models_whitelisted():
                        self.vision_model_num_context = None  # Pollinations models don't have context length info
                    else:
                        self.vision_model_num_context = self.get_ollama_model_num_context(self.vision_model)
                print()
                self.print_info(
                    f"Current vision model: {self.vision_model}{self.format_num_context(self.vision_model_num_context)}"
                )
                print()
                return True

            # Handle model selection with prefix
            if arg.startswith("ollama/") or arg.startswith("pollinations/"):
                model_name = arg.split("/", 1)[1] if "/" in arg else arg
            else:
                model_name = arg
            
            self.set_vision_model(model_name, persist=True)
            return True

        if command == "/attach":
            self.handle_attach_command(arg)
            return True

        if command == "/paste":
            self.handle_paste_command()
            return True

        if command == "/web":
            target = arg.strip()
            if not target:
                self.print_error("Usage: /web <url or search query>")
                return True

            self.add_web_context(target)
            self.session.app.invalidate()
            return True

        if command == "/models":
            self.show_models()
            return True

        if command == "/persona":
            self.open_persona_manager()
            return True

        if command == "/history":
            with patch_stdout(raw=True):
                self.show_history()
            return True

        if command == "/workflow":
            with patch_stdout(raw=True):
                self.show_workflow_help()
            return True

        if command == "/workflow-install":
            with patch_stdout(raw=True):
                self.install_workflow_and_create_persona(arg)
            return True

        if command == "/workflow-debug":
            with patch_stdout(raw=True):
                self.show_workflow_debug()
            return True

        if command == "/response-debug":
            with patch_stdout(raw=True):
                self.show_response_debug()
            return True

        if command == "/markup":
            self.toggle_agent_reply_markup()
            return True

        if command == "/clear":
            self.clear_session_state()

            with patch_stdout(raw=True):
                self.clear_screen()
                self.show_landing_page()
                
                # Check connectivity
                pollinations_configured = bool(os.getenv("POLLINATIONS_API_KEY") or self.config.get("pollinations_api_key"))
                ollama_available = self.client.is_available()
                
                self.print_info(f"Workspace: {self.workspace_dir}")
                if pollinations_configured:
                    self.print_info("Pollinations: Ready")
                else:
                    self.print_info("Pollinations: Not Configured")
                if ollama_available:
                    self.print_info("Ollama: Ready")
                else:
                    self.print_info("Ollama: Unavailable")
                self.print_info(f"Model: {self.model}{self.format_num_context(self.model_num_context)}")
                self.print_info(f"Embedding: {self.embedding_model}{self.format_num_context(self.embedding_model_num_context)}")
                self.print_info(f"Vision: {self.vision_model}{self.format_num_context(self.vision_model_num_context)}")
                self.print_dim("Type /help for commands. F1 to collapse/expand thinking. Type /exit to quit.\n")
                self.print_info("Session history, memory, attachments, web context, and screen cleared.")
                print()

            return True

        if command == "/api-pollinations":
            self.authenticate_with_pollinations()
            return True

        self.print_error(f"Unknown command: {command}")
        self.print_dim("Type /help to see available commands.")
        print()
        return True

    def parse_command(self, raw: str) -> tuple[str, str]:
        raw = raw.strip()
        if " " not in raw:
            return raw, ""

        command, arg = raw.split(" ", 1)
        return command.strip(), arg.strip()

    def clear_session_state(self) -> None:
        self.messages.clear()
        self.memory_items.clear()
        self.memory_index.clear()
        self.attachment_context_items.clear()
        self.image_attachment_context_items.clear()
        self.text_attachment_full_context_items.clear()
        self.web_context_items.clear()
        self.web_sources.clear()
        self.web_index.clear()
        self.last_workflow_messages.clear()
        self.last_thinking = ""
        self.last_visible_reply = ""
        self.streaming_thinking_line_count = 0
        self.streaming_thinking_last_block = ""
        self.streaming_thinking_closed = True
        self.reset_streaming_reply_buffer()
        self.delete_temp_files()
        self.attachments.clear()
        self.next_input_prefill = ""

    # -----------------------------
    # Attachments
    # -----------------------------
    def delete_temp_files(self) -> None:
        """
        Delete files saved inside the app attachment folder.
        Original files attached from elsewhere are not deleted unless they are inside ATTACHMENTS_DIR.
        """
        if not TEMP_DIR.exists():
            return

        for path in TEMP_DIR.iterdir():
            if not path.is_file():
                continue

            try:
                path.unlink()
            except OSError:
                continue

    def handle_attach_command(self, arg: str) -> None:
        if not arg:
            print()
            self.print_error("No attachment path provided.")
            self.print_dim("Usage: /attach <path...>")
            print()
            return

        paths = self.parse_attachment_paths(arg)
        if not paths:
            self.print_error("Could not parse attachment path.")
            print()
            return

        added_count = 0
        for path in paths:
            if self.add_attachment(path):
                added_count += 1

        if added_count:
            self.print_info(f"Attached {added_count} file(s).")
            print()
            self.session.app.invalidate()

    def parse_attachment_paths(self, raw_paths: str) -> list[Path]:
        try:
            parts = shlex.split(raw_paths)
        except ValueError as error:
            self.print_error(f"Could not parse paths: {error}")
            print()
            return []

        paths: list[Path] = []

        for part in parts:
            path = Path(part).expanduser()

            if not path.is_absolute():
                path = self.workspace_dir / path

            paths.append(path)

        return paths

    def add_attachment(self, path: Path) -> bool:
        path = path.expanduser()

        if not path.exists():
            self.print_error(f"Attachment not found: {path}")
            print()
            return False

        if not path.is_file():
            self.print_error(f"Attachment is not a file: {path}")
            print()
            return False

        if not self.is_supported_attachment(path):
            self.print_error(f"Unsupported attachment type: {path.name}")
            self.print_dim("Supported: text/code files, images .png/.jpg/.jpeg, PDF, CSV/TSV, Excel .xls/.xlsx")
            print()
            return False

        resolved_path = str(path.resolve())
        if any(item.get("source_path") == resolved_path for item in self.attachments):
            self.print_dim(f"Already attached: {path.name}")
            print()
            return False

        embedded_context_items = self.embed_attachment(path)
        if not embedded_context_items:
            return False

        mime_type, _ = mimetypes.guess_type(resolved_path)
        attachment = {
            "source_path": resolved_path,
            "filename": path.name,
            "extension": path.suffix.lower(),
            "mime_type": mime_type or "application/octet-stream",
            "sha256": utils.calculate_file_sha256(path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.attachments.append(attachment)
        self.store_attachment_context_items(embedded_context_items)
        self.store_full_text_attachment_context(path)
        return True

    def refresh_changed_attachments_before_prompt(self) -> None:
        """
        Re-read and re-embed attached files if their SHA-256 changed since attachment.

        This keeps attachment context fresh when the user edits a file outside the app
        before sending the next prompt.
        """
        if not self.attachments:
            return

        changed_attachments: list[dict[str, str]] = []

        for attachment in self.attachments:
            source_path = attachment.get("source_path", "")
            stored_sha256 = attachment.get("sha256", "")

            if not source_path or not stored_sha256:
                continue

            path = Path(source_path)
            if not path.exists() or not path.is_file():
                self.print_error(f"Attached file is no longer available: {path}")
                continue

            try:
                current_sha256 = utils.calculate_file_sha256(path)
            except Exception as error:
                self.print_error(f"Could not check attachment hash for {path.name}: {error}")
                continue

            if current_sha256 != stored_sha256:
                changed_attachments.append(attachment)

        if not changed_attachments:
            return

        changed_names = ", ".join(item.get("filename", "attachment") for item in changed_attachments)
        print()
        self.print_dim(f"Detected changed attachment(s): {changed_names}")
        self.print_dim("Refreshing attachment context before processing prompt...")

        self.rebuild_attachment_contexts()

    def rebuild_attachment_contexts(self) -> None:
        """
        Rebuild all attachment contexts and vector indexes from the current attachment list.

        Rebuilding all attachments is simpler and safer than trying to surgically remove
        old chunks from the vector index, because VectorMemoryIndex is append-oriented.
        """
        current_attachments = list(self.attachments)

        self.attachment_context_items.clear()
        self.image_attachment_context_items.clear()
        self.text_attachment_full_context_items.clear()
        self.attachment_index.clear()

        refreshed_attachments: list[dict[str, str]] = []

        for attachment in current_attachments:
            source_path = attachment.get("source_path", "")
            if not source_path:
                continue

            path = Path(source_path)
            if not path.exists() or not path.is_file():
                self.print_error(f"Skipping missing attachment during refresh: {source_path}")
                continue

            embedded_context_items = self.embed_attachment(path)
            if not embedded_context_items:
                self.print_error(f"Skipping unreadable attachment during refresh: {path.name}")
                continue

            mime_type, _ = mimetypes.guess_type(str(path.resolve()))
            refreshed_attachment = dict(attachment)
            refreshed_attachment.update(
                {
                    "source_path": str(path.resolve()),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "mime_type": mime_type or "application/octet-stream",
                    "sha256": utils.calculate_file_sha256(path),
                    "refreshed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

            refreshed_attachments.append(refreshed_attachment)
            self.store_attachment_context_items(embedded_context_items)
            self.store_full_text_attachment_context(path)

        self.attachments = refreshed_attachments

        self.print_info(f"Attachment context refreshed for {len(self.attachments)} file(s).")
        print()
        self.session.app.invalidate()

    def embed_attachment(self, path: Path) -> list[dict]:
        """
        Extract, chunk, and embed one attachment immediately when it is attached.
        """
        print()
        self.print_dim(f"Reading attachment: {path.name}")

        # Check if this is a Pollinations embedding model
        # Handle both "pollinations/openai-3-small" and "openai-3-small" formats
        embedding_model_name = self.embedding_model
        if embedding_model_name.startswith("pollinations/"):
            embedding_model_name = embedding_model_name.split("/", 1)[1]
        
        if embedding_model_name in self.pollinations_client.list_models_whitelisted():
            # For Pollinations models, we need to handle embedding differently
            try:
                # First get the text content
                content = utils.read_attachment_to_string(path)
                if not content:
                    self.print_error(f"No readable content found in attachment: {path.name}")
                    print()
                    return []
                
                # Split content into chunks
                chunks = utils.split_with_recursive_text_splitter(content)
                
                # Create embedded context items using Pollinations embeddings
                embedded_context_items = []
                for i, chunk in enumerate(chunks):
                    try:
                        embedding = self.pollinations_client.create_embeddings(chunk, model=embedding_model_name)
                        if utils.is_embedding_vector(embedding):
                            embedded_context_items.append({
                                "source_type": "attachment_chunk",
                                "source_path": str(path.resolve()),
                                "title": path.name,
                                "content": chunk,
                                "chunk_index": i,
                                "embedding": embedding,
                                "extension": path.suffix.lower(),
                                "mime_type": mimetypes.guess_type(str(path))[0] or "application/octet-stream",
                            })
                    except Exception:
                        # Skip chunks that fail to embed
                        continue
                        
            except Exception as error:
                self.print_error(f"Could not read/embed attachment {path.name}: {error}")
                print()
                return []
        else:
            # Use Ollama client for embeddings
            try:
                # Check if we need to use Ollama client for vision model too
                vision_model_to_use = None
                # Handle both "pollinations/mistral" and "mistral" formats for vision model
                vision_model_name = self.vision_model
                if vision_model_name.startswith("pollinations/"):
                    vision_model_name = vision_model_name.split("/", 1)[1]
                
                if vision_model_name in self.pollinations_client.list_models_whitelisted():
                    # For Pollinations vision models, we don't use Ollama client for vision
                    pass
                else:
                    vision_model_to_use = self.vision_model if path.suffix.lower() in IMAGE_ATTACHMENT_EXTENSIONS else None
                    
                embedded_context_items = utils.attachment_to_embedded_context_items(
                    client=self.client,
                    file_path=path,
                    model=embedding_model_name,
                    vision_model=vision_model_to_use,
                )
            except Exception as error:
                self.print_error(f"Could not read/embed attachment {path.name}: {error}")
                print()
                return []

        if not embedded_context_items:
            self.print_error(f"No readable content found in attachment: {path.name}")
            print()
            return []

        self.print_dim(f"Embedded {len(embedded_context_items)} attachment chunk(s): {path.name}")
        print()
        return embedded_context_items

    def store_attachment_context_items(self, context_items: list[dict]) -> None:
        """
        Store embedded attachment chunks.

        Image attachments are stored separately and are always injected in full.
        Non-image attachments are stored in the vector index for relevance ranking.
        """
        for context_item in context_items:
            stored_item = dict(context_item)
            extension = str(stored_item.get("extension") or "").lower()

            if extension in IMAGE_ATTACHMENT_EXTENSIONS or stored_item.get("source_type") == "image_attachment":
                stored_item.pop("embedding", None)
                self.image_attachment_context_items.append(stored_item)
                continue

            self.attachment_context_items.append(stored_item)
            self.attachment_index.add_item(stored_item)

    def store_full_text_attachment_context(self, path: Path) -> None:
        """
        Store full source text for text-like attachments.

        Text/code/config questions often need the full file for edits, rewrites,
        debugging, and source-of-truth context. We still keep ranked embedding chunks,
        but full text is also injected for complete context.
        """
        extension = path.suffix.lower()
        if extension not in utils.TEXT_ATTACHMENT_EXTENSIONS:
            return

        resolved_path = str(path.resolve())
        if any(item.get("source_path") == resolved_path for item in self.text_attachment_full_context_items):
            return

        try:
            content = utils.read_attachment_to_string(path)
        except Exception as error:
            self.print_error(f"Could not read full text context for {path.name}: {error}")
            return

        content = content.strip()
        if not content:
            return

        metadata = utils.build_attachment_metadata(path)
        self.text_attachment_full_context_items.append(
            {
                "source_type": "text_attachment_full_context",
                "source_path": resolved_path,
                "title": path.name,
                "content": content,
                **metadata,
            }
        )

    def is_supported_attachment(self, path: Path) -> bool:
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in {".gitignore", ".env"}:
            return True
        return suffix in SUPPORTED_ATTACHMENT_EXTENSIONS

    def handle_paste_command(self) -> None:
        pasted_text = self.read_clipboard_text()
        if pasted_text:
            self.next_input_prefill = pasted_text
            self.print_info("Clipboard text loaded into the next prompt.")
            print()
            return

        image_path = self.save_clipboard_image()
        if image_path and self.add_attachment(image_path):
            self.print_info(f"Clipboard image attached: {image_path.name}")
            print()
            self.session.app.invalidate()
            return

        self.print_error("Clipboard does not contain supported text or image data.")
        print()

    def read_clipboard_text(self) -> str:
        system_name = platform.system()
        commands: list[list[str]] = []

        if system_name == "Darwin":
            commands = [["pbpaste"]]
        elif system_name == "Linux":
            commands = [["wl-paste", "--no-newline"], ["xclip", "-selection", "clipboard", "-out"], ["xsel", "--clipboard", "--output"]]
        elif system_name == "Windows":
            commands = [["powershell", "-NoProfile", "-Command", "Get-Clipboard"]]

        for command in commands:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=3)
            except (FileNotFoundError, subprocess.SubprocessError):
                continue

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout

        return ""

    def save_clipboard_image(self) -> Path | None:
        system_name = platform.system()
        if system_name in {"Darwin", "Windows"}:
            return self.save_clipboard_image_with_pillow()

        self.print_dim("Image clipboard paste currently supports macOS/Windows with Pillow.")
        self.print_dim("Install it with: pip install pillow")
        print()
        return None

    def save_clipboard_image_macos(self) -> Path | None:
        return self.save_clipboard_image_with_pillow()

    def save_clipboard_image_with_pillow(self) -> Path | None:
        try:
            clipboard_data = ImageGrab.grabclipboard()
        except Exception as error:
            self.print_error(f"Could not read clipboard image: {error}")
            print()
            return None

        if clipboard_data is None:
            return None

        # Some platforms return a list of file paths when files are copied, not raw image data.
        # /paste intentionally ignores copied files; use /attach <path> for files instead.
        if isinstance(clipboard_data, list):
            return None

        if not isinstance(clipboard_data, Image.Image):
            return None

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("clipboard_%Y%m%d_%H%M%S.png")
        output_path = TEMP_DIR / filename

        try:
            clipboard_data.save(output_path, "PNG")
        except Exception as error:
            self.print_error(f"Could not save clipboard image: {error}")
            print()
            return None

        if not output_path.exists():
            return None

        return output_path

    def build_attachment_toolbar_lines(self, max_visible: int = 3) -> list[str]:
        if not self.attachments:
            return []

        visible_attachments = self.attachments[-max_visible:]
        hidden_count = max(0, len(self.attachments) - len(visible_attachments))

        lines: list[str] = []

        for display_index, attachment in enumerate(
            visible_attachments,
            start=len(self.attachments) - len(visible_attachments) + 1,
        ):
            filename = attachment.get("filename") or Path(attachment.get("source_path", "")).name
            extension = attachment.get("extension") or "file"
            lines.append(
                f"attachment {display_index} {extension} · {self.clip_text(filename, 72)}"
            )

        if hidden_count:
            lines.insert(0, f"+{hidden_count} more attachment(s)")

        return lines

    # -----------------------------
    # Display
    # -----------------------------

    def show_landing_page(self) -> None:
        for line in MASCOT_LINES:
            print(self.blue(line))

    def show_about(self) -> None:
        print()
        print(self.bold(APP_NAME))
        print("A Claude Code-style terminal interface for your SimpleAgent variant.")
        print()
        print("Backend:")
        # Check connectivity
        pollinations_configured = bool(os.getenv("POLLINATIONS_API_KEY") or self.config.get("pollinations_api_key"))
        ollama_available = self.client.is_available()
        if pollinations_configured:
            print("  Pollinations: Ready")
        else:
            print("  Pollinations: Not Configured")
        if ollama_available:
            print("  Ollama: Ready")
        else:
            print("  Ollama: Unavailable")
        print(f"  Model:       {self.model}{self.format_num_context(self.model_num_context)}")
        print(f"  Embeddings:  {self.embedding_model}{self.format_num_context(self.embedding_model_num_context)}")
        print(f"  Vision:      {self.vision_model}{self.format_num_context(self.vision_model_num_context)}")
        print("  Streaming:   always on")
        print(f"  Recent chat: {len(self.messages)}/{MAX_RECENT_MESSAGES}")
        print(f"  Memory items:{len(self.memory_items)}/{MAX_RELEVANT_MEMORY_ITEMS}")
        print()

    def set_model(self, model: str, persist: bool = True) -> None:
        self.model = model
        # Check if this is a Pollinations model
        if model in self.pollinations_client.list_models_whitelisted():
            # For Pollinations models, we don't use the Ollama client for context info
            self.model_num_context = None
        else:
            # For Ollama models, we would use the Ollama client
            self.model_num_context = self.get_ollama_model_num_context(model)

        if persist:
            self.config["model"] = model
            save_config(self.config)
            self.print_info(
                f"Model changed to: {self.model}{self.format_num_context(self.model_num_context)} and saved to {CONFIG_FILE}"
            )
        else:
            self.print_info(f"Model changed to: {self.model}{self.format_num_context(self.model_num_context)}")

    def set_embedding_model(self, model: str, persist: bool = True) -> None:
        self.embedding_model = model
        # Check if this is a Pollinations model
        # Handle both "pollinations/openai-3-small" and "openai-3-small" formats
        model_name = model
        if model_name.startswith("pollinations/"):
            model_name = model_name.split("/", 1)[1]
        
        if model_name in self.pollinations_client.list_models_whitelisted():
            self.embedding_model_num_context = None
        else:
            self.embedding_model_num_context = self.get_ollama_model_num_context(model)

        if persist:
            self.config["embedding_model"] = self.embedding_model
            save_config(self.config)
            self.print_info(
                f"Embeddings model changed to: {self.embedding_model}{self.format_num_context(self.embedding_model_num_context)} and saved to {CONFIG_FILE}"
            )
        else:
            self.print_info(
                f"Embeddings model changed to: {self.embedding_model}{self.format_num_context(self.embedding_model_num_context)}"
            )

    def set_vision_model(self, model: str, persist: bool = True) -> None:
        self.vision_model = model
        # Check if this is a Pollinations model
        # Handle both "pollinations/mistral" and "mistral" formats
        model_name = model
        if model_name.startswith("pollinations/"):
            model_name = model_name.split("/", 1)[1]
        
        if model_name in self.pollinations_client.list_models_whitelisted():
            self.vision_model_num_context = None
        else:
            self.vision_model_num_context = self.get_ollama_model_num_context(model)

        if persist:
            self.config["vision_model"] = self.vision_model
            save_config(self.config)
            self.print_info(
                f"Vision model changed to: {self.vision_model}{self.format_num_context(self.vision_model_num_context)} and saved to {CONFIG_FILE}"
            )
        else:
            self.print_info(
                f"Vision model changed to: {self.vision_model}{self.format_num_context(self.vision_model_num_context)}"
            )

    def show_models(self) -> None:
        # Show Ollama models if available
        ollama_models = []
        try:
            ollama_models = self.client.list_models()
        except Exception as exc:
            self.print_dim(f"Could not list Ollama models: {exc}")
        
        # Show Pollinations models
        pollinations_models = self.pollinations_client.list_models_whitelisted()
        
        if not ollama_models and not pollinations_models:
            self.print_dim("No models found.")
            return

        print()
        if ollama_models:
            print(self.bold("Installed Ollama models:"))
            for model in ollama_models:
                markers = []
                if model == self.model:
                    markers.append("chat")
                if model == self.embedding_model:
                    markers.append("embed")
                if model == self.vision_model:
                    markers.append("vision")
                marker_text = f" * [{' / '.join(markers)}]" if markers else ""
                num_context = self.get_ollama_model_num_context(model)
                print(f"  ollama/{model}{self.format_num_context(num_context)}{marker_text}")
        
        if pollinations_models:
            print(self.bold("Pollinations models:"))
            for model in pollinations_models:
                markers = []
                if model == self.model:
                    markers.append("chat")
                if model == self.embedding_model:
                    markers.append("embed")
                if model == self.vision_model:
                    markers.append("vision")
                marker_text = f" * [{' / '.join(markers)}]" if markers else ""
                # Pollinations models don't have context length info
                print(f"  pollinations/{model}{marker_text}")
        print()

    def show_workflow_help(self) -> None:
        print()
        print(self.bold("Workflow assignment"))
        print(self.dim("Workflows are assigned per persona."))
        print()

        self.print_info("To choose a workflow for a new persona:")
        self.print_dim("  /persona → Add → enter name → enter prompt → select workflow")
        print()

        self.print_info("To change a workflow for an existing custom persona:")
        self.print_dim("  /persona → Edit → select persona → update prompt → select workflow")
        print()

        self.print_info("To install a workflow .md and create a persona for it:")
        self.print_dim("  /workflow-install path/to/workflow.md")
        print()

        self.print_dim("Built-in personas cannot be edited. Create a custom persona if you want a different workflow.")
        self.print_dim(f"Current persona: {self.active_persona}")
        self.print_dim(f"Current workflow: {self.persona_workflows.get(self.active_persona, DEFAULT_WORKFLOW_NAME)}")
        print()

    def show_response_debug(self) -> None:
        print()
        print(self.bold("Raw model responses from last interaction"))
        print(self.dim(f"Model: {self.model}"))
        print(self.dim(f"Total raw responses: {len(self.raw_model_responses)}"))
        print()

        if not self.raw_model_responses:
            self.print_dim("No raw model responses available.")
            self.print_dim("Run a prompt first to capture raw responses.")
            print()
            return

        # Show raw responses
        # Combine all responses into one for better viewing
        combined_response = "".join(str(r) for r in self.raw_model_responses)
        
        print(self.blue("[01] Combined Raw Response"))
        print(self.dim("-" * 88))
        print(combined_response)
        print(self.dim("-" * 88))
        print()
        
        # Also show individual responses if there are multiple
        if len(self.raw_model_responses) > 1:
            print(self.blue("Individual Responses:"))
            for index, response in enumerate(self.raw_model_responses, start=1):
                print(self.blue(f"[{index:02d}] Raw Response"))
                print(self.dim("-" * 88))
                
                # Pretty print the response
                if isinstance(response, dict):
                    # Pretty print dictionary responses
                    try:
                        import json
                        pretty_json = json.dumps(response, indent=2, ensure_ascii=False)
                        print(pretty_json)
                    except:
                        print(str(response))
                else:
                    # Print string responses
                    print(str(response))
                
                print(self.dim("-" * 88))
                print()

    def show_workflow_debug(self) -> None:
        if not self.last_workflow_messages:
            print()
            self.print_dim("No workflow prompt has been built yet.")
            self.print_dim("Send a normal prompt first, then run /workflow-debug.")
            print()
            return

        print()
        print(self.bold("Last workflow prompt messages"))
        print(self.dim(f"Persona: {self.active_persona.lower()}"))
        print(self.dim(f"Workflow: {self.persona_workflows.get(self.active_persona, DEFAULT_WORKFLOW_NAME)}"))
        print(self.dim(f"Messages: {len(self.last_workflow_messages)}"))
        print()

        for index, message in enumerate(self.last_workflow_messages, start=1):
            role = str(message.get("role") or "unknown")
            content = str(message.get("content") or "")
            token_estimate = self.estimate_text_tokens(content)

            print(self.blue(f"[{index:02d}] {role} · ~{token_estimate:,} token(s)"))
            print(self.dim("-" * 88))

            if content:
                print(content)
            else:
                self.print_dim("<empty>")

            print(self.dim("-" * 88))
            print()

    def show_history(self) -> None:
        if not self.messages and not self.memory_items:
            self.print_dim("No messages in this session yet.")
            return

        print()
        print(self.bold("Session history:"))
        print(self.dim(f"Recent chat: {len(self.messages)}/{MAX_RECENT_MESSAGES}"))
        print(
            self.dim(
                f"Memory items: {len(self.memory_items)} stored · "
                f"retrieval top-k {MAX_RELEVANT_MEMORY_ITEMS} · "
                f"index {self.memory_index.backend}"
            )
        )

        if self.messages:
            print()
            print(self.bold("Recent full message:"))
            for index, message in enumerate(self.messages, start=1):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                preview = content.replace("\n", " ")
                if len(preview) > 300:
                    preview = preview[:297] + "..."
                print(f"{index:02d}. {role}: {preview}")

        if self.memory_items:
            print()
            print(self.bold("Compacted memory:"))
            for index, item in enumerate(self.memory_items, start=1):
                role = item.get("role", "unknown")
                content = item.get("content", "")
                preview = content.replace("\n", " | ")
                if len(preview) > 220:
                    preview = preview[:217] + "..."
                has_embedding = "embedded" if item.get("embedding") else "no embedding"
                print(f"{index:02d}. {role} [{has_embedding}]: {preview}")

        print()

    def print_agent_header(self) -> None:
        print(self.blue("SimpleAgent"), self.dim(f"({self.model})"))
        #print(self.dim("-" * 48))

    def get_bottom_toolbar(self):
        toolbar_lines = []
        for line in self.build_attachment_toolbar_lines():
            toolbar_lines.append(f"<ansiyellow> ■ {self.escape_toolbar_html(line)} </ansiyellow>")

        for line in self.format_web_sources_for_toolbar():
            toolbar_lines.append(
                f"<ansigreen> ◆ {self.escape_toolbar_html(line)} </ansigreen>"
            )

        active_persona = self.escape_toolbar_html(getattr(self, "active_persona", "Default"))
        active_workflow = self.escape_toolbar_html(
            getattr(self, "persona_workflows", {}).get(
                getattr(self, "active_persona", "Default"),
                DEFAULT_WORKFLOW_NAME,
            )
        )
        toolbar_lines.append(
            f"<ansiblue> @ persona: {active_persona.lower()} + workflow: {active_workflow.lower()} </ansiblue> "
            f"<ansigray>■ {self.escape_toolbar_html(self.model)} ■ type / for commands</ansigray>"
        )
        return HTML("\n".join(toolbar_lines))

    def format_web_sources_for_toolbar(self, max_visible: int = 3) -> list[str]:
        if not self.web_sources:
            return []

        visible_sources = self.web_sources[-max_visible:]
        hidden_count = max(0, len(self.web_sources) - len(visible_sources))

        lines: list[str] = []

        for display_index, item in enumerate(visible_sources, start=len(self.web_sources) - len(visible_sources) + 1):
            label = item.get("label", "web")
            source_type = item.get("source_type", "web")
            prefix = "search" if source_type == "search" else "url"

            lines.append(
                f"web {display_index} {prefix} · {self.clip_text(label, 72)}"
            )

        if hidden_count:
            lines.insert(0, f"+{hidden_count} more web source(s)")

        return lines

    def escape_toolbar_html(self, text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def show_command_preview(self) -> None:
        print()
        print(self.blue("Slash commands:"))
        for command, description in COMMANDS.items():
            preview = command_preview(command, description)
            print(f"  {self.blue(command):<18} {self.dim(preview)}")
        print()

    def read_user_input(self) -> str:
        default_text = self.next_input_prefill
        self.next_input_prefill = ""
        return self.session.prompt(
            HTML("<ansicyan>❯ </ansicyan>"),
            complete_while_typing=True,
            default=default_text,
        )

    def authenticate_with_pollinations(self) -> None:
        """
        Authenticate with Pollinations API using the Bring Your Own Pollen device flow.
        """
        print()
        self.print_info("Starting Pollinations API authentication...")
        self.print_dim("This will open a browser window for you to authorize the app.")
        self.print_dim("After authorization, your API key will be saved to your config.")
        print()

        try:
            # Request device code
            device_code_response = self.pollinations_client.request_device_code()
            device_code = device_code_response.get("device_code")
            user_code = device_code_response.get("user_code")
            verification_uri = device_code_response.get("verification_uri")
            
            if not device_code or not user_code or not verification_uri:
                self.print_error("Failed to get device code from Pollinations API")
                return
            
            # Display instructions
            self.print_info(f"Please visit: {verification_uri}")
            self.print_info(f"Enter this code: {user_code}")
            self.print_dim("Then click 'Allow' to authorize this app.")
            print()
            
            # Poll for the access token
            self.print_dim("Waiting for authorization...")
            token_response = self.pollinations_client.poll_for_device_token(device_code)
            
            if "access_token" in token_response:
                access_token = token_response["access_token"]
                
                # Get user info to confirm successful auth
                user_info = self.pollinations_client.get_user_info(access_token)
                
                # Save the API key to environment variable and config
                os.environ["POLLINATIONS_API_KEY"] = access_token
                
                # Update config with the API key
                self.config["pollinations_api_key"] = access_token
                save_config(self.config)
                
                # Set the API key on the pollinations client
                self.pollinations_client.set_api_key(access_token)
                
                self.print_info("Successfully authenticated with Pollinations API!")
                self.print_info(f"User: {user_info.get('name', 'Anonymous')}")
                self.print_dim("Your API key has been saved to config.")
                print()
            else:
                self.print_error("Authentication failed. Please try again.")
                print()
                
        except Exception as e:
            self.print_error(f"Authentication error: {e}")
            print()

    def clear_screen(self) -> None:
        # Clear visible screen, clear scrollback buffer where supported, then move cursor home.
        print("\033[2J\033[3J\033[H", end="", flush=True)



    def refresh_model_context_lengths(self) -> None:
        self.model_num_context = self.get_ollama_model_num_context(self.model)
        self.embedding_model_num_context = self.get_ollama_model_num_context(self.embedding_model)
        self.vision_model_num_context = self.get_ollama_model_num_context(self.vision_model)

    def format_num_context(self, num_context: int | None) -> str:
        if not num_context:
            return ""
        return f" [{num_context:,} ctx]"

    def get_ollama_model_num_context(self, model: str) -> int | None:
        if not model:
            return None

        # Check if this is a Pollinations model - if so, return None since Pollinations models don't have context lengths
        # Handle both "pollinations/mistral" and "mistral" formats
        model_name = model
        if model_name.startswith("pollinations/"):
            model_name = model_name.split("/", 1)[1]
        
        if model_name in self.pollinations_client.list_models_whitelisted():
            return None

        try:
            response = self.ollama_show_model(model)
        except Exception:
            return None

        model_info = response.get("model_info") or {}
        context_from_info = self.extract_context_length_from_model_info(model_info)
        if context_from_info:
            return context_from_info

        parameters = response.get("parameters") or ""
        return self.extract_num_context_from_parameters(parameters)

    def ollama_show_model(self, model: str) -> dict:
        url = self.host.rstrip("/") + "/api/show"
        payload = json.dumps({"model": model}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")

        return json.loads(body)

    def extract_context_length_from_model_info(self, model_info: dict) -> int | None:
        context_keys = [
            "context_length",
            "llama.context_length",
            "qwen2.context_length",
            "qwen3.context_length",
            "gemma.context_length",
            "mistral.context_length",
            "deepseek2.context_length",
            "deepseek3.context_length",
            "granite.context_length",
            "granite-vision.context_length",
            "nomic-bert.context_length",
        ]

        for key in context_keys:
            value = model_info.get(key)
            parsed = self.parse_positive_int(value)
            if parsed:
                return parsed

        for key, value in model_info.items():
            if key.endswith(".context_length"):
                parsed = self.parse_positive_int(value)
                if parsed:
                    return parsed

        return None

    def extract_num_context_from_parameters(self, parameters: str) -> int | None:
        if not parameters:
            return None

        match = re.search(r"(?:^|\n)\s*num_ctx\s+(\d+)\b", parameters)
        if not match:
            return None

        return self.parse_positive_int(match.group(1))

    def parse_positive_int(self, value) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None

        if parsed <= 0:
            return None

        return parsed


def main() -> None:
    app = SimpleAgentTUI()
    app.run()


if __name__ == "__main__":
    main()
