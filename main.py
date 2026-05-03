# main.py
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application.run_in_terminal import run_in_terminal

from ollama import OllamaClient, OllamaConfig


APP_NAME = "SimpleAgent TUI"
DEFAULT_MODEL = "nemotron-3-nano:4b"
DEFAULT_EMBEDDING_MODEL = "ordis/jina-embeddings-v2-base-code:latest"
MAX_RECENT_MESSAGES = 2
MAX_MEMORY_TEXT_LENGTH = 900
MAX_RELEVANT_MEMORY_ITEMS = 8

CONFIG_DIR = Path.home() / ".simpleagent-cli"
CONFIG_FILE = CONFIG_DIR / "config.json"

COMMANDS = {
    "/help": "Show this help menu",
    "/model": "Show or change the Ollama chat model",
    "/embedding": "Show or change the Ollama embeddings model",
    "/models": "List installed Ollama models",
    "/select-model": "Select and persist an installed Ollama chat model",
    "/select-embedding": "Select and persist an installed Ollama embeddings model",
    "/system": "Show or set the system prompt",
    "/system-reset": "Reset the system prompt",
    "/stream": "Enable streaming",
    "/no-stream": "Disable streaming",
    "/history": "Show current session history",
    "/reset": "Clear current session history",
    "/about": "Show app info",
    "/exit": "Exit app",
    "/quit": "Exit app",
    "/q": "Exit app",
}


COMMAND_USAGE = {
    "/model": "/model <name>",
    "/embedding": "/embedding <name>",
    "/select-model": "/select-model",
    "/select-embedding": "/select-embedding",
    "/system": "/system <prompt>",
}

THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
STREAM_THINK_START = "<think>"
STREAM_THINK_END = "</think>"

def command_preview(command: str, description: str) -> str:
    usage = COMMAND_USAGE.get(command, command)
    return f"{usage} — {description}"

MASCOT = r"""
          ╭─────┴─────╮
         ╭┤  ◉     ◉  ├╮
         ││     ▾     ││
         ╰┤  ╰─────╯  ├╯
           ╰─────────╯

          Simple Agent
"""


def build_help_text() -> str:
    lines = ["", "Available commands:", ""]

    for command, description in COMMANDS.items():
        usage = COMMAND_USAGE.get(command, command)
        lines.append(f"  {usage:<22} {description}")

    lines.extend([
        "",
        "Normal usage:",
        "  Type anything and press Enter to chat with the model.",
        "  Type / to open command suggestions.",
    ])

    return "\n".join(lines)

# -----------------------------
# Slash command completion
# -----------------------------

class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event: CompleteEvent):
        text = document.text_before_cursor

        if not text.startswith("/"):
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


DEFAULT_SYSTEM_PROMPT = """
You are SimpleAgent, a lightweight local AI assistant running through Ollama.

Rules:
- Be useful, direct, and practical.
- Keep answers clear and structured.
- If the user asks for code, provide working code.
- Do not pretend to have tools that are not currently connected.
- If more context is needed, ask a short follow-up question.
""".strip()


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


class SimpleAgentTUI:
    def __init__(self) -> None:
        self.config = load_config()
        self.model = os.getenv("SIMPLEAGENT_MODEL") or self.config.get("model", DEFAULT_MODEL)
        self.host = os.getenv("OLLAMA_HOST") or self.config.get("host", "http://localhost:11434")
        self.stream = True
        self.embedding_model = self.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self.system_prompt = self.config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.messages: list[dict[str, str]] = []
        self.memory_items: list[dict] = []
        self.last_thinking: str = ""
        self.last_visible_reply: str = ""
        self.show_thinking: bool = False
        self.is_streaming_response: bool = False

        self.key_bindings = KeyBindings()

        @self.key_bindings.add("/")
        def _(event):
            buffer = event.current_buffer
            buffer.insert_text("/")
            buffer.start_completion(select_first=False)
            event.app.invalidate()

        @self.key_bindings.add("c-o")
        def _(event):
            run_in_terminal(self.toggle_thinking)
            event.app.invalidate()

        self.session = PromptSession(
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            editing_mode=EditingMode.EMACS,
            key_bindings=self.key_bindings,
            reserve_space_for_menu=3,
            bottom_toolbar=self.get_bottom_toolbar,
        )

        self.client = OllamaClient(
            OllamaConfig(
                model=self.model,
                host=self.host,
                temperature=0.7,
                top_p=0.9,
                timeout=180,
            )
        )

    # -----------------------------
    # App lifecycle
    # -----------------------------

    def run(self) -> None:
        self.clear_screen()
        self.show_landing_page()

        if not self.client.is_available():
            self.print_error("Ollama is not reachable. Start it with: ollama serve")
            return

        self.print_info("Connected to Ollama.")
        self.print_info(f"Model: {self.model}")
        self.print_dim("Type /help for commands. Ctrl-O collapse/expand thinking. Type /exit to quit.\n")

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
                self.chat(user_input)

        self.print_dim("\nGoodbye. Keep building. 小步快跑.\n")

    # -----------------------------
    # Chat
    # -----------------------------

    def chat(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": user_input})
        self.compact_messages()

        chat_messages = self.build_chat_messages(user_input)

        print()
        self.print_agent_header()

        try:
            if self.stream:
                raw_reply = self.stream_chat_reply(chat_messages)
                assistant_reply = self.extract_and_store_thinking(raw_reply)
                self.last_visible_reply = assistant_reply
            else:
                raw_reply = self.client.chat(
                    chat_messages,
                    stream=False,
                    model=self.model,
                )
                assistant_reply = self.extract_and_store_thinking(raw_reply)
                self.last_visible_reply = assistant_reply
                self.print_model_reply(assistant_reply)
                print()

            self.messages.append(
                {"role": "assistant", "content": assistant_reply}
            )
            self.compact_messages()

        except Exception as exc:
            self.is_streaming_response = False
            self.print_error(f"Model call failed: {exc}")
            self.messages.pop()

    def build_chat_messages(self, user_input: str) -> list[dict[str, str]]:
        chat_messages = [
            {"role": "system", "content": self.build_system_prompt()},
        ]

        relevant_memory = self.get_relevant_memory_context(user_input)
        if relevant_memory:
            chat_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Compacted older conversation context selected by embeddings. "
                        "Treat this as background memory. The latest user message below is higher priority.\n\n"
                        f"{relevant_memory}"
                    ),
                }
            )

        chat_messages.extend(self.messages[-MAX_RECENT_MESSAGES:])
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

        try:
            embedding = self.client.embed(memory_text, model=self.embedding_model)
        except Exception:
            embedding = []

        self.memory_items.append(
            {
                "role": role,
                "content": content,
                "embedding": embedding,
            }
        )

    def get_relevant_memory_context(self, query_text: str) -> str:
        if not self.memory_items:
            return ""

        try:
            query_embedding = self.client.embed(query_text, model=self.embedding_model)
        except Exception:
            return ""

        if not query_embedding:
            return ""

        scored_items = []
        for item in self.memory_items:
            item_embedding = item.get("embedding") or []
            score = self.cosine_similarity(query_embedding, item_embedding)
            if score > 0:
                scored_items.append((score, item))

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        selected_items = [item for _, item in scored_items[:MAX_RELEVANT_MEMORY_ITEMS]]

        if not selected_items:
            return ""

        lines = []
        for item in selected_items:
            role = item.get("role", "unknown")
            content = item.get("content", "").strip()
            if len(content) > 700:
                content = content[:697].rstrip() + "..."
            lines.append(f"- {role}: {content}")

        return "\n".join(lines)

    def cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)

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

    def stream_chat_reply(self, chat_messages: list[dict[str, str]]) -> str:
        thinking_text = ""
        reply_text = ""
        self.reset_streaming_reply_buffer()
        previous_show_thinking = self.show_thinking
        self.show_thinking = False
        self.streaming_thinking_line_count = 0
        self.streaming_thinking_last_block = ""
        self.streaming_thinking_closed = False

        pending = ""
        in_thinking = False
        thinking_started = False

        response_stream = self.client.chat(
            chat_messages,
            stream=True,
            model=self.model,
        )
        self.is_streaming_response = True

        for chunk in response_stream:
            pending += chunk

            while pending:
                lower_pending = pending.lower()

                if in_thinking:
                    end_index = lower_pending.find(STREAM_THINK_END)

                    if end_index == -1:
                        previous_thinking = thinking_text
                        thinking_text += pending
                        self.render_streaming_thinking(previous_thinking, thinking_text)
                        pending = ""
                        break

                    thinking_fragment = pending[:end_index]
                    previous_thinking = thinking_text
                    thinking_text += thinking_fragment
                    self.render_streaming_thinking(previous_thinking, thinking_text)
                    #self.finish_streaming_thinking_display(thinking_text)

                    pending = pending[end_index + len(STREAM_THINK_END):]
                    in_thinking = False
                    continue

                start_index = lower_pending.find(STREAM_THINK_START)

                if start_index == -1:
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
                    #print(self.dim("Thinking (streaming)"))
                    #print(self.dim("-" * 48))
                    thinking_started = True

                pending = pending[start_index + len(STREAM_THINK_START):]
                in_thinking = True

        if pending:
            if in_thinking:
                previous_thinking = thinking_text
                thinking_text += pending
                self.render_streaming_thinking(previous_thinking, thinking_text)
            else:
                self.print_streaming_reply_text(pending)
                reply_text += pending

        thinking = self.normalise_thinking_text(thinking_text)
        visible_reply = reply_text.strip()

        if thinking_started and not self.streaming_thinking_closed:
            self.finish_streaming_thinking_display(thinking_text)

        if thinking:
            self.last_thinking = thinking
            self.show_thinking = False
            self.last_visible_reply = visible_reply
        else:
            self.show_thinking = previous_show_thinking

        if thinking_started and not self.streaming_thinking_closed:
            #print(self.dim("-" * 48))
            self.streaming_thinking_line_count = 0
            self.streaming_thinking_last_block = ""
            self.streaming_thinking_closed = True

        self.flush_streaming_reply_buffer()
        self.is_streaming_response = False
        print()
        print()

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

        if current_block == self.streaming_thinking_last_block:
            return

        self.clear_streaming_thinking_block()
        sys.stdout.write(self.dim(current_block))
        sys.stdout.write("\n")
        sys.stdout.flush()

        self.streaming_thinking_last_block = current_block
        self.streaming_thinking_line_count = self.get_visual_line_count(current_block)

    def clear_streaming_thinking_block(self) -> None:
        line_count = getattr(self, "streaming_thinking_line_count", 0)

        if line_count <= 0:
            return

        for _ in range(line_count):
            sys.stdout.write("\033[1A\r\033[2K")

        sys.stdout.flush()

    def finish_streaming_thinking_display(self, thinking_text: str) -> None:
        if getattr(self, "streaming_thinking_closed", False):
            return

        current_block = self.build_streaming_thinking_block(thinking_text)

        if current_block and current_block != self.streaming_thinking_last_block:
            self.clear_streaming_thinking_block()
            sys.stdout.write(self.dim(current_block))
            sys.stdout.write("\n")
            self.streaming_thinking_last_block = current_block
            self.streaming_thinking_line_count = current_block.count("\n") + 1

        #sys.stdout.write(self.dim("-" * 48) + "\n")
        sys.stdout.flush()
        self.streaming_thinking_line_count = 0
        self.streaming_thinking_last_block = ""
        self.streaming_thinking_closed = True

    def build_thinking_display_text(self, thinking: str) -> str:
        thinking = thinking.strip()

        if not thinking:
            return ""

        words = re.findall(r"\S+", thinking)
        collapse_word_limit = 45

        if len(words) > collapse_word_limit and not self.show_thinking:
            preview = " ".join(words[:collapse_word_limit])
            hidden_word_count = len(words[collapse_word_limit:])
            return f"{preview}\n+ {hidden_word_count} more word(s) (Ctrl-O to expand)"

        return thinking

    def build_streaming_thinking_block(self, thinking_text: str) -> str:
        thinking = self.normalise_thinking_text(thinking_text)
        return self.build_thinking_display_text(thinking)

    def get_hidden_thinking_word_count(self, thinking: str) -> int:
        words = re.findall(r"\S+", thinking.strip())
        collapse_word_limit = 45

        if len(words) <= collapse_word_limit:
            return 0

        return len(words[collapse_word_limit:])

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
            self.show_thinking = False
        elif not self.stream:
            self.last_thinking = ""
            self.show_thinking = False

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
            self.print_dim("No visible response returned.")

    def print_thinking_block(self) -> None:
        thinking = self.last_thinking.strip()
        display_text = self.build_thinking_display_text(thinking)

        if display_text:
            print(self.dim(display_text))

    def toggle_thinking(self) -> None:
        if not self.last_thinking:
            return

        self.show_thinking = not self.show_thinking

        self.clear_screen()
        self.show_landing_page()
        self.print_info("Connected to Ollama.")
        self.print_info(f"Model: {self.model}")
        self.print_dim("Type /help for commands. Ctrl-O collapse/expand thinking. Type /exit to quit.\n")

        self.print_agent_header()
        self.print_model_reply(self.last_visible_reply)
        print()
        print()

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

        if command == "/help":
            with patch_stdout(raw=True):
                print(build_help_text())
            return True

        if command == "/about":
            with patch_stdout(raw=True):
                self.show_about()
            return True

        if command == "/model":
            if not arg:
                self.print_info(f"Current model: {self.model}")
                return True

            self.set_model(arg, persist=True)
            return True

        if command == "/embedding":
            if not arg:
                self.print_info(f"Current embeddings model: {self.embedding_model}")
                return True

            self.set_embedding_model(arg, persist=True)
            return True

        if command == "/models":
            self.show_models()
            return True

        if command == "/select-model":
            self.select_model()
            return True

        if command == "/select-embedding":
            self.select_embedding_model()
            return True

        if command == "/stream":
            self.stream = True
            self.print_info("Streaming enabled.")
            return True

        if command == "/no-stream":
            self.stream = False
            self.print_info("Streaming disabled.")
            return True

        if command == "/system":
            if not arg:
                print("\nCurrent system prompt:\n")
                print(self.system_prompt)
                print()
                return True

            self.system_prompt = arg.strip()
            self.config["system_prompt"] = self.system_prompt
            save_config(self.config)
            self.print_info("System prompt updated and saved.")
            return True

        if command == "/system-reset":
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
            self.config.pop("system_prompt", None)
            save_config(self.config)
            self.print_info("System prompt reset and saved.")
            return True

        if command == "/history":
            with patch_stdout(raw=True):
                self.show_history()
            return True

        if command == "/reset":
            self.messages.clear()
            self.memory_items.clear()
            self.last_thinking = ""
            self.last_visible_reply = ""
            self.show_thinking = False
            self.streaming_thinking_line_count = 0
            self.streaming_thinking_last_block = ""
            self.streaming_thinking_closed = True
            self.reset_streaming_reply_buffer()

            with patch_stdout(raw=True):
                self.clear_screen()
                self.show_landing_page()
                self.print_info("Connected to Ollama.")
                self.print_info(f"Model: {self.model}")
                self.print_dim("Type /help for commands. Ctrl-O collapse/expand thinking. Type /exit to quit.\n")
                self.print_info("Session history and screen cleared.")

            return True

        self.print_error(f"Unknown command: {command}")
        self.print_dim("Type /help to see available commands.")
        return True

    def parse_command(self, raw: str) -> tuple[str, str]:
        raw = raw.strip()
        if " " not in raw:
            return raw, ""

        command, arg = raw.split(" ", 1)
        return command.strip(), arg.strip()

    # -----------------------------
    # Display
    # -----------------------------

    def show_landing_page(self) -> None:
        print(self.blue(MASCOT))
        print(self.bold("Welcome to SimpleAgent TUI"))
        print(self.dim("A tiny local AI agent shell powered by Ollama.\n"))

    def show_about(self) -> None:
        print()
        print(self.bold(APP_NAME))
        print("A Claude Code-style terminal interface for your SimpleAgent variant.")
        print()
        print("Backend:")
        print(f"  Ollama host: {self.host}")
        print(f"  Model:       {self.model}")
        print(f"  Embeddings:  {self.embedding_model}")
        print(f"  Streaming:   {self.stream}")
        print(f"  Recent chat: {len(self.messages)}/{MAX_RECENT_MESSAGES}")
        print(f"  Memory items:{len(self.memory_items)}/{MAX_RELEVANT_MEMORY_ITEMS}")
        print()

    def set_model(self, model: str, persist: bool = True) -> None:
        self.model = model
        self.client.config.model = model

        if persist:
            self.config["model"] = model
            self.config["host"] = self.host
            save_config(self.config)
            self.print_info(f"Model changed to: {self.model} and saved to {CONFIG_FILE}")
        else:
            self.print_info(f"Model changed to: {self.model}")

    def set_embedding_model(self, model: str, persist: bool = True) -> None:
        self.embedding_model = model

        if persist:
            self.config["embedding_model"] = self.embedding_model
            self.config["host"] = self.host
            save_config(self.config)
            self.print_info(f"Embeddings model changed to: {self.embedding_model} and saved to {CONFIG_FILE}")
        else:
            self.print_info(f"Embeddings model changed to: {self.embedding_model}")

    def select_model(self) -> None:
        try:
            models = self.client.list_models()
        except Exception as exc:
            self.print_error(f"Could not list models: {exc}")
            return

        if not models:
            self.print_dim("No Ollama models found. Try: ollama pull nemotron-3-nano:4b")
            return

        print()
        print(self.bold("Select an Ollama model:"))
        for index, model in enumerate(models, start=1):
            marker = "*" if model == self.model else " "
            print(f"  {index:02d}. {marker} {model}")

        print()
        choice = input(self.blue("model number > ")).strip()
        if not choice.isdigit():
            self.print_error("Invalid selection.")
            return

        index = int(choice)
        if index < 1 or index > len(models):
            self.print_error("Invalid selection.")
            return

        self.set_model(models[index - 1], persist=True)

    def select_embedding_model(self) -> None:
        try:
            models = self.client.list_models()
        except Exception as exc:
            self.print_error(f"Could not list models: {exc}")
            return

        if not models:
            self.print_dim(f"No Ollama models found. Try: ollama pull {DEFAULT_EMBEDDING_MODEL}")
            return

        print()
        print(self.bold("Select an Ollama embeddings model:"))
        for index, model in enumerate(models, start=1):
            markers = []
            if model == self.model:
                markers.append("chat")
            if model == self.embedding_model:
                markers.append("embed")
            marker_text = f" [{' / '.join(markers)}]" if markers else ""
            print(f"  {index:02d}. {model}{marker_text}")

        print()
        choice = input(self.blue("embedding model number > ")).strip()
        if not choice.isdigit():
            self.print_error("Invalid selection.")
            return

        index = int(choice)
        if index < 1 or index > len(models):
            self.print_error("Invalid selection.")
            return

        self.set_embedding_model(models[index - 1], persist=True)

    def show_models(self) -> None:
        try:
            models = self.client.list_models()
        except Exception as exc:
            self.print_error(f"Could not list models: {exc}")
            return

        if not models:
            self.print_dim("No Ollama models found. Try: ollama pull nemotron-3-nano:4b")
            return

        print()
        print(self.bold("Installed Ollama models:"))
        for model in models:
            markers = []
            if model == self.model:
                markers.append("chat")
            if model == self.embedding_model:
                markers.append("embed")
            marker_text = f" * [{' / '.join(markers)}]" if markers else ""
            print(f"  {model}{marker_text}")
        print()

    def show_history(self) -> None:
        if not self.messages and not self.memory_items:
            self.print_dim("No messages in this session yet.")
            return

        print()
        print(self.bold("Session history:"))
        print(self.dim(f"Recent chat: {len(self.messages)}/{MAX_RECENT_MESSAGES}"))
        print(self.dim(f"Memory items: {len(self.memory_items)}/{MAX_RELEVANT_MEMORY_ITEMS}"))

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
        stream_status = "stream" if self.stream else "no-stream"
        return HTML(
            f"<ansiblue> @ {APP_NAME} </ansiblue> "
            f"<ansigray>■ {self.model} ■ {stream_status} ■ type / for commands</ansigray>"
        )

    def show_command_preview(self) -> None:
        print()
        print(self.blue("Slash commands:"))
        for command, description in COMMANDS.items():
            preview = command_preview(command, description)
            print(f"  {self.blue(command):<18} {self.dim(preview)}")
        print()

    def read_user_input(self) -> str:
        return self.session.prompt(
            HTML("<ansicyan>❯ </ansicyan>"),
            complete_while_typing=True,
        )

    def clear_screen(self) -> None:
        # Clear visible screen, clear scrollback buffer where supported, then move cursor home.
        print("\033[2J\033[3J\033[H", end="", flush=True)

    # -----------------------------
    # Text styling
    # -----------------------------

    def supports_colour(self) -> bool:
        return True

    def colour(self, text: str, code: str) -> str:
        if not self.supports_colour():
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self.colour(text, "1")

    def dim(self, text: str) -> str:
        return self.colour(text, "2")

    def blue(self, text: str) -> str:
        return self.colour(text, "96")

    def green(self, text: str) -> str:
        return self.colour(text, "92")

    def red(self, text: str) -> str:
        return self.colour(text, "91")

    def print_info(self, text: str) -> None:
        print(self.blue("[info]"), text)

    def print_error(self, text: str) -> None:
        print(self.red("[error]"), text)

    def print_dim(self, text: str) -> None:
        print(self.dim(text))

    def print_tui_markdown(self, text: str) -> None:
        lines = text.strip("\n").splitlines()
        index = 0

        while index < len(lines):
            line = lines[index]

            if self.is_markdown_table_start(lines, index):
                table_lines = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_lines.append(lines[index])
                    index += 1
                self.print_tui_table(table_lines)
                continue

            self.print_tui_line(line)
            index += 1


    def print_tui_line(self, line: str) -> None:
        stripped = line.strip()

        if not stripped:
            print()
            return

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            print(self.heading_colour(level, self.apply_inline_styles(title, allow_colour=False)))
            return

        if stripped in {"---", "***", "___"}:
            print(self.dim("─" * min(72, self.safe_terminal_width())))
            return

        if stripped.startswith(">"):
            quote_text = stripped.lstrip("> ").strip()
            print(self.colour("│ ", "90") + self.dim(self.apply_inline_styles(quote_text, allow_colour=False)))
            return

        bullet_match = re.match(r"^(\s*)([-*+] |\d+\.\s+)(.*)$", line)
        if bullet_match:
            indent, bullet, content = bullet_match.groups()
            marker = self.colour(bullet.strip(), "94")
            print(f"{indent}{marker} {self.apply_inline_styles(content)}")
            return

        print(self.apply_inline_styles(line))

    def is_markdown_table_start(self, lines: list[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False

        current = lines[index].strip()
        separator = lines[index + 1].strip()

        if not current.startswith("|") or not current.endswith("|"):
            return False

        return bool(re.match(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", separator))

    def print_tui_table(self, table_lines: list[str]) -> None:
        rows = [self.parse_table_row(line) for line in table_lines]

        if len(rows) < 2:
            for line in table_lines:
                self.print_tui_line(line)
            return

        header = rows[0]
        body = rows[2:]
        column_count = max(len(row) for row in rows if row)

        normalised_rows = [self.pad_row(header, column_count), *[self.pad_row(row, column_count) for row in body]]
        widths = self.calculate_table_column_widths(normalised_rows, column_count)

        top = "╭" + "┬".join("─" * (width + 2) for width in widths) + "╮"
        sep = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
        bottom = "╰" + "┴".join("─" * (width + 2) for width in widths) + "╯"

        print(self.dim(top))
        self.print_tui_table_row(normalised_rows[0], widths, is_header=True)
        print(self.dim(sep))
        for row in normalised_rows[1:]:
            self.print_tui_table_row(row, widths, is_header=False)
        print(self.dim(bottom))

    def calculate_table_column_widths(self, rows: list[list[str]], column_count: int) -> list[int]:
        max_width = self.safe_terminal_width()
        frame_width = 2 + column_count + 1
        padding_width = column_count * 2
        separator_width = max(0, column_count - 1)
        available_width = max_width - frame_width - padding_width - separator_width

        if column_count <= 0:
            return []

        min_width = 8
        available_width = max(column_count * min_width, available_width)

        natural_widths = [0] * column_count
        for row in rows:
            for col_index, cell in enumerate(row):
                plain = self.strip_ansi(self.apply_inline_styles(cell))
                longest_word = max((len(word) for word in re.findall(r"\S+", plain)), default=0)
                natural_widths[col_index] = max(natural_widths[col_index], min(max(len(plain), longest_word), 40))

        widths = [max(min_width, min(width, 40)) for width in natural_widths]

        while sum(widths) > available_width:
            widest = max(range(column_count), key=lambda index: widths[index])
            if widths[widest] <= min_width:
                break
            widths[widest] -= 1

        while sum(widths) < available_width:
            expandable = [index for index, width in enumerate(widths) if width < natural_widths[index]]
            if not expandable:
                break
            target = max(expandable, key=lambda index: natural_widths[index] - widths[index])
            widths[target] += 1

        return widths

    def parse_table_row(self, line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    def pad_row(self, row: list[str], column_count: int) -> list[str]:
        return [*row, *([""] * (column_count - len(row)))]

    def print_tui_table_row(self, row: list[str], widths: list[int], is_header: bool) -> None:
        wrapped_cells = [self.wrap_table_cell(cell, widths[col_index]) for col_index, cell in enumerate(row)]
        row_height = max((len(lines) for lines in wrapped_cells), default=1)

        for line_index in range(row_height):
            cells = []
            for col_index, lines in enumerate(wrapped_cells):
                width = widths[col_index]
                cell_line = lines[line_index] if line_index < len(lines) else ""
                styled = self.apply_inline_styles(cell_line)
                styled = self.table_column_colour(col_index, styled)
                if is_header:
                    styled = self.bold(styled)
                padding = " " * max(0, width - len(self.strip_ansi(styled)))
                cells.append(f" {styled}{padding} ")
            print(self.dim("│") + self.dim("│").join(cells) + self.dim("│"))

    def wrap_table_cell(self, cell: str, width: int) -> list[str]:
        plain = cell.strip()

        if not plain:
            return [""]

        wrapped = textwrap.wrap(
            plain,
            width=max(1, width),
            break_long_words=False,
            break_on_hyphens=False,
        )

        return wrapped or [""]

    def apply_inline_styles(self, text: str, allow_colour: bool = True) -> str:
        def bold_replacer(match):
            inner = match.group(1)
            if allow_colour:
                return self.bold(self.colour(inner, "94"))
            return self.bold(inner)

        return re.sub(r"\*\*(.+?)\*\*", bold_replacer, text)

    def heading_colour(self, level: int, text: str) -> str:
        codes = {
            1: "94;1",
            2: "94",
            3: "36",
            4: "96",
            5: "90",
            6: "2",
        }
        prefix = "▌" if level <= 2 else "•"
        return self.colour(f"{prefix} {text}", codes.get(level, "90"))

    def table_column_colour(self, index: int, text: str) -> str:
        pastel_codes = ["95", "96", "92", "93", "94", "91"]
        return self.colour(text, pastel_codes[index % len(pastel_codes)])

    def safe_terminal_width(self) -> int:
        try:
            return max(40, min(os.get_terminal_size().columns, 120))
        except OSError:
            return 88

    def clip_text(self, text: str, width: int) -> str:
        plain = text.strip()
        if len(plain) <= width:
            return plain
        if width <= 1:
            return plain[:width]
        return plain[: max(1, width - 1)] + "…"

    def strip_ansi(self, text: str) -> str:
        return re.sub(r"\033\[[0-9;]*m", "", text)


def main() -> None:
    app = SimpleAgentTUI()
    app.run()


if __name__ == "__main__":
    main()
