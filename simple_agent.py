from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
import skills
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

try:
    from PIL import Image, ImageGrab
except Exception:
    Image = None
    ImageGrab = None

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None


MODEL_LIBRARY = [
    {
        "key": "qwen3-4b-thinking-2507",
        "id": "lmstudio-community/Qwen3-4B-Thinking-2507-MLX-4bit",
        "category": "orchestrator",
        "runtime": "mlx-lm",
        "specialty": "Primary thinking model for coding, reasoning, routing, lightweight planning, and multi-step control flow.",
        "why_selected": "MLX-ready 4-bit Qwen3 thinking model for Apple Silicon, selected to replace the older Qwen2.5 text and coder models with a stronger under-6GB coding-focused reasoning model.",
        "download_url": "https://huggingface.co/lmstudio-community/Qwen3-4B-Thinking-2507-MLX-4bit/resolve/main/README.md",
    },
    {
        "key": "qwen2.5-vl-3b-instruct",
        "id": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        "category": "specialist",
        "runtime": "mlx-vlm",
        "specialty": "Specialised model for vision-language tasks.",
        "why_selected": "MLX-ready 4-bit multimodal model for Apple Silicon image and OCR-style tasks.",
        "download_url": "https://huggingface.co/mlx-community/Qwen2.5-VL-3B-Instruct-4bit/resolve/main/README.md",
    },
    {
        "key": "rag-all-minilm-l6-v2",
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "category": "rag",
        "runtime": "sentence-transformers",
        "specialty": "Lean embedding model for ranking retrieved text snippets for RAG-style search grounding.",
        "why_selected": "Small, fast, widely used sentence-transformer suitable for lightweight local relevance ranking.",
        "download_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    },
]


@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str


@dataclass
class AgentState:
    status: str = "Idle"
    logs: list[LogEntry] = field(default_factory=list)
    chats: list[dict[str, Any]] = field(default_factory=list)
    current_chat_id: str | None = None

    def add_log(self, level: str, message: str) -> None:
        self.logs.append(
            LogEntry(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                level=level,
                message=message,
            )
        )


class SimpleAgentGUI:
    def __init__(self) -> None:
        self.state = AgentState()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.chats_dir = Path("chats")
        self.chats_dir.mkdir(exist_ok=True)
        self.temp_attachments_dir = Path("temp_attachments")
        self.temp_attachments_dir.mkdir(exist_ok=True)
        self.active_prompt_model_key = "qwen3-4b-thinking-2507"
        self.python_executable = sys.executable or "python"
        self.debug = True
        self.max_memory_items = 5
        self.is_generating = False
        self.loading_animation_job: str | None = None
        self.loading_base = "Thinking"
        self.loading_frames = ["", ".", "..", "..."]
        self.loading_index = 0
        self.min_response_tokens = 256
        self.default_response_tokens = 32768
        self.max_response_tokens = 131072
        self.unlimited_response_tokens = 0
        self.selected_response_tokens = self.default_response_tokens
        self.last_response_token_budget = self.default_response_tokens
        self.last_thinking_text = ""
        self.generation_started_at: float | None = None
        self.last_response_seconds: float | None = None
        self.console_window: tk.Toplevel | None = None
        self.console_text: tk.Text | None = None
        self.console_backlog: list[str] = []
        self.pending_attachments: list[dict[str, str]] = []
        self.image_attachment_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        self.video_attachment_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        self.vl_supported_extensions = self.image_attachment_extensions | self.video_attachment_extensions
        self.text_attachment_extensions = {
            ".txt", ".md", ".markdown", ".rst", ".log",
            ".csv", ".tsv",
            ".json", ".jsonl",
            ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
            ".xml",
            ".tex", ".bib",
            ".gitignore", ".dockerignore",
            ".editorconfig", ".requirements", ".lock",
        }
        self.pdf_attachment_extensions = {".pdf"}
        self.code_attachment_extensions = {
            ".py", ".pyw", ".ipynb", ".sql",
            ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
            ".html", ".htm", ".css", ".scss", ".sass", ".less",
            ".sh", ".bash", ".zsh", ".fish",
            ".bat", ".ps1",
            ".java", ".kt", ".kts",
            ".c", ".h", ".cpp", ".hpp", ".cc", ".cs",
            ".go", ".rs", ".swift", ".php", ".rb", ".lua", ".r", ".m",
            ".scala", ".dart",
            ".vue", ".svelte", ".astro",
        }

        self.attachment_skill_map: dict[str, int] = {
            **{extension: 4 for extension in self.image_attachment_extensions},
            **{extension: 4 for extension in self.video_attachment_extensions},
            **{extension: 5 for extension in self.text_attachment_extensions},
            **{extension: 6 for extension in self.pdf_attachment_extensions},
            **{extension: 7 for extension in self.code_attachment_extensions},
        }

        self.attachment_handler_map: dict[int, str] = {
            4: "attachment_vision",
            5: "text_file_reader",
            6: "pdf_reader",
            7: "code_reader",
        }

        os.environ.setdefault("HF_HOME", str(self.models_dir / ".hf_cache"))

        self.state.add_log("INFO", "Simple Agent GUI initialised.")

        if TkinterDnD is not None:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
        self.root.title("Simple Agent")
        self.root.geometry("1280x820")
        self.root.minsize(960, 640)

        self._build_menu_bar()
        self._configure_styles()
        self._build_gui()
        self._load_chats()
        self._ensure_chat_exists()
        self._refresh_chat_list()
        self._open_current_chat()

    def run(self) -> None:
        self.root.mainloop()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.configure(bg="#1f1f1f")
        style.configure("Sidebar.TFrame", background="#171717")
        style.configure("Main.TFrame", background="#202123")
        style.configure("Header.TFrame", background="#202123")
        style.configure("Composer.TFrame", background="#202123")
        style.configure(
            "Sidebar.TButton",
            font=("Aptos", 13),
            padding=10,
        )
        style.configure(
            "Primary.TButton",
            font=("Aptos", 12, "bold"),
            padding=6,
        )
        style.configure(
            "Sidebar.TLabel",
            background="#171717",
            foreground="#f5f5f5",
            font=("Aptos", 12),
        )
        style.configure(
            "MainTitle.TLabel",
            background="#202123",
            foreground="#ffffff",
            font=("Aptos", 22, "bold"),
        )
        style.configure(
            "Meta.TLabel",
            background="#202123",
            foreground="#b8b8b8",
            font=("Aptos", 11),
        )
        style.configure(
            "Status.TLabel",
            background="#202123",
            foreground="#d7d7d7",
            font=("Aptos", 11),
        )

    def _build_menu_bar(self) -> None:
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Chat", command=self._create_new_chat)
        file_menu.add_command(label="Delete Current Chat", command=self._delete_current_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Download Models", command=self._download_text_models_async)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Temp Attachments", command=self._clear_temp_attachments)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.destroy)

        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Show Console Output", command=self._show_console_window)
        view_menu.add_command(label="Clear Console Output", command=self._clear_console_output)

        menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_cascade(label="View", menu=view_menu)
        self.root.config(menu=menu_bar)


    def _show_console_window(self) -> None:
        if self.console_window is not None and self.console_window.winfo_exists():
            self.console_window.deiconify()
            self.console_window.lift()
            return

        self.console_window = tk.Toplevel(self.root)
        self.console_window.title("Console Output")
        self.console_window.geometry("1050x650")
        self.console_window.configure(bg="#111111")
        self.console_window.protocol("WM_DELETE_WINDOW", self.console_window.withdraw)

        frame = ttk.Frame(self.console_window)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.console_text = tk.Text(
            frame,
            wrap="word",
            bg="#0f1115",
            fg="#e6edf3",
            insertbackground="#e6edf3",
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=12,
            font=("Courier", 11),
            state="disabled",
        )
        self.console_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.console_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.console_text.configure(yscrollcommand=scrollbar.set)

        if self.console_backlog:
            self._append_console("".join(self.console_backlog), store=False)
        else:
            self._append_console("Console output window opened.\n", store=False)

    def _clear_console_output(self) -> None:
        self.console_backlog.clear()
        if self.console_text is None:
            return
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", tk.END)
        self.console_text.configure(state="disabled")

    def _clear_temp_attachments(self) -> None:
        if self.is_generating:
            messagebox.showinfo(
                "Busy",
                "Please wait for the current response to finish before clearing temp attachments.",
            )
            return

        if not self.temp_attachments_dir.exists():
            self.temp_attachments_dir.mkdir(exist_ok=True)
            self._set_status("Temp attachments folder is already clean.")
            return

        files = [path for path in self.temp_attachments_dir.iterdir() if path.is_file()]
        if not files:
            self._set_status("Temp attachments folder is already clean.")
            return

        confirmed = messagebox.askyesno(
            "Clear Temp Attachments",
            f"Delete {len(files)} file(s) from temp_attachments? This cannot be undone.",
        )
        if not confirmed:
            return

        temp_root = self.temp_attachments_dir.resolve()
        deleted_count = 0
        failed_count = 0

        for path in files:
            try:
                path.unlink()
                deleted_count += 1
            except Exception:
                failed_count += 1

        self.pending_attachments = [
            attachment
            for attachment in self.pending_attachments
            if not str(attachment.get("path", "")).startswith(str(temp_root))
        ]
        self._render_attachment_bar()

        if failed_count:
            self._set_status(
                f"Deleted {deleted_count} temp attachment(s), failed to delete {failed_count}."
            )
        else:
            self._set_status(f"Deleted {deleted_count} temp attachment(s).")

    def _append_console(self, text: str, store: bool = True) -> None:
        if store:
            self.console_backlog.append(text)
            max_backlog_items = 300
            if len(self.console_backlog) > max_backlog_items:
                self.console_backlog = self.console_backlog[-max_backlog_items:]

        if self.console_text is None:
            return
        if self.console_window is not None and not self.console_window.winfo_exists():
            self.console_window = None
            self.console_text = None
            return

        self.console_text.configure(state="normal")
        self.console_text.insert(tk.END, text)
        self.console_text.see(tk.END)
        self.console_text.configure(state="disabled")

    def _print_debug(self, text: str) -> None:
        print(text)
        if hasattr(self, "root"):
            self.root.after(0, partial(self._append_console, text if text.endswith("\n") else text + "\n"))

    def _build_gui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=280)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.rowconfigure(3, weight=1)
        self.sidebar.columnconfigure(0, weight=1)

        self.main = ttk.Frame(self.root, style="Main.TFrame")
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(1, weight=1)

        self.new_chat_button = ttk.Button(
            self.sidebar,
            text="+ New chat",
            command=self._create_new_chat,
            style="Primary.TButton",
        )
        self.new_chat_button.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))


        self.sidebar_hint = ttk.Label(
            self.sidebar,
            text="Chats are saved locally in the chats folder.",
            style="Sidebar.TLabel",
            wraplength=240,
            justify="left",
        )
        self.sidebar_hint.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        self.chat_listbox = tk.Listbox(
            self.sidebar,
            bg="#111111",
            fg="#f0f0f0",
            selectbackground="#2f6fed",
            selectforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            activestyle="none",
            font=("Aptos", 12),
        )
        self.chat_listbox.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.chat_listbox.bind("<<ListboxSelect>>", self._on_chat_selected)
        self.chat_listbox.bind("<Delete>", self._delete_current_chat_from_event)
        self.chat_listbox.bind("<BackSpace>", self._delete_current_chat_from_event)

        self.header = ttk.Frame(self.main, style="Header.TFrame")
        self.header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 10))
        self.header.columnconfigure(0, weight=1)

        self.chat_title_label = ttk.Label(
            self.header,
            text="Simple Agent",
            style="MainTitle.TLabel",
        )
        self.chat_title_label.grid(row=0, column=0, sticky="w")

        self.chat_meta_label = ttk.Label(
            self.header,
            text="Memory-aware local chat",
            style="Meta.TLabel",
        )
        self.chat_meta_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.token_selector_frame = ttk.Frame(self.header, style="Header.TFrame")
        self.token_selector_frame.grid(row=0, column=1, rowspan=2, sticky="e")

        self.token_selector_label = ttk.Label(
            self.token_selector_frame,
            text="Response size",
            style="Meta.TLabel",
        )
        self.token_selector_label.grid(row=0, column=0, sticky="e")

        self.token_options = [
            ("Small (8192)", 8192),
            ("Medium (16384)", 16384),
            ("Large (32768)", 32768),
            ("Mega (65536)", 65536),
            ("Ultra (131072)", 131072),
            ("Unlimited", self.unlimited_response_tokens),
        ]
        self.token_option_map = {label: value for label, value in self.token_options}
        default_token_label = next(
            label for label, value in self.token_options if value == self.selected_response_tokens
        )
        self.token_selector_var = tk.StringVar(value=default_token_label)
        self.token_selector = ttk.Combobox(
            self.token_selector_frame,
            textvariable=self.token_selector_var,
            values=[label for label, _ in self.token_options],
            state="readonly",
            width=16,
        )
        self.token_selector.grid(row=1, column=0, sticky="e", pady=(4, 0))
        self.token_selector.bind("<<ComboboxSelected>>", self._on_token_selection_changed)

        self.transcript = tk.Text(
            self.main,
            wrap="word",
            bg="#202123",
            fg="#f7f7f8",
            insertbackground="#f7f7f8",
            relief="flat",
            borderwidth=0,
            padx=20,
            pady=20,
            font=("Aptos", 15),
            state="normal",
            spacing1=4,
            spacing2=3,
            spacing3=12,
        )
        self.transcript.grid(row=1, column=0, sticky="nsew", padx=20, pady=0)
        self.transcript.bind("<Key>", self._block_transcript_edit)
        self.transcript.bind("<<Paste>>", self._block_transcript_edit)
        self.transcript.bind("<<Cut>>", self._block_transcript_edit)
        self.transcript.bind("<BackSpace>", self._block_transcript_edit)
        self.transcript.bind("<Delete>", self._block_transcript_edit)
        self.transcript.tag_configure(
            "user_name",
            font=("Aptos", 18, "bold"),
            foreground="#8ab4ff",
            spacing1=10,
            spacing3=4,
        )
        self.transcript.tag_configure(
            "assistant_name",
            font=("Aptos", 18, "bold"),
            foreground="#7ee787",
            spacing1=10,
            spacing3=4,
        )
        self.transcript.tag_configure(
            "body",
            font=("Aptos", 15),
            foreground="#f7f7f8",
            lmargin1=0,
            lmargin2=0,
            spacing1=2,
            spacing3=2,
        )
        self.transcript.tag_configure(
            "heading1",
            font=("Aptos", 24, "bold"),
            foreground="#9fd8a5",
            spacing1=14,
            spacing3=8,
        )
        self.transcript.tag_configure(
            "heading2",
            font=("Aptos", 20, "bold"),
            foreground="#98cfa0",
            spacing1=12,
            spacing3=6,
        )
        self.transcript.tag_configure(
            "heading3",
            font=("Aptos", 17, "bold"),
            foreground="#90c69a",
            spacing1=10,
            spacing3=5,
        )
        self.transcript.tag_configure(
            "heading4",
            font=("Aptos", 15, "bold"),
            foreground="#88bd93",
            spacing1=8,
            spacing3=4,
        )
        self.transcript.tag_configure(
            "bullet",
            font=("Aptos", 15),
            foreground="#bfd8c3",
            lmargin1=18,
            lmargin2=38,
            spacing1=1,
            spacing3=1,
        )
        self.transcript.tag_configure(
            "numbered",
            font=("Aptos", 15),
            foreground="#bfd8c3",
            lmargin1=18,
            lmargin2=38,
            spacing1=1,
            spacing3=1,
        )
        self.transcript.tag_configure(
            "blockquote",
            font=("Aptos", 15, "italic"),
            foreground="#a9cdb0",
            lmargin1=22,
            lmargin2=34,
            spacing1=3,
            spacing3=3,
        )
        self.transcript.tag_configure(
            "code_block",
            font=("Courier", 11),
            foreground="#f3f3f3",
            background="#151618",
            lmargin1=18,
            lmargin2=18,
            rmargin=18,
            spacing1=8,
            spacing3=8,
        )
        self.transcript.tag_configure(
            "inline_code",
            font=("Courier", 11),
            foreground="#ffd580",
            background="#2b2c2f",
        )
        self.transcript.tag_configure(
            "link",
            foreground="#8ab4ff",
            underline=True,
        )
        self.transcript.tag_bind("link", "<Button-1>", self._open_link_from_transcript)
        self.transcript.tag_bind("link", "<Enter>", lambda event: self.transcript.config(cursor="hand2"))
        self.transcript.tag_bind("link", "<Leave>", lambda event: self.transcript.config(cursor=""))
        self.transcript.tag_configure(
            "bold",
            font=("Aptos", 15, "bold"),
            foreground="#9fd8a5",
        )
        self.transcript.tag_configure(
            "italic",
            font=("Aptos", 15, "italic"),
            foreground="#a9cdb0",
        )
        self.transcript.tag_configure(
            "bold_italic",
            font=("Aptos", 15, "bold", "italic"),
            foreground="#98cfa0",
        )
        self.transcript.tag_configure(
            "separator",
            spacing1=10,
            spacing3=16,
        )
        self.transcript.tag_configure(
            "horizontal_rule",
            font=("Aptos", 15),
            foreground="#5f6f64",
            spacing1=10,
            spacing3=12,
        )

        self.composer = ttk.Frame(self.main, style="Composer.TFrame")
        self.composer.grid(row=2, column=0, sticky="ew", padx=20, pady=(6, 12))
        self.composer.columnconfigure(0, weight=1)

        self.input_box = tk.Text(
            self.composer,
            height=2,
            wrap="word",
            bg="#2b2c2f",
            fg="#f7f7f8",
            insertbackground="#f7f7f8",
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=14,
            font=("Aptos", 15),
        )
        self.input_box.grid(row=0, column=0, sticky="ew")
        self.input_box.bind("<Control-Return>", self._send_from_shortcut)
        self.input_box.bind("<Command-Return>", self._send_from_shortcut)
        self.input_box.bind("<KeyRelease>", self._auto_resize_input_box)
        self.input_box.bind("<<Paste>>", self._handle_input_paste)
        self._setup_attachment_drop_target()
        self.root.bind("<Configure>", self._on_root_resized)
        self.root.after(0, self._auto_resize_input_box)

        self.attachment_bar = ttk.Frame(self.composer, style="Composer.TFrame")
        self.attachment_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.attachment_bar.columnconfigure(0, weight=1)

        self.attachment_list_frame = ttk.Frame(self.attachment_bar, style="Composer.TFrame")
        self.attachment_list_frame.grid(row=0, column=0, sticky="ew")

        self.attach_button = ttk.Button(
            self.attachment_bar,
            text="Attach",
            command=self._choose_attachment,
            style="Primary.TButton",
            width=8,
        )
        self.attach_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.button_row = ttk.Frame(self.composer, style="Composer.TFrame")
        self.button_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.button_row.columnconfigure(0, weight=1)

        self.shortcut_label = ttk.Label(
            self.button_row,
            text="Ctrl+Enter / Cmd+Enter to send",
            style="Status.TLabel",
        )
        self.shortcut_label.grid(row=0, column=0, sticky="w")

        self.send_button = ttk.Button(
            self.button_row,
            text="Send",
            command=self._send_message,
            style="Primary.TButton",
            width=8,
        )
        self.send_button.grid(row=0, column=1, sticky="e")

        self.status_label = ttk.Label(
            self.main,
            text="Ready",
            style="Status.TLabel",
        )
        self.status_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 16))
        self._render_attachment_bar()

    def _send_from_shortcut(self, event: tk.Event) -> str:
        self._send_message()
        return "break"

    def _setup_attachment_drop_target(self) -> None:
        if DND_FILES is None or not hasattr(self.input_box, "drop_target_register"):
            return
        try:
            self.input_box.drop_target_register(DND_FILES)
            self.input_box.dnd_bind("<<Drop>>", self._handle_file_drop)
        except Exception as exc:
            self._log("WARN", f"Drag and drop unavailable: {exc}")

    def _handle_file_drop(self, event: tk.Event) -> str:
        raw_data = getattr(event, "data", "")
        for file_path in self._parse_dropped_files(raw_data):
            self._add_attachment(file_path)
        return "break"

    def _parse_dropped_files(self, raw_data: str) -> list[str]:
        if not raw_data:
            return []
        try:
            return [path for path in self.root.tk.splitlist(raw_data) if path]
        except Exception:
            return [raw_data.strip()] if raw_data.strip() else []

    def _choose_attachment(self) -> None:
        paths = filedialog.askopenfilenames(title="Choose attachments")
        for file_path in paths:
            self._add_attachment(file_path)

    def _attachment_extension(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix:
            return suffix
        return path.name.lower()

    def _skill_id_for_attachment_extension(self, extension: str) -> int | None:
        return self.attachment_skill_map.get(extension.lower())

    def _handler_for_attachment_skill(self, skill_id: int | None) -> str:
        if skill_id is None:
            return "pending"
        return self.attachment_handler_map.get(skill_id, "pending")

    def _add_attachment(self, file_path: str) -> None:
        path = Path(file_path).expanduser()
        if not path.exists() or not path.is_file():
            self._set_status(f"Attachment not found: {file_path}")
            return

        resolved_path = str(path.resolve())
        if any(item.get("path") == resolved_path for item in self.pending_attachments):
            self._set_status(f"Attachment already added: {path.name}")
            return

        extension = self._attachment_extension(path)
        skill_id = self._skill_id_for_attachment_extension(extension)
        handler = self._handler_for_attachment_skill(skill_id)

        self.pending_attachments.append(
            {
                "path": resolved_path,
                "name": path.name,
                "extension": extension,
                "handler": handler,
                "skill_id": skill_id,
                "pinned": False,
            }
        )

        self._render_attachment_bar()

        if skill_id == 4:
            self._set_status(f"Attached visual file for VL analysis: {path.name}")
        elif skill_id == 5:
            self._set_status(f"Attached text file for reading: {path.name}")
        elif skill_id == 6:
            self._set_status(f"Attached PDF file for reading: {path.name}")
        elif skill_id == 7:
            self._set_status(f"Attached code file for coding analysis: {path.name}")
        else:
            self._set_status(f"Attached file path only: {path.name}")

    def _remove_attachment_at(self, index: int) -> None:
        if 0 <= index < len(self.pending_attachments):
            removed = self.pending_attachments.pop(index)
            self._render_attachment_bar()
            self._set_status(f"Removed attachment: {removed.get('name', 'file')}")

    def _toggle_attachment_pin_at(self, index: int) -> None:
        if 0 <= index < len(self.pending_attachments):
            attachment = self.pending_attachments[index]
            attachment["pinned"] = not bool(attachment.get("pinned", False))
            self._render_attachment_bar()
            state = "Pinned" if attachment.get("pinned") else "Unpinned"
            self._set_status(f"{state} attachment: {attachment.get('name', 'file')}")

    def _render_attachment_bar(self) -> None:
        for child in self.attachment_list_frame.winfo_children():
            child.destroy()

        if not self.pending_attachments:
            hint = ttk.Label(
                self.attachment_list_frame,
                text="Drag files here, paste an image, or use Attach.",
                style="Meta.TLabel",
            )
            hint.grid(row=0, column=0, sticky="w")
            return

        for index, attachment in enumerate(self.pending_attachments):
            chip = ttk.Frame(self.attachment_list_frame, style="Composer.TFrame")
            chip.grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 8), pady=(0, 4))

            skill_id = attachment.get("skill_id")
            if skill_id == 4:
                handler_label = "vision"
            elif skill_id == 5:
                handler_label = "text"
            elif skill_id == 6:
                handler_label = "pdf"
            elif skill_id == 7:
                handler_label = "code"
            else:
                handler_label = "path only"
            pinned = bool(attachment.get("pinned", False))
            pin_marker = " pinned" if pinned else ""
            label = ttk.Label(
                chip,
                text=f"{attachment.get('name', 'file')} ({handler_label}{pin_marker})",
                style="Meta.TLabel",
            )
            label.grid(row=0, column=0, sticky="w")

            pin_button = ttk.Button(
                chip,
                text="Unpin" if pinned else "Pin",
                width=6,
                command=partial(self._toggle_attachment_pin_at, index),
            )
            pin_button.grid(row=0, column=1, sticky="w", padx=(4, 0))

            remove_button = ttk.Button(
                chip,
                text="X",
                width=3,
                command=partial(self._remove_attachment_at, index),
            )
            remove_button.grid(row=0, column=2, sticky="w", padx=(4, 0))

    def _handle_input_paste(self, event: tk.Event | None = None) -> str | None:
        if ImageGrab is None or Image is None:
            return None

        try:
            clipboard_content = ImageGrab.grabclipboard()
        except Exception:
            return None

        if clipboard_content is None:
            return None

        if isinstance(clipboard_content, Image.Image):
            saved_path = self._save_clipboard_image(clipboard_content)
            self._add_attachment(str(saved_path))
            return "break"

        if isinstance(clipboard_content, list):
            handled_any = False
            for item in clipboard_content:
                item_path = Path(str(item))
                if item_path.exists() and item_path.is_file():
                    self._add_attachment(str(item_path))
                    handled_any = True
            return "break" if handled_any else None

        return None

    def _save_clipboard_image(self, image: Any) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_path = self.temp_attachments_dir / f"pasted_image_{timestamp}.png"
        image.save(target_path, format="PNG")
        return target_path

    def _block_transcript_edit(self, event: tk.Event | None = None) -> str:
        if event is not None:
            if (event.state & 0x4) and event.keysym.lower() in {"c", "a"}:
                return None
            if (event.state & 0x8) and event.keysym.lower() in {"c", "a"}:
                return None
            if event.keysym in {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}:
                return None
        return "break"

    def _open_link_from_transcript(self, event: tk.Event) -> str:
        index = self.transcript.index(f"@{event.x},{event.y}")
        tags = self.transcript.tag_names(index)
        for tag in tags:
            if tag.startswith("url:"):
                url = tag[4:]
                if url:
                    webbrowser.open(url)
                return "break"
        return "break"

    def _on_root_resized(self, event: tk.Event | None = None) -> None:
        if event is not None and event.widget is not self.root:
            return
        self._auto_resize_input_box()

    def _auto_resize_input_box(self, event: tk.Event | None = None) -> None:
        self.input_box.update_idletasks()

        min_visible_lines = 1
        line_height = max(1, tkfont.Font(font=self.input_box.cget("font")).metrics("linespace"))
        vertical_padding = 28

        display_line_count = None
        try:
            count_result = self.input_box.count("1.0", "end-1c", "displaylines")
            if count_result:
                display_line_count = int(count_result[0])
        except Exception:
            display_line_count = None

        if display_line_count is None or display_line_count <= 0:
            display_line_count = 0
            index = "1.0"
            while True:
                dline = self.input_box.dlineinfo(index)
                if dline is None:
                    break
                display_line_count += 1
                index = self.input_box.index(f"{index}+1displaylines")

        if display_line_count <= 0:
            logical_line_count = int(self.input_box.index("end-1c").split(".")[0])
            display_line_count = max(min_visible_lines, logical_line_count)

        desired_height_pixels = max(
            line_height * max(min_visible_lines, display_line_count) + vertical_padding,
            56,
        )
        max_height_pixels = max(80, self.root.winfo_height() // 2)
        final_height_pixels = min(desired_height_pixels, max_height_pixels)

        text_units = max(min_visible_lines, round(final_height_pixels / line_height))
        self.input_box.configure(height=text_units)

    def _log(self, level: str, message: str) -> None:
        self.state.add_log(level, message)
        log_line = f"[{self.state.logs[-1].timestamp}] {level}: {message}"
        if self.debug:
            self._print_debug(log_line)
        elif hasattr(self, "root"):
            self.root.after(0, partial(self._append_console, log_line + "\n"))

    def _set_status(self, message: str) -> None:
        self.state.status = message
        self.status_label.config(text=message)
        self._log("INFO", message)

    def _start_loading_animation(self) -> None:
        self.loading_index = 0
        self.send_button.config(state="disabled")
        self._animate_loading()

    def _set_loading_base(self, text: str) -> None:
        self.loading_base = text
        self.loading_index = 0

    def _animate_loading(self) -> None:
        if not self.is_generating:
            self.loading_animation_job = None
            return

        suffix = self.loading_frames[self.loading_index % len(self.loading_frames)]
        self.loading_index += 1
        display = f"{self.loading_base}{suffix}"
        self.send_button.config(text=display)
        self.status_label.config(text=f"{display} with {self.active_prompt_model_key}")
        self.loading_animation_job = self.root.after(350, self._animate_loading)

    def _stop_loading_animation(self) -> None:
        if self.loading_animation_job is not None:
            self.root.after_cancel(self.loading_animation_job)
            self.loading_animation_job = None
        self.send_button.config(text="Send")

    def _load_chats(self) -> None:
        self.state.chats = []
        for path in sorted(self.chats_dir.glob("*.txt")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            data.setdefault("id", path.stem)
            data.setdefault("title", "New Chat")
            data.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
            data.setdefault("updated_at", data["created_at"])
            data.setdefault("messages", [])
            data.setdefault("memory", [])
            data["file_path"] = str(path)
            self.state.chats.append(data)

        self.state.chats.sort(
            key=lambda chat: chat.get("updated_at", ""),
            reverse=True,
        )

    def _ensure_chat_exists(self) -> None:
        if self.state.chats:
            self.state.current_chat_id = self.state.chats[0]["id"]
            return
        chat = self._build_new_chat_payload()
        self._save_chat(chat)
        self.state.chats.append(chat)
        self.state.current_chat_id = chat["id"]

    def _build_new_chat_payload(self) -> dict[str, Any]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        now = datetime.now().isoformat(timespec="seconds")
        return {
            "id": stamp,
            "title": "New Chat",
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "memory": [],
        }

    def _slugify_title(self, title: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
        return slug or "chat"

    def _chat_filename(self, chat: dict[str, Any]) -> str:
        return f"{chat['id']}__{self._slugify_title(chat['title'])}.txt"

    def _save_chat(self, chat: dict[str, Any]) -> None:
        chat["updated_at"] = datetime.now().isoformat(timespec="seconds")
        current_path_value = chat.get("file_path")
        current_path = Path(current_path_value) if current_path_value else None
        target_path = self.chats_dir / self._chat_filename(chat)

        if current_path and current_path.exists() and current_path != target_path:
            current_path.rename(target_path)

        chat["file_path"] = str(target_path)
        target_path.write_text(json.dumps(chat, indent=2, ensure_ascii=False), encoding="utf-8")

    def _refresh_chat_list(self) -> None:
        self.state.chats.sort(
            key=lambda chat: chat.get("updated_at", ""),
            reverse=True,
        )

        self.chat_listbox.delete(0, tk.END)
        for chat in self.state.chats:
            title = chat.get("title", "New Chat")
            self.chat_listbox.insert(tk.END, title[:90])

        current_index = self._current_chat_index()
        if current_index is not None:
            self.chat_listbox.selection_clear(0, tk.END)
            self.chat_listbox.selection_set(current_index)
            self.chat_listbox.activate(current_index)

    def _chat_preview(self, chat: dict[str, Any]) -> str:
        messages = chat.get("messages", [])
        if not messages:
            return "Empty"
        preview = messages[-1].get("content", "").strip().replace("\n", " ")
        return preview[:35]

    def _current_chat_index(self) -> int | None:
        if not self.state.current_chat_id:
            return None
        for index, chat in enumerate(self.state.chats):
            if chat["id"] == self.state.current_chat_id:
                return index
        return None

    def _on_chat_selected(self, event: tk.Event | None = None) -> None:
        selection = self.chat_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if index >= len(self.state.chats):
            return
        self.state.current_chat_id = self.state.chats[index]["id"]
        self._open_current_chat()

    def _open_current_chat(self) -> None:
        chat = self._get_current_chat()
        if chat is None:
            return

        self.chat_title_label.config(text=chat.get("title", "New Chat"))
        memory_count = len(chat.get("memory", []))
        message_count = len(chat.get("messages", []))
        self.chat_meta_label.config(
            text=(
                f"{message_count} messages • "
                f"{memory_count}/{self.max_memory_items} short-term memory items • "
                f"Model: {self.active_prompt_model_key} • "
                f"Tokens: {self._response_token_label()}"
            )
        )
        self._render_transcript(chat)

    def _response_token_label(self) -> str:
        if self.selected_response_tokens == self.unlimited_response_tokens:
            return "Unlimited"
        return str(self.selected_response_tokens)

    def _format_token_budget(self, token_budget: int) -> str:
        if token_budget == self.unlimited_response_tokens:
            return "Unlimited"
        return str(token_budget)

    def _format_response_time(self, seconds: float | None) -> str:
        if seconds is None:
            return "Unknown"
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

    def _on_token_selection_changed(self, event: tk.Event | None = None) -> None:
        selected_label = self.token_selector_var.get().strip()
        selected_value = self.token_option_map.get(selected_label, self.default_response_tokens)
        self.selected_response_tokens = self._clamp_response_tokens(selected_value)
        self._open_current_chat()
        self._set_status(f"Response size set to {self._response_token_label()} tokens")

    def _render_transcript(self, chat: dict[str, Any]) -> None:
        self.transcript.delete("1.0", tk.END)

        messages = chat.get("messages", [])
        if not messages:
            self.transcript.insert(
                tk.END,
                "Start a new conversation. Each chat keeps its own local memory and is saved to the chats folder.\n",
                "body",
            )
        else:
            for message in messages:
                role = message.get("role", "assistant")
                name = "You" if role == "user" else "SimpleAgent"
                tag = "user_name" if role == "user" else "assistant_name"
                self.transcript.insert(tk.END, f"{name}\n", tag)
                self._insert_formatted_message(message.get("content", "").strip())
                self._insert_message_attachments(message.get("attachments", []))
                self.transcript.insert(tk.END, "\n", "separator")

        self.transcript.see(tk.END)

    def _insert_message_attachments(self, attachments: list[dict[str, str]]) -> None:
        if not attachments:
            return

        self.transcript.insert(tk.END, "Attachments:\n", "heading4")
        for attachment in attachments:
            path = attachment.get("path", "")
            name = attachment.get("name", Path(path).name if path else "file")
            skill_id = attachment.get("skill_id")
            if skill_id == 4:
                handler_label = "vision"
            elif skill_id == 5:
                handler_label = "text"
            elif skill_id == 6:
                handler_label = "pdf"
            elif skill_id == 7:
                handler_label = "code"
            else:
                handler_label = "path only"

            pinned = bool(attachment.get("pinned", False))
            pin_marker = ", pinned" if pinned else ""
            self.transcript.insert(tk.END, f"• {name} ({handler_label}{pin_marker}) - ", "bullet")
            if path:
                self._insert_clickable_link(path, f"file://{path}", "bullet")
            self.transcript.insert(tk.END, "\n")

    def _insert_formatted_message(self, content: str) -> None:
        if not content:
            self.transcript.insert(tk.END, "\n", "body")
            return

        lines = content.splitlines()
        index = 0
        in_code_block = False
        code_lines: list[str] = []

        while index < len(lines):
            raw_line = lines[index]
            # Markdown table detection (only outside code block)
            if not in_code_block:
                table_result = self._collect_markdown_table(lines, index)
                if table_result is not None:
                    table_rows, next_index = table_result
                    self._insert_table_widget(table_rows)
                    index = next_index
                    continue
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code_block:
                    code_text = "\n".join(code_lines).rstrip()
                    if code_text:
                        self.transcript.insert(tk.END, f"{code_text}\n", "code_block")
                    else:
                        self.transcript.insert(tk.END, "\n", "code_block")
                    code_lines = []
                    in_code_block = False
                else:
                    in_code_block = True
                    code_lines = []
                index += 1
                continue

            if in_code_block:
                code_lines.append(line)
                index += 1
                continue

            if not stripped:
                self.transcript.insert(tk.END, "\n", "body")
                index += 1
                continue

            if self._is_markdown_horizontal_rule(stripped):
                self._insert_horizontal_rule()
                index += 1
                continue

            if stripped.startswith("#### "):
                self._insert_heading_line(stripped[5:].strip(), "heading4")
                index += 1
                continue

            if stripped.startswith("### "):
                self._insert_heading_line(stripped[4:].strip(), "heading3")
                index += 1
                continue

            if stripped.startswith("## "):
                self._insert_heading_line(stripped[3:].strip(), "heading2")
                index += 1
                continue

            if stripped.startswith("# "):
                self._insert_heading_line(stripped[2:].strip(), "heading1")
                index += 1
                continue

            if re.match(r"^[-*•]\s+", stripped):
                bullet_text = re.sub(r"^[-*•]\s+", "• ", stripped)
                self._insert_inline_formatted_line(bullet_text, "bullet")
                self.transcript.insert(tk.END, "\n")
                index += 1
                continue

            if re.match(r"^\d+\.\s+", stripped):
                self._insert_inline_formatted_line(stripped, "numbered")
                self.transcript.insert(tk.END, "\n")
                index += 1
                continue

            if stripped.startswith(">"):
                quote_text = stripped[1:].strip()
                self._insert_inline_formatted_line(quote_text, "blockquote")
                self.transcript.insert(tk.END, "\n")
                index += 1
                continue

            if stripped.endswith(":") and len(stripped) <= 80 and stripped == line.strip():
                self.transcript.insert(tk.END, f"{stripped}\n", "heading4")
                index += 1
                continue

            self._insert_inline_formatted_line(line.strip(), "body")
            self.transcript.insert(tk.END, "\n")
            index += 1

        if in_code_block:
            code_text = "\n".join(code_lines).rstrip()
            if code_text:
                self.transcript.insert(tk.END, f"{code_text}\n", "code_block")


    def _is_markdown_horizontal_rule(self, stripped_line: str) -> bool:
        compact = stripped_line.replace(" ", "")
        return bool(re.fullmatch(r"(-{3,}|_{3,}|\*{3,})", compact))

    def _insert_horizontal_rule(self) -> None:
        self.transcript.insert(tk.END, "\n", "body")
        self.transcript.insert(tk.END, "─" * 72 + "\n", "horizontal_rule")
        self.transcript.insert(tk.END, "\n", "body")

    def _insert_heading_line(self, text: str, heading_tag: str) -> None:
        cleaned_text = self._normalize_inline_markdown(text)
        self.transcript.insert(tk.END, f"{cleaned_text}\n", heading_tag)

    def _collect_markdown_table(self, lines: list[str], start_index: int) -> tuple[list[list[str]], int] | None:
        if start_index + 1 >= len(lines):
            return None

        header_line = lines[start_index].strip()
        separator_line = lines[start_index + 1].strip()

        if "|" not in header_line or "|" not in separator_line:
            return None
        if not self._is_markdown_table_separator(separator_line):
            return None

        table_lines: list[str] = [lines[start_index], lines[start_index + 1]]
        next_index = start_index + 2

        while next_index < len(lines):
            candidate = lines[next_index].strip()
            if not candidate or "|" not in candidate:
                break
            table_lines.append(lines[next_index])
            next_index += 1

        rows = [
            [self._normalize_inline_markdown(cell) for cell in self._split_markdown_table_row(line)]
            for line in table_lines
        ]
        if len(rows) < 2:
            return None

        header_row = rows[0]
        data_rows = rows[2:]
        column_count = max(len(header_row), *(len(row) for row in data_rows)) if data_rows else len(header_row)
        if column_count <= 0:
            return None

        normalized_rows: list[list[str]] = []
        normalized_rows.append(header_row + [""] * (column_count - len(header_row)))
        for row in data_rows:
            normalized_rows.append(row + [""] * (column_count - len(row)))

        return normalized_rows, next_index

    def _is_markdown_table_separator(self, line: str) -> bool:
        compact = line.replace(" ", "")
        if not compact:
            return False
        return all(char in "|-:" for char in compact) and "-" in compact

    def _split_markdown_table_row(self, line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]



    def _insert_table_widget(self, rows: list[list[str]]) -> None:
        if not rows:
            return

        table_frame = tk.Frame(
            self.transcript,
            bg="#202123",
            bd=0,
            highlightthickness=0,
            padx=18,
            pady=10,
        )

        header_bg = "#243126"
        cell_bg = "#202123"
        border_color = "#314235"
        header_fg = "#d8f0dc"
        cell_fg = "#e7eee8"

        column_count = max(len(row) for row in rows)
        for column_index in range(column_count):
            table_frame.grid_columnconfigure(column_index, weight=1, uniform="table")

        for row_index, row in enumerate(rows):
            is_header = row_index == 0
            for column_index, cell_text in enumerate(row):
                cell = tk.Label(
                    table_frame,
                    text=cell_text,
                    justify="left",
                    anchor="w",
                    bg=header_bg if is_header else cell_bg,
                    fg=header_fg if is_header else cell_fg,
                    font=("Aptos", 13, "bold") if is_header else ("Aptos", 13),
                    wraplength=320,
                    padx=10,
                    pady=8,
                    bd=1,
                    relief="solid",
                    highlightthickness=0,
                )
                cell.grid(row=row_index, column=column_index, sticky="nsew", padx=1, pady=1)
                cell.configure(highlightbackground=border_color)

        self.transcript.insert(tk.END, "\n")
        self.transcript.window_create(tk.END, window=table_frame, padx=0, pady=4)
        self.transcript.insert(tk.END, "\n")

    def _normalize_inline_markdown(self, text: str) -> str:
        if not text:
            return ""
        normalized = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        normalized = re.sub(r"\*(.*?)\*", r"\1", normalized)
        normalized = re.sub(r"`(.*?)`", r"\1", normalized)
        return normalized.strip()

    def _insert_inline_formatted_line(self, text: str, base_tag: str) -> None:
        if not text:
            self.transcript.insert(tk.END, "", base_tag)
            return

        pattern = r"(\[[^\]]+\]\(https?://[^\s)]+\)|https?://[^\s)]+|\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)"
        last_index = 0

        for match in re.finditer(pattern, text):
            start, end = match.span()
            if start > last_index:
                self.transcript.insert(tk.END, text[last_index:start], base_tag)

            token = match.group(0)
            markdown_link_match = re.match(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", token)
            if markdown_link_match:
                label = markdown_link_match.group(1).strip()
                url = markdown_link_match.group(2).strip()
                self._insert_clickable_link(label or url, url, base_tag)
            elif re.match(r"https?://", token):
                clean_url = token.rstrip(".,;:!?")
                trailing = token[len(clean_url):]
                self._insert_clickable_link(clean_url, clean_url, base_tag)
                if trailing:
                    self.transcript.insert(tk.END, trailing, base_tag)
            elif token.startswith("`") and token.endswith("`"):
                inline_code_text = token[1:-1]
                self.transcript.insert(tk.END, inline_code_text, (base_tag, "inline_code"))
            elif token.startswith("***") and token.endswith("***") and len(token) >= 6:
                styled_text = token[3:-3]
                self.transcript.insert(tk.END, styled_text, (base_tag, "bold_italic"))
            elif token.startswith("**") and token.endswith("**") and len(token) >= 4:
                styled_text = token[2:-2]
                self.transcript.insert(tk.END, styled_text, (base_tag, "bold"))
            elif token.startswith("*") and token.endswith("*") and len(token) >= 2:
                styled_text = token[1:-1]
                self.transcript.insert(tk.END, styled_text, (base_tag, "italic"))
            else:
                self.transcript.insert(tk.END, token, base_tag)

            last_index = end

        if last_index < len(text):
            self.transcript.insert(tk.END, text[last_index:], base_tag)

    def _insert_clickable_link(self, label: str, url: str, base_tag: str) -> None:
        safe_url = url.strip()
        if not safe_url:
            return
        url_tag = f"url:{safe_url}"
        self.transcript.tag_configure(url_tag)
        self.transcript.insert(tk.END, label, (base_tag, "link", url_tag))

    def _create_new_chat(self) -> None:
        if self.is_generating:
            messagebox.showinfo("Busy", "Please wait for the current response to finish.")
            return

        chat = self._build_new_chat_payload()
        self._save_chat(chat)
        self.state.chats.insert(0, chat)
        self.state.current_chat_id = chat["id"]
        self._refresh_chat_list()
        self._open_current_chat()
        self.input_box.focus_set()
        self._set_status("Created a new chat.")

    def _delete_current_chat_from_event(self, event: tk.Event | None = None) -> str:
        self._delete_current_chat()
        return "break"

    def _delete_current_chat(self) -> None:
        if self.is_generating:
            messagebox.showinfo("Busy", "Please wait for the current response to finish.")
            return

        chat = self._get_current_chat()
        if chat is None:
            return

        title = chat.get("title", "New Chat")
        confirmed = messagebox.askyesno(
            "Delete Chat",
            f"Delete chat '{title}'? This cannot be undone.",
        )
        if not confirmed:
            return

        file_path_value = chat.get("file_path")
        if file_path_value:
            file_path = Path(file_path_value)
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as exc:
                    messagebox.showerror("Delete Failed", f"Could not delete the chat file.\n\n{exc}")
                    return

        self.state.chats = [c for c in self.state.chats if c.get("id") != chat.get("id")]

        if not self.state.chats:
            new_chat = self._build_new_chat_payload()
            self._save_chat(new_chat)
            self.state.chats.append(new_chat)
            self.state.current_chat_id = new_chat["id"]
        else:
            self.state.current_chat_id = self.state.chats[0]["id"]

        self._refresh_chat_list()
        self._open_current_chat()
        self.input_box.focus_set()
        self._set_status(f"Deleted chat: {title}")

    def _get_current_chat(self) -> dict[str, Any] | None:
        for chat in self.state.chats:
            if chat["id"] == self.state.current_chat_id:
                return chat
        return None

    def _send_message(self) -> None:
        if self.is_generating:
            return

        chat = self._get_current_chat()
        if chat is None:
            return

        prompt = self.input_box.get("1.0", tk.END).strip()
        if not prompt:
            return

        self.input_box.delete("1.0", tk.END)
        self._auto_resize_input_box()
        attachments_snapshot = [dict(item) for item in self.pending_attachments]
        chat.setdefault("messages", []).append(
            {
                "role": "user",
                "content": prompt,
                "attachments": attachments_snapshot,
            }
        )
        self.pending_attachments = [
            attachment for attachment in self.pending_attachments
            if bool(attachment.get("pinned", False))
        ]
        self._render_attachment_bar()
        self._save_chat(chat)
        self._refresh_chat_list()
        self._open_current_chat()

        self.is_generating = True
        self.generation_started_at = time.perf_counter()
        self.last_response_seconds = None
        self.new_chat_button.config(state="disabled")
        self._start_loading_animation()

        memory_snapshot = list(chat.get("memory", []))
        chat_id = chat["id"]
        thread = threading.Thread(
            target=self._generate_response_worker,
            args=(chat_id, prompt, memory_snapshot, attachments_snapshot),
            daemon=True,
        )
        thread.start()

    def _generate_response_worker(
        self,
        chat_id: str,
        prompt: str,
        memory_snapshot: list[dict[str, str]],
        attachments_snapshot: list[dict[str, str]],
    ) -> None:
        try:
            response = self._run_prompt(prompt, memory_snapshot, attachments_snapshot)
            title = ""
            chat = self._chat_by_id(chat_id)
            if chat is not None and self._should_generate_title(chat):
                title = self._generate_chat_title(prompt)

            self.root.after(
                0,
                partial(self._complete_response, chat_id, prompt, response, title, None),
            )
        except Exception as exc:
            error_message = str(exc)
            self.root.after(
                0,
                partial(self._complete_response, chat_id, prompt, "", "", error_message),
            )

    def _complete_response(
        self,
        chat_id: str,
        prompt: str,
        response: str,
        title: str,
        error: str | None,
    ) -> None:
        self.is_generating = False
        self._stop_loading_animation()
        self.send_button.config(state="normal", text="Send")
        self.new_chat_button.config(state="normal")
        if self.generation_started_at is not None:
            self.last_response_seconds = time.perf_counter() - self.generation_started_at
            self.generation_started_at = None

        chat = self._chat_by_id(chat_id)
        if chat is None:
            self._set_status("The active chat could not be found.")
            return

        if error:
            chat.setdefault("messages", []).append(
                {"role": "assistant", "content": f"Error: {error}"}
            )
            self._save_chat(chat)
            self._refresh_chat_list()
            self._open_current_chat()
            self._set_status(
                f"Generation failed. Last token budget: {self._format_token_budget(self.last_response_token_budget)} • "
                f"Time: {self._format_response_time(self.last_response_seconds)}"
            )
            return

        chat.setdefault("messages", []).append({"role": "assistant", "content": response})
        self._remember_turn(chat, prompt, response)

        if self._should_generate_title(chat) and title:
            chat["title"] = title

        self._save_chat(chat)
        self._refresh_chat_list()
        if self.state.current_chat_id == chat_id:
            self._open_current_chat()
        self._set_status(
            f"Response generated. Tokens used allowance: {self._format_token_budget(self.last_response_token_budget)} • "
            f"Time: {self._format_response_time(self.last_response_seconds)}"
        )

    def _chat_by_id(self, chat_id: str) -> dict[str, Any] | None:
        for chat in self.state.chats:
            if chat["id"] == chat_id:
                return chat
        return None

    def _should_generate_title(self, chat: dict[str, Any]) -> bool:
        title = chat.get("title", "").strip().lower()
        user_messages = [m for m in chat.get("messages", []) if m.get("role") == "user"]
        return title in {"", "new chat"} and len(user_messages) == 1

    def _build_agent_loop_plan(
            self,
            prompt: str,
            skill_ids: list[int],
            attachments: list[dict[str, str]],
    ) -> str:
        if not skill_ids and not attachments:
            return ""

        skill_names = {
            0: "no_skill",
            1: "internet_search",
            2: "scrape_url",
            3: "memory_rag",
            4: "attachment_vision",
            5: "text_file_reader",
            6: "pdf_reader",
            7: "code_reader",
        }

        selected_skills = [skill_names.get(skill_id, f"unknown_{skill_id}") for skill_id in skill_ids]
        attachment_names = [attachment.get("name", "file") for attachment in attachments]

        lines = [
            "Agent loop plan:",
            "- Think: identify the user's actual goal and the evidence needed to answer it.",
        ]

        if selected_skills:
            lines.append(f"- Act: run selected skills: {', '.join(selected_skills)}.")

        if attachment_names:
            lines.append(f"- Act: inspect attached files: {', '.join(attachment_names)}.")

        lines.extend(
            [
                "- Observe: extract only information that is relevant to the user's prompt.",
                "- Respond: give the final answer directly; do not narrate this loop unless the user asks how the answer was produced.",
            ]
        )

        return "\n".join(lines)

    def _build_agent_observation_summary(
            self,
            prompt: str,
            skill_context: str,
            attachment_context: str,
    ) -> str:
        observation_source_parts: list[str] = []
        if skill_context:
            observation_source_parts.append(skill_context)
        if attachment_context and self._attachment_context_should_be_observed(attachment_context):
            observation_source_parts.append(attachment_context)
        if not observation_source_parts:
            return ""

        model = self._resolve_model(self.active_prompt_model_key)
        if model is None or model["runtime"] != "mlx-lm":
            return ""

        model_dir = self.models_dir / model["key"]
        local_model_path = model_dir / "model"
        if not local_model_path.exists():
            return ""

        observation_source_text = "\n\n".join(observation_source_parts)
        observation_prompt = (
            "You are the observation step of a lightweight agent loop.\n"
            "The agent already acted by running tools. Your job is to inspect the tool output and produce a compact observation summary for the final answer.\n\n"
            "Rules:\n"
            "- Focus only on information relevant to the original user prompt.\n"
            "- Preserve source URLs, dates, names, numbers, rankings, and concrete facts.\n"
            "- Prefer reputable or primary sources when the tool output includes source quality information.\n"
            "- Flag weak, suspicious, incomplete, or undated sources instead of treating them as verified.\n"
            "- If the tool output is insufficient, say exactly what is missing.\n"
            "- Do not answer the user fully. Return only observations that will help the final response.\n"
            "- For resumes, documents, or code reviews, extract strengths, weaknesses, missing details, and specific improvement opportunities.\n"
            "- Use concise bullets or a compact table when useful.\n\n"
            f"Original user prompt:\n{prompt}\n\n"
            "Context output to observe:\n"
            f"{observation_source_text}\n"
        )
    def _build_code_conversation_context(self, current_prompt: str, max_messages: int = 8) -> str:
        current_chat = self._get_current_chat()
        if current_chat is None:
            return ""

        messages = current_chat.get("messages", [])
        if not messages:
            return ""

        code_related_messages: list[dict[str, Any]] = []
        for message in reversed(messages):
            role = message.get("role", "")
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "user" and content == current_prompt.strip():
                continue
            if self._message_is_code_related(message):
                code_related_messages.append(message)
            if len(code_related_messages) >= max_messages:
                break

        if not code_related_messages:
            return ""

        code_related_messages.reverse()
        lines = [
            "Recent code conversation context:",
            "Use this only to preserve continuity for ongoing code edits. Attached code files remain the source of truth.",
        ]
        for index, message in enumerate(code_related_messages, start=1):
            role = message.get("role", "unknown")
            content = self._compact_code_context_text(str(message.get("content", "")))
            if not content:
                continue
            lines.append(f"{index}. {role}: {content}")
            attachments = message.get("attachments", [])
            if attachments:
                attachment_bits = []
                for attachment in attachments:
                    name = attachment.get("name", "file")
                    skill_id = attachment.get("skill_id")
                    handler = attachment.get("handler", "pending")
                    attachment_bits.append(f"{name} (skill {skill_id}, {handler})")
                lines.append(f"   Attachments: {', '.join(attachment_bits)}")

        return "\n".join(lines)

    def _message_is_code_related(self, message: dict[str, Any]) -> bool:
        content = str(message.get("content", "")).lower()
        attachments = message.get("attachments", [])
        if any(attachment.get("skill_id") == 7 for attachment in attachments):
            return True

        code_markers = (
            "code", "coding", "script", "function", "class", "method", "bug", "debug",
            "traceback", "refactor", "implement", "implementation", "patch", "edit file",
            "modify", "fix this", "fix it", "unit test", "pytest", "syntax", "simple_agent.py",
            "skills.py", "python", "javascript", "typescript", "sql", "tkinter", "pyinstaller",
        )
        return any(marker in content for marker in code_markers)

    def _compact_code_context_text(self, text: str, max_chars: int = 900) -> str:
        cleaned = self._strip_thinking_for_storage(text)
        cleaned = self._plain_text_compact(cleaned)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        try:
            self.root.after(0, lambda: self._set_loading_base("Observing skill results"))
            observation = self._generate_text_with_budget(
                local_model_path=local_model_path,
                prompt=observation_prompt,
                max_tokens=4096,
                env=env,
                show_thinking=False,
            )
        except Exception:
            return ""

        observation = observation.strip()
        if not observation:
            return ""

        return "Agent observations from tool results:\n" + observation

    def _attachment_context_should_be_observed(self, attachment_context: str) -> bool:
        if not attachment_context:
            return False

        lowered = attachment_context.lower()
        important_markers = (
            "pdf attachment content",
            "text attachment content",
            "resume",
            "cv",
            "code",
            "script",
            "function",
            "class ",
            "def ",
            "error",
            "traceback",
        )
        if any(marker in lowered for marker in important_markers):
            return True

        return len(attachment_context) > 6000

    def _build_agent_identity_context(self) -> str:
        return (
            "SimpleAgent context:\n"
            "- You are SimpleAgent, a local-first AI agent running on the user's Mac. The base model may be Qwen, but speak as SimpleAgent unless asked about the model.\n"
            "- Available context may include memory, skill results, web/search output, URLs, attachments, PDF text, vision analysis, code files, and recent code conversation history. Treat provided context as grounding.\n"
            "- Answer the user's actual request directly. Do not merely summarise tool output. Do not expose internal planning or agent-loop steps unless asked.\n"
            "- Be factual: do not invent details beyond provided context; if context is missing, failed, weak, truncated, or uncertain, say so clearly.\n"
            "- Prefer practical output: tables for comparisons/multiple items, concise bullets for actions, and patch-style snippets for code changes.\n"
            "- For coding, attached code files are the source of truth; use recent code context only for continuity across follow-ups. Do not claim edits were applied unless the app actually edited files.\n"
            "- For file analysis, name the file used and mention truncation or extraction limits when relevant.\n"
        )

    def _build_datetime_context(self) -> str:
        now = datetime.now().astimezone()
        return (
            "Current date/time context:\n"
            f"- Local date and time: {now.strftime('%A, %d %B %Y, %H:%M:%S %Z')}\n"
            f"- ISO timestamp: {now.isoformat(timespec='seconds')}\n"
            "Use this as today's date/time when interpreting relative dates like today, tomorrow, yesterday, this week, this month, or this year."
        )

    def _build_attachment_context(self, prompt: str, attachments: list[dict[str, str]]) -> str:
        if not attachments:
            return ""

        context_parts: list[str] = []
        pending_files: list[str] = []

        for attachment in attachments:
            path = attachment.get("path", "")
            name = attachment.get("name", Path(path).name if path else "file")
            extension = attachment.get("extension", Path(path).suffix.lower() if path else "")
            skill_id = attachment.get("skill_id")

            if skill_id == 4 and path:
                vl_result = self._run_vl_attachment_analysis(prompt, path)
                if vl_result:
                    context_parts.append(f"Vision attachment analysis for {name}:\n{vl_result}")
                else:
                    pending_files.append(f"{name} ({path}) - vision analysis unavailable")
            elif skill_id == 5 and path:
                text_result = self._read_text_attachment(path)
                if text_result:
                    context_parts.append(f"Text attachment content for {name}:\n{text_result}")
                else:
                    pending_files.append(f"{name} ({path}) - text file could not be read")
            elif skill_id == 6 and path:
                pdf_result = self._read_pdf_attachment(path)
                if pdf_result:
                    context_parts.append(f"PDF attachment content for {name}:\n{pdf_result}")
                else:
                    pending_files.append(f"{name} ({path}) - PDF could not be read")
            elif skill_id == 7 and path:
                code_result = self._read_text_attachment(path)
                if code_result:
                    context_parts.append(f"Code attachment content for {name}:\n{code_result}")
                else:
                    pending_files.append(f"{name} ({path}) - code file could not be read")
            else:
                handler = attachment.get("handler", "pending")
                pending_files.append(
                    f"{name} ({path}) - attached path only; no automatic analysis implemented yet for "
                    f"{extension or 'unknown extension'} using handler {handler}"
                )

        if pending_files:
            context_parts.append(
                "Attached files without automatic analysis:\n" + "\n".join(f"- {item}" for item in pending_files)
            )

        if not context_parts:
            return ""

        return "Attachment context:\n" + "\n\n".join(context_parts)

    def _read_text_attachment(self, file_path: str, max_chars: int | None = None) -> str:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return ""

        try:
            file_size = path.stat().st_size
        except Exception:
            file_size = 0

        encodings = ("utf-8", "utf-8-sig", "utf-16", "latin-1")
        content = ""
        used_encoding = ""

        for encoding in encodings:
            try:
                content = path.read_text(encoding=encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                return f"Text read failed: {exc}"

        if not content:
            return ""

        truncated = max_chars is not None and len(content) > max_chars
        if truncated:
            content = content[:max_chars]

        metadata = (
            f"Path: {path}\n"
            f"Extension: {self._attachment_extension(path)}\n"
            f"Size bytes: {file_size}\n"
            f"Encoding used: {used_encoding or 'unknown'}\n"
            f"Truncated: {'yes' if truncated else 'no'}\n"
            "--- file content start ---\n"
        )

        footer = "\n--- file content end ---"
        if truncated:
            footer = "\n--- file content truncated because it exceeded the text attachment limit ---" + footer

        return metadata + content + footer

    def _read_pdf_attachment(self, file_path: str, max_chars: int | None = None) -> str:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return ""

        try:
            file_size = path.stat().st_size
        except Exception:
            file_size = 0

        pdf_warning_buffer = io.StringIO()
        try:
            with contextlib.redirect_stderr(pdf_warning_buffer):
                reader = PdfReader(str(path), strict=False)
        except Exception as exc:
            return f"PDF read failed: {exc}"

        pages: list[str] = []
        page_count = len(reader.pages)

        for page_index, page in enumerate(reader.pages, start=1):
            try:
                with contextlib.redirect_stderr(pdf_warning_buffer):
                    page_text = page.extract_text() or ""
            except Exception as exc:
                page_text = f"[Page {page_index} extraction failed: {exc}]"

            page_text = page_text.strip()
            if page_text:
                pages.append(f"--- page {page_index} ---\n{page_text}")

        content = "\n\n".join(pages).strip()

        if not content:
            return (
                f"PDF read warning: no extractable text found in {path.name}. "
                "This PDF may be scanned or image-based. Use the vision attachment skill by converting pages to images."
            )

        truncated = max_chars is not None and len(content) > max_chars
        if truncated:
            content = content[:max_chars]

        metadata = (
            f"Path: {path}\n"
            f"Extension: {self._attachment_extension(path)}\n"
            f"Size bytes: {file_size}\n"
            f"Page count: {page_count}\n"
            f"Truncated: {'yes' if truncated else 'no'}\n"
            "--- pdf content start ---\n"
        )

        footer = "\n--- pdf content end ---"
        if truncated:
            footer = "\n--- pdf content truncated because it exceeded the PDF attachment limit ---" + footer

        return metadata + content + footer

    def _run_vl_attachment_analysis(self, user_prompt: str, file_path: str) -> str:
        model = self._resolve_model("qwen2.5-vl-3b-instruct")
        if model is None or model.get("runtime") != "mlx-vlm":
            return ""

        model_dir = self.models_dir / model["key"]
        local_model_path = model_dir / "model"
        if not local_model_path.exists():
            return ""

        extension = Path(file_path).suffix.lower()
        media_arg = "--video" if extension in self.video_attachment_extensions else "--image"

        vl_prompt = (
            "Analyse this attachment for the user's request. "
            "Extract visible text, describe important visual details, and mention anything relevant to the prompt. "
            "Be concise and factual.\n\n"
            f"User prompt:\n{user_prompt}"
        )

        command = [
            self.python_executable,
            "-m",
            "mlx_vlm",
            "generate",
            "--model",
            str(local_model_path),
            media_arg,
            file_path,
            "--prompt",
            vl_prompt,
            "--max-tokens",
            "2048",
        ]

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        try:
            self.root.after(0, lambda: self._set_loading_base("Analysing attachment"))
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        except Exception as exc:
            return f"Vision analysis failed: {exc}"

        if result.returncode != 0:
            error_output = result.stderr.strip() or result.stdout.strip() or "Unknown VL error"
            return f"Vision analysis failed: {error_output}"

        output = result.stdout.strip()

        if self.debug:
            raw_debug_output = (
                "\n===== VL MODEL OUTPUT START =====\n"
                f"Attachment: {file_path}\n"
                f"{output}\n"
                "===== VL MODEL OUTPUT END =====\n"
            )
            self._print_debug(raw_debug_output)

        cleaned_output = self._strip_mlx_output_headers(output)
        cleaned_output = self._remove_hidden_prompt_thinking_leak(cleaned_output)
        return cleaned_output.strip()

    def _run_prompt(
        self,
        prompt: str,
        memory_items: list[dict[str, str]],
        attachments: list[dict[str, str]] | None = None,
    ) -> str:
        model = self._resolve_model(self.active_prompt_model_key)
        if model is None:
            raise RuntimeError(f"Prompt model not found: {self.active_prompt_model_key}")

        if model["runtime"] != "mlx-lm":
            raise RuntimeError(
                f"Prompt mode currently supports mlx-lm text models only. Active model runtime: {model['runtime']}"
            )

        model_dir = self.models_dir / model["key"]
        metadata_path = model_dir / "metadata.json"
        local_model_path = model_dir / "model"
        if not metadata_path.exists() or not local_model_path.exists():
            raise RuntimeError("Prompt model is not downloaded locally yet. Use the download button first.")

        self.root.after(0, lambda: self._set_loading_base("Preparing context"))
        memory_block = self._build_memory_block(memory_items)

        self.root.after(0, lambda: self._set_loading_base("Routing skills"))
        skill_ids = self._decide_skill_ids(prompt, memory_items, attachments or [])

        self.root.after(0, lambda: self._set_loading_base("Executing skills"))
        skill_context = self._execute_skills(skill_ids, prompt)

        self.root.after(0, lambda: self._set_loading_base("Handling attachments"))
        attachment_context = self._build_attachment_context(prompt, attachments or [])

        self.root.after(0, lambda: self._set_loading_base("Planning response"))
        agent_loop_plan = self._build_agent_loop_plan(prompt, skill_ids, attachments or [])
        observation_context = self._build_agent_observation_summary(prompt, skill_context, attachment_context)

        self.root.after(0, lambda: self._set_loading_base("Building prompt"))
        prompt_parts: list[str] = [
            self._build_agent_identity_context(),
            self._build_datetime_context(),
        ]
        if memory_block:
            prompt_parts.append(memory_block)

        if 7 in skill_ids:
            code_conversation_context = self._build_code_conversation_context(prompt)
            if code_conversation_context:
                prompt_parts.append(code_conversation_context)

        if agent_loop_plan:
            prompt_parts.append(agent_loop_plan)

        if observation_context:
            prompt_parts.append(observation_context)
        elif skill_context:
            prompt_parts.append(skill_context)

        if attachment_context:
            prompt_parts.append(attachment_context)

        if skill_context:
            prompt_parts.append(
                "Skill result rules:\n"
                "- Use the skill results as grounding context.\n"
                "- Do not invent facts that are not present in the skill results.\n"
                "- For web/news/current-event answers, prefer reputable sources and be cautious with unknown domains.\n"
                "- If dates are missing from the skill results, say that the date was not provided instead of guessing.\n"
                "- If the skill results only provide source pages but not the exact requested answer, say that and point the user to the best source URLs.\n"
                "- Do not call something verified unless the supplied context supports it."
            )

        if attachment_context:
            prompt_parts.append(
                "Attachment result rules:\n"
                "- Use the attachment analysis as grounding context.\n"
                "- If an attached file was marked path only, acknowledge that it was attached but not automatically analysed yet.\n"
                "- If attached content was truncated, mention that your answer is based only on the available extracted content.\n"
                "- For resumes, documents, or code reviews, provide a direct evaluation, a score when requested, strengths, weaknesses, and specific improvements.\n"
                "- Do not claim to see an image, read a PDF, or inspect a file if the relevant attachment analysis failed."
            )

        if 7 in skill_ids:
            prompt_parts.append(
                "Code skill rules:\n"
                "- Treat attached code files as source-of-truth context.\n"
                "- Use recent code conversation context to preserve continuity across multi-turn edits.\n"
                "- When the user says 'this', 'that', 'it', 'try again', 'fix it', or 'continue', resolve the reference from the recent code context and attached files.\n"
                "- When the user asks to code, debug, refactor, or modify, identify the relevant functions/classes/files first.\n"
                "- Prefer patch-style edits with exact replacement snippets when changing existing code.\n"
                "- If multiple code files are attached, explain which file each change belongs to.\n"
                "- Do not claim a change has been applied unless the file was actually edited by the app.\n"
                "- Keep explanations short unless the change is complex."
            )

        if agent_loop_plan or observation_context:
            prompt_parts.append(
                "Agent loop response rules:\n"
                "- Use the plan and observations to improve answer quality.\n"
                "- Do not expose the internal loop unless the user asks for it.\n"
                "- If the observation summary and raw context conflict, prefer the raw context and mention uncertainty.\n"
                "- Give the final answer in the most useful format for the user's request."
            )

        prompt_parts.append(f"Current user prompt:\n{prompt}")
        final_prompt = "\n\n".join(prompt_parts)
        response_token_budget = self.selected_response_tokens
        if response_token_budget != self.unlimited_response_tokens:
            response_token_budget = self._clamp_response_tokens(response_token_budget)
        self.last_response_token_budget = response_token_budget

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        self.root.after(0, lambda: self._set_loading_base("Generating answer"))
        return self._generate_text_with_budget(
            local_model_path=local_model_path,
            prompt=final_prompt,
            max_tokens=response_token_budget,
            env=env,
        )

    def _generate_text_with_budget(
        self,
        local_model_path: Path,
        prompt: str,
        max_tokens: int,
        env: dict[str, str],
        show_thinking: bool = True,
    ) -> str:
        effective_prompt = prompt if show_thinking else self._build_no_think_prompt(prompt)
        command = [
            self.python_executable,
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            str(local_model_path),
            "--prompt",
            effective_prompt,
        ]

        if max_tokens > 0:
            command.extend(["--max-tokens", str(max_tokens)])

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        if result.returncode != 0:
            error_output = result.stderr.strip() or result.stdout.strip() or "Unknown MLX error"
            raise RuntimeError(error_output)

        output = result.stdout.strip()

        if self.debug:
            raw_debug_output = (
                "\n===== RAW MODEL OUTPUT START =====\n"
                f"{output}\n"
                "===== RAW MODEL OUTPUT END =====\n"
            )
            self._print_debug(raw_debug_output)

        if not output:
            raise RuntimeError("Prompt execution returned no output.")

        visible_response = self._extract_visible_response(output, show_thinking=show_thinking)
        if not visible_response:
            raise RuntimeError("Prompt execution returned no visible response.")
        return visible_response

    def _strip_mlx_output_headers(self, output: str) -> str:
        hidden_prefixes = (
            "==========",
            "Prompt:",
            "Generation:",
            "Peak memory:",
        )
        visible_lines = [
            line for line in output.splitlines() if not line.startswith(hidden_prefixes)
        ]
        return "\n".join(visible_lines).strip()

    def _extract_visible_response(self, output: str, show_thinking: bool = True) -> str:
        cleaned_output = self._strip_mlx_output_headers(output)
        answer, thinking = self._split_thinking_output(cleaned_output)
        self.last_thinking_text = thinking

        if thinking and show_thinking:
            thinking_preview = self._thinking_preview(thinking)
            if thinking_preview:
                self.root.after(0, partial(self._set_loading_base, thinking_preview))

        if not show_thinking:
            answer = self._remove_hidden_prompt_thinking_leak(answer)

        return answer.strip()

    def _build_no_think_prompt(self, prompt: str) -> str:
        return (
            f"{prompt.strip()}\n\n"
            "System instruction for this internal helper call:\n"
            "Do not output reasoning, analysis, planning notes, or <think> tags. "
            "Return only the final requested value.\n"
            "/no_think"
        )

    def _remove_hidden_prompt_thinking_leak(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""

        cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()
        cleaned = re.sub(r"(?is)^.*?</think>", "", cleaned).strip()

        leak_patterns = [
            r"(?is)^okay,?\s+the user.*?(?:\n\s*\n|$)",
            r"(?is)^let me .*?(?:\n\s*\n|$)",
            r"(?is)^i need to .*?(?:\n\s*\n|$)",
            r"(?is)^we need to .*?(?:\n\s*\n|$)",
            r"(?is)^first,.*?(?:\n\s*\n|$)",
            r"(?is)^final response:?",
        ]
        for pattern in leak_patterns:
            cleaned = re.sub(pattern, "", cleaned).strip()

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) > 1:
            filtered_lines = [
                line for line in lines
                if not re.match(r"(?i)^(okay|let me|i need|we need|first|then|wait|make sure|the user)", line)
            ]
            if filtered_lines:
                cleaned = "\n".join(filtered_lines).strip()

        return cleaned

    def _split_thinking_output(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""

        think_start_match = re.search(r"<think>", text, flags=re.IGNORECASE)
        think_end_match = re.search(r"</think>", text, flags=re.IGNORECASE)

        if think_end_match:
            thinking = ""
            if think_start_match and think_start_match.start() < think_end_match.start():
                thinking = text[think_start_match.end():think_end_match.start()].strip()
            else:
                thinking = text[:think_end_match.start()].strip()
            answer = text[think_end_match.end():].strip()
            return answer, thinking

        if think_start_match:
            thinking = text[think_start_match.end():].strip()
            return "", thinking

        if self._looks_like_unclosed_thinking(text):
            return "", text.strip()

        return text.strip(), ""

    def _looks_like_unclosed_thinking(self, text: str) -> bool:
        compact = text.strip().lower()
        if not compact:
            return False

        thinking_starts = (
            "okay, the user",
            "okay, user",
            "hmm, the user",
            "hmm, user",
            "let me",
            "i need to",
            "we need to",
            "first,",
        )
        if compact.startswith(thinking_starts):
            return True

        thinking_markers = (
            "the user wants",
            "the user is asking",
            "i should",
            "i need",
            "let me",
            "final response",
        )
        marker_count = sum(1 for marker in thinking_markers if marker in compact[:600])
        return marker_count >= 2

    def _thinking_preview(self, thinking: str) -> str:
        if not thinking:
            return "Thinking"

        compact = re.sub(r"\s+", " ", thinking).strip()
        compact = re.sub(r"^okay,?\s*", "", compact, flags=re.IGNORECASE)
        if not compact:
            return "Thinking"

        max_length = 80
        if len(compact) > max_length:
            compact = compact[: max_length - 3].rstrip() + "..."
        return f"Thinking: {compact}"

    def _attachment_skill_ids(self, attachments: list[dict[str, str]] | None = None) -> list[int]:
        attachment_items = attachments if attachments is not None else self.pending_attachments
        skill_ids: list[int] = []

        for attachment in attachment_items:
            skill_id = attachment.get("skill_id")
            if isinstance(skill_id, int):
                skill_ids.append(skill_id)

        return list(dict.fromkeys(skill_ids))

    def _current_prompt_has_vl_attachments(self, attachments: list[dict[str, str]] | None = None) -> bool:
        return 4 in self._attachment_skill_ids(attachments)

    def _decide_skill_ids(
        self,
        prompt: str,
        memory_items: list[dict[str, str]],
        attachments: list[dict[str, str]] | None = None,
    ) -> list[int]:
        model = self._resolve_model(self.active_prompt_model_key)
        if model is None or model["runtime"] != "mlx-lm":
            return []

        model_dir = self.models_dir / model["key"]
        local_model_path = model_dir / "model"
        if not local_model_path.exists():
            return []

        deterministic_ids: list[int] = []

        if self._prompt_needs_memory(prompt):
            deterministic_ids.append(3)

        if self._prompt_contains_url(prompt):
            deterministic_ids.append(2)

        if self._prompt_needs_code_skill(prompt):
            deterministic_ids.append(7)

        deterministic_ids.extend(self._attachment_skill_ids(attachments))

        if deterministic_ids:
            selected_ids = list(dict.fromkeys(deterministic_ids))
            self._debug_skill_decision("rule", selected_ids, prompt, "Rule-based skill selection")
            return selected_ids

        if not self._prompt_explicitly_needs_internet(prompt):
            self._debug_skill_decision("rule", [], prompt, "No explicit internet or memory intent detected")
            return []

        skill_catalog = skills.get_all_skills()
        memory_block = self._build_memory_block(memory_items)
        decision_prompt = (
            "You are a very conservative skill router for a local assistant.\n"
            "Available skills:\n"
            f"{skill_catalog}\n\n"
            "Choose which skills should run before answering the user.\n"
            "Return only comma-separated integers, with no explanation.\n"
            "Use 1 only when the user explicitly asks to search, look up, browse, get latest/current/today information, check prices, check news, or verify facts online.\n"
            "Use 2 only when the user provides a URL/link and asks about its content, wants it summarised, or wants information extracted from it.\n"
            "Use 4 when the user has attached an image/video or asks to analyse an attached visual file.\n"
            "Use 5 when the user has attached a readable text/code/config file such as txt, md, py, json, csv, html, css, js, sql, sh, yaml, or similar.\n"
            "Use 6 when the user has attached a PDF file or asks to read, summarise, extract, or analyse an attached PDF.\n"
            "Use 7 when the user asks to code, debug, refactor, edit, implement, patch, review code, or when a code file is attached.\n"
            "Use 0 for normal conversation, opinions, follow-up questions, UI changes, reasoning, writing, summarising, or anything answerable from the existing chat without tools.\n"
            "For short follow-ups like 'what do you think', 'explain more', 'continue', or 'is it bullish or bearish', choose 0 unless the user also explicitly asks to search online.\n"
            "When unsure, choose 0.\n\n"
        )
        if memory_block:
            decision_prompt += f"Memory context:\n{memory_block}\n\n"
        decision_prompt += f"User prompt:\n{prompt}"

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        try:
            self.root.after(0, lambda: self._set_loading_base("Choosing tools"))
            decision_text = self._generate_text_with_budget(
                local_model_path=local_model_path,
                prompt=decision_prompt,
                max_tokens=self.default_response_tokens,
                env=env,
                show_thinking=False,
            )
        except Exception:
            return []

        parsed_ids = self._parse_skill_ids(decision_text)

        if self._prompt_contains_url(prompt) and 2 not in parsed_ids:
            parsed_ids.append(2)

        if 1 in parsed_ids and not self._prompt_explicitly_needs_internet(prompt):
            parsed_ids = [skill_id for skill_id in parsed_ids if skill_id != 1]

        self._debug_skill_decision("model", parsed_ids, prompt, decision_text)

        return parsed_ids

    def _debug_skill_decision(
            self,
            source: str,
            skill_ids: list[int],
            prompt: str,
            raw_decision: str,
    ) -> None:
        if not self.debug:
            return

        skill_catalog = {
            0: "no_skill",
            1: "internet_search",
            2: "scrape_url",
            3: "memory_rag",
            4: "attachment_vision",
            5: "text_file_reader",
            6: "pdf_reader",
            7: "code_reader",
        }
        readable_skills = [skill_catalog.get(skill_id, f"unknown_{skill_id}") for skill_id in skill_ids]
        if not readable_skills:
            readable_skills = ["no_skill"]

        debug_text = (
            "\n===== SKILL ROUTER DECISION START =====\n"
            f"Source: {source}\n"
            f"Prompt: {prompt}\n"
            f"Selected skill ids: {skill_ids if skill_ids else [0]}\n"
            f"Selected skills: {', '.join(readable_skills)}\n"
            "Raw/router decision:\n"
            f"{raw_decision}\n"
            "===== SKILL ROUTER DECISION END =====\n"
        )
        self._print_debug(debug_text)

    def _prompt_contains_url(self, prompt: str) -> bool:
        return re.search(r"https?://[^\s)\]>\"']+", prompt) is not None


    def _prompt_needs_memory(self, prompt: str) -> bool:
        p = prompt.lower().strip()

        triggers = [
            "earlier",
            "previous",
            "before",
            "last time",
            "you said",
            "we talked",
            "based on our chat",
            "from memory",
            "what did",
            "didn't you say",
            "remind me",
        ]

        if any(t in p for t in triggers):
            return True

        # implicit short follow-ups
        if len(p.split()) <= 6 and any(w in p for w in ["think", "why", "how", "again"]):
            return True

        return False

    def _prompt_explicitly_needs_internet(self, prompt: str) -> bool:
        prompt_lower = prompt.strip().lower()
        if not prompt_lower:
            return False

        explicit_search_phrases = {
            "search",
            "search online",
            "search the internet",
            "browse",
            "browse online",
            "look up",
            "lookup",
            "google",
            "check online",
            "find online",
            "verify online",
            "latest",
            "current",
            "today",
            "recent",
            "newest",
            "up to date",
            "up-to-date",
            "news",
            "price now",
            "current price",
            "stock price",
            "crypto price",
            "weather",
        }

        if any(phrase in prompt_lower for phrase in explicit_search_phrases):
            return True

        time_sensitive_patterns = [
            r"\b20\d{2}\b",
            r"\bnow\b",
            r"\bright now\b",
            r"\bthis week\b",
            r"\bthis month\b",
            r"\bthis year\b",
            r"\bas of\b",
        ]

        if any(re.search(pattern, prompt_lower) for pattern in time_sensitive_patterns):
            return True

        return False

    def _prompt_needs_code_skill(self, prompt: str) -> bool:
        prompt_lower = prompt.strip().lower()
        if not prompt_lower:
            return False

        code_markers = {
            "code", "coding", "script", "function", "class", "method",
            "bug", "debug", "error", "traceback",
            "refactor", "implement", "implementation", "patch",
            "edit file", "modify", "fix this", "fix it", "add feature",
            "unit test", "pytest", "lint", "syntax",
            "python", "javascript", "typescript", "sql", "html", "css",
            "tkinter", "pyinstaller",
        }
        return any(marker in prompt_lower for marker in code_markers)

    def _parse_skill_ids(self, decision_text: str) -> list[int]:
        valid_skill_ids = skills.get_valid_skill_ids()
        selected_ids: list[int] = []

        for match in re.findall(r"\b\d+\b", decision_text):
            skill_id = int(match)
            if skill_id in valid_skill_ids:
                selected_ids.append(skill_id)

        decision_lower = decision_text.lower()
        skill_name_map = {
            "internet_search": 1,
            "internet search": 1,
            "web_search": 1,
            "web search": 1,
            "search online": 1,
            "online search": 1,
            "search the internet": 1,
            "scrape_url": 2,
            "scrape url": 2,
            "direct url": 2,
            "provided url": 2,
            "website link": 2,
            "link provided": 2,
            "memory_rag": 3,
            "memory rag": 3,
            "search memory": 3,
            "past messages": 3,
            "previous messages": 3,
            "old messages": 3,
            "attachment_vision": 4,
            "attachment vision": 4,
            "vision attachment": 4,
            "image attachment": 4,
            "attached image": 4,
            "attached video": 4,
            "visual file": 4,
            "text_file_reader": 5,
            "text file reader": 5,
            "text file": 5,
            "read text file": 5,
            "attached text": 5,
            "markdown file": 5,
            "pdf_reader": 6,
            "pdf reader": 6,
            "pdf file": 6,
            "read pdf": 6,
            "attached pdf": 6,
            "analyse pdf": 6,
            "analyze pdf": 6,
            "summarise pdf": 6,
            "summarize pdf": 6,
            "code_reader": 7,
            "code reader": 7,
            "code skill": 7,
            "coding": 7,
            "code file": 7,
            "attached code": 7,
            "debug code": 7,
            "refactor code": 7,
            "edit code": 7,
            "patch code": 7,
            "review code": 7,
        }

        for name, skill_id in skill_name_map.items():
            if name in decision_lower and skill_id in valid_skill_ids:
                selected_ids.append(skill_id)

        no_skill_phrases = {
            "no_skill",
            "no skill",
            "none",
            "no tools",
            "no tool",
            "no skills needed",
            "no skill needed",
        }

        if 0 in selected_ids or any(phrase in decision_lower for phrase in no_skill_phrases):
            return []

        return list(dict.fromkeys(skill_id for skill_id in selected_ids if skill_id != 0))

    def _generate_search_query(self, prompt: str) -> str:
        model = self._resolve_model(self.active_prompt_model_key)
        if model is None or model["runtime"] != "mlx-lm":
            return prompt

        model_dir = self.models_dir / model["key"]
        local_model_path = model_dir / "model"
        if not local_model_path.exists():
            return prompt

        query_prompt = (
            "You create search engine queries for an assistant.\n"
            "Given the user's request, output one concise search query that finds concrete answer data, not just generic landing pages.\n"
            "The query must be generic to the user's domain: products, prices, people, facts, rankings, documents, releases, tutorials, code, or whatever the user actually needs.\n"
            "Prefer authoritative sources, specific list pages, names, dates, and terms like latest, current, ranking, or list when relevant.\n"
            "Do not answer the user.\n"
            "Do not include explanations.\n"
            "Return only the search query.\n\n"
            f"User request:\n{prompt}"
        )

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        try:
            self.root.after(0, lambda: self._set_loading_base("Creating search query"))
            query = self._generate_text_with_budget(
                local_model_path=local_model_path,
                prompt=query_prompt,
                max_tokens=self.default_response_tokens,
                env=env,
                show_thinking=False,
            )
        except Exception:
            return prompt

        cleaned_query = query.strip().replace("\n", " ")
        cleaned_query = re.sub(r"\s+", " ", cleaned_query)
        cleaned_query = cleaned_query.strip("`'\"")
        return cleaned_query or prompt

    def _execute_skills(self, skill_ids: list[int], prompt: str) -> str:
        if not skill_ids:
            return ""

        outputs: list[str] = []
        for skill_id in skill_ids:
            try:
                skill_input = prompt
                if skill_id == 1:
                    if not self._prompt_explicitly_needs_internet(prompt):
                        continue
                    skill_input = self._generate_search_query(prompt)
                    self.root.after(0, lambda: self._set_loading_base("Searching internet"))
                elif skill_id == 2:
                    skill_input = prompt
                    self.root.after(0, lambda: self._set_loading_base("Scraping URL"))
                elif skill_id == 3:
                    skill_input = prompt
                    self.root.after(0, lambda: self._set_loading_base("Searching memory"))
                elif skill_id == 4:
                    self.root.after(0, lambda: self._set_loading_base("Preparing attachment analysis"))
                    result = "Attachment vision analysis will run through the local VL attachment pipeline."
                    if self.debug:
                        debug_text = (
                            "\n===== SKILL EXECUTION START =====\n"
                            f"Skill id: {skill_id}\n"
                            "Skill input: attached visual files\n"
                            "Skill output:\n"
                            f"{result}\n"
                            "===== SKILL EXECUTION END =====\n"
                        )
                        self._print_debug(debug_text)
                    outputs.append(f"Skill {skill_id} output:\n{result}")
                    continue
                elif skill_id == 5:
                    self.root.after(0, lambda: self._set_loading_base("Preparing text file reading"))
                    result = "Text file reading will run through the local attachment text pipeline."

                    if self.debug:
                        debug_text = (
                            "\n===== SKILL EXECUTION START =====\n"
                            f"Skill id: {skill_id}\n"
                            "Skill input: attached text/code files\n"
                            "Skill output:\n"
                            f"{result}\n"
                            "===== SKILL EXECUTION END =====\n"
                        )
                        self._print_debug(debug_text)

                    outputs.append(f"Skill {skill_id} output:\n{result}")
                    continue
                elif skill_id == 6:
                    self.root.after(0, lambda: self._set_loading_base("Preparing PDF reading"))
                    result = "PDF reading will run through the local attachment PDF pipeline."

                    if self.debug:
                        debug_text = (
                            "\n===== SKILL EXECUTION START =====\n"
                            f"Skill id: {skill_id}\n"
                            "Skill input: attached PDF files\n"
                            "Skill output:\n"
                            f"{result}\n"
                            "===== SKILL EXECUTION END =====\n"
                        )
                        self._print_debug(debug_text)

                    outputs.append(f"Skill {skill_id} output:\n{result}")
                    continue
                elif skill_id == 7:
                    self.root.after(0, lambda: self._set_loading_base("Preparing code analysis"))
                    result = "Code reading and coding guidance will run through the local attachment code pipeline."

                    if self.debug:
                        debug_text = (
                            "\n===== SKILL EXECUTION START =====\n"
                            f"Skill id: {skill_id}\n"
                            "Skill input: code intent or attached code files\n"
                            "Skill output:\n"
                            f"{result}\n"
                            "===== SKILL EXECUTION END =====\n"
                        )
                        self._print_debug(debug_text)

                    outputs.append(f"Skill {skill_id} output:\n{result}")
                    continue

                if self.debug:
                    debug_text = (
                        "\n===== SKILL EXECUTION START =====\n"
                        f"Skill id: {skill_id}\n"
                        f"Skill input: {skill_input}\n"
                    )
                    self._print_debug(debug_text)

                current_chat = self._get_current_chat()
                current_memory = current_chat.get("memory", []) if current_chat is not None else []
                result = skills.execute_skill(skill_id, skill_input, current_memory)

                if self.debug:
                    debug_text = (
                        "Skill output:\n"
                        f"{result if result.strip() else '<empty>'}\n"
                        "===== SKILL EXECUTION END =====\n"
                    )
                    self._print_debug(debug_text)
            except Exception as exc:
                result = f"Skill {skill_id} failed: {exc}"

            if result.strip():
                outputs.append(f"Skill {skill_id} output:\n{result.strip()}")

        if not outputs:
            if self.debug:
                debug_text = (
                    "\n===== SKILL EXECUTION SUMMARY =====\n"
                    "No skill outputs were added to the final prompt.\n"
                    "===== SKILL EXECUTION SUMMARY END =====\n"
                )
                self._print_debug(debug_text)
            return ""

        return "Skill results to use as grounding context:\n" + "\n\n".join(outputs)

    def _clamp_response_tokens(self, value: int) -> int:
        if value == self.unlimited_response_tokens:
            return value
        return max(self.min_response_tokens, min(self.max_response_tokens, value))

    def _build_memory_block(self, memory_items: list[dict[str, str]]) -> str:
        if not memory_items:
            return ""

        lines = ["Recent memory:"]
        for index, item in enumerate(memory_items, start=1):
            lines.append(f"{index}. User summary: {item['user_summary']}")
            lines.append(f"   Assistant summary: {item['assistant_summary']}")
        return "\n".join(lines)

    def _remember_turn(self, chat: dict[str, Any], user_prompt: str, model_response: str) -> None:
        self._set_loading_base("Saving memory")
        user_summary, assistant_summary = self._summarise_turn_for_memory(user_prompt, model_response)

        user_summary = self._clean_hidden_value(user_summary)
        assistant_summary = self._clean_hidden_value(assistant_summary)

        if not user_summary or self._summary_looks_bad(user_summary):
            user_summary = self._fallback_user_memory_summary(user_prompt)
        if not assistant_summary or self._summary_looks_bad(assistant_summary):
            assistant_summary = self._fallback_assistant_memory_summary(model_response)

        memory = chat.setdefault("memory", [])
        memory.append(
            {
                "user_summary": user_summary,
                "assistant_summary": assistant_summary,
            }
        )
        chat["memory"] = memory[-self.max_memory_items :]


    def _summarise_turn_for_memory(self, user_prompt: str, model_response: str) -> tuple[str, str]:
        model = self._resolve_model(self.active_prompt_model_key)
        if model is None:
            return "", ""
        if model["runtime"] != "mlx-lm":
            return "", ""

        model_dir = self.models_dir / model["key"]
        local_model_path = model_dir / "model"
        if not local_model_path.exists():
            return "", ""

        memory_prompt = (
            "Create compact memory summaries for one conversation turn.\n"
            "Return exactly two lines and nothing else.\n"
            "Line 1 must start with `user_summary:` and contain one short sentence summarising the user's durable intent, key constraints, and important facts.\n"
            "Line 2 must start with `assistant_summary:` and contain one short sentence summarising only the assistant's main outcome, decision, or useful result.\n"
            "Do not include reasoning, analysis, markdown, bullets, JSON, or <think> tags.\n\n"
            f"User prompt:\n{user_prompt}\n\n"
            f"Assistant response:\n{model_response}"
        )

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        try:
            visible_response = self._generate_text_with_budget(
                local_model_path=local_model_path,
                prompt=memory_prompt,
                max_tokens=256,
                env=env,
                show_thinking=False,
            )
        except Exception:
            return "", ""

        return self._parse_turn_memory_summary(visible_response)

    def _parse_turn_memory_summary(self, text: str) -> tuple[str, str]:
        cleaned = self._remove_hidden_prompt_thinking_leak(text)
        cleaned = re.sub(r"```(?:json|text)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        user_summary = ""
        assistant_summary = ""

        for line in cleaned.splitlines():
            stripped = line.strip().strip("-• ")
            if not stripped:
                continue
            user_match = re.match(r"(?i)^user_summary\s*[:=-]\s*(.+)$", stripped)
            if user_match:
                user_summary = user_match.group(1).strip()
                continue
            assistant_match = re.match(r"(?i)^assistant_summary\s*[:=-]\s*(.+)$", stripped)
            if assistant_match:
                assistant_summary = assistant_match.group(1).strip()
                continue

        if user_summary and assistant_summary:
            return user_summary, assistant_summary

        json_like_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if json_like_match:
            try:
                data = json.loads(json_like_match.group(0))
                user_summary = str(data.get("user_summary", "")).strip()
                assistant_summary = str(data.get("assistant_summary", "")).strip()
                if user_summary and assistant_summary:
                    return user_summary, assistant_summary
            except Exception:
                pass

        compact_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(compact_lines) >= 2:
            return compact_lines[0], compact_lines[1]

        return "", ""

    def _generate_chat_title(self, first_prompt: str) -> str:
        self.root.after(0, lambda: self._set_loading_base("Generating chat title"))
        title = self._summarise_text(
            text=first_prompt,
            instruction=(
                "Create a very short chat title based on this user prompt. "
                "Use 2 to 5 words only. "
                "Return title only, with no punctuation except hyphens if absolutely needed."
            ),
            max_tokens="256",
        )
        title = self._clean_hidden_value(title)
        if not title or self._title_looks_bad(title):
            title = self._fallback_chat_title(first_prompt)
        return title[:60] or "New Chat"

    def _title_looks_bad(self, title: str) -> bool:
        cleaned = title.strip()
        if not cleaned:
            return True
        lowered = cleaned.lower()
        bad_markers = (
            "the user",
            "hmm",
            "okay",
            "looking at",
            "options could be",
            "title",
            "return only",
            "thinking",
        )
        if any(marker in lowered for marker in bad_markers):
            return True
        if len(cleaned.split()) > 6:
            return True
        if len(cleaned) > 60:
            return True
        return False

    def _clean_hidden_value(self, text: str) -> str:
        cleaned = self._remove_hidden_prompt_thinking_leak(text)
        cleaned = cleaned.strip().replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip("`'\".,:;!?- ")
        return cleaned

    def _fallback_chat_title(self, text: str) -> str:
        cleaned = re.sub(r"https?://\S+", "", text)
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", cleaned)
        words = [word for word in cleaned.split() if len(word) > 1]
        stop_words = {
            "what", "your", "name", "and", "can", "you", "the", "for", "with",
            "this", "that", "from", "into", "about", "explain", "detail", "please",
        }
        meaningful_words = [word for word in words if word.lower() not in stop_words]
        selected_words = meaningful_words[:5] or words[:5]
        if not selected_words:
            return "New Chat"
        return " ".join(word.capitalize() for word in selected_words[:5])

    def _summary_looks_bad(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return True

        lowered = cleaned.lower()
        thinking_leaks = (
            "okay, the user",
            "hmm, the user",
            "looking at the text",
            "i need to",
            "i should",
            "let me",
            "the provided text",
        )
        if any(leak in lowered for leak in thinking_leaks):
            return True

        if len(cleaned) > 260:
            return True

        markdown_or_copy_markers = ("###", "####", "---", "**", "```", "🔍", "✅", "🛠️")
        if any(marker in cleaned for marker in markdown_or_copy_markers):
            return True

        return False

    def _fallback_user_memory_summary(self, text: str, max_chars: int = 260) -> str:
        cleaned = self._strip_thinking_for_storage(text)
        cleaned = self._plain_text_compact(cleaned)
        if not cleaned:
            return "User made a general request."
        return cleaned[:max_chars]

    def _fallback_assistant_memory_summary(self, text: str, max_chars: int = 420) -> str:
        cleaned = self._strip_thinking_for_storage(text)
        cleaned = self._plain_text_compact(cleaned)
        if not cleaned:
            return "Assistant provided a response."

        identity_match = re.search(
            r"(?i)\bmy name is\s+([A-Za-z0-9_.-]+).*?(?:i can|i am able to|i help|capabilities|functionality)",
            cleaned,
        )
        if identity_match:
            name = identity_match.group(1).strip("`'\".,:;!?- ")
            return (
                f"Assistant introduced itself as {name} and explained its main capabilities, "
                "including conversation, writing, reasoning, coding help, analysis, and limitations."
            )

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        useful_sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
            and not sentence.strip().lower().startswith(("hello", "sure", "of course"))
        ]
        selected = " ".join(useful_sentences[:2]).strip() or cleaned
        return selected[:max_chars]

    def _plain_text_compact(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
        cleaned = re.sub(r"\*\*\*([^*]*)\*\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*\*([^*]*)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]*)\*", r"\1", cleaned)
        cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"[-_]{3,}", " ", cleaned)
        cleaned = re.sub(r"[🔍✅🛠️🔑📌⚠️🚀]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _strip_thinking_for_storage(self, text: str) -> str:
        cleaned = self._strip_mlx_output_headers(text)
        answer, thinking = self._split_thinking_output(cleaned)
        if answer:
            return answer.strip()
        if thinking:
            return ""
        return cleaned.strip()

    def _summarise_text(self, text: str, instruction: str, max_tokens: str = "64") -> str:
        model = self._resolve_model(self.active_prompt_model_key)
        if model is None:
            return ""
        if model["runtime"] != "mlx-lm":
            return ""

        model_dir = self.models_dir / model["key"]
        local_model_path = model_dir / "model"
        if not local_model_path.exists():
            return ""

        summary_prompt = (
            f"{instruction}\n\n"
            "Return only the requested output. "
            "Do not explain. Do not include reasoning. Do not include <think> tags.\n\n"
            f"Text:\n{text}"
        )

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        try:
            visible_response = self._generate_text_with_budget(
                local_model_path=local_model_path,
                prompt=summary_prompt,
                max_tokens=self.default_response_tokens,
                env=env,
                show_thinking=False,
            )
        except Exception:
            return ""

        cleaned_value = self._clean_hidden_value(visible_response)
        if not cleaned_value or self._summary_looks_bad(cleaned_value):
            return ""
        return cleaned_value[:420]

    def _download_text_models_async(self) -> None:
        if self.is_generating:
            return
        self.is_generating = True
        self.send_button.config(state="disabled")
        self.new_chat_button.config(state="disabled")
        self._set_loading_base("Downloading models")
        self._set_status("Downloading models...")
        threading.Thread(target=self._download_text_models_worker, daemon=True).start()

    def _download_text_models_worker(self) -> None:
        messages: list[str] = []
        for model in MODEL_LIBRARY:
            if model["runtime"] not in {"mlx-lm", "mlx-vlm", "sentence-transformers"}:
                messages.append(f"Skipped unsupported download runtime: {model['id']}")
                continue
            self.root.after(0, lambda model_key=model["key"]: self._set_loading_base(f"Downloading {model_key}"))
            message = self._download_single_model(model)
            messages.append(message)

        self.root.after(0, lambda: self._finish_downloads(messages))

    def _finish_downloads(self, messages: list[str]) -> None:
        self.is_generating = False
        self.send_button.config(state="normal", text="Send")
        self.new_chat_button.config(state="normal")
        joined = " | ".join(messages[-3:]) if messages else "Download complete."
        self._set_status(joined)

    def _download_single_model(self, model: dict[str, str]) -> str:
        model_dir = self.models_dir / model["key"]
        model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = model_dir / "model"
        metadata_path = model_dir / "metadata.json"

        if metadata_path.exists() and local_model_path.exists():
            return f"Already installed: {model['key']}"

        if model["runtime"] in {"sentence-transformers", "mlx-vlm"}:
            try:
                from huggingface_hub import snapshot_download
            except Exception:
                return "Install huggingface-hub to download the RAG model: pip install huggingface-hub"


            try:
                if local_model_path.exists():
                    shutil.rmtree(local_model_path, ignore_errors=True)

                snapshot_download(
                    repo_id=model["id"],
                    local_dir=str(local_model_path),
                    repo_type="model",
                )
            except Exception as exc:
                if local_model_path.exists():
                    shutil.rmtree(local_model_path, ignore_errors=True)
                return f"Download failed for {model['key']}: {exc}"

            metadata = {
                "id": model["id"],
                "key": model["key"],
                "category": model["category"],
                "runtime": model["runtime"],
                "specialty": model["specialty"],
                "why_selected": model["why_selected"],
                "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                "local_model_path": str(local_model_path),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return f"Downloaded: {model['key']}"

        command = [
            self.python_executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            model["id"],
            "--mlx-path",
            str(local_model_path),
            "-q",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            return "Python executable not found for model download."
        except Exception as exc:
            return f"Download failed for {model['key']}: {exc}"

        if result.returncode != 0:
            if local_model_path.exists():
                shutil.rmtree(local_model_path, ignore_errors=True)
            error_output = result.stderr.strip() or result.stdout.strip() or "Unknown MLX conversion error"
            return f"Download failed for {model['key']}: {error_output}"

        metadata = {
            "id": model["id"],
            "key": model["key"],
            "category": model["category"],
            "runtime": model["runtime"],
            "specialty": model["specialty"],
            "why_selected": model["why_selected"],
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "local_model_path": str(local_model_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return f"Downloaded: {model['key']}"

    def _resolve_model(self, selector: str) -> dict[str, str] | None:
        stripped = selector.strip().lower()

        if stripped.isdigit():
            index = int(stripped) - 1
            if 0 <= index < len(MODEL_LIBRARY):
                return MODEL_LIBRARY[index]
            return None

        for model in MODEL_LIBRARY:
            if stripped == model["key"].lower() or stripped == model["id"].lower():
                return model

        return None


def main() -> None:
    app = SimpleAgentGUI()
    app.run()


if __name__ == "__main__":
    main()