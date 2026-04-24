from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk
import skills


MODEL_LIBRARY = [
    {
        "key": "qwen2.5-3b-instruct",
        "id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "category": "orchestrator",
        "runtime": "mlx-lm",
        "specialty": "Primary orchestration model for routing, lightweight planning, and multi-step control flow.",
        "why_selected": "MLX-ready 4-bit model for Apple Silicon orchestration with a good balance of speed and quality.",
        "download_url": "https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit/resolve/main/README.md",
    },
    {
        "key": "qwen2.5-coder-7b-instruct",
        "id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "category": "specialist",
        "runtime": "mlx-lm",
        "specialty": "Specialised model for code generation and understanding.",
        "why_selected": "Upgraded to the 7B 4-bit MLX coder model for much stronger local coding performance while staying around the upper end of a 4-5 GB memory budget on Apple Silicon.",
        "download_url": "https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit/resolve/main/README.md",
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
        self.active_prompt_model_key = "qwen2.5-3b-instruct"
        self.python_executable = sys.executable or "python"
        self.debug = True
        self.max_memory_items = 5
        self.is_generating = False
        self.loading_animation_job: str | None = None
        self.loading_base = "Thinking"
        self.loading_frames = ["", ".", "..", "..."]
        self.loading_index = 0
        self.min_response_tokens = 256
        self.default_response_tokens = 1024
        self.max_response_tokens = 16384
        self.selected_response_tokens = self.default_response_tokens
        self.last_response_token_budget = self.default_response_tokens

        os.environ.setdefault("HF_HOME", str(self.models_dir / ".hf_cache"))

        self.state.add_log("INFO", "Simple Agent GUI initialised.")

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
        file_menu.add_command(label="Quit", command=self.root.destroy)

        menu_bar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu_bar)

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
            ("Minimal (256)", 256),
            ("Small (512)", 512),
            ("Medium (1024)", 1024),
            ("Large (2048)", 2048),
            ("XL (4096)", 4096),
            ("XXL (8192)", 8192),
            ("Max (16384)", 16384),
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
        self.root.bind("<Configure>", self._on_root_resized)
        self.root.after(0, self._auto_resize_input_box)

        self.button_row = ttk.Frame(self.composer, style="Composer.TFrame")
        self.button_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
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

    def _send_from_shortcut(self, event: tk.Event) -> str:
        self._send_message()
        return "break"

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
        if self.debug:
            print(f"[{self.state.logs[-1].timestamp}] {level}: {message}")

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
                f"Tokens: {self.selected_response_tokens}"
            )
        )
        self._render_transcript(chat)


    def _on_token_selection_changed(self, event: tk.Event | None = None) -> None:
        selected_label = self.token_selector_var.get().strip()
        selected_value = self.token_option_map.get(selected_label, self.default_response_tokens)
        self.selected_response_tokens = self._clamp_response_tokens(selected_value)
        self._open_current_chat()
        self._set_status(f"Response size set to {self.selected_response_tokens} tokens")

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
                name = "You" if role == "user" else "Qwen"
                tag = "user_name" if role == "user" else "assistant_name"
                self.transcript.insert(tk.END, f"{name}\n", tag)
                self._insert_formatted_message(message.get("content", "").strip())
                self.transcript.insert(tk.END, "\n", "separator")

        self.transcript.see(tk.END)

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

            if stripped.startswith("#### "):
                self.transcript.insert(tk.END, f"{stripped[5:].strip()}\n", "heading4")
                index += 1
                continue

            if stripped.startswith("### "):
                self.transcript.insert(tk.END, f"{stripped[4:].strip()}\n", "heading3")
                index += 1
                continue

            if stripped.startswith("## "):
                self.transcript.insert(tk.END, f"{stripped[3:].strip()}\n", "heading2")
                index += 1
                continue

            if stripped.startswith("# "):
                self.transcript.insert(tk.END, f"{stripped[2:].strip()}\n", "heading1")
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
        chat.setdefault("messages", []).append({"role": "user", "content": prompt})
        self._save_chat(chat)
        self._refresh_chat_list()
        self._open_current_chat()

        self.is_generating = True
        self.new_chat_button.config(state="disabled")
        self._start_loading_animation()

        memory_snapshot = list(chat.get("memory", []))
        chat_id = chat["id"]
        thread = threading.Thread(
            target=self._generate_response_worker,
            args=(chat_id, prompt, memory_snapshot),
            daemon=True,
        )
        thread.start()

    def _generate_response_worker(
        self,
        chat_id: str,
        prompt: str,
        memory_snapshot: list[dict[str, str]],
    ) -> None:
        try:
            response = self._run_prompt(prompt, memory_snapshot)
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
            self._set_status(f"Generation failed. Last token budget: {self.last_response_token_budget}")
            return

        chat.setdefault("messages", []).append({"role": "assistant", "content": response})
        self._remember_turn(chat, prompt, response)

        if self._should_generate_title(chat) and title:
            chat["title"] = title

        self._save_chat(chat)
        self._refresh_chat_list()
        if self.state.current_chat_id == chat_id:
            self._open_current_chat()
        self._set_status(f"Response generated. Tokens used allowance: {self.last_response_token_budget}")

    def _chat_by_id(self, chat_id: str) -> dict[str, Any] | None:
        for chat in self.state.chats:
            if chat["id"] == chat_id:
                return chat
        return None

    def _should_generate_title(self, chat: dict[str, Any]) -> bool:
        title = chat.get("title", "").strip().lower()
        user_messages = [m for m in chat.get("messages", []) if m.get("role") == "user"]
        return title in {"", "new chat"} and len(user_messages) == 1

    def _run_prompt(self, prompt: str, memory_items: list[dict[str, str]]) -> str:
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
        skill_ids = self._decide_skill_ids(prompt, memory_items)

        self.root.after(0, lambda: self._set_loading_base("Executing skills"))
        raw_skill_context = self._execute_skills(skill_ids, prompt)

        self.root.after(0, lambda: self._set_loading_base("Extracting relevant tool info"))
        skill_context = self._refine_skill_context(
            prompt=prompt,
            raw_skill_context=raw_skill_context,
            local_model_path=local_model_path,
            env={
                **os.environ.copy(),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        )

        self.root.after(0, lambda: self._set_loading_base("Building prompt"))
        prompt_parts: list[str] = []
        if memory_block:
            prompt_parts.append(memory_block)
        if skill_context:
            prompt_parts.append(skill_context)
        if skill_context:
            prompt_parts.append(
                "The tool information below has already been filtered for relevance to the user's original request. "
                "Use it as supporting context to answer the user's actual question directly. "
                "Do not merely summarise the tool information. "
                "If the filtered tool information is incomplete, clearly say what is missing."
            )
        prompt_parts.append(f"Current user prompt:\n{prompt}")
        final_prompt = "\n\n".join(prompt_parts)
        response_token_budget = self._clamp_response_tokens(self.selected_response_tokens)
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

    def _refine_skill_context(
        self,
        prompt: str,
        raw_skill_context: str,
        local_model_path: Path,
        env: dict[str, str],
    ) -> str:
        if not raw_skill_context.strip():
            return ""

        self.root.after(0, lambda: self._set_loading_base("Filtering tool results"))

        refinement_prompt = (
            "You are a retrieval filtering layer for a local AI agent.\n"
            "Your job is NOT to answer the user yet.\n"
            "Your job is to read raw tool results and extract only the information that helps answer the user's original request.\n\n"
            "Rules:\n"
            "1. Focus on the user's original intent.\n"
            "2. Keep concrete facts, names, numbers, lists, dates, claims, and source URLs.\n"
            "3. Remove irrelevant search-result descriptions, navigation text, boilerplate, repeated links, and generic summaries.\n"
            "4. If the raw tool results do not contain the answer, say what is missing.\n"
            "5. Do not invent information.\n"
            "6. Return compact grounding notes only.\n\n"
            f"Original user request:\n{prompt}\n\n"
            f"Raw tool results:\n{raw_skill_context}\n\n"
            "Filtered relevant tool information:"
        )

        try:
            refined = self._generate_text_with_budget(
                local_model_path=local_model_path,
                prompt=refinement_prompt,
                max_tokens=min(1024, self._clamp_response_tokens(self.selected_response_tokens)),
                env=env,
            )
        except Exception as exc:
            if self.debug:
                print("\n===== TOOL RESULT REFINEMENT FAILED =====")
                print(exc)
                print("Falling back to raw skill context.")
                print("===== TOOL RESULT REFINEMENT FAILED END =====\n")
            return raw_skill_context

        refined = refined.strip()
        if self.debug:
            print("\n===== REFINED TOOL CONTEXT START =====")
            print(refined if refined else "<empty>")
            print("===== REFINED TOOL CONTEXT END =====\n")

        return refined or raw_skill_context

    def _generate_text_with_budget(
        self,
        local_model_path: Path,
        prompt: str,
        max_tokens: int,
        env: dict[str, str],
    ) -> str:
        command = [
            self.python_executable,
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            str(local_model_path),
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
        ]

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
            print("\n===== RAW MODEL OUTPUT START =====")
            print(output)
            print("===== RAW MODEL OUTPUT END =====\n")

        if not output:
            raise RuntimeError("Prompt execution returned no output.")

        hidden_prefixes = (
            "==========",
            "Prompt:",
            "Generation:",
            "Peak memory:",
        )
        visible_lines = [
            line for line in output.splitlines() if not line.startswith(hidden_prefixes)
        ]
        visible_response = "\n".join(visible_lines).strip()
        if not visible_response:
            raise RuntimeError("Prompt execution returned no visible response.")
        return visible_response

    def _decide_skill_ids(self, prompt: str, memory_items: list[dict[str, str]]) -> list[int]:
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
            "Use 0 for normal conversation, opinions, follow-up questions, local code edits, UI changes, reasoning, writing, summarising, or anything answerable from the existing chat.\n"
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
                max_tokens=32,
                env=env,
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
        }
        readable_skills = [skill_catalog.get(skill_id, f"unknown_{skill_id}") for skill_id in skill_ids]
        if not readable_skills:
            readable_skills = ["no_skill"]

        print("\n===== SKILL ROUTER DECISION START =====")
        print(f"Source: {source}")
        print(f"Prompt: {prompt}")
        print(f"Selected skill ids: {skill_ids if skill_ids else [0]}")
        print(f"Selected skills: {', '.join(readable_skills)}")
        print("Raw/router decision:")
        print(raw_decision)
        print("===== SKILL ROUTER DECISION END =====\n")

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
                max_tokens=64,
                env=env,
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

                if self.debug:
                    print("\n===== SKILL EXECUTION START =====")
                    print(f"Skill id: {skill_id}")
                    print(f"Skill input: {skill_input}")

                current_chat = self._get_current_chat()
                current_memory = current_chat.get("memory", []) if current_chat is not None else []
                result = skills.execute_skill(skill_id, skill_input, current_memory)

                if self.debug:
                    print("Skill output:")
                    print(result if result.strip() else "<empty>")
                    print("===== SKILL EXECUTION END =====\n")
            except Exception as exc:
                result = f"Skill {skill_id} failed: {exc}"

            if result.strip():
                outputs.append(f"Skill {skill_id} output:\n{result.strip()}")

        if not outputs:
            if self.debug:
                print("\n===== SKILL EXECUTION SUMMARY =====")
                print("No skill outputs were added to the final prompt.")
                print("===== SKILL EXECUTION SUMMARY END =====\n")
            return ""

        return "Skill results to use as grounding context:\n" + "\n\n".join(outputs)


    def _clamp_response_tokens(self, value: int) -> int:
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
        user_summary = self._summarise_text(
            text=user_prompt,
            instruction=(
                "Summarise this user prompt for future memory. "
                "Keep only the durable intent, key constraints, and important facts. "
                "Use one short sentence only."
            ),
            max_tokens="64",
        )
        assistant_summary = self._summarise_text(
            text=model_response,
            instruction=(
                "Summarise this assistant response for future memory. "
                "Keep only the main outcome, decision, or useful result. "
                "Use one short sentence only."
            ),
            max_tokens="64",
        )

        if not user_summary:
            user_summary = user_prompt.strip().replace("\n", " ")[:200]
        if not assistant_summary:
            assistant_summary = model_response.strip().replace("\n", " ")[:200]

        memory = chat.setdefault("memory", [])
        memory.append(
            {
                "user_summary": user_summary,
                "assistant_summary": assistant_summary,
            }
        )
        chat["memory"] = memory[-self.max_memory_items :]

    def _generate_chat_title(self, first_prompt: str) -> str:
        self.root.after(0, lambda: self._set_loading_base("Generating chat title"))
        title = self._summarise_text(
            text=first_prompt,
            instruction=(
                "Create a very short chat title based on this user prompt. "
                "Use 2 to 5 words only. "
                "Return title only, with no punctuation except hyphens if absolutely needed."
            ),
            max_tokens="16",
        )
        title = title.strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)
        return title[:60] or "New Chat"

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
            "Return only the requested output.\n\n"
            f"Text:\n{text}"
        )
        command = [
            self.python_executable,
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            str(local_model_path),
            "--prompt",
            summary_prompt,
            "--max-tokens",
            max_tokens,
        ]

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        except Exception:
            return ""

        if result.returncode != 0:
            return ""

        output = result.stdout.strip()
        if not output:
            return ""

        hidden_prefixes = (
            "==========",
            "Prompt:",
            "Generation:",
            "Peak memory:",
        )
        cleaned_lines = [
            line.strip()
            for line in output.splitlines()
            if line.strip() and not line.startswith(hidden_prefixes)
        ]
        return " ".join(cleaned_lines).strip()[:300]

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
            if model["runtime"] not in {"mlx-lm", "sentence-transformers"}:
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

        if model["runtime"] == "sentence-transformers":
            try:
                from huggingface_hub import snapshot_download
            except Exception:
                return "Install huggingface-hub to download the RAG model: pip install huggingface-hub"

            try:
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