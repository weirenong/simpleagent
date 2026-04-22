from __future__ import annotations

import json
import subprocess
import shutil
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


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
        "key": "qwen2.5-coder-1.5b-instruct",
        "id": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        "category": "specialist",
        "runtime": "mlx-lm",
        "specialty": "Specialised model for code generation and understanding.",
        "why_selected": "MLX-ready 4-bit coding specialist suitable for lightweight local code tasks on Apple Silicon.",
        "download_url": "https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit/resolve/main/README.md",
    },
    {
        "key": "qwen2.5-math-1.5b-instruct",
        "id": "mlx-community/Qwen2.5-Math-1.5B-Instruct-4bit",
        "category": "specialist",
        "runtime": "mlx-lm",
        "specialty": "Specialised model for mathematical reasoning and problem solving.",
        "why_selected": "MLX-ready 4-bit maths specialist that stays compact for local reasoning experiments.",
        "download_url": "https://huggingface.co/mlx-community/Qwen2.5-Math-1.5B-Instruct-4bit/resolve/main/README.md",
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
    running: bool = True
    mode: str = "menu"
    memory: list[dict[str, str]] = field(default_factory=list)

    def add_log(self, level: str, message: str) -> None:
        self.logs.append(
            LogEntry(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                level=level,
                message=message,
            )
        )


class SimpleAgentCLI:
    def __init__(self) -> None:
        self.state = AgentState()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.active_prompt_model_key = "qwen2.5-3b-instruct"
        self.python_executable = sys.executable or "python"
        os.environ.setdefault("HF_HOME", str(self.models_dir / ".hf_cache"))
        self.debug = True # set to False to hide debug messages
        self.max_memory_items = 5
        self.state.add_log("INFO", "Simple Agent CLI initialised.")
        self.state.add_log("INFO", "Type a command and press Enter. Type 'help' for options.")

    def run(self) -> None:
        self._print_welcome()

        while self.state.running:
            try:
                command = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self.state.add_log("INFO", "Shutting down agent.")
                self.state.running = False
                break

            if not command:
                continue

            self.state.status = "Processing"
            self.state.add_log("USER", command)
            if self.state.mode == "prompt":
                self._handle_prompt_input(command)
            else:
                self._process_command(command)
            self.state.status = "Idle"

    def _print_welcome(self) -> None:
        print("=" * 72)
        print("Simple Agent CLI (Apple Silicon MLX-first)")
        print("Commands: help | models | download | delete model <number|key> | prompt | quit")
        print("=" * 72)
        self._print_logs()

    def _print_logs(self) -> None:
        for entry in self.state.logs:
            print(f"[{entry.timestamp}] {entry.level}: {entry.message}")

    def _print_log_entry(self, entry: LogEntry) -> None:
        if not self.debug:
            if entry.level == "MODEL":
                hidden_prefixes = (
                    "==========",
                    "Prompt:",
                    "Generation:",
                    "Peak memory:",
                )
                if entry.message.startswith(hidden_prefixes):
                    return
                print(entry.message)
                return

            # keep system logs visible
            print(f"[{entry.timestamp}] {entry.level}: {entry.message}")
            return

        print(f"[{entry.timestamp}] {entry.level}: {entry.message}")

    def _process_command(self, command: str) -> None:
        lowered = command.lower()

        if lowered == "help":
            self._print_help()
        elif lowered == "models":
            self._list_models()
        elif lowered == "download":
            self._download_all_models()
        elif lowered.startswith("delete model"):
            self._delete_model_command(command)
        elif lowered == "prompt":
            self._enter_prompt_mode()
        elif lowered == "quit":
            self._add_and_print("INFO", "Quit command received.")
            self.state.running = False
        else:
            self._add_and_print(
                "ERROR",
                f"Unknown command: {command}. Type 'help' to see available commands.",
            )

    def _print_help(self) -> None:
        self._add_and_print("INFO", "Available commands:")
        self._add_and_print("INFO", "- help")
        self._add_and_print("INFO", "- models")
        self._add_and_print("INFO", "- download  (pre-download local MLX models)")
        self._add_and_print("INFO", "- delete model <number|key>")
        self._add_and_print("INFO", "- prompt")
        self._add_and_print("INFO", f"- memory: automatic summaries of the last {self.max_memory_items} prompt/response pairs")
        self._add_and_print("INFO", "- quit")

    def _list_models(self) -> None:
        self._add_and_print("INFO", "Committed MLX model selection:")

        installed_keys = self._installed_model_keys()
        for index, model in enumerate(MODEL_LIBRARY, start=1):
            status = "installed" if model["key"] in installed_keys else "not installed locally"
            self._add_and_print(
                "INFO",
                (
                    f"{index}. {model['id']} | key: {model['key']} | "
                    f"type: {model['category']} | runtime: {model['runtime']} | status: {status}"
                ),
            )
            self._add_and_print("INFO", f"   speciality: {model['specialty']}")
            self._add_and_print("INFO", f"   why selected: {model['why_selected']}")

    def _enter_prompt_mode(self) -> None:
        self.state.mode = "prompt"
        self._add_and_print("INFO", "Entered prompt mode.")
        self._add_and_print("INFO", "Type /exit to return to the main menu.")
        self._add_and_print(
            "INFO",
            f"Prompt model: {self.active_prompt_model_key}",
        )
        self._add_and_print(
            "INFO",
            f"Memory items stored: {len(self.state.memory)}/{self.max_memory_items}",
        )

    def _handle_prompt_input(self, command: str) -> None:
        if command.strip() == "/exit":
            self.state.mode = "menu"
            self._add_and_print("INFO", "Exited prompt mode and returned to the main menu.")
            return

        self._run_prompt(command)

    def _run_prompt(self, prompt: str) -> None:
        model = self._resolve_model(self.active_prompt_model_key)
        if model is None:
            self._add_and_print("ERROR", f"Prompt model not found: {self.active_prompt_model_key}")
            return

        if model["runtime"] != "mlx-lm":
            self._add_and_print(
                "ERROR",
                f"Prompt mode currently supports mlx-lm text models only. Active model runtime: {model['runtime']}",
            )
            return

        model_dir = self.models_dir / model["key"]
        metadata_path = model_dir / "metadata.json"
        local_model_path = model_dir / "model"
        if not metadata_path.exists() or not local_model_path.exists():
            self._add_and_print(
                "ERROR",
                "Prompt model is not downloaded locally yet. Run 'download' first.",
            )
            return

        if self.debug:
            self._add_and_print("INFO", f"Prompting local model {model['id']}...")

        memory_block = self._build_memory_block()
        final_prompt = prompt if not memory_block else f"{memory_block}\n\nCurrent user prompt:\n{prompt}"

        command = [
            self.python_executable,
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            str(local_model_path),
            "--prompt",
            final_prompt,
            "--max-tokens",
            "256",
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
        except FileNotFoundError:
            self._add_and_print(
                "ERROR",
                "Python executable was not found while trying to run mlx_lm.generate.",
            )
            return
        except Exception as exc:
            self._add_and_print("ERROR", f"Prompt execution failed: {exc}")
            return

        if result.returncode != 0:
            error_output = result.stderr.strip() or result.stdout.strip() or "Unknown MLX error"
            self._add_and_print("ERROR", f"Prompt execution failed: {error_output}")
            return

        output = result.stdout.strip()
        if not output:
            self._add_and_print("ERROR", "Prompt execution returned no output.")
            return

        visible_lines: list[str] = []
        hidden_prefixes = (
            "==========",
            "Prompt:",
            "Generation:",
            "Peak memory:",
        )
        for line in output.splitlines():
            self._add_and_print("MODEL", line)
            if not line.startswith(hidden_prefixes):
                visible_lines.append(line)

        visible_response = "\n".join(visible_lines).strip()
        if visible_response:
            self._remember_turn(prompt, visible_response)

    def _build_memory_block(self) -> str:
        if not self.state.memory:
            return ""

        lines = ["Recent memory:"]
        for index, item in enumerate(self.state.memory, start=1):
            lines.append(f"{index}. User summary: {item['user_summary']}")
            lines.append(f"   Assistant summary: {item['assistant_summary']}")
        return "\n".join(lines)

    def _remember_turn(self, user_prompt: str, model_response: str) -> None:
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

        self.state.memory.append(
            {
                "user_summary": user_summary,
                "assistant_summary": assistant_summary,
            }
        )
        self.state.memory = self.state.memory[-self.max_memory_items :]

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
            "Return only the summary sentence.\n\n"
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

    def _download_all_models(self) -> None:
        self._add_and_print("INFO", "Downloading all models locally in MLX format...")
        for model in MODEL_LIBRARY:
            self._download_single_model(model)


    def _download_single_model(self, model: dict[str, str]) -> None:
        model_dir = self.models_dir / model["key"]
        model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = model_dir / "model"
        metadata_path = model_dir / "metadata.json"

        if metadata_path.exists() and local_model_path.exists():
            self._add_and_print("INFO", f"Model already downloaded locally: {model['id']}")
            return

        self._add_and_print("INFO", f"Downloading local MLX model for {model['id']}...")

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
            self._add_and_print(
                "ERROR",
                "Python executable was not found while trying to run mlx_lm.convert.",
            )
            return
        except Exception as exc:
            self._add_and_print("ERROR", f"Download failed: {exc}")
            return

        if result.returncode != 0:
            if local_model_path.exists():
                shutil.rmtree(local_model_path, ignore_errors=True)
            error_output = result.stderr.strip() or result.stdout.strip() or "Unknown MLX conversion error"
            self._add_and_print("ERROR", f"Download failed: {error_output}")
            return

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

        self._add_and_print("INFO", f"Downloaded locally: {model['id']}")
        self._add_and_print("INFO", f"Stored model in: {local_model_path}")
        self._add_and_print("INFO", f"Runtime for this model: {model['runtime']}")

    def _delete_model_command(self, command: str) -> None:
        parts = command.split(maxsplit=2)
        if len(parts) < 3:
            self._add_and_print(
                "ERROR",
                "Usage: delete model <number|key>. Run 'models' to see choices.",
            )
            return

        selector = parts[2].strip()
        model = self._resolve_model(selector)
        if model is None:
            self._add_and_print("ERROR", f"Unknown model selection: {selector}")
            return

        model_dir = self.models_dir / model["key"]
        if not model_dir.exists():
            self._add_and_print("ERROR", f"Model is not downloaded: {model['id']}")
            return

        shutil.rmtree(model_dir)
        self._add_and_print("INFO", f"Deleted local model files for: {model['id']}")

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

    def _installed_model_keys(self) -> set[str]:
        installed: set[str] = set()
        for model in MODEL_LIBRARY:
            metadata_path = self.models_dir / model["key"] / "metadata.json"
            local_model_path = self.models_dir / model["key"] / "model"
            if metadata_path.exists() and local_model_path.exists():
                installed.add(model["key"])
        return installed


    def _add_and_print(self, level: str, message: str) -> None:
        self.state.add_log(level, message)
        self._print_log_entry(self.state.logs[-1])


def main() -> None:
    app = SimpleAgentCLI()
    app.run()


if __name__ == "__main__":
    main()