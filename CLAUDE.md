# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimpleAgent TUI is a lightweight terminal user interface for interacting with local Ollama models. It is designed as a tiny local AI agent shell with:

- Chat model selection and persistence.
- Embeddings model selection and persistence.
- Streaming and non-streaming model responses.
- Collapsible thinking output using `Ctrl-T`.
- Markdown-style terminal rendering for replies, including headings, bold text, bullet lists, and responsive tables.
- Session compaction using embeddings so the app can keep prompts lightweight while still recalling relevant older context.

## Key Components

1. **`SimpleAgentTUI`** in `main.py`
   - Main application class.
   - Handles the TUI lifecycle, command parsing, display rendering, thinking toggle, session state, compacted memory, and model selection.

2. **`OllamaClient`** in `ollama.py`
   - Handles communication with the Ollama API.
   - Supports chat requests, model listing, and embeddings through Ollama.

## Common Commands

### Development

- Run the application:

  ```bash
  python main.py
  ```

### Usage

- Show help: `/help`
- Show or change chat model: `/model <model_name>`
- Select chat model from installed Ollama models: `/select-model`
- Show or change embeddings model: `/embedding <model_name>`
- Select embeddings model from installed Ollama models: `/select-embedding`
- List installed Ollama models: `/models`
- Set system prompt: `/system <prompt>`
- Reset persisted system prompt: `/reset-system`
- Enable streaming: `/stream`
- Disable streaming: `/no-stream`
- Show active and compacted history: `/history`
- Reset session history, compacted memory, and clear the terminal: `/reset`
- Show app info: `/about`
- Exit: `/exit`, `/quit`, or `/q`

> Note: `/clear` has been removed. Use `/reset` to clear both session state and visible terminal output.

## Configuration

The app stores persistent configuration in:

```text
~/.simpleagent-cli/config.json
```

Persisted values include:

- Chat model.
- Embeddings model.
- Ollama host.
- Custom system prompt.

Environment variables:

- `SIMPLEAGENT_MODEL`: Set the default chat model.
- `OLLAMA_HOST`: Set the Ollama server host, defaulting to `http://localhost:11434`.

Current default models:

- Chat model: `nemotron-3-nano:4b`
- Embeddings model: `ordis/jina-embeddings-v2-base-code:latest`

## Session Memory and Compaction

SimpleAgent does not rely on Ollama or the model to store memory. Memory is handled by the app.

Runtime memory layers:

1. **Recent raw messages**
   - Only the most recent raw messages are kept directly in `self.messages`.
   - This keeps prompts lightweight for small local models.

2. **Compacted memory items**
   - Older messages are compacted into memory items.
   - User + assistant pairs are combined into a single `turn` item when possible:

     ```text
     User asked: ...
     Assistant answered: ...
     ```

3. **Embedding retrieval**
   - Compacted memory items are embedded using the configured embeddings model.
   - For each new user prompt, the app embeds the latest prompt and retrieves relevant older memory items.
   - Retrieved memory is injected as background system context before the latest raw message.

Prompt structure:

```text
system prompt
compacted relevant memory selected by embeddings
latest raw message(s)
```

`/reset` clears both recent messages and compacted memory.

## Thinking Display

Some models emit thinking text inside thinking tags. SimpleAgent separates thinking text from visible reply text.

- `thinking_text`: accumulated model thinking content.
- `reply_text`: accumulated visible assistant reply.
- `Ctrl-T`: toggles collapsed/full thinking display.
- Thinking display should reuse shared formatting logic so streaming and toggled views stay consistent.

The thinking display is collapsed by default to keep the TUI readable.

## Markdown and Table Rendering

The TUI includes lightweight markdown rendering for assistant replies:

- Headings are styled with blue/cyan terminal colours.
- `**bold**` text is highlighted.
- Bullet and numbered lists are styled for readability.
- Markdown tables are rendered as framed TUI tables.
- Table columns use rotating pastel colours.
- Long table cells are wrapped inside their own columns so content stays aligned when the terminal width is smaller than the raw table width.

Table rendering should account for terminal width using `os.get_terminal_size()` and avoid overflowing the frame.

## Architecture

The project follows a simple separation of concerns:

1. **Terminal Interface (`main.py`)**
   - Prompt loop.
   - Commands.
   - Display formatting.
   - Chat history and compacted memory.
   - Model and embeddings model selection.

2. **Ollama Client (`ollama.py`)**
   - Chat API calls.
   - Streaming response handling.
   - Model listing.
   - Embedding API calls.

## Development Practices

- Keep the app lightweight and local-first.
- Preserve a clear split between TUI logic and Ollama API logic.
- Use type hints for clarity.
- Prefer shared formatting helpers over duplicated display logic.
- Keep command behaviour reflected in this file whenever commands change.
- Be careful with terminal rendering: cursor movement, flushing, and scrollback clearing can differ between terminals.

## Notes

- Ensure Ollama is running before starting the application.
- If model calls fail, verify the selected chat model exists in Ollama.
- If memory embedding fails, verify the selected embeddings model exists in Ollama.
- The app-side session history can grow heavy if not compacted; keep compaction logic active for small models.