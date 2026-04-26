# SimpleAgent — Lightweight Local AI Agent

<p align="center">
  <img src="simpleagent_logo.png" alt="SimpleAgent screenshot" width="300">
</p>

SimpleAgent is a local-first desktop AI agent built for Apple Silicon.

It runs small local models with MLX and gives them a practical agent system around them: memory, skills, file attachments, web search, vision analysis, text-file reading, and a simple GUI.

The goal is simple:

> Make small local models more useful by giving them structure, tools, and context.

![SimpleAgent screenshot](screenshot.png)

---

## What SimpleAgent Can Do

SimpleAgent is more than a local chatbot.

It can:

- chat with a local Qwen model
- use a thinking model for stronger reasoning
- search the web when needed
- scrape user-provided URLs
- remember recent conversation summaries
- read attached text and code files
- analyse attached images and videos with a vision model
- paste images directly from clipboard as attachments
- drag and drop files into the chat
- pin attachments so they stay available across prompts
- display debug logs and raw model outputs in a console window
- show response time and token allowance after each reply
- render markdown-style formatting in the GUI

---

## Why This Exists

Small local models are useful, but they often struggle with agent behaviour.

They may:

- forget context quickly
- call tools when they should not
- fail to call tools when they should
- break strict JSON formats
- hallucinate when they lack information
- struggle with messy tool results

SimpleAgent avoids relying on fragile complex tool calling.

Instead, it uses a controlled skill-routing system, simple IDs, deterministic rules, memory summaries, and clear debugging output.

This makes a small model behave more like a practical assistant without needing a large cloud model.

---

## Main Features

### Local Chat

SimpleAgent runs a local Qwen model through MLX.

The current main model is a Qwen3 thinking model, which gives better reasoning than a normal instruction model.

It is used for:

- normal chat
- reasoning
- coding help
- final response generation
- skill routing
- search query generation
- chat titles
- memory summaries

Thinking output is handled safely. The app keeps the final answer clean instead of showing the full thinking trace in the chat.

---

### Date and Time Awareness

Every main prompt includes the current local date and time.

This helps the model understand phrases like:

- today
- tomorrow
- yesterday
- this week
- this month
- this year
- latest
- current

---

### Skill Routing

SimpleAgent uses numbered skills instead of complex JSON tool calls.

| Skill ID | Skill | What it does |
|---:|---|---|
| 0 | `no_skill` | Normal answer without tools |
| 1 | `internet_search` | Search online for current or factual information |
| 2 | `scrape_url` | Read and extract information from a URL |
| 3 | `memory_rag` | Search previous memory summaries |
| 4 | `attachment_vision` | Analyse attached images or videos |
| 5 | `text_file_reader` | Read attached text, code, markdown, config, CSV, JSON, SQL, HTML, CSS, shell, YAML, and similar files |

The router can select one or more skills before generating the final answer.

Example:

```text
User asks for latest news
↓
Skill 1 runs internet search
↓
Search results are added to the prompt
↓
The model answers using the search context
```

---

### Internet Search

The internet search skill is used when the user asks for current, latest, factual, or online information.

It can be triggered by prompts like:

- search online
- latest news
- current price
- look this up
- what is happening today
- recent updates

The search flow is:

```text
User prompt
↓
Generate search query
↓
Search online
↓
Rank results
↓
Extract relevant snippets
↓
Add search context to final prompt
↓
Generate answer
```

Search result ranking can use MiniLM embeddings when available, with a keyword fallback.

---

### URL Scraping

If the user provides a URL, SimpleAgent can scrape it and use the page content as context.

The flow is:

```text
User gives URL
↓
Extract URL
↓
Fetch page
↓
Use Playwright when JavaScript rendering is needed
↓
Clean page text
↓
Add relevant excerpt to prompt
```

This is useful for:

- summarising webpages
- extracting important points
- reading documentation pages
- asking questions about a specific link

---

### Memory

SimpleAgent stores lightweight memory after each turn.

After every response, it creates:

```json
{
  "user_summary": "...",
  "assistant_summary": "..."
}
```

These summaries are used in two ways:

| Memory type | Purpose |
|---|---|
| Recent memory injection | Adds recent summaries to the prompt automatically |
| Memory RAG | Searches older summaries when the user refers to previous discussion |

This gives the assistant continuity without storing huge full-history prompts.

---

### Attachments

SimpleAgent supports attachments through:

- drag and drop
- file picker
- image paste from clipboard

Attached files appear as chips in the composer.

Each attachment can be:

- removed with `X`
- pinned for reuse in the next prompt

Pinned attachments stay attached after sending a message. Unpinned attachments are cleared after the message is sent.

---

### Image and Video Attachments

Images and videos are routed to the vision skill.

Supported visual extensions include:

```text
.png, .jpg, .jpeg, .webp, .bmp, .gif,
.mp4, .mov, .avi, .mkv, .webm
```

The vision flow is:

```text
Attach image or video
↓
Skill 4 selected
↓
Qwen2.5-VL analyses the file
↓
Vision summary is added to the main prompt
↓
Qwen3 answers using that visual context
```

This can be used for:

- screenshots
- UI images
- charts
- visual documents
- candlestick chart screenshots
- pasted clipboard images

The vision model is run in a separate subprocess, so it is released from memory after analysis finishes.

---

### Text and Code File Attachments

Text-like files are routed to the text file reader skill.

Supported examples include:

```text
.txt, .md, .markdown, .rst, .log,
.py, .js, .ts, .html, .css, .sql,
.json, .yaml, .yml, .toml, .ini,
.csv, .tsv,
.sh, .bash, .zsh,
.java, .cpp, .cs, .go, .rs, .php,
.env, .gitignore, .dockerignore
```

The text file flow is:

```text
Attach text/code file
↓
Skill 5 selected
↓
File content is read locally
↓
Content is added to prompt
↓
Model answers using the file content
```

This makes it possible to attach multiple scripts and ask the model to:

- explain code
- find bugs
- suggest refactors
- compare files
- write new functions
- propose patch-style edits

---

### Attachment Pinning

Attachments can be pinned.

This is useful when you want to keep asking questions about the same file.

Example:

```text
Attach simple_agent.py
Pin it
Ask: explain this file
Ask: where should I add PDF support?
Ask: suggest a cleaner module split
```

The file stays available until you unpin or remove it.

---

### Temporary Attachments

Pasted clipboard images are saved into:

```text
temp_attachments/
```

The app includes a menu option to clear these temporary files.

This helps prevent pasted images from piling up over time.

---

### Console Output Window

SimpleAgent has a GUI console window.

It shows useful debugging information such as:

- raw model output
- skill router decisions
- selected skill IDs
- skill inputs
- skill outputs
- model errors
- response logs

This makes the agent easier to debug without relying only on the PyCharm terminal.

---

### Markdown Formatting

The chat display supports common markdown-style formatting:

- headings
- bullets
- numbered lists
- bold text
- italic text
- inline code
- code blocks
- blockquotes
- horizontal dividers
- basic tables
- clickable links

This makes model responses easier to read in the GUI.

---

### Response Controls

The GUI includes response token options.

This is useful because thinking models can use many tokens before producing the final answer.

The app also shows response time after each reply, for example:

```text
Response generated. Tokens used allowance: 32768 • Time: 46.9s
```

---

## Model Setup

Current model roles:

| Role | Model | Runtime |
|---|---|---|
| Main reasoning model | Qwen3-4B-Thinking-2507 MLX 4-bit | `mlx-lm` |
| Vision model | Qwen2.5-VL-3B-Instruct MLX 4-bit | `mlx-vlm` |
| Embedding model | all-MiniLM-L6-v2 | `sentence-transformers` |

The main model handles chat and reasoning.

The vision model handles image and video attachments.

The embedding model helps with semantic ranking for memory and search results.

---

## Requirements

### Hardware

Recommended:

- Apple Silicon Mac: M1, M2, M3, or M4
- 16GB RAM or more

### Software

- macOS
- Python 3.10+
- Git

### Python packages

Core:

```bash
pip install mlx-lm
```

Recommended:

```bash
pip install sentence-transformers huggingface-hub
```

For vision attachments:

```bash
pip install mlx-vlm pillow torch torchvision
```

For drag and drop:

```bash
pip install tkinterdnd2
```

For webpage scraping with JavaScript rendering:

```bash
pip install playwright
python -m playwright install chromium
```

---

## Setup

### 1. Clone the project

```bash
git clone <your-repo-url>
cd simpleagent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install mlx-lm mlx-vlm sentence-transformers huggingface-hub pillow tkinterdnd2 torch torchvision
```

Optional for webpage scraping:

```bash
pip install playwright
python -m playwright install chromium
```

### 4. Run the app

```bash
python simple_agent.py
```

### 5. Download models

Use the app menu:

```text
File → Download Models
```

Models are saved under:

```text
models/<model_key>/model/
```

---

## Local Folders

SimpleAgent stores local data in these folders:

| Folder | Purpose |
|---|---|
| `chats/` | Saved chat files |
| `models/` | Downloaded local models |
| `temp_attachments/` | Pasted clipboard images and temporary attachments |

Recommended `.gitignore` entries:

```text
models/
.venv/
chats/
temp_attachments/
```

---

## How The Agent Works

High-level flow:

```text
User sends message
↓
Save user message
↓
Prepare date/time context
↓
Prepare recent memory context
↓
Route skills
↓
Run selected skills
↓
Read or analyse attachments
↓
Build final prompt
↓
Generate answer with local model
↓
Save assistant response
↓
Summarise turn into memory
↓
Update GUI
```

This structure lets a small model behave more like an agent.

---

## Design Philosophy

SimpleAgent follows a few simple principles:

### 1. Local first

The app is designed to run locally on a Mac.

### 2. Small model, better system

Instead of relying only on a huge model, SimpleAgent gives a small model better context and tools.

### 3. Use deterministic logic where possible

File extensions, URLs, and obvious tool triggers should be handled by code, not guessed by the model.

### 4. Make everything visible

The console window shows what the agent is doing internally.

This makes debugging much easier.

### 5. Keep it hackable

The project is intentionally simple enough to modify and extend.

---

## Current Limitations

SimpleAgent is still experimental.

Known limitations:

- Web search quality depends on available search results.
- Some websites may block scraping.
- JavaScript-heavy pages need Playwright.
- Vision analysis requires the correct MLX-VLM model and dependencies.
- Text attachments can become very large and slow down responses.
- The model can suggest code changes, but automatic patch application is not yet part of the core workflow.
- PDF, DOCX, XLSX, and PPTX files are not deeply parsed yet unless converted or handled through future skills.

---

## Run

```bash
python simple_agent.py
```

Start chatting, attach files, search online, and experiment.
