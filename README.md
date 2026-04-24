# SimpleAgent — (Low-Resource AI Agent Framework)

SimpleAgent is a lightweight local-first AI agent built around one core idea:

> Small models can become much more useful when they are given structure, memory, and carefully controlled tools.

This project experiments with running a small local Qwen model on Apple Silicon using MLX, then wrapping it with a simple agent system that can route tasks, call skills, search memory, scrape pages, and generate better responses without relying on a large cloud model.

---

## The Problem We Are Solving

Most small local models are good at simple chat, but weak at agent behaviour.

The biggest missing piece is reliable tool calling.

Large cloud models can usually:
- decide when to use tools
- output structured JSON
- browse or retrieve information
- remember context across turns
- recover from messy tool outputs

Small models struggle because they often:
- break JSON formats
- overuse tools when not needed
- fail to call tools when needed
- hallucinate tool results
- lose context quickly
- produce weaker answers when information is not directly in the prompt

SimpleAgent is designed to solve this by using a lightweight, controlled system around the model.

Instead of expecting a small model to behave like GPT-4, we give it guardrails.

---

## Core Design Thinking

### 1. Keep the model small

The main orchestrator model is intentionally small so it can run locally on consumer Apple Silicon hardware.

The goal is not to beat large cloud models directly. The goal is to maximise capability per unit of compute.

### 2. Use structure instead of model size

Rather than making the model responsible for everything, SimpleAgent separates responsibilities:

| Layer | Responsibility |
|---|---|
| GUI | Chat interface and user interaction |
| Orchestrator model | Reasoning, final responses, skill routing |
| Skills | External actions such as search, scraping, memory retrieval |
| Memory | Summarised context from previous turns |
| Parser | Converts messy model routing output into safe skill calls |

This makes the system easier to debug and more reliable.

### 3. Avoid fragile JSON tool calling

Small models often fail strict JSON output.

SimpleAgent currently uses integer-based skill routing instead:

```text
0 = no skill
1 = internet search
2 = scrape URL
3 = memory RAG
```

The model is prompted to return skill IDs, and the parser safely handles imperfect outputs.

Examples:

| Model output | Parsed result |
|---|---|
| `1` | internet search |
| `1,2` | internet search + scrape URL |
| `use internet_search` | internet search |
| `no skill needed` | no skill |
| random invalid numbers | ignored |

### 4. Use deterministic rules where possible

Some tool calls should not be left entirely to model discretion.

Examples:
- If a prompt contains a URL, use the URL scraping skill.
- If a prompt clearly asks for latest/current/news/price information, allow internet search.
- If a prompt references earlier discussion, use memory RAG.
- If a prompt is a normal follow-up like “what do you think”, avoid internet search.

This reduces unnecessary tool calls and improves answer quality.

### 5. Make the agent observable

SimpleAgent prints debug information to the console so we can inspect:
- raw model outputs
- skill router decisions
- selected skill IDs
- generated search queries
- scraped page content
- skill outputs
- final responses

The GUI also shows loading stages such as:
- Preparing context
- Routing skills
- Choosing tools
- Searching internet
- Scraping URL
- Searching memory
- Building prompt
- Generating answer
- Saving memory

This makes the agent easier to understand and improve.

---

## Current Skills

SimpleAgent currently supports the following skills.

| Skill ID | Skill | Purpose |
|---:|---|---|
| 0 | `no_skill` | Do not call any skill. Used for normal reasoning, writing, coding help, or conversation. |
| 1 | `internet_search` | Searches the web, ranks results, extracts snippets/page excerpts, and returns grounding context. |
| 2 | `scrape_url` | Extracts relevant information from one or more user-provided URLs. |
| 3 | `memory_rag` | Retrieves relevant past memory summaries using semantic search when the user references earlier conversation. |

---

## Skill Details

### Skill 1 — Internet Search

The internet search skill is used when the user explicitly needs current, external, or factual online information.

Flow:

```text
User prompt
↓
Model generates a concise search query
↓
Search engine results are fetched
↓
Top results are ranked
↓
Top pages are optionally fetched for deeper excerpts
↓
Relevant search context is appended to the final prompt
↓
Qwen answers using the retrieved context
```

This skill currently uses DuckDuckGo HTML search as a lightweight search source.

Search result ranking uses:
- MiniLM sentence-transformer embeddings when available
- keyword overlap fallback when the embedding model is unavailable

### Skill 2 — URL Scraping

The URL scraping skill is used when the user provides a link.

Flow:

```text
User provides URL
↓
URL is extracted from prompt
↓
Page is fetched
↓
JavaScript rendering is attempted using Playwright when available
↓
Fallback to lightweight urllib fetch if needed
↓
Page text is cleaned and condensed
↓
Relevant excerpt is appended to final prompt
```

This allows the agent to answer questions about specific webpages.

### Skill 3 — Memory RAG

The memory RAG skill retrieves relevant previous memory summaries.

Flow:

```text
User references earlier context
↓
Memory RAG skill is triggered
↓
Stored memory summaries are embedded
↓
Relevant memories are semantically ranked
↓
Top matching memory items are appended to prompt
↓
Qwen answers with past context
```

This improves follow-up answers like:
- “What do you think?”
- “Based on what we discussed earlier…”
- “Remind me what we decided.”
- “Why did we choose that approach?”

---

## Memory System

SimpleAgent has a lightweight summarised memory system.

After each assistant response, the model performs two internal summarisation steps:

1. Summarise the user prompt
2. Summarise the assistant response

The summaries are stored as memory pairs:

```json
{
  "user_summary": "...",
  "assistant_summary": "..."
}
```

The system keeps a rolling window of recent memory items.

Memory is used in two ways:

| Memory Layer | Behaviour |
|---|---|
| Recent memory injection | Recent summaries are automatically included in prompts. |
| Memory RAG skill | Relevant memories are retrieved when the user references past discussion. |

This keeps the agent lightweight while still giving it useful continuity.

---

## Why This Is Interesting

Offline local LLM apps already exist.

The unusual part of this project is not just offline chat.

The interesting part is:

> Low-resource tool calling for small local models.

SimpleAgent attempts to make small models behave more like agents by combining:
- structured routing
- deterministic rules
- robust parsing
- memory summaries
- semantic memory retrieval
- web search
- URL scraping
- visible debugging

This is a low-resource agent architecture rather than just a local chatbot.

---

## Architecture

Current high-level flow:

```text
User message
↓
Save message to local chat file
↓
Build recent memory context
↓
Decide which skills are needed
↓
Run selected skills
↓
Append skill outputs to prompt
↓
Generate final answer with local Qwen model
↓
Save assistant response
↓
Summarise turn into memory
↓
Update GUI
```

---

## Model Setup

Current primary model:

| Role | Model | Runtime |
|---|---|---|
| Orchestrator | Qwen2.5-3B-Instruct 4-bit | MLX |
| Code Specialist | Qwen2.5-Coder-7B-Instruct 4-bit | MLX |
| Vision Specialist | Qwen2.5-VL-3B-Instruct 4-bit | MLX-VLM |
| RAG Embedding | all-MiniLM-L6-v2 | sentence-transformers |

The orchestrator model handles:
- final response generation
- title generation
- memory summarisation
- skill routing
- search query generation

The MiniLM embedding model handles:
- web result ranking
- semantic memory retrieval

---

## Requirements

### Hardware

- Apple Silicon Mac: M1 / M2 / M3 / M4
- Recommended: 16GB RAM or more

### Software

- macOS
- Python 3.10+
- Git

### Python Packages

Core:

```bash
pip install mlx-lm
```

Recommended for RAG and model downloads:

```bash
pip install sentence-transformers huggingface-hub
```

Recommended for JavaScript-rendered webpage scraping:

```bash
pip install playwright
python -m playwright install chromium
```

---

## Setup

### 1. Clone project

```bash
git clone <your-repo-url>
cd simpleagent
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install mlx-lm sentence-transformers huggingface-hub
```

Optional for JavaScript website scraping:

```bash
pip install playwright
python -m playwright install chromium
```

### 4. Start app

```bash
python simple_agent.py
```

Use the menu option:

```text
Download Models
```

This downloads local models into:

```text
models/<model_key>/model/
```

---

## GUI Features

SimpleAgent includes a Tkinter-based desktop GUI.

Current GUI features:
- Local chat interface
- Individual chat files
- Auto-generated chat titles
- Markdown-like formatting
- Headings, bullets, bold, italic, code formatting
- Basic table rendering
- Clickable links
- Auto-expanding input box
- Loading/status stages
- Delete chats
- Model response token-size dropdown

---

## Local Files

Chats are stored locally under:

```text
chats/
```

Each chat is saved as an individual text/JSON-like file.

Models are stored locally under:

```text
models/
```

---

## Local-First Philosophy

SimpleAgent is designed to prioritise:

- local execution
- low resource usage
- clear architecture
- visible debugging
- hackability
- modular skills
- small-model capability

The goal is not to create the largest model.

The goal is to build a system where a small model can do more by using the right structure.

---

## Current Limitations

This is still an experimental prototype.

Known limitations:
- Search scraping can fail if sites block requests.
- JavaScript scraping requires Playwright.
- Memory RAG currently searches stored summaries, not a full vector database of all chat messages.
- Small models may still make weak routing or reasoning decisions.
- Web extraction is heuristic and may miss important information.
- Tkinter UI has visual limitations compared with modern web UI frameworks.

---

## Future Improvements

High-value future upgrades:

- Full chat-history vector memory
- FAISS or SQLite vector storage
- Better webpage extraction
- Multi-step planner/executor loop
- Skill chaining
- Confidence scoring for tool outputs
- Better table rendering
- Streaming model output
- Runtime abstraction for llama.cpp / GGUF on Windows
- Better packaging for macOS

---

## Design Philosophy

> Build intelligence through structure, not size.

SimpleAgent is an experiment in making small models useful by surrounding them with:
- memory
- tools
- routing
- retrieval
- debugging
- deterministic control

Small models alone are limited.

Small models inside a good system can become powerful.

---

## Git Ignore Reminder

Ensure this is in `.gitignore`:

```text
models/
.venv/
chats/
```

---

## Run

```bash
python simple_agent.py
```

Start experimenting.
