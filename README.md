# SimpleAgent (Local MLX Agent)

A lightweight local-first AI agent framework built to experiment with **small models, orchestration, and specialised delegation** using Apple Silicon (MLX).

---

## 🧠 Overview

This project is designed to:
- Run **fully locally** (no cloud inference)
- Use **small Qwen models** in MLX format
- Explore **multi-agent orchestration** using a primary model + specialists
- Stay lightweight and hackable

---

## ⚙️ Requirements

### Hardware
- Apple Silicon Mac (M1 / M2 / M3)
- Recommended: **16GB RAM or more**

### Software
- macOS
- Python **3.10+**
- Git

---

## 🧱 Setup Instructions

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
pip install mlx-lm
```

(Optional but recommended)
```bash
pip install -U pip
```

### 4. Verify installation
```bash
python -m mlx_lm.generate --help
```

---

## 📦 Model Setup

Models are downloaded and converted locally using MLX.

Run:
```bash
python simple_agent.py
```

Then inside CLI:
```text
download
```

This will:
- Download Hugging Face model
- Convert it to MLX format
- Store locally under:

```
models/<model_key>/model/
```

---

## 🚀 Usage

### Start CLI
```bash
python simple_agent.py
```

### Available commands
| Command | Description |
|--------|------------|
| `help` | Show commands |
| `models` | List available models |
| `download` | Download all models locally (MLX format) |
| `delete model <id>` | Delete a model |
| `prompt` | Enter prompt mode |
| `quit` | Exit |

---

## 🧠 Memory System

The agent includes a lightweight **short-term memory system**.

### How it works
- After each prompt, the model runs **2 internal summarisation steps**:
  - Summarises the user prompt
  - Summarises the model response
- These summaries are **not shown to the user**
- Stored as memory pairs in the format:
  - User summary
  - Assistant summary
- The system keeps up to **5 memory items** (rolling window)

### How memory is used
- Memory is automatically injected into every new prompt
- Format:

```text
Recent memory:
1. User summary: ...
   Assistant summary: ...

Current user prompt:
...
```

### Notes
- Memory starts working after the first interaction
- Designed to be lightweight for small models
- Summaries may be imperfect (small model limitation)

---
---

## 💬 Prompt Mode

Enter prompt mode:
```text
prompt
```

Exit prompt mode:
```text
/exit
```

All prompts use:
```
Qwen2.5-3B-Instruct (MLX, local)
```

- Automatically uses memory context from previous interactions

---

## 🧠 Architecture

Current model setup:

| Role | Model |
|-----|------|
| Orchestrator | Qwen2.5-3B-Instruct (MLX 4-bit) |
| Code Specialist | Qwen2.5-Coder-1.5B |
| Math Specialist | Qwen2.5-Math-1.5B |
| Vision Specialist | Qwen2.5-VL-3B |

All models are:
- Local
- MLX-optimised
- 4-bit quantised (via mlx-community)

---

## 🔒 Local-Only Guarantee

This project enforces:
- No cloud inference
- No API calls
- Offline inference only

Environment variables used:
```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

---

## ⚠️ Notes

- First download may take time (model conversion)
- Models consume RAM (3B ≈ ~2GB)
- Vision model may require separate runtime support later
 - Each prompt runs the model up to 3 times (response + 2 memory summaries)

---

## 🧩 Future Improvements

- Multi-agent orchestration loop
- Tool calling
- Streaming outputs
- Better model routing

---

## 💡 Philosophy

> Build intelligence through structure, not size.

Small models + good orchestration = powerful systems.

---

## 🧼 Git Ignore Reminder

Ensure this is in your `.gitignore`:
```
models/
.venv/
```

---

## 🏁 You're Ready

```bash
python simple_agent.py
```

Start experimenting 🚀
