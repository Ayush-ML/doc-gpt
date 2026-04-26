<div align="center">

![Banner](./banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-blueviolet?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Runs%20on-Ollama-black?style=for-the-badge)](https://ollama.ai)

**A self-learning clinical diagnosis agent that runs entirely in your terminal.**

No frontend. No browser. No server. Describe your symptoms, and klini walks them through a four-step diagnostic pipeline — analysis, data scoring, evidence verification, and a final patient report — backed by real clinical tools and peer-reviewed literature. After every session it writes a skill file and embeds the conversation into a local vector database so it gets sharper the more you use it.

[Installation](#installation) · [Usage](#usage) · [How it works](#how-it-works) · [Configuration](#configuration) · [Architecture](#architecture)

</div>

---

## Overview

|  |  |
| --- | --- |
| **Four-step diagnostic pipeline** | Analysis → data scoring → evidence verification → final report. Each step builds on the last. A gatekeeper LLM runs between every step and must approve the response before the agent can proceed. |
| **Structured skepticism** | The verification step is explicitly tasked with challenging and refuting claims from earlier steps using PubMed and web evidence — not confirming them. |
| **Self-learning memory** | After every session, klini writes a skill file to disk and embeds the full session into ChromaDB. Future sessions load relevant skills automatically and search past cases semantically. |
| **Runs locally or in the cloud** | Use Ollama with small quantized models (phi3:mini, gemma2:2b) on a CPU-only machine with no API key, or switch to OpenRouter for access to hundreds of cloud models with a single key. |
| **Real clinical tools** | DuckDuckGo web search, PubMed literature lookup, OpenFDA drug information, and Infermedica ML symptom classification — all free or free-tier, no extra accounts required beyond setup. |
| **Persistent patient profile** | A `USER.md` file tracks clinical history, medications, allergies, and family history across sessions. Updated automatically after every session with newly confirmed information. |

---

## Installation

**Prerequisites:** Python 3.11+, and either [Ollama](https://ollama.ai) (local, no API key) or an [OpenRouter](https://openrouter.ai) API key (cloud).

```bash
git clone https://github.com/Ayush-ML/doc-gpt
cd doc-gpt
pip install -e .
```

Run first-time setup:

```bash
agent init
```

This creates `~/.klini/` with the required directory structure, an empty patient profile, and walks you through provider configuration.

> **Ollama users:** Pull a model before your first session — `ollama pull phi3:mini`. klini works on CPU with no GPU required.

---

## Usage

### Starting a session

```bash
agent diagnose                    # use the default patient profile
agent diagnose --profile john     # use a named profile
```

klini opens a free-form conversation where it asks clarifying questions and builds clinical context. When it has enough information it emits a `<READY/>` signal and automatically begins the formal diagnostic pipeline — no command needed.

### Skills and memory

```bash
agent skills list                 # list all learned skill files
agent skills show <title>         # display a specific skill file
agent sessions list               # browse past session history
```

### Configuration

```bash
agent config set provider ollama       # switch to local inference
agent config set provider openrouter   # switch to cloud inference
agent config set model phi3:mini       # change the active model
```

---

## How it works

### Pre-session

Before the pipeline begins, klini runs a free-form conversation loop to gather symptoms and clinical context. When it determines it has enough information it signals readiness and the LangGraph pipeline starts — with the full conversation already loaded into state. No clarification tools are needed inside the pipeline.

### The pipeline

```
Preprocess ─────────────────────────────────────────────────────────────────
  Load skill index · Read patient profile · Semantic search past sessions
  │
  ▼
Step 1 · Analysis ──────────────────────────────────────────────────────────
  Phase A: select relevant skill files from index
  Phase B: candidate conditions, red flags, information gaps
  Tools: web_search · semantic_search
  │
[Gatekeeper]
  │
  ▼
Step 2 · Data ──────────────────────────────────────────────────────────────
  ML probability scores · Drug interaction analysis · Past session comparison
  Tools: ml_classifier · drug_lookup · semantic_search
  │
[Gatekeeper]
  │
  ▼
Step 3 · Verification ──────────────────────────────────────────────────────
  Every claim from Steps 1–2 challenged with external evidence
  Every condition requires at least one PubMed search
  Tools: web_search · pubmed · drug_lookup · semantic_search
  │
[Gatekeeper]
  │
  ▼
Step 4 · Diagnosis ─────────────────────────────────────────────────────────
  Final report written directly to the patient
  Primary diagnosis · differentials · red flags · recommended next steps
  No tools — pure synthesis of all prior steps
  │
[Gatekeeper]
  │
  ├── Profile Updater ─── rewrites USER.md with newly confirmed clinical info
  │
  └── Skill Writer ─────── writes skill.md · updates index.jsonl · embeds session to ChromaDB
```

### The gatekeeper

A separate, smaller LLM call runs between every step. It reads the agent's response and the reason it gave for ending, then returns an `approved` boolean with an explanation. If rejected, the agent retries up to `MAX_RETRIES` times before being forced forward. The agent can request to go back any number of steps at any time — backward movement is always approved without a gatekeeper check.

### Self-learning loop

```
Session ends
  └─ profile_updater   updates USER.md with confirmed new information
  └─ skill_writer      writes skill.md capturing diagnostic patterns learned
  └─ skill_writer      embeds the full session into ChromaDB

Next session
  └─ preprocess        loads updated skill index — new skill is available
  └─ preprocess        searches ChromaDB — this session is now searchable
  └─ Step 1            selects and reads relevant skills before analysis begins
```

Every session produces a skill file covering clinical patterns, diagnostic approach, key findings, common pitfalls, and verified evidence. The skill index grows over time, and the agent's reasoning improves for conditions it has encountered before.

---

## Tools

Each pipeline step binds only the tools it is permitted to use. Tool results flow back into the conversation as standard LangChain tool messages in a standard ReAct loop.

| Tool | Source | API key |
|------|--------|---------|
| Web search | DuckDuckGo | None |
| Literature search | PubMed via LangChain | Email address only |
| Drug information | OpenFDA | None |
| Symptom classification | Infermedica | Free tier · 100 calls/day |
| Semantic search | ChromaDB (local, embedded) | None |

---

## Configuration

Provider and model are set with `agent config set` and stored in `~/.klini/config.json`.

**Ollama — local, private, CPU-friendly**

```bash
ollama pull phi3:mini              # or gemma2:2b, llama3.2:3b, etc.
agent config set provider ollama
agent config set model phi3:mini
```

No API key. No data leaves your machine. Works on any laptop.

**OpenRouter — cloud, hundreds of models**

```bash
export OPENROUTER_API_KEY=sk-or-...
agent config set provider openrouter
agent config set model mistralai/mistral-7b-instruct
```

Free-tier models available. Full list at [openrouter.ai/models](https://openrouter.ai/models).

---

## Architecture

```
agent/
├── config.py               configuration, paths, API keys, prompts
├── utils.py                shared helpers
├── main/
│   ├── state.py            AgentState TypedDict — single source of truth
│   ├── router.py           LLM factory, returns correct client based on config
│   ├── graph.py            builds and compiles the LangGraph StateGraph
│   └── edges.py            routing logic after each gatekeeper decision
├── nodes/
│   ├── preprocess.py       loads skills, patient profile, ChromaDB results
│   ├── step1.py            analysis — two-phase LLM execution
│   ├── step2.py            data — ML classification and drug analysis
│   ├── step3.py            verification — evidence-based claim checking
│   ├── step4.py            diagnosis — final patient report
│   ├── gatekeeper.py       quality gate between every step
│   ├── profile_updater.py  rewrites USER.md after session ends
│   └── skill_writer.py     writes skill file, updates index, embeds to ChromaDB
├── tools/
│   ├── web_search.py       DuckDuckGo
│   ├── pubmed.py           PubMed via LangChain
│   ├── semantic_search.py  ChromaDB keyword search
│   ├── drug_lookup.py      OpenFDA
│   └── ml_classifier.py   Infermedica symptom classification
├── steps/
│   └── prompts.py          system prompts for all nodes
└── memory/
    └── chroma.py           ChromaDB client — write and search functions
```

**Data storage** — all persistent data lives under `~/.klini/`:

```
~/.klini/
├── skills/
│   ├── index.jsonl                      one line per skill: { title: summary }
│   ├── chest_pain_differential.md
│   └── diabetic_ketoacidosis_workup.md
├── users/
│   └── default/
│       └── USER.md                      patient clinical profile
└── memory/
    ├── chroma/                          ChromaDB vector store
    └── checkpoint.db                    LangGraph session checkpoints (SQLite)
```

**Stack**

| Component | Library |
|-----------|---------|
| Agent framework | LangGraph |
| LLM clients | LangChain — ChatOllama, ChatOpenAI |
| CLI | Typer + Rich |
| Vector database | ChromaDB (embedded, no server) |
| Session checkpoints | SQLite via LangGraph |

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first.

```bash
git clone https://github.com/Ayush-ML/doc-gpt
cd doc-gpt
pip install -e ".[dev]"
```

---

## Disclaimer

klini is an AI-assisted tool intended to support clinical reasoning. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Never disregard or delay seeking professional medical advice based on output from this tool. Always consult a qualified healthcare provider.

---

## License

MIT — see [LICENSE](LICENSE).