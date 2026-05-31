<div align="center">

![Project Banner](assets/banner.png)

# klini

**A self-learning clinical diagnosis agent that runs entirely in your terminal.**

</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org) [![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-blueviolet?style=for-the-badge)](https://github.com/langchain-ai/langgraph)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Why `klini` Exists](#why-klini-exists)
4. [Architecture](#architecture)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Command Reference](#command-reference)
8. [How the Diagnostic Pipeline Works](#how-the-diagnostic-pipeline-works)
9. [Memory, Skills, and Learning](#memory-skills-and-learning)
10. [Supported Providers and Tooling](#supported-providers-and-tooling)
11. [Configuration](#configuration)
12. [Project Structure](#project-structure)
13. [Contributing](#contributing)
14. [Security and Safety](#security-and-safety)
15. [License](#license)
16. [What to Add](#what-to-add)

---

## Project Overview

`klini` is a CLI-first, terminal-native clinical reasoning assistant designed to support conversational intake and structured medical analysis in a local environment.

It is built for:
- clinicians and healthcare-focused developers who want a command-line clinical workflow
- users who need a self-contained system without frontend or browser dependencies
- those who want incremental learning behavior through local memory and skills

`klini` is not a replacement for licensed medical professionals. It is an AI-assisted clinical reasoning companion meant for structured exploratory workflows, hypothesis generation, and evidence-backed diagnostic reasoning.

---

## Key Features

- **Terminal-native UX**
  - 100% command-line interface
  - No frontend, no browser, no remote server required

- **Four-step clinical pipeline**
  - Analysis
  - Data scoring
  - Evidence verification
  - Final patient report

- **Gatekeeper supervision**
  - Every pipeline step is reviewed by a quality gate
  - Ensures safety, completeness, and clinical relevance

- **Local self-learning memory**
  - Session embeddings stored in ChromaDB
  - Learned skills stored as local markdown files
  - Future sessions retrieve relevant experience automatically

- **Multi-user support**
  - User-specific profiles, sessions, skills, and vector memory
  - Supports multiple clinician or patient identities within one installation

- **Flexible model support**
  - Local inference via Ollama-compatible models
  - Cloud inference via OpenRouter and multiple provider adapters
  - Pluggable provider abstraction for Ollama, Google, OpenAI, Anthropic, Cohere, Mistral, Groq, and Azure

- **Clinical tool integration**
  - Drug lookup
  - PubMed literature search
  - Web search
  - Symptom classifier
  - Semantic retrieval over prior sessions

- **Structured clinical output**
  - Candidate conditions
  - Red flag identification
  - Test and investigation recommendations
  - Evidence-backed final report

---

## Why `klini` Exists

The modern clinical decision support landscape often depends on web apps, dashboards, and proprietary systems. `klini` was created to deliver a lightweight, terminal-based alternative that:

- keeps patient context local
- scales from a laptop to a remote VM
- supports operators who favor CLI workflows
- enables experimentation with custom clinical pipelines
- preserves auditability through plain-text session and skill artifacts

This makes `klini` particularly useful for:
- clinician-developers prototyping AI-assisted workflows
- research teams building explainable prompt pipelines
- educators teaching structured reasoning with LLMs
- health-tech builders wanting a self-contained proof of concept

---

## Architecture

### High-level design

`klini` is organized into the following logical layers:

- `agent/cli.py`
  - CLI entrypoint built with `Typer`
  - Exposes commands for init, register, start, sessions, skills, profile, config, set, delete, and more

- `agent/config.py`
  - Defines directory layout and user-specific config objects
  - Loads active user, provider settings, model names, API keys, and clinical metadata

- `agent/main/router.py`
  - Abstracts the LLM provider layer
  - Supports multiple providers through a single unified interface

- `agent/main/state.py`
  - Captures session state, pipeline progress, tool results, gatekeeper decisions, and control flags

- `agent/steps/prompts.py`
  - Houses the structured prompt templates for Step 1, Step 2, Step 3, and the gatekeeper
  - Contains strict response formats and approval requirements

- `agent/main/graph.py`
  - Orchestrates the reasoning pipeline using LangGraph
  - Connects nodes, steps, and tool integration

- `agent/tools/*`
  - Encapsulates all external data sources and tool wrappers

---

## Installation

### Prerequisites

- Python 3.11+
- `pip`

### Install from PyPi

```bash
pip install klini
```

### First-time setup

```bash
klini init
klini register
```

`klini init` creates the following local runtime structure:
- `.klini/app_config.toml`
- `.klini/users/<username>/user_config.toml`
- `.klini/users/<username>/history.json`
- `.klini/users/<username>/skills/index.jsonl`
- `.klini/users/<username>/memory/chroma`

> If you are using Ollama for local inference, run `ollama pull <model>` before starting your first session.

`klini register` collects:
- username
- age
- sex
- provider settings
- model choices
- API token(s)
- any additional profile metadata

---

## Quick Start

### Start a session

```bash
klini start
```

The agent enters an interactive session, gathers symptoms and clinical history, then transitions into the structured 4-step diagnostic pipeline automatically.

### List sessions

```bash
klini sessions
```

### Resume a session

```bash
klini sessions "session title"
```

### List learned skills

```bash
klini skills
```

### View user profile

```bash
klini profile
```

---

## Command Reference

### `klini init`
Initializes the `.klini` workspace and user directories.

### `klini register`
Populates the active user's configuration and clinical metadata.

### `klini start`
Begins a new diagnostic session.

### `klini sessions`
Lists and optionally resumes previous sessions.

### `klini skills`
Lists available skills or shows skill details.

### `klini profile`
Displays the current `USER.md` patient profile.

### `klini config`
Shows current configuration values for the active user.

### `klini set`
Updates configuration values such as provider, model, gatekeeper, API key, and user metadata.

Example:

```bash
klini set --key provider --value ollama
klini set --key model --value phi3:mini
klini set --key gatekeeper --value phi3:mini
klini set --key openrouter_api_key --value sk-or-...
klini set --key age --value 32
```

### `klini delete`
Deletes a session, skill, user, or the entire local workspace.

Examples:

```bash
klini delete --mode session --name "session title"
klini delete --mode skill --name "skill title"
klini delete --mode user --name john
klini delete --mode all
```

---

## How the Diagnostic Pipeline Works

`klini` follows a rigid four-step clinical reasoning pipeline, each with a dedicated role and approval workflow.

### Step 1 - Analysis

- Performs the first clinical review
- Summarizes the patient presentation
- Breaks down symptoms
- Generates candidate conditions
- Identifies red flags
- Lists gaps in the clinical picture

### Step 2 - Data

- Scores candidate conditions with an ML classifier
- Assesses medications and drug interactions
- Searches prior sessions for relevant history
- Updates condition likelihoods based on data evidence

### Step 3 - Verification

- Verifies every condition with external evidence
- Uses PubMed, web search, and drug tools
- Confirms or challenges prior assumptions
- Refines differential diagnoses

### Step 4 - Diagnosis

- Produces the final clinical summary
- Recommends next steps and safety guidance
- Provides differential diagnoses
- Includes an AI-assisted disclaimer

### Gatekeeper oversight

A smaller, specialized LLM evaluates the output after each step to ensure:
- clinical completeness
- safety
- required content
- adherence to the correct reasoning format

The gatekeeper can approve, reject, or request revisions. It also allows backward movement when the agent decides it must revisit prior steps.

---

## Memory, Skills, and Learning

`klini` is built to learn from every session.

### Skill files

- Each completed session can produce a skill file
- Skill files capture patterns, reasoning, and verified evidence
- Stored under `.klini/users/<username>/skills/`
- Indexed in `index.jsonl`

### Vector memory

- Session transcripts are embedded into ChromaDB
- Local vector memory lives under `.klini/users/<username>/memory/chroma`
- Past sessions are searched semantically to inform future reasoning

### User profile persistence

- `USER.md` stores active clinical profile content
- Automatically updated with confirmed history, medications, allergies, and family history after each session

---

## Supported Providers and Tooling

`klini` supports a broad provider layer through `langchain` adapters:

- Ollama
- Google GenAI
- OpenAI
- Azure OpenAI
- Anthropic
- Cohere
- Mistral
- Groq

### Core dependencies

- `typer`
- `rich`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langgraph`
- `chromadb`
- `requests`
- `tomli-w`

### Tools currently used by the pipeline

- `classifier`
- `drug_lookup`
- `semantic_search`
- `pubmed`
- `web_search`

These tools are integrated to provide:
- probability scoring
- drug safety checks
- evidence retrieval
- past case recall

---

## Configuration

`klini` keeps config in TOML files under `.klini/`.

### Global app config
- `.klini/app_config.toml`
- stores the active user

### User config
- `.klini/users/<username>/user_config.toml`
- stores provider, model, gatekeeper, API keys, email, age, sex, and other per-user settings

### User profile
- `.klini/users/<username>/USER.md`
- stores ongoing patient profile and confirmed data

### Session history
- `.klini/users/<username>/history.json`
- tracks past sessions and metadata

---

## Project Structure

```text
klini/
+-- LICENSE
+-- pyproject.toml
+-- README.md
+-- agent/
-   +-- __init__.py
-   +-- cli.py
-   +-- config.py
-   +-- utils.py
-   +-- main/
-   -   +-- __init__.py
-   -   +-- edges.py
-   -   +-- graph.py
-   -   +-- router.py
-   -   +-- state.py
-   +-- memory/
-   -   +-- __init__.py
-   -   +-- chroma.py
-   +-- steps/
-   -   +-- __init__.py
-   -   +-- prompts.py
-   -   +-- nodes/
-   -   -   +-- gatekeeper.py
-   -   -   +-- preprocess.py
-   -   -   +-- profile_updater.py
-   -   -   +-- skill_updater.py
-   -   -   +-- step1.py
-   -   -   +-- step2.py
-   -   -   +-- step3.py
-   -   -   +-- step4.py
-   +-- tools/
-   -   +-- __init__.py
-   -   +-- drug_lookup.py
-   -   +-- ml_classifier.py
-   -   +-- pubmed.py
-   -   +-- semantic_search.py
-   -   +-- web_search.py
+-- assets/
-   +-- banner.png
```

---

## Contributing

Contributions are welcome and encouraged. Recommended areas of contribution include:

- improving the diagnostic prompt templates
- expanding supported tool integrations
- adding support for additional inference providers
- improving session persistence and memory retrieval
- enhancing the CLI experience
- adding automated tests and CI validation

If you want to contribute:
1. fork the repository
2. create a feature branch
3. open a pull request with a clear description

---

## Security and Safety

`klini` is explicitly designed as an AI-assisted reasoning tool, not a medical device.

Important safety principles:
- do not use for final clinical decision-making
- always verify with licensed providers
- never rely on it for medication dosing
- avoid using it with real patient identifiers unless your environment is HIPAA-compliant
- treat outputs as advisory and educational

---

## License

`klini` is released under the **MIT License**.

See `LICENSE` for full terms.

---