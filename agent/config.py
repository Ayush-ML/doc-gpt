# This is a Python Script That contains all values that are needed for The Agent
# For Example:
    # Models
    # File Paths
    # Values
    # etc

# Models

PROVIDER = "" # Can be either Local or Cloud, Recommended - Llama: Local, Openrouter: Cloud
AGENT = "" # Main Agent Model
GATEKEEPER = "" # Gatekeeper for end_response, keep as same as agent or choose different
OLLAMA_BASE_URL = "http://localhost:11434"  # The base URL for the Ollama Model
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = "" # IF the provider is cloud based and requires an API key, Set its Corssponding API key

# File Paths

SKILLS = r"agent\skills"
USERS = r"agent\users"
CHECKPOINTS = r"agent\memory\checkpoint.db"
INDEX = r"agent\skills\index.json"
HISTORY = r"agent\memory\history"

# Values

TEMPERATURE = 0.3

# Prompts

STEP_1_PHASE_1 = """

You are a clinical skill selector. Your only job is to read a list of available skills and select the ones relevant to the patient's case.

## Instructions
- You will be given a dictionary of skills in the format { title: summary }
- You will be given the patient's message
- Read each skill title and summary carefully
- Select NOT the skills that are directly relevant to the symptoms or context described
- Return ONLY a plain list of selected skill titles
- Do NOT explain your choices
- Do NOT return anything else
- If no skills are relevant, return the word NONE

## Example Output

[chest_pain_differential, hypertensive_crisis_management, diabetic_workup]

The title of the skill and the Skill name given in the List Should ALWAYS match EXACTLY

"""

