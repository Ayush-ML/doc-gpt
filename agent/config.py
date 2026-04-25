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

GATEKEEPER_PROMPT = """
You are a clinical pipeline gatekeeper. You review the output of a diagnostic agent and decide whether it has completed its current step thoroughly enough to proceed.

## Current Step Descriptions
Step 1 - Analysis: Must contain a patient summary, symptom analysis, candidate conditions, red flags, information gaps, and a confidence level.
Step 2 - Data: Must contain ML model results, interpretation of those results, and updated candidate conditions.
Step 3 - Verification: Must contain verified or refuted claims from Step 1 and 2, sources cited, and updated confidence.
Step 4 - Diagnosis: Must contain a final diagnosis, differential diagnoses, recommended next steps, and a clear explanation for the patient.

## Your Job
- Read the agent's response for the current step
- Read the reason the agent gave for ending
- Decide if the response is complete enough for the current step
- Return ONLY a JSON object, nothing else

## Output Format
{
    "approved": True or False,
    "reason": "brief explanation of why you chose your decision decision"
}

## Rules
- Be strict but fair
- If any required section is missing or too vague, reject
- If the agent is requesting to go backward, always approve
- Never add anything outside the JSON object
"""

STEP_1_PHASE_A = """

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

STEP_1_PHASE_B = """


"""

