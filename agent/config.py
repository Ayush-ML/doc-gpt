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

# File Paths

SKILLS = r"agent\skills"
USERS = r"agent\users"
CHECKPOINTS = r"agent\memory\checkpoint.db"
INDEX = r"agent\skills\index.json"
HISTORY = r"agent\memory\history"

# Values

TEMPERATURE = 0.3
N_RESULTS = 5

# User Info

EMAIL = "ayushmanoj279@gmail.com"
OPENROUTER_API_KEY = "" # IF the provider is cloud based and requires an API key, Set its Corssponding API key
# Infermedica
INFERMEDICA_APP_ID = ""
INFERMEDICA_APP_KEY = ""
PARSE_URL = "https://api.infermedica.com/v3/parse"
DIAGNOSIS_URL = "https://api.infermedica.com/v3/diagnosis"
