# This is a Python Script That contains all values that are needed for The Agent
# For Example:
    # Models
    # File Paths
    # Values
    # etc
# Imported Libraries

import tomllib
from pathlib import Path

# Load User Config Data

USER_CONFIG = r"agent\user_config.toml"
USER_CONFIG = Path(USER_CONFIG)
_data = {}
if USER_CONFIG.exists():
    with open(USER_CONFIG, "rb") as f:
        _data = tomllib.load(f)
_cfg = _data.get("config", {})
_usr = _data.get("user", {})

# Models

PROVIDER = _cfg.get("provider", "")
AGENT = _cfg.get("model", "")
GATEKEEPER = _cfg.get("gatekeeper", "")
OLLAMA_BASE_URL = "http://localhost:11434"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = _cfg.get("openrouter_api_key", "")

# File Paths

SKILLS = r"agent\skills"
USERS = r"agent\users"
MEMORY = r"agent\memory"
INDEX = r"agent\skills\index.jsonl"
HISTORY = r"agent\memory\history.json"
CHROMA = r"agent\memory\chroma"
CHECKPOINTS = r"agent\memory\checkpoint.db"

# Values

TEMPERATURE = 0.3
N_RESULTS = 5
MAX_RETRIES = 3

# User Info

EMAIL = _cfg.get("email", "")
AGE = _usr.get("age", "")
SEX = _usr.get("sex", "")
INFERMEDICA_APP_ID = _cfg.get("infermedica_app_id", "")
INFERMEDICA_APP_KEY = _cfg.get("infermedica_app_key", "")
PARSE_URL = "https://api.infermedica.com/v3/parse"
DIAGNOSIS_URL = "https://api.infermedica.com/v3/diagnosis"
ACTIVE_USER = _usr.get("active_user", "default")

