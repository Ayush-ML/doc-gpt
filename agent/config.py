import tomllib
from pathlib import Path

# Base Paths
BASE = Path(".klini")
USERS_DIR = BASE / "users"
APP_CONFIG = BASE / "app_config.toml"

# Path Helpers
def user_dir(username: str) -> Path:
    return USERS_DIR / username

def user_config(username: str) -> Path:
    return user_dir(username) / "user_config.toml"

def user_history(username: str) -> Path:
    return user_dir(username) / "history.json"

def user_profile(username: str) -> Path:
    return user_dir(username) / "USER.md"

def user_skills_dir(username: str) -> Path:
    return user_dir(username) / "skills"

def user_skills_index(username: str) -> Path:
    return user_skills_dir(username) / "index.jsonl"

def user_memory_dir(username: str) -> Path:
    return user_dir(username) / "memory"

def user_chroma(username: str) -> Path:
    return user_memory_dir(username) / "chroma"

def user_checkpoints(username: str) -> Path:
    return user_memory_dir(username) / "checkpoint.db"

# Load Active User from App Config
_app_data = {}
if APP_CONFIG.exists():
    with open(APP_CONFIG, "rb") as f:
        _app_data = tomllib.load(f)

ACTIVE_USER = _app_data.get("config", {}).get("active_user", "")

# Load User Config
_cfg = {}
_usr = {}
if ACTIVE_USER:
    _user_config_path = user_config(ACTIVE_USER)
    if _user_config_path.exists():
        with open(_user_config_path, "rb") as f:
            _data = tomllib.load(f)
            _cfg = _data.get("config", {})
            _usr = _data.get("user", {})

# Models
PROVIDER = _cfg.get("provider", "")
AGENT = _cfg.get("model", "")
GATEKEEPER = _cfg.get("gatekeeper", "")
API_KEY = _cfg.get("api_key", "")
EMAIL = _usr.get("email", "")


# Values
TEMPERATURE = 0.3
N_RESULTS = 5
MAX_RETRIES = 3

# User Info
AGE = _usr.get("age", "")
SEX = _usr.get("sex", "")