# This is a Script that handles the LLM used for the Agent and Gatekeeper
# It ensures that changing the LLM does not require change of Code

from agent.config import PROVIDER, AGENT, GATEKEEPER, OPENROUTER_API_KEY, OLLAMA_BASE_URL, TEMPERATURE, OPENROUTER_BASE_URL
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama

def _openrouter(model: str) -> ChatOpenRouter:
    return ChatOpenRouter(model=model, temperature=TEMPERATURE, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

def _ollama(model: str) -> ChatOllama:
    return ChatOllama(model=model, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)

def _get_model(model: str) -> ChatOllama | ChatOpenRouter:
    providers = {
        'ollama': _ollama,
        'openrouter': _openrouter
    }
    if PROVIDER not in providers:
        raise ValueError(f"Unknown provider: '{PROVIDER}'. Choose from: {list(providers.keys())}")
    
    return providers[PROVIDER](model)

def get_agent() -> ChatOllama | ChatOpenRouter:
    return _get_model(model=AGENT)

def get_gatekeeper() -> ChatOllama | ChatOpenRouter:
    return _get_model(model=GATEKEEPER)