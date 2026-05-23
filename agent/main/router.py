# This is a Script that handles the LLM used for the Agent and Gatekeeper
# It ensures that changing the LLM does not require change of Code

from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from agent.config import PROVIDER, AGENT, GATEKEEPER, API_KEY, TEMPERATURE

def _google_genai(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=TEMPERATURE, google_api_key=API_KEY)


def _openai(model: str) -> ChatOpenAI:
    return ChatOpenAI(model_name=model, temperature=TEMPERATURE, openai_api_key=API_KEY)


def _get_model(model: str) -> Any:
    providers = {
        'google': _google_genai,
        'openai': _openai,
    }
    if PROVIDER not in providers:
        raise ValueError(f"Unknown provider: '{PROVIDER}'. Choose from: {list(providers.keys())}")
    
    return providers[PROVIDER](model)

def get_agent() -> Any:
    return _get_model(model=AGENT)

def get_gatekeeper() -> Any:
    return _get_model(model=GATEKEEPER)