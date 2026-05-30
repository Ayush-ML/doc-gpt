# This is a Script that handles the LLM used for the Agent and Gatekeeper
# It ensures that changing the LLM does not require change of Code

from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from agent.config import PROVIDER, AGENT, GATEKEEPER, API_KEY, TEMPERATURE

def _google_genai(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=TEMPERATURE, google_api_key=API_KEY)


def _openai(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=TEMPERATURE, api_key=API_KEY)


def _anthropic(model: str) -> ChatAnthropic:
    return ChatAnthropic(model_name=model, temperature=TEMPERATURE, api_key=API_KEY, timeout=None, stop=None)


def _cohere(model: str) -> ChatCohere:
    return ChatCohere(model=model, temperature=TEMPERATURE, cohere_api_key=API_KEY)


def _azure_openai(model: str) -> AzureChatOpenAI:
    return AzureChatOpenAI(model=model, temperature=TEMPERATURE, api_key=API_KEY)


def _mistral(model: str) -> ChatMistralAI:
    return ChatMistralAI(name=model, temperature=TEMPERATURE, api_key=API_KEY)


def _groq(model: str) -> ChatGroq:
    return ChatGroq(model=model, temperature=TEMPERATURE, api_key=API_KEY)


def _get_model(model: str) -> Any:
    providers = {
        'google': _google_genai,
        'openai': _openai,
        'anthropic': _anthropic,
        'cohere': _cohere,
        'azure': _azure_openai,
        'mistral': _mistral,
        'groq': _groq,
    }
    if PROVIDER not in providers:
        raise ValueError(f"Unknown provider: '{PROVIDER}'. Choose from: {list(providers.keys())}")
    
    return providers[PROVIDER](model)

def get_agent() -> Any:
    return _get_model(model=AGENT)

def get_gatekeeper() -> Any:
    return _get_model(model=GATEKEEPER)