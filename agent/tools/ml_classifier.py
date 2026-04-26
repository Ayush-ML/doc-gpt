# This is a Script That handles Th classifier tool of The Agent
# This gives the Agent access to predictive models that it can use to get predictions for given diseases
# Imported Libraries

from langchain_core.tools import tool
import requests
from agent.config import INFERMEDICA_APP_ID, INFERMEDICA_APP_KEY, PARSE_URL, DIAGNOSIS_URL

# Lazy Initialize Headers

_headers = None

def _get_headers() -> dict:
    global _headers
    if _headers is None:
        _headers = {
            "App-Id": INFERMEDICA_APP_ID,
            "App-Key": INFERMEDICA_APP_KEY,
            "Content-Type": "application/json"
        }
    return _headers

# Parse Plain Text into User Symptoms

def _parse(symptoms: str, age: int, sex: int) -> list[dict]:
    payload = {
        "text": symptoms,
        "age": {"value": age},
        "sex": sex
    }
    parsed = requests.post(url=PARSE_URL, headers=_get_headers(), json=payload)
    parsed.raise_for_status()
    data = parsed.json()

    evidence = []
    for mention in data.get("mentions", []):
        evidence.append({
            "id": mention["id"],
            "choice_id": "present",
        })

    return evidence

# Diagnose Diseases using Symptoms

def _diagnose(evidence: list[dict], age: int, sex: str) -> list[dict]:
    payload = {
        "sex": sex,
        "age": {"value": age},
        "evidence": evidence
    }
    response = requests.post(url=DIAGNOSIS_URL, json=payload, headers=_get_headers())
    response.raise_for_status()
    data = response.json()

    results = []
    for condition in data.get("conditions", []):
        results.append({
            "condition": condition["name"],
            "probability": round(condition["probability"], 3),
            "severity": condition.get("severity", "unknown")
        })

    return results

# Create the tool that will be passed to the model

@tool
def classifier(symptoms: str, age: int, sex: str) -> list[dict]:
    """
    Run a clinical diagnosis using patient symptoms.

    Use this when you need to:
    - Get probability scores for candidate conditions
    - Validate your candidate conditions list from Step 1
    - Identify conditions you may have missed

    Args:
        symptoms: plain text description of patient symptoms
        age: patient age as integer
        sex: patient sex as "male" or "female"

    Returns:
        list of conditions with probability scores
    """
    try:
        evidence = _parse(symptoms=symptoms, age=age, sex=sex)
        if not evidence:
            return [{"condition": "No conditions identified", "probability": 0.0, "severity": "unknown"}]
        
        results = _diagnose(evidence=evidence, age=age, sex=sex)
        return results
    
    except requests.exceptions.HTTPError as e:
        return [{"condition": "API error", "probability": 0.0, "severity": str(e)}]
    except Exception as e:
        return [{"condition": "Classifier failed", "probability": 0.0, "severity": str(e)}]