# This is a Script That handles the classifier tool of The Agent
# Uses LLM prompting instead of Infermedica to classify symptoms and return probability scores
# Imported Libraries

import json
from langchain_core.tools import tool
from agent.config import AGE, SEX
from agent.steps.prompts import CLASSIFIER_PROMPT
from agent.main.router import get_agent
from langchain_core.messages import SystemMessage, HumanMessage

# The Function that is passed to The model as a tool

@tool
def classifier(symptoms: str) -> list[dict]:
    """
    Perform a specialist-level clinical differential diagnosis using LLM reasoning.

    This tool deconstructs patient symptoms and returns a probability-weighted list
    of candidate conditions ranked by likelihood. It accounts for patient demographics
    (age and sex loaded from config), applies threat stratification to ensure
    life-threatening conditions are never missed, and calibrates probabilities against
    real-world disease prevalence.

    Use this tool when you need to:
        - Generate or validate a differential diagnosis from patient-reported symptoms
        - Obtain probability scores for candidate conditions identified in Step 1
        - Identify conditions that may have been missed in earlier analysis
        - Stratify conditions by clinical severity for prioritisation
        - Cross-check your reasoning against an independent diagnostic assessment

    Do NOT use this tool when:
        - You already have a confirmed diagnosis and are only seeking treatment guidance
        - The symptoms are too vague or sparse to support meaningful classification
        - You are looking for drug interactions or literature evidence — use the appropriate tools for those

    Args:
        symptoms (str): Plain text description of the patient's symptoms. Be as detailed
                        as possible — include onset, duration, character, location,
                        radiation, severity, aggravating and relieving factors, and
                        any associated symptoms. The more detail provided, the more
                        accurate the classification will be.

    Returns:
        list[dict]: A list of 5 to 8 candidate conditions, each represented as a dict
                    with the following keys:

                    - condition (str): The name of the candidate condition
                    - probability (float): Estimated probability between 0.0 and 1.0.
                                          All probabilities sum to 1.0.
                    - severity (str): Clinical severity of the condition — one of:
                                      "mild", "moderate", "severe", or "critical"

                    Example:
                    [
                        {"condition": "Acute Myocardial Infarction", "probability": 0.35, "severity": "critical"},
                        {"condition": "Unstable Angina", "probability": 0.25, "severity": "severe"},
                        {"condition": "Pulmonary Embolism", "probability": 0.15, "severity": "critical"},
                        {"condition": "GERD", "probability": 0.10, "severity": "mild"},
                        {"condition": "Musculoskeletal Chest Pain", "probability": 0.15, "severity": "mild"}
                    ]

    Raises:
        Returns a single-item error list if classification fails — the pipeline
        will not crash but the calling node should handle a failed classification
        gracefully and proceed with its own reasoning rather than relying on these results.
    """
    try:
        # Build Agent
        agent = get_agent()

        # Build Context
        prompt = f"Patient is {AGE} year old {SEX} with Symptoms: {symptoms}"

        messages = [
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = agent.invoke(messages)
        response = response.content if hasattr(response, 'content') else str(response)
        if isinstance(response, list):
                response = " ".join(
                    block.get("text", "") for block in response 
                    if isinstance(block, dict) and "text" in block
                )
        raw = response.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        clean = raw.strip()

        results = json.loads(clean)
        return results

    except json.JSONDecodeError:
        return [{"condition": "Classifier failed — invalid JSON returned", "probability": 0.0, "severity": "unknown"}]
    except Exception as e:
        return [{"condition": "Classifier failed", "probability": 0.0, "severity": str(e)}]