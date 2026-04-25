# This is a Script that Contains the Drug Look up tool 
# It uses the OpenFDA APIin order to fetch information about certain drugs
# Imported Libraries

from langchain_core.tools import tool
import requests

# Create the Function that is passed to model

@tool
def drug_lookup(drug_name: str) -> dict | str:
    """
    Look up drug information from the FDA database.

    Use this when you need to:
    - Check a drug's side effects or adverse reactions
    - Look up drug interactions
    - Find dosage information
    - Verify if a medication could be causing symptoms

    Do NOT use this for general information — use web_search for that.
    Do NOT use this for clinical literature — use pubmed for that.

    Args:
        drug_name: name of the drug, brand or generic

    Returns:
        dict containing drug purpose, warnings, adverse effects and interactions
    """
    # Set The URL where requests fetch's the data
    url = "https://api.fda.gov/drug/label.json" 

    # Search for the drug by Drug name
    params = {'search': f'openfda.brand_name:"{drug_name}"', 'limit': 1}

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data['results'][0]
        else:
            return "API Error"
    except requests.exceptions.RequestException:
        return "Request failed"
    
    return {
    "spl_product_data_elements": results.get("spl_product_data_elements", "Not available"),
    "indications_and_usage": results.get("indications_and_usage", "Not available"),
    "warnings": results.get("warnings", "Not available"),
    "do_not_use": results.get("do_not_use", "Not available"),
    "ask_doctor": results.get("ask_doctor", "Not available"),
    "when_using": results.get("when_using", "Not available"),
    "stop_use": results.get("stop_use", "Not available"),
    "pregnancy_or_breast_feeding": results.get("pregnancy_or_breast_feeding", "Not available"),
    "keep_out_of_reach_of_children": results.get("keep_out_of_reach_of_children", "Not available"),
    "dosage_and_administration": results.get("dosage_and_administration", "Not available"),
    }