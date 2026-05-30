# This is a Utils Script that contains any helper function for the Agent
# Imported Libraries

import uuid, re, json
from ast import literal_eval

# A Function that Return an Unpredictable and Secure String of Letters and Numbers
# Used for the Session ID

def id() -> str:

    return str(uuid.uuid4())

# A function responsible for Parsing the End Response tag at the end of the Agents Response

def parse_end_response(response: str) -> tuple[str, str, int | None]:
    pattern = r'<END_RESPONSE\s+reason="([^"]*)"\s+next="([^"]*)"\s*(?:target_step="([^"]*)")?\s*/>'
    match = re.search(pattern, response)
    if match:
        reason = match.group(1)
        next_dir = match.group(2)
        target = int(match.group(3)) if match.group(3) and match.group(3) != "null" else None
        return reason, next_dir, target
    # fallback if model forgets the tag
    return "Response complete", "forward", None

# A function for stripping End Response Tag

def strip_end_response(response: str) -> str:
    pattern = r'<END_RESPONSE[^/]*/>'
    return re.sub(pattern, "", response).strip()

# A function for Extracting the Gatekeeper's Response 

def extract_gatekeeper_response(content):
    """
    Attempts to extract a valid dict from the model's response content.
    Handles JSON, Python dict, and error cases robustly.
    Returns a dict with 'approved' and 'reason' keys, or an error message if extraction fails.
    """
    if not content or not str(content).strip():
        return {
            "approved": False,
            "reason": "Gatekeeper response content is empty."
        }
    # Try JSON first
    try:
        return json.loads(content)
    except Exception:
        pass
    # Try Python dict (sometimes models return single quotes)
    try:
        return literal_eval(content)
    except Exception:
        pass
    # Try to extract JSON substring if extra text is present
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            sub = content[start:end+1]
            try:
                return json.loads(sub)
            except Exception:
                return literal_eval(sub)
    except Exception:
        pass
    # Fallback: return error
    return {
        "approved": False,
        "reason": f"Could not parse gatekeeper response: {content}"
    }
    
# A function to clean Models Skill Selection

def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\[\]]', '', name).strip()
