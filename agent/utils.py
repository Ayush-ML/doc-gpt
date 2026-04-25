# This is a Utils Script that contains any helper function for the Agent
# Imported Libraries

import uuid, re

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

# A function to parse all Tools and execute them in the models response

def parse_tools(response: str) -> None:
    pass

