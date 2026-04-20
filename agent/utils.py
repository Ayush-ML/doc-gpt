# This is a Utils Script that contains any helper function for the Agent
# Imported Libraries

import uuid

# A Function that Return an Unpredictable and Secure String of Letters and Numbers
# Used for the Session ID

def id() -> str:

    return str(uuid.uuid4())