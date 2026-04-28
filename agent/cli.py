# This is The Final Script that works to create a CLI (command Line Interface) between the agent and the user
# It works to make the agent a python package that gives the user certain commands to interact with the agent
# It consists of multiple sub agents such as :
    # -- Diagnosis
# Imported Libraries

from agent.config import CHROMA, SKILLS, USERS, INDEX, MEMORY, HISTORY, USER_CONFIG
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.main.graph import get_graph
from agent.main.router import get_agent
from agent.main.state import AgentState
from agent.utils import generate_id
from agent.steps.prompts import CONVERSATION_PROMPT
import tomli_w

# Register The App and Console

app = typer.Typer(name='klini', help="a self learn medical assistant", add_completion=False)
console = Console()

# These are the commands that the user can run to interact with the App
# The Commands help for the easy navigation of the User
# The Following Commands have been Made or are planned :
    # klini init - Initializes All Directories
    # klini register - The user can input their own info such as API Key's, Email, basic Clinical Data etc
    # klini start - Creates a Brand New session with the agent for the user (uses deafult user)
    # klini start - Creats a Brand new session in the specified 
    # klini sessions - To view the titles of all past sessions
    # klini sessions -- {session name} -  To resume that specific session that they have entered
    # klini profile -  To view the Clinical Profile of the user that the agent has made or they themselves itself has made
    # klini skills - To view the titles and a brief summary of all skills
    # klini skills -- {skill name} - To view the entire content of the skill name that is specified
    # klini users - To view the list of created users
    # klini users -- {user name} - To switch to the specified user name, further actions will be carried out in this user
    # klini config - Shows All the registered data of the user
    # klini config set {config name} {change} -- To change the value of the specific config to thee value the user gave
    # klini delete user --{username} - Deletes a user profile and all associated data.
    # klini delete skill --{skill_name} - Deletes a specific skill file if the agent learned something incorrect.
    # klini status - Shows a quick overview of the current state of the agent.

# The Initialization function

@app.command()
def init() -> None:
    console.print()
    console.print(Panel("[bold]Initializing Klini Agent... [/]"))
    console.print()

    # Register Strings as Path
    directories = [
        Path(SKILLS),
        Path(USERS),
        Path(MEMORY),
        Path(CHROMA)
    ]
 
    # Create Necessary Directories
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            console.print(f"    [green]✓ Created {directory} [/]")
        else:
            console.print(f"[yellow]→ Skipped {directory} (already exists)[/]")

    # Create The Skill Index
    index_path = Path(INDEX)
    if not index_path.exists():
        index_path.touch()
        console.print(f"    [green]✓ Created Skill Index at {index_path} [/]")
    else:
        console.print(f"    [yellow]→ Skipped Creation of Skill Index, already exists at path: {index_path} [/]")

    # Create The User Accessed History
    history_path = Path(HISTORY)
    if not history_path.exists():
        history_path.touch()
        history_path.write_text("[]")
        console.print(f"    [green]✓ Created History File at {history_path} [/]")
    else:
        console.print(f"    [yellow]→ Skipped Creation of History File, already exists at path: {history_path} [/]")

    # Create The User Config
    config_path = Path(USER_CONFIG)
    if not config_path.exists():
        default_config = {
            "config": {
                "provider": "",
                "model": "",
                "gatekeeper": "",
                "openrouter_api_key": "",
                "email": "",
                "infermedica_app_id": "",
                "infermedica_app_key": "",
            },
            "user": {
                "active_user": "",
            }
        }
        with open(config_path, "wb") as f:
            tomli_w.dump(default_config, f)
        console.print(f"    [green]✓ Created User Config at {config_path} [/]")
    else:
        console.print(f"    [yellow]→ Skipped Creation of User Config, already exists at path: {config_path} [/]")

    # Finished Initialization

    console.print()
    console.print("[bold green]Klini initialized successfully.[/]")
    console.print()
    console.print("Next step: run [bold cyan]klini register[/] to set up your profile.")

# The Registration Function

@app.command()
def register() -> None:
    console.print()
    console.print(Panel("[bold]Klini Registration[/bold]", expand=False))
    console.print()

    # Check if User is Already Registered