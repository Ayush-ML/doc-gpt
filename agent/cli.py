# This is The Final Script that works to create a CLI (command Line Interface) between the agent and the user
# It works to make the agent a python package that gives the user certain commands to interact with the agent
# It consists of multiple sub agents such as :
    # -- Diagnosis
# Imported Libraries

import warnings
warnings.filterwarnings("ignore")
from agent.config import USERS_DIR, user_config, user_dir, user_chroma, user_history, user_skills_dir, user_skills_index, user_memory_dir, PROVIDER, AGENT, GATEKEEPER, API_KEY, AGE, SEX, ACTIVE_USER, APP_CONFIG, user_profile
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.main.graph import get_graph
from agent.main.router import get_agent
from agent.main.state import AgentState
from agent.utils import id
from agent.steps.prompts import CONVERSATION_PROMPT, SESSION_TITLE_PROMPT
import tomli_w, typer, json, shutil, tomllib
from datetime import datetime
import traceback

# Register The App and Console

app = typer.Typer(name='klini', help="a self learn medical assistant", add_completion=False)
console = Console()

# These are the commands that the user can run to interact with the App
# The Commands help for the easy navigation of the User
# The Following Commands have been Made or are planned :
    # klini init - Initializes All Directories
    # klini register - The user can input their own info such as API Key's, Email, basic Clinical Data etc
    # klini start - Creates a Brand New session with the agent for the user (uses deafult user)
    # klini sessions - To view the titles of all past sessions
    # klini sessions {session name} -  To resume that specific session that they have entered
    # klini profile -  To view the Clinical Profile of the user that the agent has made or they themselves itself has made
    # klini skills - To view the titles and a brief summary of all skills
    # klini skills {skill name} - To view the entire content of the skill name that is specified
    # klini users - To view the list of created users
    # klini config - Shows All the registered data of the user
    # klini set --mode {mode} --change {change} -- To change the value of the specific Mode to the given Change
    # klini delete user {username} - Deletes a user profile and all associated data.
    # klini delete skill {skill_name} - Deletes a specific skill file if the agent learned something incorrect.
    # klini delete session {session_name} - Deletes a specific session and all associated data if the user wants to clear up space or remove old sessions.
    # klini delete all - Deletes all data i mean all data related to klini inlcuding users, sessions, skills, configs everything and resets the agent to a fresh state, have to initialize and set up all again. Use with caution.

# The Initialization function

@app.command()
def init() -> None:
    try:
        console.print()
        console.print(Panel("[bold]Initializing Klini Agent... [/]"))
        console.print()


        # Get Username
        if USERS_DIR.exists():
            names = [p.name for p in USERS_DIR.iterdir() if p.is_dir()]
        else:
            names = None

        while True:
            name = console.input(f" Please Enter your Username: ").strip()
            if names:
                if name in names:
                    console.print(f"  [red]Username already exists. Please choose a different name.[/]")
                    continue
            else:
                if name:
                    break
                else:
                    console.print("  [red]Name cannot be empty.[/]")

        # Create App Config
        config = {
            "config": {
                "active_user": name
            }
        }
        APP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with open(APP_CONFIG, "wb") as f:
            tomli_w.dump(config, f)

        # Register Strings as Path
        directories = [
            user_skills_dir(name),
            user_memory_dir(name),
            user_chroma(name),
        ]
    
        # Create Necessary Directories
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                console.print(f"    [green]✓ Created {directory} [/]")
            else:
                console.print(f"    [yellow]→ Skipped {directory} (already exists)[/]")

        # Create The Skill Index
        index_path = user_skills_index(name)
        if not index_path.exists():
            index_path.touch()
            console.print(f"    [green]✓ Created Skill Index at {index_path} [/]")
        else:
            console.print(f"    [yellow]→ Skipped Creation of Skill Index, already exists at path: {index_path} [/]")

        # Create The User Accessed History
        history_path = user_history(name)
        if not history_path.exists():
            history_path.touch()
            history_path.write_text("[]")
            console.print(f"    [green]✓ Created History File at {history_path} [/]")
        else:
            console.print(f"    [yellow]→ Skipped Creation of History File, already exists at path: {history_path} [/]")

        # Create The User Config
        config_path = user_config(name)
        if not config_path.exists():
            default_config = {
                "config": {
                    "provider": "",
                    "model": "",
                    "gatekeeper": "",
                    "api_key": "",
                    "email": "",
                    "age": "",
                    "sex": ""
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
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/]")
        console.print(traceback.format_exc())
        return None

# The Registration Function

@app.command()
def register() -> None:
    try:
        console.print()
        console.print(Panel("[bold]Klini Registration[/bold]", expand=False))
        console.print()

        config_path = user_config(ACTIVE_USER)

        # Check if Initialization is done
        if not config_path.exists():
            console.print("[red]Please initialize the agent first using [bold cyan]klini init[/bold cyan].[/red]")
            return None
            
        # Start Registration of Personal Info
        console.print(f"    [bold] Personal Information [/]")
        console.print()

        # Name
        names = [p.name for p in USERS_DIR.iterdir() if p.is_dir()]
        while True:
            if len(names) == 1:
                name = names[0]
                break
            name = console.input(f"    Name: ").strip()
            if name in names:
                console.print(f"  [red]Username already exists. Please choose a different name.[/]")
            elif not name.isalpha():
                console.print(f"  [red]Name cannot be empty and must only contain letters.[/]")
            else:
                break
            
        # Age
        while True:
            age = console.input(f"  Age: ").strip()
            if age.isdigit():
                age = int(age)
                if age > 0:
                    break
                else:
                    console.print("  [red]Age must be a positive number.[/]")
            else:
                console.print("  [red]Age should Always be a Number[/]")

        # Sex
        while True:
            sex = console.input(f"  Sex(can be 'male', 'female' or 'other'): ")
            if sex.lower() in ['male', 'female', 'other']:
                break
            else:
                console.print("  [red]Please enter male, female, or other.[/]")

        # Email
        while True:
            email = console.input(f"  Email: ")
            if "@" in email and "." in email:
                break
            else:
                console.print("  [red]Please enter a valid email address.[/]")


        console.print()

        # Configuration Inputs
        console.print(f"    [bold] Agent Configuration [/]")
        console.print()

        console.print("[bold underline]Supported Providers and Models[/]\n")

        console.print("[cyan]- google[/]")
        console.print("    [green]• gemini-2.5-pro[/]")
        console.print("    [green]• gemini-2.5-flash[/]")
        console.print("    [green]• gemini-2.0-flash[/]")
        console.print("    [green]• gemini-2.0-flash-lite[/]")
        console.print("    [green]• gemini-1.5-pro[/]")
        console.print("    [green]• gemini-1.5-flash[/]")
        console.print("    [green]• gemini-1.5-flash-8b[/]")
        console.print("    [green]• gemini-1.0-pro[/]")
        console.print()

        console.print("[cyan]- openai[/]")
        console.print("    [green]• gpt-4.1[/]")
        console.print("    [green]• gpt-4.1-mini[/]")
        console.print("    [green]• gpt-4.1-nano[/]")
        console.print("    [green]• gpt-4o[/]")
        console.print("    [green]• gpt-4o-mini[/]")
        console.print("    [green]• gpt-4-turbo[/]")
        console.print("    [green]• gpt-4[/]")
        console.print("    [green]• gpt-3.5-turbo[/]")
        console.print("    [green]• o4-mini[/]")
        console.print("    [green]• o3[/]")
        console.print("    [green]• o3-mini[/]")
        console.print("    [green]• o1[/]")
        console.print("    [green]• o1-mini[/]")
        console.print()

        console.print("[cyan]- anthropic[/]")
        console.print("    [green]• claude-opus-4-7[/]")
        console.print("    [green]• claude-opus-4-6[/]")
        console.print("    [green]• claude-sonnet-4-6[/]")
        console.print("    [green]• claude-haiku-4-5-20251001[/]")
        console.print()

        console.print("[cyan]- cohere[/]")
        console.print("    [green]• command-a-03-2025[/]")
        console.print("    [green]• command-r-plus[/]")
        console.print("    [green]• command-r[/]")
        console.print("    [green]• command-r7b-12-2024[/]")
        console.print("    [green]• command-light[/]")
        console.print()

        console.print("[cyan]- azure[/]")
        console.print("    [green]• gpt-4.1[/]")
        console.print("    [green]• gpt-4o[/]")
        console.print("    [green]• gpt-4o-mini[/]")
        console.print("    [green]• gpt-4-turbo[/]")
        console.print("    [green]• gpt-4[/]")
        console.print("    [green]• gpt-3.5-turbo[/]")
        console.print("    [green]• o3[/]")
        console.print("    [green]• o3-mini[/]")
        console.print("    [green]• o1[/]")
        console.print()

        console.print("[cyan]- mistral[/]")
        console.print("    [green]• mistral-large-latest[/]")
        console.print("    [green]• mistral-medium-latest[/]")
        console.print("    [green]• mistral-small-latest[/]")
        console.print("    [green]• codestral-latest[/]")
        console.print("    [green]• ministral-8b-latest[/]")
        console.print("    [green]• ministral-3b-latest[/]")
        console.print("    [green]• open-mistral-nemo[/]")
        console.print()

        console.print("[cyan]- groq[/]")
        console.print("    [green]• llama-3.3-70b-versatile[/]")
        console.print("    [green]• llama-3.1-8b-instant[/]")
        console.print("    [green]• llama3-70b-8192[/]")
        console.print("    [green]• llama3-8b-8192[/]")
        console.print("    [green]• gemma2-9b-it[/]")
        console.print("    [green]• mixtral-8x7b-32768[/]")
        console.print("    [green]• deepseek-r1-distill-llama-70b[/]")
        console.print()

        # Provider
        console.print("  The provider is the company that provides the language model API.")
        console.print()
        while True:
            provider = console.input("  Model Provider: ").strip().lower()
            if provider in ['google', 'openai', 'anthropic', 'cohere', 'azure', 'mistral', 'groq']:
                break
            else:
                console.print("  [red]Currently, only 'google' and 'openai' are supported as model providers.[/]")

        # API Key
        console.print()
        while True:
            api_key = console.input("  The API key for your selected provider: ").strip()
            if api_key:
                break
            console.print("  [red]API key cannot be empty.[/red]")
        console.print()

        # Model
        while True:
            model = console.input("  Model name for Specific Provider: ").strip()
            if model:
                break
            console.print("  [red]Model name cannot be empty.[/]")

        console.print()

        # Gatekeeper
        console.print("  The gatekeeper is a separate model that validates each step.")
        console.print("  It can be the same as your main model or a smaller faster one.")
        console.print()
        while True:
            gatekeeper_input = console.input(f"  Gatekeeper model (press enter to use {model}): ").strip()
            if gatekeeper_input:
                gatekeeper = gatekeeper_input
                break
            else:
                gatekeeper = model
                break


        console.print()

        # Update Config

        updated_config = {
            "config": {
                "provider": provider,
                "model": model,
                "gatekeeper": gatekeeper,
                "api_key": api_key,
                },
            "user": {
                "age": age,
                "sex": sex,
                "email": email
            }
        }

        with open(config_path, "wb") as f:
            tomli_w.dump(updated_config, f)

        # Create User's and USER.md
        profile_dir = user_dir(ACTIVE_USER)

        profile_path = profile_dir / "USER.md"
        profile_path.write_text(
            f"# Patient Profile - {name}\n\n"
            f"## Personal Information\n"
            f"Name: {name}\n"
            f"Age: {age}\n"
            f"Sex: {sex}\n\n"
            f"## Medical History\n\n"
            f"## Current Medications\n\n"
            f"## Allergies\n\n"
            f"## Family History\n\n"
            f"## Lifestyle\n",
            encoding="utf-8"
        )

        # Registration Complete

        console.print(Panel(
            f"[bold green]Registration complete.[/]\n\n"
            f"  Name:      {name}\n"
            f"  Age:       {age}\n"
            f"  Sex:       {sex}\n"
            f"  Provider:  {provider}\n"
            f"  Model:     {model}\n"
            f"  Profile:   {profile_path}",
            expand=False
        ))
        console.print()
        console.print("Run [bold cyan]klini start[/] to begin your first ever session!")
        console.print()
    except Exception as e:
        console.print(f"[red]Registration failed: {e}[/]")
        console.print(traceback.format_exc())
        return None

# The Configuration Function to view all the registered data of the user and the agent

@app.command()
def config() -> None:
    try:
        console.print(f"    [bold]Your Current Configuration Setting [/]")
        console.print()
        console.print(Panel(
            f"  Current User: {ACTIVE_USER}\n"
            f"  Age of User: {AGE}\n"
            f"  User's Sex / Gender: {SEX}\n"
            f"  Model Provider: {PROVIDER}\n"
            f"  Agent Model: {AGENT}\n"
            f"  Gatekeeper Model: {GATEKEEPER}\n"
            f"  API Key for Model: {API_KEY}\n",
            expand=False
        ))
        console.print(" To change any of these settings, use [bold cyan]klini config set {config name} {new value}[/]")
        console.print()
    except Exception as e:
        console.print(f"[red]Config Loading failed: {e}[/]")
        console.print(traceback.format_exc())
        return None
# The Function to view all skills and summaries of the skills

@app.command()
def skills(skill_name = typer.Argument(None)) -> None:
    try:
        if skill_name is None:
            console.print(f"    [bold]Agent Skills[/]")
            console.print()
            index_path = user_skills_index(ACTIVE_USER)
            if not index_path.exists():
                console.print("[red]The Skills index file does not exsist, please run [bold cyan]klini init[/] to create all directories and files and then run [bold cyan]klini register[/] to set up the Config![/]")
                return None
            index_data = []
            with open(index_path, 'r', encoding='utf-8') as f:
                for line in f:
                    index_data.append(json.loads(line))
            if not index_data:
                console.print("[yellow]No skills found. Start a session with [bold cyan]klini start[/] to let the agent learn new skills![/]")
                return None
            skills = [item['title'] for item in index_data]
            summaries = [item['summary'] for item in index_data]

            for skill, summary in zip(skills, summaries):
                console.print(f"  [cyan]→ {skill}[/]")
                console.print(f"  [white]→ {summary}[/]")
                console.print()
            skill_path = user_skills_dir(ACTIVE_USER) / f"{skill_name}.md"
            if not skill_path.exists():
                console.print(f"[red]Skill '{skill_name}' not found.[/]")
                return None
            else:
                skill_content = skill_path.read_text(encoding='utf-8')
                console.print(Markdown(skill_content))
    except Exception as e:
        console.print(f"[red]Skills view failed: {e}[/]")
        console.print(traceback.format_exc())
        return None

# The Function to view the Clinical Profile of the user that the agent has made or they themselves itself has made

@app.command()
def profile() -> None:
    try:
        console.print(f"    [bold]Patient Clinical Profile[/]")
        console.print()
        if not ACTIVE_USER:
            console.print("[red]No active user found. Please run [bold cyan]klini klini and klini register[/] to set up your profile.[/]")
            return None
        profile_path = user_profile(ACTIVE_USER)
        if not profile_path.exists():
            console.print("[red]Profile file not found. Please run [bold cyan]klini register[/] to set up your profile.[/]")
            return None
        profile_content = profile_path.read_text(encoding='utf-8')
        console.print(Markdown(profile_content))
    except Exception as e:
        console.print(f"[red]Profile view failed: {e}[/]")
        console.print(traceback.format_exc())
        return None

# Function to view all created users

@app.command()
def users() -> None:
    try:
        console.print(f"    [bold]Registered Users[/]")
        console.print()
        users_path = USERS_DIR
        if not users_path.exists():
            console.print("[red]Users directory not found. Please run [bold cyan]klini init[/] to create all directories and files and then run [bold cyan]klini register[/] to set up the Config![/]")
            return None
        user_dirs = [d for d in users_path.iterdir() if d.is_dir()]
        if not user_dirs:
            console.print("[red]No users found. Please run [bold cyan]klini register[/] to create a user profile![/]")
            return None
        for user_dir in user_dirs:
            console.print(f"  [cyan]→ {user_dir.name}[/]")
    except Exception as e:
        console.print(f"[red]Users view failed: {e}[/]")
        console.print(traceback.format_exc())
        return None

# The Function for Deletion

@app.command()
def delete(mode: str = typer.Option('all', "--mode", "-m"), name: str = typer.Option(None, "--name", "-n")) -> None:
    try:
        if mode == "user":
            if not name:
                console.print("[red]Please specify the username to delete using [bold cyan]klini delete user {username}[/]")
                return None
            else:
                user_path = USERS_DIR / name
                if not user_path.exists():
                    console.print(f"[red]User '{name}' not found.[/]")
                    return None
                else:
                    console.print(f" Are You absolutely sure you want to delete user '{name}' and all associated data? This action cannot be undone.")
                    confirmation = console.input(f" y/n: ").strip().lower()
                    if confirmation != "y":
                        console.print("[yellow]Deletion cancelled.[/]")
                        return None
                    else:
                        try:
                            shutil.rmtree(user_path)
                        except Exception as e:
                            console.print(f"[red]An error occurred while deleting user '{name}': {e}[/]")
                            return None
                        console.print("[green]User deleted successfully.[/]")

        elif mode == "skill":
            if not name:
                console.print("[red]Please specify the skill name to delete using [bold cyan]klini delete skill skill_name[/][/]")
                return None
            skill_path = user_skills_dir(ACTIVE_USER) / f"{name}.md"
            if not skill_path.exists():
                console.print(f"[red]Skill '{name}' not found.[/]")
                return None
            else:
                console.print(f" Are You absolutely sure you want to delete skill '{name}'? This action cannot be undone.")
                confirmation = console.input(f" y/n: ").strip().lower()
                if confirmation != "y":
                    console.print("[yellow]Deletion cancelled.[/]")
                    return None
                else:
                    skill_path.unlink()
                    console.print(f"[green]Skill '{name}' deleted successfully.[/]")

        elif mode == "session":
            if not name:
                console.print("[red]Please specify the session title to delete using [bold cyan]klini delete session session_title[/][/]")
                return None
            
            history_path = user_history(ACTIVE_USER)
            if not history_path.exists():
                console.print("[red]History file not found. Please run [bold cyan]klini init[/] to create all directories and files and then run [bold cyan]klini register[/] to set up the Config![/]")
                return None
            
            with open(history_path, "r") as file:
                history = json.load(file)
            session_to_delete = None

            for session in history:
                if session['session_title'].lower() == name.lower():
                    session_to_delete = session
                    break

            if not session_to_delete:
                console.print(f"[red]Session '{name}' not found.[/]")
                return None
            
            else:
                console.print(f" Are You absolutely sure you want to delete session '{name}' and all associated data? This action cannot be undone.")
                confirmation = console.input(f" y/n: ").strip().lower()
                if confirmation != "y":
                    console.print("[yellow]Deletion cancelled.[/]")
                    return None
                
                else:
                    history.remove(session_to_delete)
                    with open(history_path, "w") as file:
                        json.dump(history, file, indent=4)
                    console.print(f"[green]Session '{name}' and all associated data deleted successfully.[/]")
                    return None

        elif mode == "all":
            console.print(f" Are You absolutely sure you want to delete all data related to Klini including users, sessions, skills, configs everything and reset the agent to a fresh state? This action CANNOT be undone.")
            confirmation = console.input(f" y/n: ").strip().lower()
            if confirmation != "y":
                console.print("[yellow]Deletion cancelled.[/]")
                return None
            else:
                # Delete Users
                users_path = USERS_DIR
                if users_path.exists():
                    try:
                        shutil.rmtree(users_path)
                    except Exception as e:
                        console.print(f"[red]An error occurred while deleting users: {e}[/]")
                        return None
                else:
                    console.print(f"[yellow]→ Users directory not found, Skipping.[/]")
                
                # Delete App Config
                if APP_CONFIG.exists():
                    try:
                        APP_CONFIG.unlink()
                    except Exception as e:
                        console.print(f"[red]An error occurred while deleting app config: {e}[/]")
                        return None
                else:
                    console.print(f"[yellow]→ App config file not found, Skipping.[/]")
                    
                console.print(f"[bold green]All data deleted and agent reset to fresh state. Please run [bold cyan]klini init[/] to initialize and then [bold cyan]klini register[/] to set up your account.")
                return None
    except Exception as e:
        console.print(f"[red]Deletion failed: {e}[/]")
        console.print(traceback.format_exc())
        return None
    
# Create A function that is used to run a session witha given state, used to resume sessions or start a brand new one

def _run_session(initial_state: AgentState) -> None:
    # Initialize Necessary Variables
    try:
        graph = get_graph()
        agent = get_agent()
        state = initial_state
    except Exception as e:
        console.print(f"[red]Failed to initialize agent or graph: {e}[/]")
        console.print(traceback.format_exc())
        return None

    console.print(f"[green]Session started. Use Ctrl + C to exit.[/]")
    console.print()

    while True:
        try:
            user_input = console.input("[blue]You:[/] ").strip()
            if not user_input:
                continue
            state['messages'].append(HumanMessage(content=user_input))
            if not state['diagnosis_started']:
                user_message = f"User's Clinical Profile: {state['clinical_profile']}, Current Session Messages: {state['messages']}"
                context = [
                    SystemMessage(content=CONVERSATION_PROMPT),
                    HumanMessage(content=user_message)
                ]
                console.print("[green]Klini:[/] ", end="")
                response = ""
                clean = "Message Generation Failed"
                try:
                    clean = "Message Generation Failed"
                    for chunk in agent.stream(context):
                        chunk = chunk.content or ""
                        if isinstance(chunk, list):
                            chunk = " ".join(
                                block.get("text", "") for block in chunk 
                                if isinstance(block, dict) and "text" in block
                            )
                        if "<DIAGNOSE/>" in chunk:
                            state['diagnosis_started'] = True
                            clean = chunk.replace("<DIAGNOSE/>", "").strip()
                        else:
                            clean = chunk.strip()
                        console.print(Markdown(clean), end="")
                        response += clean
                except Exception as e:
                    console.print(f"[red]An error occurred while streaming response: {e}[/]")
                    response += clean
                console.print()
                state['messages'].append(AIMessage(content=response))

            elif state['diagnosis_started']:
                with console.status("[green]Klini is diagnosing...[/]"):
                    result = graph.invoke(state, config={"configurable": {"thread_id": state['session_id']}})
                console.print("[green]Diagnosis Completed Successfully.[/]")
                state = result
                last_message = state['messages'][-1].content
                console.print(Markdown(last_message))
                state['diagnosis_started'] = False
                state['ever_diagnosed'] = True
        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Session ended by user.[/]")
            if state['messages']:
                with console.status("[green]Saving session...[/]"):
                    history_path = user_history(ACTIVE_USER)
                    if history_path.exists():
                        with open(history_path, "r") as file:
                            history = json.load(file)
                    else:
                        history = []
                    user_message = f"All of Session Messages: {state['messages']}"
                    context = [
                        SystemMessage(content=SESSION_TITLE_PROMPT),
                        HumanMessage(content=user_message)
                    ]
                    try:
                        session_title = (agent.invoke(context)).content
                    except Exception as e:
                        console.print(f"[red]An error occurred while generating session title: {e}[/]")
                        session_title = "Untitled Session"
                    updated_session = {
                        "session_title": session_title,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "session_id": state['session_id'],
                        "messages": [{"role": "human" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in state['messages']],
                        "diagnosis_started": state['diagnosis_started'],
                        "ever_diagnosed": state['ever_diagnosed']
                    }
                    
                    # Remove any existing session with the same session_id to avoid duplicates
                    history = [s for s in history if s['session_id'] != state['session_id']]
                    history.append(updated_session)
                    
                    with open(history_path, "w") as file:
                        json.dump(history, file, indent=4)

                console.print(f"[green]Session saved with title: {session_title}[/]")
            return None
        
        except Exception as e:
            console.print(f"[red]An error occurred during the session: {e}[/]")
            console.print(traceback.format_exc())
            continue

# Create The Sessions Command to view all sessions or resume a specific session

@app.command()
def sessions(session_title: str = typer.Argument(None)) -> None:
    try:
        history_path = user_history(ACTIVE_USER)
        if not history_path.exists():
            console.print("[red]History file not found. Please run [bold cyan]klini init[/] to create all directories and files and then run [bold cyan]klini register[/] to set up the Config![/]")
            return None
        else:
            with open(history_path, "r") as file:
                history = json.load(file)
            if not history:
                console.print("[red]No sessions found. Start a session with [bold cyan]klini start[/] to create sessions![/]")
                return None
        
        if session_title is None:
            console.print(f"    [bold]Your Sessions[/]")
            console.print()
            count = 1
            for session in history:
                console.print(f"    {count}. [cyan] {session['session_title']}[/]")
                count += 1
            console.print()
            console.print("To resume a session, run [bold cyan]klini sessions {session name}[/]")
            return None
        
        if session_title:
            session_data = None
            for session in history:
                if session['session_title'].lower() == session_title.lower():
                    console.print(f"Resuming session: [cyan]{session['session_title']}[/]")
                    console.print()
                    session_data = session

                    for message in session['messages']:
                        role = 'You' if message['role'] == 'human' else 'Klini'
                        content = message['content']
                        if role == "You":
                            console.print(f"[blue]{role}:[/] {content}")
                        else:
                            console.print(f"[green]{role}:[/] {content}")
                    break

            if not session_data:
                console.print(f"[red]Session '{session_title}' not found.[/]")
                return None
            else:
                with open(user_profile(ACTIVE_USER), "r") as file: # Load Clinical Profile
                    clinical_profile = file.read()
                state = AgentState(
                    session_id=session_data['session_id'],
                    user_id=ACTIVE_USER,
                    clinical_profile=clinical_profile,
                    current_step=1,
                    max_reached_step=1,
                    retries={},
                    messages=[HumanMessage(content=m['content']) if m['role'] == 'human' else AIMessage(content=m['content']) for m in session_data['messages']],
                    end_response_reason="No Specific Reason",
                    requested_next="forward",
                    requested_target_step=None,
                    all_skills={},
                    semantic_search=[],
                    used_skills=[],
                    skill_contents=[],
                    gatekeeper_decision=False,
                    gatekeeper_reason="",
                    diagnosis_started=session_data.get('diagnosis_started', True),
                    ever_diagnosed=session_data.get('ever_diagnosed', False)
                )

                _run_session(initial_state=state)
    except Exception as e:
        console.print(f"[red]Failed to load sessions: {e}[/]")
        console.print(traceback.format_exc())
        return None
            
# The Function to start a new session with the agent for the user

@app.command()
def start() -> None:
    try:
        console.print(f"    [bold]Starting New Session with Klini Agent[/]")
        console.print()

        session_id = id()
        with open(user_profile(ACTIVE_USER), "r") as file: # Load Clinical Profile
            clinical_profile = file.read()

        state = AgentState(
                    session_id=session_id,
                    user_id=ACTIVE_USER,
                    clinical_profile=clinical_profile,
                    current_step=1,
                    max_reached_step=1,
                    retries={},
                    messages=[],
                    end_response_reason="No Specific Reason",
                    requested_next="forward",
                    requested_target_step=None,
                    all_skills={},
                    semantic_search=[],
                    used_skills=[],
                    skill_contents=[],
                    gatekeeper_decision=False,
                    gatekeeper_reason="",
                    diagnosis_started=False,
                    ever_diagnosed=False
        )
    except Exception as e:
        console.print(f"[red]Failed to start session: {e}[/]")
        console.print(traceback.format_exc())
        return None

    _run_session(initial_state=state)

# The Function used to Change a Specific Value

@app.command()
def update(key: str = typer.Option(..., "--key", "-k"), value: str = typer.Option(..., "--value", "-v")) -> None:
    try:
        console.print()
        user_config_keys = ["provider", "model", "gatekeeper", "api_key", "email", "infermedica_app_id", "infermedica_app_key", "age", "sex"]

        if key == 'user':
            console.print(f"[bold]Updating Active User[/]")
            console.print()
            user_path = USERS_DIR / value
            if not user_path.exists():
                console.print(f"[red]User '{value}' not found. Please enter a valid username.[/]")
                return None

            with open(APP_CONFIG, "rb") as f:
                app_data = tomllib.load(f)

            if app_data.get("config", {}).get("active_user") == value:
                console.print(f"[yellow]'{value}' is already the active user.[/] Skipping...")
                return None

            app_data.setdefault("config", {})["active_user"] = value
            with open(APP_CONFIG, "wb") as f:
                tomli_w.dump(app_data, f)
            console.print(f"[green]Active user updated to '{value}'.")
            return None

        elif key in user_config_keys:
            console.print(f"  [bold]Updating User Config[/]")
            console.print()
            config_path = user_config(ACTIVE_USER)
            if not config_path.exists():
                console.print("[red]User config file not found. Please run [bold cyan]klini init[/] to create all directories and files and then run [bold cyan]klini register[/] to set up the Config![/]")
                return None
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)

            if not isinstance(config_data, dict):
                config_data = {}

            if key in ["age", "sex", "email"]:
                config_data.setdefault('user', {})[key] = value
            else:
                config_data.setdefault('config', {})[key] = value

            with open(config_path, "wb") as f:
                tomli_w.dump(config_data, f)
            console.print(f"[green]'{key}' successfully updated to '{value}'.[/]")
            return None

        else:
            console.print(f"[red]Invalid key: {key}[/]")
            console.print(f"[red]You can update the active user with key 'user' or the user config with keys: {', '.join(user_config_keys)}[/]")
            return None
    except Exception as e:
        console.print(f"[red]Failed to update config: {e}[/]")
        console.print(traceback.format_exc())
        return None
    
# Run the Complete Code

if __name__ == "__main__":
    app()