# This is The Final Script that works to create a CLI (command Line Interface) between the agent and the user
# It works to make the agent a python package that gives the user certain commands to interact with the agent
# It consists of multiple sub agents such as :
    # -- Diagnosis
# Imported Libraries

from agent.config import CHROMA, SKILLS, USERS, INDEX, MEMORY, HISTORY, USER_CONFIG, PROVIDER, AGENT, GATEKEEPER, OPENROUTER_API_KEY, EMAIL, AGE, SEX, INFERMEDICA_APP_ID ,INFERMEDICA_APP_KEY, ACTIVE_USER
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.main.graph import get_graph
from agent.main.router import get_agent
from agent.main.state import AgentState
from agent.utils import generate_id
from agent.steps.prompts import CONVERSATION_PROMPT, SESSION_TITLE_PROMPT
import tomli_w, typer, json, tomllib, shutil
from datetime import datetime

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
        # klini set users {user name} - To switch to the specified user name, further actions will be carried out in this user
    # klini config - Shows All the registered data of the user
        # klini set config {config name} {change} -- To change the value of the specific config to thee value the user gave
    # klini delete user {username} - Deletes a user profile and all associated data.
    # klini delete skill {skill_name} - Deletes a specific skill file if the agent learned something incorrect.
    # klini delete session {session_name} - Deletes a specific session and all associated data if the user wants to clear up space or remove old sessions.
    # klini delete all - Deletes all data i mean all data related to klini inlcuding users, sessions, skills, configs everything and resets the agent to a fresh state, have to initialize and set up all again. Use with caution.

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

    config_path = Path(USER_CONFIG)

    # Check if Initialization is done
    if not config_path.exists():
        console.print("[red]Please initialize the agent first using [bold cyan]klini init[/bold cyan].[/red]")
        return None

    # Check if User is Already Registered
    if config_path.exists():
        with open(config_path, "rb") as f:
            existing = tomllib.load(f)
        if existing.get("user", {}).get("active_user", ""):
            console.print("[yellow]You are already registered.[/yellow]")
            console.print("Use [bold cyan]klini config set[/bold cyan] to update individual values.")
            console.print()
            return None
        
    # Start Registration of Personal Info
    console.print(f"    [bold] Personal Information [/]")

    # Name
    while True:
        name = console.input(f"    Name: ").strip()
        if name.isalpha():
            break
        else:
            console.print("  [red]Name cannot be empty.[/]")

    # Age
    while True:
        age = console.input(f"  Age: ").strip()
        if age.isdigit():
            age = int(age)
            if age > 0:
                break
        else:
            console.print("  [red]Please enter a valid age That is atleast greater than 0.[/]")

    # Sex
    while True:
        sex = console.input(f"  Sex or Gender(can be 'male', 'female' or 'other'): ")
        if sex in ['male', 'female', 'other']:
            break
        else:
            console.print("  [red]Please enter male, female, or other.[/]")

    console.print()

    # Configuration Inputs
    console.print(f"    [bold] Agent Configuration [/]")
    console.print()

    # Email
    console.print(f" Your Email is required to Create a free Pubmedical Account for an API")
    while True:
        email = console.input(f"    Email: ")
        if email and '@' in email:
            break
        else:
            console.print("  [red]Please enter a valid email address.[/red]")

    # Infermedica
    console.print("  Infermedica provides clinical symptom classification (free tier).")
    console.print("  Get your free credentials at [link=https://developer.infermedica.com]https://developer.infermedica.com[/]!")
    console.print()
    while True:
        infermedica_id = console.input("  Infermedica App ID (optional): ").strip()
        infermedica_key = console.input("  Infermedica App Key (optional): ").strip()
        if infermedica_id and infermedica_key:
            break
        else:
            console.print(f"    [red] Please enter a Valid Infermedica ID and Key [/]")
        
    # Providers
    console.print("  Providers:")
    console.print("    [cyan]1[/] ollama     (local, no API key needed)")
    console.print("    [cyan]2[/] openrouter (cloud, API key required)")
    console.print()
    while True:
        provider_input = console.input("  Choose provider (1 or 2): ").strip()
        if provider_input == "1":
            provider = "ollama"
            break
        elif provider_input == "2":
            provider = "openrouter"
            break
        else:
            console.print("  [red]Please enter 1 or 2.[/]")
            console.print()

    console.print()

    # Model
    if provider == "ollama":
        console.print("  Recommended Ollama models:")
        console.print("    [cyan]phi3:mini[/]      3.8B — fast on CPU, good reasoning")
        console.print("    [cyan]gemma2:2b[/]      2B   — very fast, decent quality")
        console.print("    [cyan]mistral:7b-q4[/]  7B   — strong reasoning, slower on CPU")
        console.print()
        while True:
            model = console.input("  Model name: ").strip()
            if model:
                break
            else:
                console.print("  [red]Model name cannot be empty.[/]")
    elif provider == 'openrouter':
        # API Key
        console.print("  Get your free API key at [link=https://openrouter.ai]https://openrouter.ai[/]")
        console.print()
        while True:
            api_key = console.input("  OpenRouter API key: ").strip()
            if api_key:
                break
            console.print("  [red]API key cannot be empty for OpenRouter.[/red]")
        console.print()

        console.print("  Recommended OpenRouter free models:")
        console.print("    [cyan]meta-llama/llama-3.1-8b-instruct:free[/]")
        console.print("    [cyan]microsoft/phi-3-mini-128k-instruct:free[/]")
        console.print("    [cyan]mistralai/mistral-7b-instruct:free[/]")
        console.print("    [cyan]deepseek/deepseek-r1:free[/]")
        console.print()
        while True:
            model = console.input("  Model name: ").strip()
            if model:
                break
            console.print("  [red]Model name cannot be empty.[/]")

    console.print()

    console.print("  The gatekeeper is a separate model that validates each step.")
    console.print("  It can be the same as your main model or a smaller faster one.")
    console.print()
    gatekeeper_input = console.input(f"  Gatekeeper model (press enter to use {model}): ").strip()
    if gatekeeper_input:
        gatekeeper = gatekeeper_input
    else:
        gatekeeper = model

    console.print()

    # Update Config

    updated_config = {
        "config": {
            "provider": provider,
            "model": model,
            "gatekeeper": gatekeeper,
            "openrouter_api_key": api_key if provider == "openrouter" else "",
            "email": email,
            "infermedica_app_id": infermedica_id,
            "infermedica_app_key": infermedica_key,
        },
        "user": {
            "active_user": name.lower().replace(" ", "_"),
        }
    }

    with open(config_path, "wb") as f:
        tomli_w.dump(updated_config, f)

    # Create User's and USER.md
    user_id = name.lower().replace(" ", "_")
    profile_dir = Path(USERS) / user_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_path = profile_dir / "USER.md"
    profile_path.write_text(
        f"# Patient Profile — {name}\n\n"
        f"## Personal Information\n"
        f"Name: {name}\n"
        f"Age: {age}\n"
        f"Sex: {sex}\n\n"
        f"## Medical History\n\n"
        f"## Current Medications\n\n"
        f"## Allergies\n\n"
        f"## Family History\n\n"
        f"## Lifestyle\n"
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

# The Configuration Function to view all the registered data of the user and the agent

@app.command()
def config() -> None:
    console.print(f"    [bold]Your Current Configuration Setting [/]")
    console.print()
    console.print(Panel(
        f"  Current User: {ACTIVE_USER}\n"
        f"  Age of User: {AGE}\n"
        f"  User's Sex / Gender: {SEX}\n"
        f"  Model Provider: {PROVIDER}\n"
        f"  Agent Model: {AGENT}\n"
        f"  Gatekeeper Model: {GATEKEEPER}\n"
        f"  User's Email: {EMAIL}\n"
        f"  API Key if Provider is OpenRouter: {OPENROUTER_API_KEY}\n"
        f"  Infermedica ID: {INFERMEDICA_APP_ID}\n"
        f"  Infermedica API Key: {INFERMEDICA_APP_KEY}\n",
        expand=False
    ))
    console.print(" To change any of these settings, use [bold cyan]klini config set {config name} {new value}[/]")
    console.print()

# The Function to view all skills and summaries of the skills

@app.command()
def skills(skill_name = typer.Argument(None)) -> None:
    if skill_name is None:
        console.print(f"    [bold]Agent Skills[/]")
        console.print()
        index_path = Path(INDEX)
        if not index_path.exists():
            console.print("[red]The Skills index file does not exsist, please run [bold cyan]klini init[/] to create all directories and files and then run [bold cyan]klini register[/] to set up the Config![/]")
            return None
        index_data = []
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                index_data.append(json.loads(line))
        if not index_data:
            console.print("[red]No skills found. Start a session with [bold cyan]klini start[/] to let the agent learn new skills![/]")
            return None
        skills = [item['title'] for item in index_data]
        summaries = [item['summary'] for item in index_data]

        for skill, summary in zip(skills, summaries):
            console.print(f"  [cyan]→ {skill}[/]")
            console.print(f"  [white]→ {summary}[/]")
            console.print()
            return None
    else:
        skill_path = Path(SKILLS) / f"{skill_name}.md"
        if not skill_path.exists():
            console.print(f"[red]Skill '{skill_name}' not found.[/]")
            return None
        else:
            skill_content = skill_path.read_text(encoding='utf-8')
            console.print(Markdown(skill_content))

# The Function to view the Clinical Profile of the user that the agent has made or they themselves itself has made

@app.command()
def profile() -> None:
    console.print(f"    [bold]Patient Clinical Profile[/]")
    console.print()
    active_user = ACTIVE_USER
    if not active_user:
        console.print("[red]No active user found. Please run [bold cyan]klini register[/] to set up your profile.[/]")
        return None
    profile_path = Path(USERS) / active_user / "USER.md"
    if not profile_path.exists():
        console.print("[red]Profile file not found. Please run [bold cyan]klini register[/] to set up your profile.[/]")
        return None
    profile_content = profile_path.read_text(encoding='utf-8')
    console.print(Markdown(profile_content))

# Function to view all created users

@app.command()
def users() -> None:
    console.print(f"    [bold]Registered Users[/]")
    console.print()
    users_path = Path(USERS)
    if not users_path.exists():
        console.print("[red]Users directory not found. Please run [bold cyan]klini init[/] to create all directories and files and then run [bold cyan]klini register[/] to set up the Config![/]")
        return None
    user_dirs = [d for d in users_path.iterdir() if d.is_dir()]
    if not user_dirs:
        console.print("[red]No users found. Please run [bold cyan]klini register[/] to create a user profile![/]")
        return None
    for user_dir in user_dirs:
        console.print(f"  [cyan]→ {user_dir.name}[/]")

# The Function for Deletion

@app.command()
def delete(mode: str = typer.Argument('all'), name: str = typer.Argument(None)) -> None:
    if mode == "user":
        if not name:
            console.print("[red]Please specify the username to delete using [bold cyan]klini delete user {username}[/]")
            return None
        else:
            user_path = Path(USERS) / name
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
                    console.print(f"[green]User '{name}' and all associated data deleted successfully.[/]")

    elif mode == "skill":
        if not name:
            console.print("[red]Please specify the skill name to delete using [bold cyan]klini delete skill skill_name[/][/]")
            return None
        skill_path = Path(SKILLS) / f"{name}.md"
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
        
        history_path = Path(HISTORY)
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
            users_path = Path(USERS)
            if users_path.exists():
                try:
                    shutil.rmtree(users_path)
                except Exception as e:
                    console.print(f"[red]An error occurred while deleting users: {e}[/]")
                    return None
                console.print(f"[green]All users and associated data deleted successfully.[/]")
            else:
                console.print(f"[yellow]→ Users directory not found, skipping.[/]")

            # Delete Skills
            skills_path = Path(SKILLS)
            if skills_path.exists():
                try:
                    shutil.rmtree(skills_path)
                except Exception as e:
                    console.print(f"[red]An error occurred while deleting skills: {e}[/]")
                    return None
                console.print(f"[green]All skills deleted successfully.[/]")
            else:
                console.print(f"[yellow]→ Skills directory not found, skipping.[/]")

            # Delete Memory
            memory_path = Path(MEMORY)
            if memory_path.exists():
                try:
                    shutil.rmtree(memory_path)
                except Exception as e:
                    console.print(f"[red]An error occurred while deleting memory: {e}[/]")
                    return None
                console.print(f"[green]All memory deleted successfully.[/]")
            else:
                console.print(f"[yellow]→ Memory directory not found, skipping.[/]")

            # Delete Chroma
            chroma_path = Path(CHROMA)
            if chroma_path.exists():
                try:
                    shutil.rmtree(chroma_path)
                except Exception as e:
                    console.print(f"[red]An error occurred while deleting chroma: {e}[/]")
                    return None
                console.print(f"[green]All chroma data deleted successfully.[/]")
            else:
                console.print(f"[yellow]→ Chroma directory not found, skipping.[/]")

            # Delete Config
            config_path = Path(USER_CONFIG)
            if config_path.exists():
                try:
                    config_path.unlink()
                except Exception as e:
                    console.print(f"[red]An error occurred while deleting user config: {e}[/]")
                    return None
                console.print(f"[green]User config deleted successfully.[/]")
            else:
                console.print(f"[yellow]→ User config file not found, skipping.[/]")
            
            console.print(f"[bold green]All data deleted and agent reset to fresh state. Please run [bold cyan]klini init[/] to initialize and then [bold cyan]klini register[/] to set up your account.")
            return None
    else:
        console.print("[red]Invalid mode. Please choose from: user, skill, session, all.[/]")
        return None
    
# Create A function that is used to run a session witha given state, used to resume sessions or start a brand new one

def _run_session(initial_state: AgentState) -> None:
    # Initialize Necessary Variables
    graph = get_graph()
    agent = get_agent()
    state = initial_state

    console.print(f"[green]Session started. Use Ctrl + C to exit.[/]")
    console.print()

    while True:
        try:
            user_input = console.input("[blue]You:[/] ").strip()
            if not user_input:
                continue
            state['messages'].append(HumanMessage(content=user_input))
            if not state['diagnosis_started']:
                session_text = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
                for m in state['messages']
                ])
                user_message = f"User's Clinical Profile: {state['clinical_profile']}, Current Session Messages: {session_text}"
                context = [
                    {"role": "system", "content": CONVERSATION_PROMPT},
                    {"role": "user", "content": user_message}
                ]
                console.print("[green]Klini:[/] ", end="")
                response = ""
                for chunk in agent.stream(context):
                    if "<DIAGNOSE/>" in chunk.content:
                        state['diagnosis_started'] = True
                        chunk.content = chunk.content.replace("<DIAGNOSE/>", "").strip()
                    console.print(chunk.content, end="", flush=True)
                    response += chunk.content
                state['messages'].append(AIMessage(content=response))
                console.print(f"[green]Klini:[/] {response}")

            elif state['diagnosis_started']:
                with console.status("[green]Klini is diagnosing...[/]"):
                    result = graph.invoke(state)
                console.print(f"[green]Klini[/] Diagnosis complete.")
                state = result
                last_message = state['messages'][-1].content
                console.print(f"[green]Klini:[/] {last_message}")
                state['diagnosis_started'] = False
                state['ever_diagnosed'] = True
        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Session ended by user.[/]")
            if state['messages']:
                with console.status("[green]Saving session...[/]"):
                    history_path = Path(HISTORY)
                    if history_path.exists():
                        with open(history_path, "r") as file:
                            history = json.load(file)
                    else:
                        history = []
                    session_text = "\n".join([
                    f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
                    for m in state['messages']
                    ])
                    context = [
                        {"role": "system", "content": SESSION_TITLE_PROMPT},
                        {"role": "user", "content": session_text}
                    ]
                    session_title = (agent.invoke(context)).content
                    history.append({
                        "session_title": session_title,
                        "time": datetime.now(),
                        "session_id": state['session_id'],
                        "messages": [{"role": m.type, "content": m.content} for m in state['messages']]
                    })
                    with open(history_path, "w") as file:
                        json.dump(history, file, indent=4)

                console.print(f"[green]Session saved with title: {session_title}[/]")
            return None

# Create The Sessions Command to view all sessions or resume a specific session

@app.command()
def sessions(session_title: str = typer.Argument(None)) -> None:
    history_path = Path(HISTORY)
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
        for session in history:
            if session['session_title'].lower() == session_title.lower():
                console.print(f"Resuming session: [cyan]{session['session_title']}[/]")
                console.print()
                session_data = session

                for message in session['messages']:
                    role = 'You' if message['role'] == 'user' else 'Klini'
                    content = message['content']
                    if role == "You":
                        console.print(f"[blue]{role}:[/] {content}")
                    else:
                        console.print(f"[green]{role}:[/] {content}")
        if not session_data:
            console.print(f"[red]Session '{session_title}' not found.[/]")
            return None
        else:
            with open(f"agent\users\{ACTIVE_USER}\USER.md", "r") as file: # Load Clinical Profile
                clinical_profile = file.read()
            state = AgentState(session_id=session_data['session_id'],
                                user_id=ACTIVE_USER,
                                messages=[HumanMessage(content=m['content']) if m['role'] == 'user' else AIMessage(content=m['content']) for m in session_data['messages']],
                                clinical_profile=clinical_profile)
            _run_session(initial_state=state)
            
# The Function to start a new session with the agent for the user

@app.command()
def start() -> None:
    console.print(f"    [bold]Starting New Session with Klini Agent[/]")
    console.print()

    session_id = generate_id()
    with open(f"agent\users\{ACTIVE_USER}\USER.md", "r") as file: # Load Clinical Profile
        clinical_profile = file.read()

    state = AgentState(session_id=session_id, user_id=ACTIVE_USER, clinical_profile=clinical_profile)
    _run_session(initial_state=state)