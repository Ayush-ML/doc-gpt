from time import sleep
from rich.console import Console

console = Console()

# Start the spinner with a message
with console.status("[bold green]Working on tasks...") as status:
    # Simulate work
    for n in range(1, 4):
        sleep(1)
        console.log(f"Task {n} complete")
    
    # You can update the text or spinner style dynamically
    status.update("[bold yellow]Almost finished...", spinner="bouncingBar")
    sleep(1)

console.print("Done! :heavy_check_mark:")
