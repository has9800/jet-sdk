# src/jet/cli.py
import os
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer(help="Jet SDK CLI: train and evaluate custom LLMs with a streamlined workflow.")
console = Console()

def _quiet_hf():
    # Hide HF Hub progress bars and quiet Transformers logs
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        tlog.disable_progress_bar()
    except Exception:
        pass

def _banner():
    console.print(Panel.fit("🚀 Jet SDK • Fine-tuning made simple", style="bold magenta"))

@app.callback()
def _entry():
    _quiet_hf()
    _banner()

@app.command(help="Show version.")
def version():
    import jet
    console.print(f"Jet SDK {jet.__version__}", style="green")

@app.command(help="Interactive Menu (scaffold, train, eval).")
def menu():
    _quiet_hf()
    try:
        import questionary  # lazy import; install extra: pip install 'jet-ai-sdk[cli]'
    except ImportError:
        console.print("Tip: install CLI extras for interactive prompts: pip install 'jet-ai-sdk[cli]'", style="yellow")
        raise

    console.clear()
    _banner()
    console.print("Welcome! Choose what to do next:", style="bold cyan")

    while True:
        choice = questionary.select(
            "Select from:",
            choices=[
                "🧱 Init project",
                "📦 Prepare data",
                "🧠 Train",
                "🧪 Evaluate",
                "🚪 Exit",
            ],
            qmark="✨",
            pointer="›",
            instruction="Use ↑/↓ and Enter",
        ).ask()

        if choice == "🚪 Exit":
            console.print("Goodbye! 👋", style="bold green")
            break

        if choice == "🧱 Init project":
            _init_wizard()
        elif choice == "📦 Prepare data":
            _prepare_data()
        elif choice == "🧠 Train":
            _train_wizard()
        elif choice == "🧪 Evaluate":
            _eval_wizard()

def _init_wizard():
    import questionary
    name = questionary.text("Project name:").ask()
    model = questionary.text("Base model (e.g., sshleifer/tiny-gpt2):").ask()
    data = questionary.text("Dataset (e.g., text:./data.txt or hf_id):").ask()
    engine = questionary.select("Engine:", choices=["auto","unsloth","hf"], default="auto").ask()
    table = Table(title="Init Summary", box=box.SIMPLE)
    table.add_column("Field"); table.add_column("Value")
    for k,v in [("Project", name), ("Model", model), ("Dataset", data), ("Engine", engine)]:
        table.add_row(k, v)
    console.print(table)
    console.print("✅ Project initialized (scaffold your files as needed).", style="green")

def _prepare_data():
    from rich.status import Status
    with Status("Preparing dataset …", spinner="dots") as status:
        try:
            from jet.dataset import DatasetBuilder
            # Example: small text sample; in real use, prompt for path or HF ID
            ds = DatasetBuilder("text:./sample.txt", split="train").load()
            status.update("Dataset ready!")
        except Exception as e:
            console.print(f"❌ Data preparation failed: {e}", style="bold red")
            return
    console.print("📦 Dataset prepared.", style="green")

def _train_wizard():
    import questionary
    model = questionary.text("Base model (e.g., sshleifer/tiny-gpt2):").ask()
    data = questionary.text("Dataset (e.g., text:./data.txt or hf_id):").ask()
    engine = questionary.select("Engine:", choices=["auto","unsloth","hf"], default="auto").ask()
    epochs = int(questionary.text("Epochs:", default="1").ask() or "1")
    max_seq = int(questionary.text("Max seq length:", default="1024").ask() or "1024")
    out = questionary.text("Output dir:", default="outputs/model").ask()

    from rich.status import Status
    with Status("Setting up training …", spinner="dots"):
        from jet.options import TrainOptions
        from jet.dataset import DatasetBuilder
        from jet.train import train_with_options
        ds = DatasetBuilder(data, split="train").load()
        opts = TrainOptions(model=model, engine=engine, epochs=epochs, max_seq=max_seq, output_dir=out)

    with Status("Downloading model/tokenizer …", spinner="line"):
        pass  # hidden by HF env + quiet logging

    with Status("Training … this may take a while", spinner="earth"):
        job = train_with_options(opts, ds)

    console.print(f"✅ Trained and saved to [green]{job.model_dir}[/green]")

def _eval_wizard():
    import questionary
    model_dir = questionary.text("Path to saved model:", default="outputs/model").ask()
    prompts = questionary.text("Prompts (comma-separated):", default="Hello").ask()
    refs = questionary.text("References (comma-separated):", default="Hi").ask()
    from rich.status import Status
    with Status("Evaluating …", spinner="dots"):
        from jet.eval import Evaluator
        P = [p.strip() for p in prompts.split(",")]
        R = [r.strip() for r in refs.split(",")]
        rep = Evaluator(model_dir, bf16=False).evaluate(P, R)
    console.print(rep)

def main():
    app()
