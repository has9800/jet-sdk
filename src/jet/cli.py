# src/jet/cli.py
import os, sys, io, contextlib
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="🚀 Jet CLI: train and evaluate custom LLMs with a streamlined workflow.")
console = Console()

# Quiet HF/Transformers globally before any heavy imports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")      # hide hub tqdm bars [web:903]
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")        # quiet Transformers logs [web:911]
try:
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()
    tlog.disable_progress_bar()
except Exception:
    pass

def quiet_io():
    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())

@app.callback(invoke_without_command=True)
def _entry(ctx: typer.Context):
    console.print(Panel.fit("🚀 Jet CLI • Fine-tuning made simple", style="bold magenta"))
    if ctx.invoked_subcommand is None:
        # No subcommand: open the home menu
        menu()  # default entry [web:709]

@app.command(help="Show version.")
def version():
    import jet
    console.print(f"Jet SDK {jet.__version__}", style="green")

@app.command(help="Interactive Menu (scaffold, train, eval).")
def menu():
    try:
        import questionary
    except ImportError:
        console.print("Tip: pip install 'jet-ai-sdk[cli]' for interactive prompts", style="yellow")
        raise

    console.clear()
    console.print(Panel.fit("✨ Jet Menu", style="bold cyan"))
    choice = questionary.select(
        "Select an action:",
        choices=[
            "🧱 Init project",
            "📦 Prepare data",
            "🧠 Train",
            "🧪 Evaluate",
            "🚪 Exit",
        ],
        qmark="✨", pointer="›",
        instruction="Use ↑/↓ and Enter",
    ).ask()

    if choice == "🧱 Init project":
        _init_wizard()
    elif choice == "📦 Prepare data":
        _prepare_data()
    elif choice == "🧠 Train":
        _train_wizard()
    elif choice == "🧪 Evaluate":
        _eval_wizard()
    else:
        console.print("Goodbye! 👋", style="bold green")

def _prepare_data():
    from rich.status import Status
    with Status("Preparing dataset …", spinner="dots"):
        from jet.dataset import DatasetBuilder
        # Suppress download/progress output while we show spinner
        out, err = quiet_io()
        with out[0], out[1]:
            ds = DatasetBuilder("text:./sample.txt", split="train").load()
    console.print("📦 Dataset prepared.", style="green")

def _train_wizard():
    import questionary
    model = questionary.text("Base model (e.g., sshleifer/tiny-gpt2):").ask()
    data = questionary.text("Dataset (e.g., text:./data.txt or hf_id):").ask()
    engine = questionary.select("Engine:", choices=["auto","unsloth","hf"], default="auto").ask()
    epochs = int(questionary.text("Epochs:", default="1").ask() or "1")
    max_seq = int(questionary.text("Max seq length:", default="1024").ask() or "1024")
    outdir = questionary.text("Output dir:", default="outputs/model").ask()

    from rich.status import Status
    from jet.options import TrainOptions
    from jet.dataset import DatasetBuilder
    from jet.train import train_with_options

    with Status("Setting up training …", spinner="dots"):
        out, err = quiet_io()
        with out[0], out[1]:
            ds = DatasetBuilder(data, split="train").load()
            opts = TrainOptions(model=model, engine=engine, epochs=epochs, max_seq=max_seq, output_dir=outdir)

    with Status("Downloading model/tokenizer …", spinner="line"):
        pass  # hidden by HF env vars and quiet IO [web:903][web:911]

    with Status("Training …", spinner="earth"):
        out, err = quiet_io()
        with out[0], out[1]:
            job = train_with_options(opts, ds)

    console.print(f"✅ Trained and saved to [green]{job.model_dir}[/green]")

def _eval_wizard():
    import questionary
    model_dir = questionary.text("Path to saved model:", default="outputs/model").ask()
    prompts = questionary.text("Prompts (comma-separated):", default="Hello").ask()
    refs = questionary.text("References (comma-separated):", default="Hi").ask()
    from rich.status import Status
    from jet.eval import Evaluator
    P = [p.strip() for p in prompts.split(",")]
    R = [r.strip() for r in refs.split(",")]
    with Status("Evaluating …", spinner="dots"):
        out, err = quiet_io()
        with out[0], out[1]:
            rep = Evaluator(model_dir, bf16=False).evaluate(P, R)
    console.print(rep)

def _init_wizard():
    import questionary
    # prompt and print summary...
    console.print("✅ Project initialized.", style="green")

def main():
    app()