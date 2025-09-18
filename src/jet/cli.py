# src/jet/cli.py
import os, io, contextlib, pathlib, json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

app = typer.Typer(help="🚀 Jet CLI • train and evaluate your own custom LLMs with a streamlined workflow.")
console = Console()

# Global quiet settings for a clean UX
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # hide hub tqdm bars [web:903]
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")    # quiet Transformers logs [web:911]
try:
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()
    tlog.disable_progress_bar()
except Exception:
    pass
try:
    import datasets, datasets.logging as dlog
    dlog.set_verbosity_error()  # silence datasets logging too [web:1035][web:1036]
except Exception:
    pass

@contextlib.contextmanager
def quiet_io():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield  # silence nested stdout/stderr while spinner shows [web:1028]

@app.callback(invoke_without_command=True)
def _entry(ctx: typer.Context):
    console.print(Panel.fit("🚀 Jet CLI • Fine-tuning made simple", style="bold magenta"))
    if ctx.invoked_subcommand is None:
        return menu()  # auto-launch the home screen on plain `jet` [web:1028]

@app.command(help="Show version.")
def version():
    import jet
    console.print(f"Jet SDK {jet.__version__}", style="green")

@app.command(help="Interactive Menu (scaffold, prepare, train, eval).")
def menu():
    try:
        import questionary
    except ImportError:
        console.print("Tip: pip install 'jet-ai-sdk[cli]' for interactive prompts", style="yellow")
        raise

    while True:
        console.clear()
        console.print(Panel.fit("✨ Jet Menu", style="bold cyan"))
        choice = questionary.select(
            "Select an action:",
            choices=["🧱 Init project", "📦 Prepare data", "🧠 Train", "🧪 Evaluate", "🚪 Exit"],
            qmark="✨", pointer="›", instruction="Use ↑/↓ and Enter",
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
    name = questionary.text("Project name:", default="jet-project").ask()
    base = pathlib.Path(name)
    base.mkdir(parents=True, exist_ok=True)
    (base / "data").mkdir(exist_ok=True)
    (base / "outputs").mkdir(exist_ok=True)
    # Minimal config scaffold
    (base / "jet.config.json").write_text(json.dumps({
        "model": "sshleifer/tiny-gpt2", "engine": "auto", "max_seq": 1024, "epochs": 1,
        "dataset": "text:./data/sample.txt", "output_dir": "./outputs/model"
    }, indent=2))
    # Sample data
    (base / "data" / "sample.txt").write_text("Hello world\nSecond sample\n")
    console.print(f"✅ Created project at [green]{base}[/green] with config and sample data.")

def _prepare_data():
    from jet.dataset import DatasetBuilder
    with Status("Preparing dataset …", spinner="dots"):
        with quiet_io():  # silence HF/Datasets logs while spinner renders [web:1035][web:1028]
            if not pathlib.Path("sample.txt").exists():
                pathlib.Path("sample.txt").write_text("Hello world\nSecond sample\n")
            ds = DatasetBuilder("text:./sample.txt", split="train").load()
    console.print("📦 Dataset prepared.", style="green")

def _train_wizard():
    import questionary
    from jet.options import TrainOptions
    from jet.dataset import DatasetBuilder
    from jet.train import train_with_options

    model = questionary.text("Base model:", default="sshleifer/tiny-gpt2").ask()
    data = questionary.text("Dataset (e.g., text:./data.txt or hf_id):", default="text:./sample.txt").ask()
    engine = questionary.select("Engine:", choices=["auto","unsloth","hf"], default="auto").ask()
    epochs = int(questionary.text("Epochs:", default="1").ask() or "1")
    max_seq = int(questionary.text("Max seq length:", default="1024").ask() or "1024")
    outdir = questionary.text("Output dir:", default="outputs/model").ask()

    with Status("Setting up training …", spinner="dots"):
        with quiet_io():
            ds = DatasetBuilder(data, split="train").load()
            opts = TrainOptions(model=model, engine=engine, epochs=epochs, max_seq=max_seq, output_dir=outdir)

    with Status("Training …", spinner="earth"):
        with quiet_io():
            job = train_with_options(opts, ds)

    console.print(f"✅ Trained and saved to [green]{job.model_dir}[/green]")

def _eval_wizard():
    import questionary
    from jet.eval import Evaluator
    model_dir = questionary.text("Path to saved model:", default="outputs/model").ask()
    prompts = questionary.text("Prompts (comma-separated):", default="Hello").ask()
    refs = questionary.text("References (comma-separated):", default="Hi").ask()
    P = [p.strip() for p in prompts.split(",")]
    R = [r.strip() for r in refs.split(",")]
    with Status("Evaluating …", spinner="dots"):
        with quiet_io():
            rep = Evaluator(model_dir, bf16=False).evaluate(P, R)
    console.print(rep)

def main():
    app()
