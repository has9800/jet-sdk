import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

app = typer.Typer(help="Jet SDK CLI: train and evaluate custom LLMs with a streamlined end-to-end workflow.")
console = Console()

@app.callback()
def _banner():
    console.print(Panel.fit("🚀 Jet SDK • Fine-tuning made simple", style="bold magenta"))

@app.command(help="Show version.")
def version():
    import jet
    console.print(f"Jet SDK {jet.__version__}", style="green")

@app.command(help="Interactive setup (project name, model, dataset, engine).")
def init():
    try:
        import questionary  # lazy import
    except Exception:
        console.print("Tip: install CLI extras for interactive prompts: pip install 'jet-ai-sdk[cli]'", style="yellow")
        raise
    answers = questionary.prompt([
        {"type": "text", "name": "project", "message": "Project name:"},
        {"type": "text", "name": "model", "message": "Base model (e.g., sshleifer/tiny-gpt2):"},
        {"type": "text", "name": "data", "message": "Dataset (e.g., text:./data.txt or hf_id):"},
        {"type": "select", "name": "engine", "message": "Engine:", "choices": ["auto","unsloth","hf"], "default": "auto"},
    ])
    console.print(f"Creating project [cyan]{answers['project']}[/cyan]…")
    console.print("✅ Done.", style="green")

@app.command(help="Train a model (lazy imports inside handler).")
def train(
    model: str = typer.Option(..., help="Base model id"),
    data: str = typer.Option(..., help="Dataset source (e.g., text:path, csv:path, parquet:path, or HF id)"),
    engine: str = typer.Option("auto", help="Engine: auto|unsloth|hf"),
    epochs: int = typer.Option(1, help="Epochs"),
    max_seq: int = typer.Option(2048, help="Max sequence length"),
    output_dir: str = typer.Option("outputs/model", help="Output dir"),
    per_device_batch: int = typer.Option(1, help="Per-device batch size"),
    grad_accum: int = typer.Option(16, help="Gradient accumulation steps"),
    lr: float = typer.Option(2e-4, help="Learning rate"),
):
    from jet.options import TrainOptions  # lazy
    from jet.dataset import DatasetBuilder  # lazy
    from jet.train import train_with_options  # lazy

    with Progress() as prog:
        task = prog.add_task("Preparing…", total=3)
        ds = DatasetBuilder(data, split="train").load()
        prog.update(task, advance=1)
        opts = TrainOptions(
            model=model, engine=engine, epochs=epochs, max_seq=max_seq,
            output_dir=output_dir, per_device_batch=per_device_batch,
            grad_accum=grad_accum, lr=lr,
        )
        prog.update(task, advance=1)
        job = train_with_options(opts, ds)
        prog.update(task, advance=1)
    console.print(f"✅ Trained and saved to [green]{job.model_dir}[/green]")

@app.command(help="Evaluate a trained model (lazy import of evaluation stack).")
def eval(
    model_dir: str = typer.Option(..., help="Path to saved model"),
    prompts: str = typer.Option(..., help="Comma-separated prompts"),
    refs: str = typer.Option(..., help="Comma-separated references"),
):
    from jet.eval import Evaluator  # lazy
    P = [p.strip() for p in prompts.split(",")]
    R = [r.strip() for r in refs.split(",")]
    with Progress() as prog:
        task = prog.add_task("Evaluating…", total=1)
        report = Evaluator(model_dir, bf16=False).evaluate(P, R)
        prog.update(task, advance=1)
    console.print(report)

def app():
    return typer.run(_banner)