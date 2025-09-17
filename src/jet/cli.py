# src/jet/cli.py
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
import questionary

app = typer.Typer(help="Jet SDK CLI: Train and evaluate custom LLMs with a streamlined workflow.")

console = Console()

@app.callback()
def main():
    console.print(Panel.fit("🚀 Jet SDK CLI • Train and evaluate you custom LLMs with a streamlined workflow, one command deploy.", style="bold magenta"))

@app.command(help="Interactive setup (like create-next-app).")
def init():
    answers = questionary.prompt([
        {"type": "text", "name": "project", "message": "Project name:"},
        {"type": "text", "name": "model", "message": "Base model (e.g., sshleifer/tiny-gpt2):"},
        {"type": "text", "name": "data", "message": "Dataset (e.g., text:./data.txt or hf_id):"},
        {"type": "select", "name": "engine", "message": "Engine:", "choices": ["auto","unsloth","hf"], "default": "auto"},
    ])
    console.print(f"Creating project [cyan]{answers['project']}[/cyan]…")
    # Write a minimal config or scaffold files here
    console.print("✅ Done.", style="green")

@app.command(help="Train a model...")
def train(
    model: str = typer.Option(..., help="Base model id"),
    data: str = typer.Option(..., help="Dataset source (e.g., text:path, csv:path, hf_id)"),
    engine: str = typer.Option("auto", help="Engine: auto|unsloth|hf"),
    epochs: int = typer.Option(1, help="Epochs"),
    max_seq: int = typer.Option(2048, help="Max sequence length"),
    output_dir: str = typer.Option("outputs/model", help="Output dir"),
):
    with Progress() as prog:
        task = prog.add_task("Preparing…", total=3)
        # Lazy imports to avoid heavy deps during --help
        from jet.options import TrainOptions
        from jet.dataset import DatasetBuilder
        from jet.train import train_with_options

        prog.update(task, advance=1)
        ds = DatasetBuilder(data, split="train").load()
        prog.update(task, advance=1)
        opts = TrainOptions(model=model, engine=engine, epochs=epochs, max_seq=max_seq, output_dir=output_dir)
        job = train_with_options(opts, ds)
        prog.update(task, advance=1)
    console.print(f"✅ Trained and saved to [green]{job.model_dir}[/green]")

@app.command(help="Evaluate a trained model...")
def eval(
    model_dir: str = typer.Option(..., help="Path to saved model"),
    prompts: str = typer.Option(..., help="Comma-separated prompts"),
    refs: str = typer.Option(..., help="Comma-separated references"),
):
    # Lazy import to avoid pulling evaluate/transformers until needed
    from jet.eval import Evaluator
    P = [p.strip() for p in prompts.split(",")]
    R = [r.strip() for r in refs.split(",")]
    rep = Evaluator(model_dir, bf16=False).evaluate(P, R)
    console.print(rep)
