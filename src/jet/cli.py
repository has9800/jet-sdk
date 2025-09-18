import os, sys, io, json, contextlib, pathlib
import typer
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from importlib import resources

app = typer.Typer(help="🚀 Jet CLI • train and evaluate your custom LLMs with a streamlined workflow!")
console = Console()

# Quiet HF Hub / Transformers / Datasets logs globally [web:903][web:911][web:1035]
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()
    tlog.disable_progress_bar()
except Exception:
    pass
try:
    import datasets.logging as dlog
    dlog.set_verbosity_error()
except Exception:
    pass

@contextlib.contextmanager
def quiet_io():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield  # silence nested prints/tqdm while spinner runs [web:1028]

def _load_json_resource(package: str, name: str):
    with resources.files(package).joinpath(name).open("r", encoding="utf-8") as f:
        return json.load(f)

def _banner():
    console.print(Panel.fit("🚀 Jet CLI • Fine-tuning made simple", style="bold magenta"))

@app.callback(invoke_without_command=True)
def _entry(ctx: typer.Context):
    # Avoid interactive prompts during shell completion or when stdin isn't a TTY [web:1031]
    if os.environ.get("_TYPER_COMPLETE") or not sys.stdin.isatty():
        return
    _banner()
    if ctx.invoked_subcommand is None:
        return menu()  # auto home screen on plain `jet` [web:1028]

@app.command(help="Show version.")
def version():
    import jet
    console.print(f"Jet SDK {jet.__version__}", style="green")

@app.command(help="Interactive Menu (prepare, train, evaluate).")
def menu():
    try:
        import questionary
    except ImportError:
        console.print("Tip: pip install 'jet-ai-sdk[cli]' for interactive prompts", style="yellow")
        raise

    while True:
        console.clear()
        _banner()
        choice = questionary.select(
            "Select an action:",
            choices=["📦 Prepare data", "🧠 Train", "🧪 Evaluate", "🚪 Exit"],
            qmark="✨", pointer="›"
        ).ask()
        if choice == "🚪 Exit":
            console.print("Goodbye! 👋", style="bold green")
            break
        if choice == "📦 Prepare data":
            _prepare_data()
        elif choice == "🧠 Train":
            _train_wizard()
        elif choice == "🧪 Evaluate":
            _eval_wizard()

def _prepare_data():
    try:
        import questionary
    except ImportError:
        console.print("Tip: pip install 'jet-ai-sdk[cli]'", style="yellow")
        raise
    curated = _load_json_resource("jet.data", "curated_datasets.json")  # bundled [web:1086]
    mode = questionary.select(
        "Choose a data source:",
        choices=[
            "🧩 Curated list",
            "🧭 Paste HF dataset ID (org/name)",
            "🔗 Paste remote URL (csv/jsonl/txt/parquet)",
            "📄 Local text file",
            "🧾 Local CSV",
            "🧾 Local JSON/JSONL",
            "🧾 Local Parquet",
            "↩️ Back",
        ],
        qmark="✨", pointer="›"
    ).ask()
    if mode == "↩️ Back":
        return

    src, text_field, input_field, target_field = None, None, None, None
    if mode == "🧩 Curated list":
        import questionary
        pick = questionary.select("Pick a dataset:", choices=[c["name"] for c in curated], qmark="📦").ask()
        meta = next(c for c in curated if c["name"] == pick)
        src = f"hf:{meta['id']}"
        text_field = meta.get("text_field") or "text"
        # allow override
        text_field = questionary.text("Text field:", default=text_field).ask()
    elif mode == "🧭 Paste HF dataset ID (org/name)":
        dsid = questionary.text("Dataset ID:", default="sshleifer/tiny-shakespeare").ask()
        src = f"hf:{dsid}"
        text_field = questionary.text("Text field:", default="text").ask()
    elif mode == "🔗 Paste remote URL (csv/jsonl/txt/parquet)":
        url = questionary.text("HTTP(S) URL:", default="https://example.com/data.jsonl").ask()
        src = url
        text_field = questionary.text("Text field (if CSV/JSON/Parquet):", default="text").ask()
    elif mode == "📄 Local text file":
        path = questionary.path("Path to .txt:", default="sample.txt").ask()
        src = f"text:{path}"
        text_field = "text"
    elif mode == "🧾 Local CSV":
        path = questionary.path("Path to .csv:", default="data.csv").ask()
        src = f"csv:{path}"
        text_field = questionary.text("Text column:", default="text").ask()
    elif mode == "🧾 Local JSON/JSONL":
        path = questionary.path("Path to .json/.jsonl:", default="data.jsonl").ask()
        src = f"json:{path}"
        text_field = questionary.text("Text field:", default="text").ask()
    elif mode == "🧾 Local Parquet":
        path = questionary.path("Path to .parquet:", default="data.parquet").ask()
        src = f"parquet:{path}"
        text_field = questionary.text("Text column:", default="text").ask()

    if not src:
        return

    cfgp = pathlib.Path("jet.config.json")
    cfg = {"dataset": src, "text_field": text_field, "input_field": input_field, "target_field": target_field}
    try:
        cfgp.write_text(json.dumps(cfg, indent=2))
        console.print(f"📝 Saved config to [green]{cfgp}[/green]")
    except Exception:
        pass

    from jet.dataset import DatasetBuilder
    with Status(f"Loading dataset ({src}) …", spinner="dots"):
        with quiet_io():
            ds = DatasetBuilder(src, split="train", text_field=text_field, input_field=input_field, target_field=target_field).load()  # unified text [web:1083]

    # Optional preview
    try:
        preview = pathlib.Path("prepared_preview.txt")
        lines = []
        take_n = 5
        if hasattr(ds, "select"):
            sample = ds.select(range(min(take_n, len(ds))))
            for ex in sample:
                lines.append(ex["text"])
        else:
            # streaming
            for i, ex in enumerate(ds):
                if i >= take_n: break
                lines.append(ex.get("text", str(ex)))
        preview.write_text("\n---\n".join(lines))
        console.print(f"📦 Preview saved to [green]{preview}[/green]")
    except Exception:
        console.print("📦 Dataset prepared.", style="green")

def _train_wizard():
    try:
        import questionary
    except ImportError:
        console.print("Tip: pip install 'jet-ai-sdk[cli]'", style="yellow"); raise
    curated_models = _load_json_resource("jet.data", "curated_models.json")

    # Choose model
    model_mode = questionary.select(
        "Choose model source:", choices=["🧩 Curated models", "✏️ Enter model ID"], qmark="🧠", pointer="›"
    ).ask()
    if model_mode == "🧩 Curated models":
        pick = questionary.select("Pick a model:", choices=[m["name"] for m in curated_models]).ask()
        model_id = next(m["id"] for m in curated_models if m["name"] == pick)
    else:
        model_id = questionary.text("Model repo ID (e.g., sshleifer/tiny-gpt2):", default="sshleifer/tiny-gpt2").ask()

    # Gather training params
    engine = questionary.select("Engine:", choices=["auto","unsloth","hf"], default="auto").ask()
    epochs = int(questionary.text("Epochs:", default="1").ask() or "1")
    max_seq = int(questionary.text("Max seq length:", default="1024").ask() or "1024")
    outdir = questionary.text("Output dir:", default="outputs/model").ask()

    # Load config for dataset fields if present
    cfg = {}
    cfgp = pathlib.Path("jet.config.json")
    if cfgp.exists():
        try:
            cfg = json.loads(cfgp.read_text())
        except Exception:
            cfg = {}

    from jet.options import TrainOptions
    from jet.dataset import DatasetBuilder
    from jet.train import train_with_options

    src = cfg.get("dataset", "text:./sample.txt")
    text_field = cfg.get("text_field", "text")
    input_field = cfg.get("input_field")
    target_field = cfg.get("target_field")

    with Status("Preparing dataset …", spinner="dots"):
        with quiet_io():
            ds = DatasetBuilder(src, split="train", text_field=text_field, input_field=input_field, target_field=target_field).load()  # normalized [web:1083]

    opts = TrainOptions(
        model=model_id, engine=engine, epochs=epochs, max_seq=max_seq, output_dir=outdir,
        dataset_source=src, text_field=text_field, input_field=input_field, target_field=target_field
    )

    with Status("Training …", spinner="earth"):
        with quiet_io():
            job = train_with_options(opts, ds)  # TRL SFTConfig + processing_class [web:347]

    console.print(f"✅ Trained and saved to [green]{job.model_dir}[/green]")

def _eval_wizard():
    try:
        import questionary
    except ImportError:
        console.print("Tip: pip install 'jet-ai-sdk[cli]'", style="yellow"); raise
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