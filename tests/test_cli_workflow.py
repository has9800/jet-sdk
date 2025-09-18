import pytest
from typer.testing import CliRunner
from jet.cli import app

runner = CliRunner()

@pytest.fixture
def mock_inputs():
    return iter([
        "📦 Prepare data",
        "🧩 Curated list",
        "Tiny Shakespeare (demo)",
        "",  # default text field
        "🧠 Train",
        "🧩 Curated models",
        "Tiny GPT-2 (smoke)",
        "auto",
        "1",
        "1024",
        "outputs/model",
        "🧪 Evaluate",
        "outputs/model",
        "Hello",
        "Hi",
        "🚪 Exit"
    ])

def patch_questionary(monkeypatch, inputs):
    def fake_select(message, choices, **kwargs):
        ans = next(inputs)
        assert ans in choices
        return ans
    class FakePrompt:
        def __init__(self, default=None): self.default = default
        def ask(self): return self.default if self.default is not None else ""
    def text(message, default=None, **kwargs):
        class T(FakePrompt):
            def ask(self_inner): 
                try:
                    val = next(inputs)
                except StopIteration:
                    val = default
                return val if val != "" else (default or "")
        return T(default)
    def path(message, default=None, **kwargs):
        return text(message, default)

    monkeypatch.setattr("questionary.select", fake_select)
    monkeypatch.setattr("questionary.text", text)
    monkeypatch.setattr("questionary.path", path)

def test_full_cli_flow(monkeypatch, mock_inputs):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.delenv("_TYPER_COMPLETE", raising=False)
    patch_questionary(monkeypatch, mock_inputs)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Jet CLI" in result.output
    assert "Select an action" in result.output
    assert "✅ Trained and saved" in result.output
