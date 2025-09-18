# tests/test_cli_workflow.py

def patch_questionary(monkeypatch, inputs):
    # Fake prompt object that mimics questionary's interface
    class FakePrompt:
        def __init__(self, value): self.value = value
        def ask(self): return self.value

    def fake_select(message, choices, **kwargs):
        ans = next(inputs)
        assert ans in choices, f"Unexpected selection {ans} for {message}"
        return FakePrompt(ans)

    def fake_text(message, default=None, **kwargs):
        try:
            val = next(inputs)
        except StopIteration:
            val = default
        if val == "":  # allow pressing Enter for default
            val = default
        return FakePrompt(val)

    def fake_path(message, default=None, **kwargs):
        return fake_text(message, default=default)

    # Patch both module-level and CLI-local imports to be safe
    monkeypatch.setattr("questionary.select", fake_select, raising=True)
    monkeypatch.setattr("questionary.text", fake_text, raising=True)
    monkeypatch.setattr("questionary.path", fake_path, raising=True)
    # If your CLI imports questionary inside functions, also patch the resolved symbol:
    monkeypatch.setattr("jet.cli.questionary", type("Q", (), {
        "select": fake_select, "text": fake_text, "path": fake_path
    }))
