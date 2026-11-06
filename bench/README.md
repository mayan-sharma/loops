# bench

Runs the same tasks through `loops --print` and `pi -p`, then scores each run with pytest.

- `tasks/<slug>/`: stub, test file and `INSTRUCTIONS.md` of an Exercism Python exercise
  (from https://github.com/exercism/python, MIT). `solutions/` holds the reference
  solutions, used only to check that the tests pass with a correct answer.
- `run.py`: the runner. `python run.py [--runs 3] [--tasks a,b] [--harness loops,pi] [--timeout 300]`.
- `results/`: one JSON (everything, including logs) and one Markdown table per run.

Setup, once:

```
py -3.12 -m venv .venv && .venv/Scripts/python -m pip install pytest
cargo build --release -p loops-tui
```

pi needs an `lmstudio` provider in `~/.pi/agent/models.json` pointing at
`http://localhost:1234/v1` with the model you loaded in LM Studio.
