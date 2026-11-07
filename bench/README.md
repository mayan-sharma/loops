# bench

Runs the same tasks through `loops --print` and `pi -p`, then scores each run with pytest.

- `tasks/<slug>/`: stub, test file and `INSTRUCTIONS.md` of an Exercism Python exercise
  (from https://github.com/exercism/python, MIT). `solutions/` holds the reference
  solutions, used only to check that the tests pass with a correct answer.
- `run.py`: the runner.
  `python run.py [--runs 3] [--tasks a,b] [--harness loops,pi] [--timeout 300] [--model google/gemma-4-e4b]`.
  `--model` is the LM Studio id; pi gets the part after the slash as its model name.
- `results/`: one JSON (everything, including logs) and one Markdown table per run.

Setup, once:

```
py -3.12 -m venv .venv && .venv/Scripts/python -m pip install pytest
cargo build --release -p loops-tui
```

pi needs an `lmstudio` provider in `~/.pi/agent/models.json` pointing at
`http://localhost:1234/v1`, with one entry per model you load. Its `contextWindow` must
match the window the model is loaded with, and should be 16k or more: pi keeps a 4,096-token
safety margin inside it and sends `max_tokens: 1` once the conversation fills the rest.
