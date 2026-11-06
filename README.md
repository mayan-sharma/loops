# loops

A coding-agent harness built from scratch, in Rust, as a learning project.

The end goal is a desktop app in the style of the Claude Code desktop app and the
Codex app, with swappable backends: this custom harness and [pi](https://github.com/badlogic/pi-mono).
As the custom harness grows it gets compared against pi on the same tasks.

Right now it is a terminal app that talks to local models through LM Studio.

## Run

```
cargo install --path crates/loops-tui   # installs the `loops` binary
loops
```

Requires [LM Studio](https://lmstudio.ai) with the `lms` CLI on PATH.

## Layout

- `crates/loops-core` — providers, LM Studio control, the chat/agent loop. No UI.
- `crates/loops-tui` — the terminal UI (ratatui + crossterm). Produces the `loops` binary.
