//! Headless mode: `loops --print "<prompt>"` runs one agent turn in the
//! current directory and exits, for scripts and benchmark runners.
//!
//! There is nobody at a keyboard, so nothing here may wait on input: every
//! permission request is allowed, and the whole turn runs under a wall-clock
//! timeout. The streams are split so a caller can capture them separately:
//! stdout carries only the model's final answer, stderr a live log with one
//! line per agent event. A turn that fails or times out still prints the
//! answer text it got (e.g. a reply cut off at the output limit too often), so
//! check the exit code, not stdout, for success (see [`USAGE`]); `--stats`
//! leaves a JSON summary behind for comparison across runs.
//!
//! The summary is folded from the turn's [`AgentEvent`]s by [`fold_stats`], a
//! pure function, so it is tested without a model.

use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use loops_core::{
    Conversation, Message, Role,
    agent::{
        Agent, AgentEvent, Decision, INSTRUCTIONS_FILE, TurnOptions, default_system_prompt,
        environment_with, load_instructions,
    },
    backend::ModelBackend,
    chat::Thinking,
    lmstudio::{LmStudio, ModelInfo, ModelState},
    tools::ToolSet,
};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::app::describe_instructions;

/// What `--help` prints.
pub const USAGE: &str = "\
loops: a coding agent for models served by LM Studio.

Usage:
  loops [--resume [id]]          start the interactive TUI
  loops --print <prompt> [options]
                                 run one agent turn in the current directory and exit

Options (TUI):
      --resume [id]        open the latest session of this directory, or <id>

Options (with --print):
  -p, --print <prompt>     the user message; the final answer goes to stdout,
                           a log of the turn to stderr
      --max-rounds <n>     model requests allowed in the turn (default 20)
      --timeout <seconds>  wall-clock limit for the turn (default 300)
      --stats <path>       write a JSON summary of the run to <path>
      --model <id>         model to use (default: the loaded one; loads <id>
                           if it is not loaded)
      --think <level>      how much the model thinks: off, low, medium, high,
                           or default (the model's own choice; the default)
      --max-tokens <n>     cap each reply at <n> tokens, thinking included
                           (default: the room left in the context window)
  -h, --help               print this help

Every tool call is allowed without asking.
A failed or timed-out turn still prints any answer text it got.
Exit codes: 0 done, 1 turn failed or timed out, 2 setup error.
";

/// Rounds allowed when `--max-rounds` is not given.
const DEFAULT_MAX_ROUNDS: usize = 20;
/// Wall-clock limit when `--timeout` is not given.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
/// Longest `bash` command shown in a log line; the rest is cut.
const COMMAND_CHARS: usize = 100;

/// What the command line asks for.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    /// No `--print`: the interactive TUI.
    Tui {
        /// `--resume`: which saved session to open first, if any.
        resume: Option<Resume>,
    },
    /// `--print`: one headless turn.
    Print(PrintArgs),
    /// `--help`: print [`USAGE`].
    Help,
}

/// Which session `--resume` opens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resume {
    /// `--resume` alone: the most recently saved session of this directory.
    Latest,
    /// `--resume <id>`: that session.
    Id(String),
}

/// Options of a headless run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrintArgs {
    /// The user message.
    pub prompt: String,
    /// Model requests allowed in the turn ([`Agent::max_iterations`]).
    pub max_rounds: usize,
    /// Wall-clock limit for the turn.
    pub timeout: Duration,
    /// Where to write the JSON summary, if anywhere.
    pub stats: Option<PathBuf>,
    /// Model to use; `None` means whatever is loaded.
    pub model: Option<String>,
    /// How much the model may think.
    pub thinking: Thinking,
    /// Most tokens one reply may use; `None` for no cap beyond the window.
    pub max_tokens: Option<u64>,
}

/// Parses the arguments after the program name. Options other than `--help`
/// and `--resume` need `--print`, and `--resume` is for the TUI only;
/// anything unknown is an error, so a typo in a benchmark script fails loudly
/// instead of opening the TUI. `--resume` takes the next argument as its id
/// unless that starts with `-`.
pub fn parse(args: &[String]) -> Result<Command> {
    let mut prompt = None;
    let mut max_rounds = DEFAULT_MAX_ROUNDS;
    let mut timeout = DEFAULT_TIMEOUT;
    let mut stats = None;
    let mut model = None;
    let mut thinking = Thinking::Default;
    let mut max_tokens = None;
    let mut resume = None;
    // The first `--print` option seen, for the error if `--print` is missing.
    let mut option = None;

    let mut args = args.iter().peekable();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .with_context(|| format!("{arg} needs a value"))
                .map(String::as_str)
        };
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-p" | "--print" => prompt = Some(value()?.to_owned()),
            "--max-rounds" => max_rounds = positive(arg, value()?)?,
            "--timeout" => timeout = Duration::from_secs(positive(arg, value()?)?),
            "--stats" => stats = Some(PathBuf::from(value()?)),
            "--model" => model = Some(value()?.to_owned()),
            "--think" => {
                let level = value()?;
                thinking = Thinking::parse(level).with_context(|| {
                    format!("{arg} needs off, low, medium, high or default, got {level:?}")
                })?;
            }
            "--max-tokens" => max_tokens = Some(positive(arg, value()?)?),
            "--resume" => {
                resume = Some(match args.next_if(|next| !next.starts_with('-')) {
                    Some(id) => Resume::Id(id.clone()),
                    None => Resume::Latest,
                });
                continue;
            }
            other => bail!("unknown argument: {other}"),
        }
        if !matches!(arg.as_str(), "-p" | "--print") {
            option.get_or_insert(arg);
        }
    }

    let Some(prompt) = prompt else {
        if let Some(option) = option {
            bail!("{option} needs --print");
        }
        return Ok(Command::Tui { resume });
    };
    if resume.is_some() {
        bail!("--resume is for the TUI; it cannot be used with --print");
    }
    if prompt.trim().is_empty() {
        bail!("--print needs a non-empty prompt");
    }
    Ok(Command::Print(PrintArgs {
        prompt,
        max_rounds,
        timeout,
        stats,
        model,
        thinking,
        max_tokens,
    }))
}

/// `value` as a whole number above zero, for option `name`.
fn positive<T: std::str::FromStr + Default + PartialOrd>(name: &str, value: &str) -> Result<T> {
    match value.parse::<T>() {
        Ok(n) if n > T::default() => Ok(n),
        _ => bail!("{name} needs a whole number above 0, got {value:?}"),
    }
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The model answered in text.
    Done,
    /// The turn stopped with an error (including running out of rounds).
    Failed,
    /// The wall-clock limit ran out and the turn was aborted.
    Timeout,
}

impl Outcome {
    /// The name used in the stats file.
    fn as_str(self) -> &'static str {
        match self {
            Outcome::Done => "done",
            Outcome::Failed => "failed",
            Outcome::Timeout => "timeout",
        }
    }
}

/// One finished tool call, for the stats file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallStat {
    /// The tool's name.
    pub name: String,
    /// Its one-line result summary.
    pub summary: String,
    /// Whether it failed (or was declined).
    pub is_error: bool,
}

/// The summary of a run that `--stats` writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    /// How the run ended.
    pub outcome: Outcome,
    /// Why it failed or timed out.
    pub error: Option<String>,
    /// The model that ran the turn.
    pub model: String,
    /// Model requests made: one `Usage` event per request.
    pub rounds: usize,
    /// Replies cut off at the output limit and continued (or given up on):
    /// one per `ReplyCutOff` event.
    pub cut_offs: usize,
    /// Finished tool calls, in order. A call cut off by the timeout is not here.
    pub tool_calls: Vec<ToolCallStat>,
    /// Prompt tokens of the last request: how full the context was at the end.
    pub prompt_tokens_last: u64,
    /// Completion tokens summed over all requests.
    pub completion_tokens_total: u64,
    /// Prompt tokens summed over all requests (what was actually processed,
    /// before any prompt caching).
    pub prompt_tokens_total: u64,
    /// Wall-clock time of the turn.
    pub elapsed: Duration,
    /// The model's final text: what it said after its last tool call.
    pub answer: String,
}

impl Stats {
    /// The stats file's JSON.
    pub fn to_json(&self) -> Value {
        let tool_calls: Vec<Value> = self
            .tool_calls
            .iter()
            .map(|call| {
                json!({
                    "name": call.name,
                    "summary": call.summary,
                    "is_error": call.is_error,
                })
            })
            .collect();
        // Keys come out in alphabetical order (`serde_json` sorts them).
        json!({
            "outcome": self.outcome.as_str(),
            "error": self.error,
            "model": self.model,
            "rounds": self.rounds,
            "cut_offs": self.cut_offs,
            "tool_calls": tool_calls,
            "prompt_tokens_last": self.prompt_tokens_last,
            "completion_tokens_total": self.completion_tokens_total,
            "prompt_tokens_total": self.prompt_tokens_total,
            "elapsed_ms": u64::try_from(self.elapsed.as_millis()).unwrap_or(u64::MAX),
            "answer": self.answer,
        })
    }
}

/// Folds a turn's events into its stats. `model` is left empty for the
/// caller to fill in. A turn with no `TurnDone` or `TurnFailed` (its task
/// ended without one) counts as failed; the caller marks a timeout itself.
///
/// The answer is the text that came after the last tool event: within a
/// round all text arrives before its tools start, so text followed by tool
/// calls was narration, not the answer.
pub fn fold_stats(events: &[AgentEvent], elapsed: Duration) -> Stats {
    let mut stats = Stats {
        outcome: Outcome::Failed,
        error: Some("the turn ended without a result".to_owned()),
        model: String::new(),
        rounds: 0,
        cut_offs: 0,
        tool_calls: Vec::new(),
        prompt_tokens_last: 0,
        completion_tokens_total: 0,
        prompt_tokens_total: 0,
        elapsed,
        answer: String::new(),
    };
    for event in events {
        match event {
            AgentEvent::Text(text) => stats.answer.push_str(text),
            AgentEvent::ToolStarted { .. } | AgentEvent::PermissionRequested { .. } => {
                stats.answer.clear();
            }
            AgentEvent::ToolFinished {
                name,
                summary,
                is_error,
                ..
            } => {
                stats.answer.clear();
                stats.tool_calls.push(ToolCallStat {
                    name: name.clone(),
                    summary: summary.clone(),
                    is_error: *is_error,
                });
            }
            AgentEvent::Usage {
                prompt_tokens,
                completion_tokens,
            } => {
                stats.rounds += 1;
                stats.prompt_tokens_last = *prompt_tokens;
                stats.prompt_tokens_total += prompt_tokens;
                stats.completion_tokens_total += completion_tokens;
            }
            AgentEvent::ReplyCutOff { .. } => stats.cut_offs += 1,
            AgentEvent::Thinking => {}
            AgentEvent::TurnDone => {
                stats.outcome = Outcome::Done;
                stats.error = None;
            }
            AgentEvent::TurnFailed(err) => {
                stats.outcome = Outcome::Failed;
                stats.error = Some(err.clone());
            }
        }
    }
    stats
}

/// Runs one headless turn and returns the process exit code: 0 done, 1 failed
/// or timed out, 2 setup error (reported on stderr here). An `Err` is a setup
/// error too, such as a stats file that cannot be written.
pub async fn run(args: PrintArgs) -> Result<u8> {
    let backend: Arc<dyn ModelBackend> = Arc::new(LmStudio::default());
    let info = match pick_model(backend.as_ref(), args.model.as_deref()).await {
        Ok(info) => info,
        Err(err) => {
            eprintln!("error: {err:#}");
            return Ok(2);
        }
    };

    let cwd = std::env::current_dir()?;
    let tools = Arc::new(ToolSet::standard(&cwd));
    if !tools.has("bash") {
        eprintln!("warning: bash not found; the bash tool is off");
    }
    let instructions = match load_instructions(&cwd) {
        Ok(Some(found)) => {
            eprintln!(
                "loops: {INSTRUCTIONS_FILE} loaded, {}",
                describe_instructions(&found)
            );
            Some(found)
        }
        Ok(None) => None,
        Err(err) => {
            eprintln!("warning: {INSTRUCTIONS_FILE} could not be read: {err}");
            None
        }
    };
    let mut agent = Agent::new(backend, tools, Some(default_system_prompt()))
        .with_environment(environment_with(&cwd, instructions.as_ref()));
    agent.max_iterations = args.max_rounds;
    let options = TurnOptions {
        context_length: info.context_length(),
        max_output_tokens: args.max_tokens,
        thinking: args.thinking,
    };
    let model = info.id;
    let window = options
        .context_length
        .map_or("unknown".to_owned(), |n| n.to_string());
    eprintln!(
        "loops: model {model} (window {window}), thinking {}, up to {} rounds, timeout {}s, in {}",
        options.thinking,
        args.max_rounds,
        args.timeout.as_secs(),
        cwd.display()
    );

    let mut conversation = Conversation::new();
    conversation.push(Message::new(Role::User, args.prompt));
    let (sink, mut events) = mpsc::channel(256);
    let (decide, decisions) = mpsc::channel(16);
    let turn_model = model.clone();
    let start = Instant::now();
    let turn = tokio::spawn(async move {
        // The outcome arrives as `TurnDone` / `TurnFailed` on `sink`.
        let _ = agent
            .run_turn(&turn_model, &mut conversation, sink, decisions, &options)
            .await;
    });

    let deadline = tokio::time::sleep(args.timeout);
    tokio::pin!(deadline);
    let mut log = Vec::new();
    let timed_out = loop {
        tokio::select! {
            event = events.recv() => {
                // `None`: the turn task ended and dropped its sender.
                let Some(event) = event else { break false };
                if let Some(line) = log_line(&event) {
                    eprintln!("[{:5.1}s] {line}", start.elapsed().as_secs_f64());
                }
                if matches!(event, AgentEvent::PermissionRequested { .. }) {
                    // A benchmark has nobody to ask. The turn is waiting on
                    // exactly this answer, so the channel has room.
                    let _ = decide.send(Decision::Allow).await;
                }
                log.push(event);
            }
            () = &mut deadline => {
                // Dropping the turn's future kills a running `bash` child.
                turn.abort();
                break true;
            }
        }
    };
    let elapsed = start.elapsed();

    let mut stats = fold_stats(&log, elapsed);
    stats.model = model;
    if timed_out {
        stats.outcome = Outcome::Timeout;
        stats.error = Some(format!("timed out after {}s", args.timeout.as_secs()));
        eprintln!(
            "[{:5.1}s] timed out after {}s",
            elapsed.as_secs_f64(),
            args.timeout.as_secs()
        );
    } else if !log
        .iter()
        .any(|e| matches!(e, AgentEvent::TurnDone | AgentEvent::TurnFailed(_)))
    {
        eprintln!(
            "[{:5.1}s] the turn ended without a result",
            elapsed.as_secs_f64()
        );
    }

    // Also on failure: a reply cut off too often is still most of an answer.
    let answer = stats.answer.trim_end();
    if !answer.is_empty() {
        println!("{answer}");
    }
    if let Some(path) = &args.stats {
        let text = serde_json::to_string_pretty(&stats.to_json())?;
        std::fs::write(path, text + "\n")
            .with_context(|| format!("could not write stats to {}", path.display()))?;
    }
    Ok(if stats.outcome == Outcome::Done { 0 } else { 1 })
}

/// Starts the backend and settles on a model: `wanted` if given (loading it
/// when it is not loaded yet), else the loaded one. Returns its entry, whose
/// context window bounds the turn.
async fn pick_model(backend: &dyn ModelBackend, wanted: Option<&str>) -> Result<ModelInfo> {
    let loaded = backend.start().await.context("could not start LM Studio")?;
    let Some(wanted) = wanted else {
        return match loaded {
            Some(model) => Ok(model),
            None => bail!("no model loaded (load one in LM Studio or pass --model <id>)"),
        };
    };
    // `start` reports only the first loaded model; another may be loaded too.
    let already = backend
        .list_models()
        .await?
        .into_iter()
        .find(|m| m.id == wanted && m.state == ModelState::Loaded);
    if let Some(model) = already {
        return Ok(model);
    }
    eprintln!("loops: loading {wanted}");
    backend
        .load_model(wanted)
        .await
        .with_context(|| format!("could not load {wanted}"))
}

/// The log line for `event`, without its time prefix; `None` for events too
/// small to log one by one (streamed text).
fn log_line(event: &AgentEvent) -> Option<String> {
    let line = match event {
        AgentEvent::Text(_) => return None,
        AgentEvent::Thinking => "thinking".to_owned(),
        AgentEvent::ToolStarted {
            name, arguments, ..
        } => format!("tool {name}: {}", call_label(arguments)),
        AgentEvent::PermissionRequested { scope, .. } => format!("allowed {scope}"),
        AgentEvent::ToolFinished {
            name,
            summary,
            is_error,
            ..
        } => {
            let first = summary.lines().next().unwrap_or_default();
            let verb = if *is_error { "error" } else { "done" };
            format!("{verb} {name}: {first}")
        }
        AgentEvent::Usage {
            prompt_tokens,
            completion_tokens,
        } => format!("usage {prompt_tokens}+{completion_tokens}"),
        AgentEvent::ReplyCutOff { attempt, max } => {
            format!("reply cut off, continuing ({attempt}/{max})")
        }
        AgentEvent::TurnDone => "turn done".to_owned(),
        AgentEvent::TurnFailed(err) => format!("turn failed: {err}"),
    };
    Some(line)
}

/// A call's arguments on one line: `$ <command>` for `bash` (first line, cut
/// after [`COMMAND_CHARS`] characters), the `path` for file tools, else the
/// raw JSON.
fn call_label(arguments: &str) -> String {
    let Ok(Value::Object(map)) = serde_json::from_str::<Value>(arguments) else {
        return arguments.to_owned();
    };
    match (map.get("command"), map.get("path")) {
        (Some(Value::String(command)), _) => {
            let first = command.lines().next().unwrap_or_default();
            let mut shown: String = first.chars().take(COMMAND_CHARS).collect();
            if shown != command.trim_end() {
                shown.push_str(" …");
            }
            format!("$ {shown}")
        }
        (_, Some(Value::String(path))) => path.clone(),
        _ => arguments.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_strs(args: &[&str]) -> Result<Command> {
        let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
        parse(&args)
    }

    fn print_args(args: &[&str]) -> PrintArgs {
        match parse_strs(args).unwrap() {
            Command::Print(args) => args,
            other => panic!("expected Print, got {other:?}"),
        }
    }

    fn parse_error(args: &[&str]) -> String {
        format!("{:#}", parse_strs(args).unwrap_err())
    }

    #[test]
    fn no_arguments_run_the_tui() {
        assert_eq!(parse_strs(&[]).unwrap(), Command::Tui { resume: None });
    }

    #[test]
    fn resume_takes_an_optional_id() {
        let tui = |resume| Command::Tui {
            resume: Some(resume),
        };
        assert_eq!(parse_strs(&["--resume"]).unwrap(), tui(Resume::Latest));
        assert_eq!(
            parse_strs(&["--resume", "1794067200-042"]).unwrap(),
            tui(Resume::Id("1794067200-042".into()))
        );
        // An option after it is not taken as the id.
        assert_eq!(parse_strs(&["--resume", "-h"]).unwrap(), Command::Help);
        assert_eq!(
            parse_error(&["--resume", "--model", "m"]),
            "--model needs --print"
        );
        assert_eq!(parse_error(&["--resume", "a", "b"]), "unknown argument: b");
    }

    #[test]
    fn resume_cannot_be_used_with_print() {
        let error = "--resume is for the TUI; it cannot be used with --print";
        assert_eq!(parse_error(&["--resume", "-p", "x"]), error);
        assert_eq!(parse_error(&["-p", "x", "--resume", "abc"]), error);
    }

    #[test]
    fn print_alone_uses_the_defaults() {
        let args = print_args(&["--print", "fix it"]);
        assert_eq!(
            args,
            PrintArgs {
                prompt: "fix it".into(),
                max_rounds: 20,
                timeout: Duration::from_secs(300),
                stats: None,
                model: None,
                thinking: Thinking::Default,
                max_tokens: None,
            }
        );
        assert_eq!(print_args(&["-p", "fix it"]), args);
    }

    #[test]
    fn every_option_is_read_in_any_order() {
        let args = print_args(&[
            "--stats",
            "out/stats.json",
            "--max-rounds",
            "5",
            "-p",
            "fix the typo",
            "--timeout",
            "120",
            "--model",
            "google/gemma-4-e4b",
            "--think",
            "Off",
            "--max-tokens",
            "2048",
        ]);
        assert_eq!(
            args,
            PrintArgs {
                prompt: "fix the typo".into(),
                max_rounds: 5,
                timeout: Duration::from_secs(120),
                stats: Some(PathBuf::from("out/stats.json")),
                model: Some("google/gemma-4-e4b".into()),
                thinking: Thinking::Off,
                max_tokens: Some(2048),
            }
        );
    }

    #[test]
    fn help_wins_over_everything_else() {
        assert_eq!(parse_strs(&["--help"]).unwrap(), Command::Help);
        assert_eq!(parse_strs(&["-p", "x", "-h"]).unwrap(), Command::Help);
    }

    #[test]
    fn a_missing_value_is_an_error() {
        for option in [
            "--print",
            "-p",
            "--max-rounds",
            "--timeout",
            "--stats",
            "--model",
            "--think",
            "--max-tokens",
        ] {
            assert_eq!(
                parse_error(&["-p", "x", option]),
                format!("{option} needs a value")
            );
        }
    }

    #[test]
    fn bad_values_and_unknown_arguments_are_errors() {
        assert!(parse_error(&["-p", "x", "--max-rounds", "0"]).contains("above 0"));
        assert!(parse_error(&["-p", "x", "--timeout", "-1"]).contains("above 0"));
        assert!(parse_error(&["-p", "x", "--timeout", "1.5"]).contains("whole number"));
        assert_eq!(
            parse_error(&["-p", "x", "--fast"]),
            "unknown argument: --fast"
        );
        assert_eq!(parse_error(&["hello"]), "unknown argument: hello");
        assert!(parse_error(&["-p", "  "]).contains("non-empty"));
        assert_eq!(parse_error(&["--model", "m"]), "--model needs --print");
        assert!(parse_error(&["-p", "x", "--max-tokens", "0"]).contains("above 0"));
        assert_eq!(
            parse_error(&["-p", "x", "--think", "max"]),
            r#"--think needs off, low, medium, high or default, got "max""#
        );
        assert_eq!(parse_error(&["--think", "low"]), "--think needs --print");
    }

    fn usage(prompt_tokens: u64, completion_tokens: u64) -> AgentEvent {
        AgentEvent::Usage {
            prompt_tokens,
            completion_tokens,
        }
    }

    fn tool(id: &str, name: &str, summary: &str, is_error: bool) -> [AgentEvent; 2] {
        [
            AgentEvent::ToolStarted {
                id: id.into(),
                name: name.into(),
                arguments: "{}".into(),
            },
            AgentEvent::ToolFinished {
                id: id.into(),
                name: name.into(),
                summary: summary.into(),
                is_error,
                output: String::new(),
            },
        ]
    }

    #[test]
    fn a_done_turn_folds_rounds_totals_tools_and_the_answer() {
        let mut events = vec![
            AgentEvent::Thinking,
            AgentEvent::Text("Let me look.".into()),
            usage(1000, 50),
        ];
        events.extend(tool("1", "read_file", "4 lines", false));
        events.extend([AgentEvent::Thinking, usage(1200, 80)]);
        events.extend(tool("2", "edit_file", "old text not found", true));
        events.extend([
            AgentEvent::Text("Fixed ".into()),
            AgentEvent::Text("the typo.".into()),
            usage(1401, 131),
            AgentEvent::TurnDone,
        ]);

        let stats = fold_stats(&events, Duration::from_millis(10_300));
        assert_eq!(stats.outcome, Outcome::Done);
        assert_eq!(stats.error, None);
        assert_eq!(stats.rounds, 3);
        assert_eq!(stats.cut_offs, 0);
        assert_eq!(stats.prompt_tokens_last, 1401);
        assert_eq!(stats.prompt_tokens_total, 3601);
        assert_eq!(stats.completion_tokens_total, 261);
        assert_eq!(stats.answer, "Fixed the typo.");
        assert_eq!(
            stats.tool_calls,
            [
                ToolCallStat {
                    name: "read_file".into(),
                    summary: "4 lines".into(),
                    is_error: false,
                },
                ToolCallStat {
                    name: "edit_file".into(),
                    summary: "old text not found".into(),
                    is_error: true,
                },
            ]
        );

        let json = stats.to_json();
        assert_eq!(json["outcome"], "done");
        assert_eq!(json["error"], Value::Null);
        assert_eq!(json["elapsed_ms"], 10_300);
        assert_eq!(json["tool_calls"][1]["is_error"], true);
        assert_eq!(json["answer"], "Fixed the typo.");
        assert_eq!(json["cut_offs"], 0);
    }

    #[test]
    fn cut_offs_are_counted_and_the_continued_answer_is_whole() {
        let cut_off = |attempt| AgentEvent::ReplyCutOff { attempt, max: 2 };
        let events = [
            AgentEvent::Text("Hello, ".into()),
            usage(500, 256),
            cut_off(1),
            AgentEvent::Text("wor".into()),
            usage(780, 256),
            cut_off(2),
            AgentEvent::Text("ld.".into()),
            usage(1060, 3),
            AgentEvent::TurnDone,
        ];
        let stats = fold_stats(&events, Duration::ZERO);
        assert_eq!(stats.cut_offs, 2);
        assert_eq!(stats.rounds, 3);
        assert_eq!(stats.answer, "Hello, world.");
        assert_eq!(stats.to_json()["cut_offs"], 2);
        assert_eq!(
            log_line(&cut_off(1)).unwrap(),
            "reply cut off, continuing (1/2)"
        );
    }

    /// The agent gives up on the third cut-off without a `ReplyCutOff` for
    /// it; the text streamed so far is still the answer printed.
    #[test]
    fn a_reply_cut_off_too_often_fails_but_keeps_its_text() {
        let cut_off = |attempt| AgentEvent::ReplyCutOff { attempt, max: 2 };
        let events = [
            AgentEvent::Text("partial".into()),
            usage(500, 256),
            cut_off(1),
            AgentEvent::Text(" more".into()),
            usage(780, 256),
            cut_off(2),
            AgentEvent::Text(" end".into()),
            usage(1060, 256),
            AgentEvent::TurnFailed("reply cut off at the output limit 3 times".into()),
        ];
        let stats = fold_stats(&events, Duration::ZERO);
        assert_eq!(stats.outcome, Outcome::Failed);
        assert_eq!(stats.cut_offs, 2);
        assert_eq!(stats.answer, "partial more end");
    }

    #[test]
    fn a_failed_turn_keeps_its_error_and_has_no_answer_after_tools() {
        let mut events = vec![AgentEvent::Text("Checking.".into()), usage(900, 20)];
        events.extend(tool("1", "bash", "exit 0, 3 lines", false));
        events.push(AgentEvent::TurnFailed("stopped after 1 tool rounds".into()));

        let stats = fold_stats(&events, Duration::ZERO);
        assert_eq!(stats.outcome, Outcome::Failed);
        assert_eq!(stats.error.as_deref(), Some("stopped after 1 tool rounds"));
        assert_eq!(stats.rounds, 1);
        assert_eq!(stats.answer, "");
        assert_eq!(stats.to_json()["outcome"], "failed");
    }

    #[test]
    fn a_turn_without_a_final_event_counts_as_failed() {
        let stats = fold_stats(&[AgentEvent::Thinking], Duration::ZERO);
        assert_eq!(stats.outcome, Outcome::Failed);
        assert!(stats.error.is_some());
        assert_eq!(stats.rounds, 0);
    }

    #[test]
    fn log_lines_name_the_tool_and_its_command() {
        let started = AgentEvent::ToolStarted {
            id: "1".into(),
            name: "bash".into(),
            arguments: r#"{"command":"cargo test\necho hi"}"#.into(),
        };
        assert_eq!(log_line(&started).unwrap(), "tool bash: $ cargo test …");
        let [_, finished] = tool("1", "bash", "exit 0, 12 lines", false);
        assert_eq!(log_line(&finished).unwrap(), "done bash: exit 0, 12 lines");
        assert_eq!(log_line(&usage(1401, 131)).unwrap(), "usage 1401+131");
        assert_eq!(log_line(&AgentEvent::Text("x".into())), None);
        assert_eq!(call_label(r#"{"path":"notes.txt","old":"a"}"#), "notes.txt");
        assert_eq!(call_label("not json"), "not json");
    }
}
