//! The agent loop: ask the model, run the tools it asks for, hand it the
//! results, and repeat until it answers in plain text.
//!
//! ```text
//!  conversation ──> model ──> text only? ──yes──> done
//!       ^                        │ no
//!       └── tool results <── run each tool call
//! ```
//!
//! One pass through that loop is a *round*. A *turn* is everything from the
//! user's message to the model's final text, so it can hold many rounds.
//!
//! # Permission
//!
//! A tool that changes things ([`crate::tools::Tool::mutates`]) is not run on
//! the model's say-so. The agent works out a preview of the change, sends
//! [`AgentEvent::PermissionRequested`], and waits for a [`Decision`] from the
//! front end. A preview that fails (the change could not succeed anyway) goes
//! straight back to the model as the result, without asking. Read-only tools
//! never ask. The request carries the call's
//! [`permission_scope`](crate::tools::Tool::permission_scope), so a front end
//! that remembers "don't ask again" knows what the yes covers: a whole file
//! tool, or one exact `bash` command.

use std::{path::Path, sync::Arc};

use anyhow::{Result, bail};
use tokio::sync::mpsc;

use crate::{
    Conversation, Message, Role, ToolCall,
    backend::ModelBackend,
    chat::{ChatDelta, ToolCallAssembler},
    tools::{ToolSet, ToolSpec},
};

/// Rounds allowed in one turn before the agent gives up.
pub const DEFAULT_MAX_ITERATIONS: usize = 20;

/// [`AgentEvent::ToolFinished`] summary of a call the user declined.
pub const DECLINED: &str = "declined";

/// [`AgentEvent::ToolFinished`] summary of a call declined because nobody was
/// left to ask (the decision channel closed).
pub const DECLINED_NOBODY_TO_ASK: &str = "declined: nobody to ask";

/// What the model reads as the result of a declined call.
const DECLINED_RESULT: &str =
    "declined by the user; do not retry this change, ask what they want instead";

/// The user's answer to [`AgentEvent::PermissionRequested`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Run the tool.
    Allow,
    /// Do not run it; the model is told it was declined.
    Deny,
}

/// What the agent reports while a turn runs, for a front end to show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    /// More answer text from the model.
    Text(String),
    /// The model started thinking (sent once per round; the thinking itself is not kept).
    Thinking,
    /// A tool call is about to run.
    ToolStarted {
        /// The call's id.
        id: String,
        /// The tool's name.
        name: String,
        /// The raw JSON arguments.
        arguments: String,
    },
    /// The call just started (its `ToolStarted` came first) changes things,
    /// and the agent waits for a [`Decision`] before running it.
    PermissionRequested {
        /// The call's id.
        id: String,
        /// The tool's name.
        name: String,
        /// The raw JSON arguments.
        arguments: String,
        /// What the call would do, from [`crate::tools::Tool::preview`].
        preview: String,
        /// What "don't ask again" for this call would cover, from
        /// [`crate::tools::Tool::permission_scope`]: e.g. `edit_file`, or
        /// `bash: cargo test`.
        scope: String,
    },
    /// A tool call finished.
    ToolFinished {
        /// The call's id.
        id: String,
        /// The tool's name.
        name: String,
        /// One line about the result, e.g. `31 lines`, the error text, or
        /// [`DECLINED`] / [`DECLINED_NOBODY_TO_ASK`].
        summary: String,
        /// Whether the tool failed or was declined (the model still sees why
        /// as the result).
        is_error: bool,
    },
    /// Tokens a request used (once per round, after its stream ends). The
    /// prompt holds the whole conversation as sent, so the last of these says
    /// how full the model's context window is.
    Usage {
        /// Everything the model read: system prompt, tools, the conversation.
        prompt_tokens: u64,
        /// What it wrote, thinking included.
        completion_tokens: u64,
    },
    /// The model answered in text; the turn is over.
    TurnDone,
    /// The turn stopped early; the reason, formatted for display.
    TurnFailed(String),
}

/// A model plus the tools it may use, and the loop that connects them.
pub struct Agent {
    /// Serves the model.
    pub backend: Arc<dyn ModelBackend>,
    /// Offered to the model every round.
    pub tools: Arc<ToolSet>,
    /// Sent first in every request; never stored in the conversation.
    pub system_prompt: Option<String>,
    /// Facts about where the harness runs (directory, OS). Prepended to the
    /// first user message of every request, never stored in the conversation,
    /// so the system prompt stays identical across runs and models.
    pub environment: Option<String>,
    /// Rounds allowed per turn.
    pub max_iterations: usize,
}

/// What the model said in one round.
#[derive(Debug, Default)]
struct Reply {
    text: String,
    calls: Vec<ToolCall>,
}

/// Whether a tool call may run.
enum Verdict {
    /// Run it.
    Run,
    /// It cannot succeed; this error is the result.
    Fail(anyhow::Error),
    /// Not allowed; the summary says why ([`DECLINED`] or [`DECLINED_NOBODY_TO_ASK`]).
    Decline(&'static str),
}

impl Agent {
    /// An agent allowing [`DEFAULT_MAX_ITERATIONS`] rounds per turn.
    pub fn new(
        backend: Arc<dyn ModelBackend>,
        tools: Arc<ToolSet>,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            backend,
            tools,
            system_prompt,
            environment: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }

    /// Sets the environment block (see [`default_environment`]).
    pub fn with_environment(mut self, environment: impl Into<String>) -> Self {
        self.environment = Some(environment.into());
        self
    }

    /// Runs one turn on `conversation`, whose last message is the user's, and
    /// appends what happens: per round, the model's reply and any tool
    /// results; finally the model's text answer. Progress goes to `sink`, which
    /// always gets `TurnDone` or `TurnFailed` last.
    ///
    /// Before each mutating tool call the turn sends `PermissionRequested` and
    /// waits on `decisions` for the answer; several such calls in one round
    /// are asked one at a time, in order. If `decisions` is closed there is
    /// nobody to ask, and the call is declined.
    ///
    /// To cancel, drop the future (abort its task), also while it waits for a
    /// decision. `conversation` stays valid to send again: a round is appended
    /// only once all its tools have run (or been declined), so an interrupted
    /// round leaves nothing, never a tool call without its result.
    pub async fn run_turn(
        &self,
        model: &str,
        conversation: &mut Conversation,
        sink: mpsc::Sender<AgentEvent>,
        mut decisions: mpsc::Receiver<Decision>,
    ) -> Result<()> {
        let result = self
            .run_rounds(model, conversation, &sink, &mut decisions)
            .await;
        let last = match &result {
            Ok(()) => AgentEvent::TurnDone,
            Err(err) => AgentEvent::TurnFailed(format!("{err:#}")),
        };
        emit(&sink, last).await;
        result
    }

    /// The loop itself.
    async fn run_rounds(
        &self,
        model: &str,
        conversation: &mut Conversation,
        sink: &mpsc::Sender<AgentEvent>,
        decisions: &mut mpsc::Receiver<Decision>,
    ) -> Result<()> {
        let specs = self.tools.specs();
        for _ in 0..self.max_iterations {
            let (result, reply) = self.ask_model(model, conversation, &specs, sink).await;
            if let Err(err) = result {
                // Keep whatever text arrived; tool calls are dropped since they never ran.
                if !reply.text.is_empty() {
                    conversation.push(Message::new(Role::Assistant, reply.text));
                }
                return Err(err);
            }
            if reply.calls.is_empty() {
                conversation.push(Message::new(Role::Assistant, reply.text));
                return Ok(());
            }

            let mut results = Vec::new();
            for call in &reply.calls {
                results.push(self.run_tool(call, sink, decisions).await);
            }
            // The request and its results go in together (see `run_turn`).
            conversation.push(Message::assistant_with_tool_calls(reply.text, reply.calls));
            for result in results {
                conversation.push(result);
            }
        }
        bail!("stopped after {} tool rounds", self.max_iterations)
    }

    /// Streams one model reply, forwarding text and thinking to `sink` as it
    /// arrives. Returns the backend's result together with everything that
    /// arrived, so text before a failure is not lost.
    async fn ask_model(
        &self,
        model: &str,
        conversation: &Conversation,
        specs: &[ToolSpec],
        sink: &mpsc::Sender<AgentEvent>,
    ) -> (Result<()>, Reply) {
        let mut messages = Vec::new();
        if let Some(prompt) = &self.system_prompt {
            messages.push(Message::new(Role::System, prompt.clone()));
        }
        messages.extend_from_slice(conversation.messages());
        if let Some(environment) = &self.environment
            && let Some(first_user) = messages.iter_mut().find(|m| m.role == Role::User)
        {
            first_user.content = format!(
                "<environment>\n{environment}\n</environment>\n\n{}",
                first_user.content
            );
        }

        let (tx, mut deltas) = mpsc::channel(64);
        let collect = async {
            let mut reply = Reply::default();
            let mut calls = ToolCallAssembler::default();
            let mut thinking = false;
            while let Some(delta) = deltas.recv().await {
                match delta {
                    ChatDelta::Content(text) => {
                        reply.text.push_str(&text);
                        emit(sink, AgentEvent::Text(text)).await;
                    }
                    ChatDelta::Reasoning(_) if !thinking => {
                        thinking = true;
                        emit(sink, AgentEvent::Thinking).await;
                    }
                    ChatDelta::Reasoning(_) | ChatDelta::Done { .. } => {}
                    ChatDelta::Usage {
                        prompt_tokens,
                        completion_tokens,
                    } => {
                        let usage = AgentEvent::Usage {
                            prompt_tokens,
                            completion_tokens,
                        };
                        emit(sink, usage).await;
                    }
                    ChatDelta::ToolCall {
                        index,
                        id,
                        name,
                        arguments,
                    } => calls.push(index, id, name, &arguments),
                }
            }
            reply.calls = calls.finish();
            reply
        };
        // `chat` owns `tx` and drops it when it returns, which ends `collect`.
        tokio::join!(self.backend.chat(model, &messages, specs, tx), collect)
    }

    /// Runs one tool call, after asking if it changes things, and returns its
    /// result message. A failing or declined tool is not a failed turn: the
    /// error text, or the refusal, becomes the result the model reads.
    async fn run_tool(
        &self,
        call: &ToolCall,
        sink: &mpsc::Sender<AgentEvent>,
        decisions: &mut mpsc::Receiver<Decision>,
    ) -> Message {
        emit(
            sink,
            AgentEvent::ToolStarted {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        )
        .await;
        let failed = |err: anyhow::Error| (format!("error: {err:#}"), format!("{err:#}"), true);
        let (output, summary, is_error) = match self.verdict(call, sink, decisions).await {
            Verdict::Run => match self.tools.run(&call.name, &call.arguments).await {
                Ok(output) => {
                    let summary = self.tools.summarize(&call.name, &output);
                    (output, summary, false)
                }
                Err(err) => failed(err),
            },
            Verdict::Fail(err) => failed(err),
            Verdict::Decline(why) => (DECLINED_RESULT.to_owned(), why.to_owned(), true),
        };
        emit(
            sink,
            AgentEvent::ToolFinished {
                id: call.id.clone(),
                name: call.name.clone(),
                summary,
                is_error,
            },
        )
        .await;
        Message::tool_result(&call.id, output)
    }

    /// Whether `call` may run: always for a read-only tool; for a mutating
    /// one, only if its preview works and the user then allows it.
    async fn verdict(
        &self,
        call: &ToolCall,
        sink: &mpsc::Sender<AgentEvent>,
        decisions: &mut mpsc::Receiver<Decision>,
    ) -> Verdict {
        if !self.tools.mutates(&call.name) {
            return Verdict::Run;
        }
        let preview = match self.tools.preview(&call.name, &call.arguments).await {
            Ok(preview) => preview,
            Err(err) => return Verdict::Fail(err),
        };
        let scope = match self.tools.permission_scope(&call.name, &call.arguments) {
            Ok(scope) => scope,
            Err(err) => return Verdict::Fail(err),
        };
        emit(
            sink,
            AgentEvent::PermissionRequested {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                preview,
                scope,
            },
        )
        .await;
        match decisions.recv().await {
            Some(Decision::Allow) => Verdict::Run,
            Some(Decision::Deny) => Verdict::Decline(DECLINED),
            None => Verdict::Decline(DECLINED_NOBODY_TO_ASK),
        }
    }
}

/// Sends `event`, ignoring a closed channel: nobody listening is not a reason
/// to stop the turn.
async fn emit(sink: &mpsc::Sender<AgentEvent>, event: AgentEvent) {
    let _ = sink.send(event).await;
}

/// The system prompt the harness starts with. Static on purpose: nothing about
/// the machine or the run goes in here (see [`default_environment`]), so the
/// same text is sent to every model on every run. Kept in one place so it is
/// easy to revise. Rationale and the survey it came from: docs/notes/system-prompts.md.
pub fn default_system_prompt() -> String {
    "You are loops, a coding assistant running in a terminal. \
     Tools: list_files to see what exists, read_file to read, edit_file to \
     replace one exact unique snippet in a file, write_file to create a file or \
     replace one whole. bash runs a shell command in the working directory; use \
     it to build, run tests and check your work, and prefer read_file and \
     list_files for reading. Never run a destructive command (deleting, \
     resetting, force-pushing) unless the user asked for exactly that. Never \
     guess paths or contents: read a file before you edit it, and answer from \
     what tools returned. Prefer edit_file for changes to existing files. \
     Every change needs the user's approval; if one is declined, do not retry \
     it, ask what they want instead. If something is missing or ambiguous, say \
     so instead of inventing it. Keep replies short: lead with the answer, then \
     the evidence as `path:line` references. No preamble, no restating the \
     question."
        .to_owned()
}

/// Facts about the run, for [`Agent::environment`]. On Windows the shell is
/// Git Bash (see [`crate::tools::find_bash`]), where `\` is an escape
/// character, so the model is told to write paths with `/`.
pub fn default_environment(cwd: &Path) -> String {
    let shell = if cfg!(windows) {
        "bash (Git Bash; use forward slashes in paths)"
    } else {
        "bash"
    };
    format!(
        "working directory: {}\noperating system: {}\nshell: {shell}",
        cwd.display(),
        std::env::consts::OS
    )
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Mutex, time::Instant};

    use anyhow::anyhow;

    use super::*;
    use crate::{
        backend::BoxFuture,
        lmstudio::{LmStudio, ModelInfo},
        test_support::TempDir,
        tools::find_bash,
    };
    use serde_json::json;

    /// Replies from a script: call `n` gets round `n`, or the last round once
    /// the script runs out. Fails after sending its round if `fail` is set.
    #[derive(Default)]
    struct Scripted {
        rounds: Vec<Vec<ChatDelta>>,
        fail: Option<&'static str>,
        /// Messages and tool count of each request, in order.
        requests: Mutex<Vec<(Vec<Message>, usize)>>,
    }

    impl ModelBackend for Scripted {
        fn start(&self) -> BoxFuture<'_, Result<Option<ModelInfo>>> {
            Box::pin(async { Ok(None) })
        }

        fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn load_model<'a>(&'a self, _id: &'a str) -> BoxFuture<'a, Result<ModelInfo>> {
            Box::pin(async { Err(anyhow!("not supported")) })
        }

        fn unload_model<'a>(&'a self, _id: &'a str) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Err(anyhow!("not supported")) })
        }

        fn chat<'a>(
            &'a self,
            _model: &'a str,
            messages: &'a [Message],
            tools: &'a [ToolSpec],
            sink: mpsc::Sender<ChatDelta>,
        ) -> BoxFuture<'a, Result<()>> {
            let mut requests = self.requests.lock().unwrap();
            let round = self.rounds[requests.len().min(self.rounds.len() - 1)].clone();
            requests.push((messages.to_vec(), tools.len()));
            Box::pin(async move {
                for delta in round {
                    let _ = sink.send(delta).await;
                }
                match self.fail {
                    Some(err) => Err(anyhow!(err)),
                    None => Ok(()),
                }
            })
        }
    }

    fn call(id: &str, name: &str, arguments: &str) -> ChatDelta {
        call_at(0, id, name, arguments)
    }

    /// Call number `index` of a round asking for several.
    fn call_at(index: usize, id: &str, name: &str, arguments: &str) -> ChatDelta {
        ChatDelta::ToolCall {
            index,
            id: Some(id.into()),
            name: Some(name.into()),
            arguments: arguments.into(),
        }
    }

    fn text(text: &str) -> ChatDelta {
        ChatDelta::Content(text.into())
    }

    /// An agent over `backend` whose tools work in this crate's directory.
    fn agent(backend: Arc<Scripted>) -> Agent {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Agent::new(
            backend,
            Arc::new(ToolSet::read_only(root)),
            Some("be brief".into()),
        )
    }

    /// A decision channel already holding `answers`, then closed: once they
    /// run out, any further question finds nobody to ask.
    fn answers(answers: &[Decision]) -> mpsc::Receiver<Decision> {
        let (tx, rx) = mpsc::channel(answers.len().max(1));
        for answer in answers {
            tx.try_send(*answer).unwrap();
        }
        rx
    }

    /// Runs one turn on a conversation holding just `prompt`; returns the
    /// result, the conversation after, and every event sent.
    async fn turn(agent: &Agent, prompt: &str) -> (Result<()>, Conversation, Vec<AgentEvent>) {
        turn_answering(agent, prompt, &[]).await
    }

    /// [`turn`], answering permission requests with `decisions` in order.
    async fn turn_answering(
        agent: &Agent,
        prompt: &str,
        decisions: &[Decision],
    ) -> (Result<()>, Conversation, Vec<AgentEvent>) {
        let mut conversation = Conversation::new();
        conversation.push(Message::new(Role::User, prompt));
        let (tx, mut rx) = mpsc::channel(256);
        let result = agent
            .run_turn("m", &mut conversation, tx, answers(decisions))
            .await;
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        (result, conversation, events)
    }

    #[tokio::test]
    async fn tool_call_then_text() {
        let args = r#"{"path":"Cargo.toml"}"#;
        let backend = Arc::new(Scripted {
            rounds: vec![
                vec![
                    call("call_1", "read_file", args),
                    ChatDelta::Done {
                        finish_reason: Some("tool_calls".into()),
                    },
                ],
                vec![
                    ChatDelta::Reasoning("hm".into()),
                    text("It is "),
                    text("loops-core."),
                ],
            ],
            ..Scripted::default()
        });
        let agent = agent(backend.clone());
        let (result, conversation, events) = turn(&agent, "what crate is this?").await;
        result.unwrap();

        let msgs = conversation.messages();
        let roles: Vec<Role> = msgs.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            [Role::User, Role::Assistant, Role::Tool, Role::Assistant]
        );
        let asked = ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: args.into(),
        };
        assert_eq!(msgs[1].tool_calls, [asked]);
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("call_1"));
        assert!(msgs[2].content.starts_with("1: [package]\n"));
        assert_eq!(msgs[3], Message::new(Role::Assistant, "It is loops-core."));

        let lines = msgs[2].content.lines().count();
        assert_eq!(
            events,
            [
                AgentEvent::ToolStarted {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: args.into(),
                },
                AgentEvent::ToolFinished {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    summary: format!("{lines} lines"),
                    is_error: false,
                },
                AgentEvent::Thinking,
                AgentEvent::Text("It is ".into()),
                AgentEvent::Text("loops-core.".into()),
                AgentEvent::TurnDone,
            ]
        );

        // Round 2 saw the system prompt, the call and its result, and the tools.
        let requests = backend.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let (second, tools) = &requests[1];
        assert_eq!(*tools, 2);
        assert_eq!(second[0], Message::new(Role::System, "be brief"));
        assert_eq!(&second[1..], &msgs[..3]);
    }

    /// Every round's usage is passed on as it arrives, so the last one before
    /// `TurnDone` describes the request that ended the turn.
    #[tokio::test]
    async fn usage_is_reported_once_per_round_before_turn_done() {
        let usage = |prompt_tokens, completion_tokens| ChatDelta::Usage {
            prompt_tokens,
            completion_tokens,
        };
        let backend = Arc::new(Scripted {
            rounds: vec![
                vec![
                    call("c", "list_files", ""),
                    ChatDelta::Done {
                        finish_reason: Some("tool_calls".into()),
                    },
                    usage(100, 10),
                ],
                vec![
                    text("ok"),
                    ChatDelta::Done {
                        finish_reason: Some("stop".into()),
                    },
                    usage(150, 2),
                ],
            ],
            ..Scripted::default()
        });
        let (result, _conversation, events) = turn(&agent(backend), "look").await;
        result.unwrap();

        let steps: Vec<String> = events
            .iter()
            .map(|event| match event {
                AgentEvent::Usage {
                    prompt_tokens,
                    completion_tokens,
                } => format!("usage {prompt_tokens}+{completion_tokens}"),
                AgentEvent::ToolStarted { id, .. } => format!("start {id}"),
                AgentEvent::ToolFinished { id, .. } => format!("finish {id}"),
                AgentEvent::Text(text) => format!("text {text}"),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            steps,
            [
                "usage 100+10",
                "start c",
                "finish c",
                "text ok",
                "usage 150+2",
                "TurnDone",
            ]
        );
    }

    #[tokio::test]
    async fn tool_errors_are_results_not_failures() {
        let backend = Arc::new(Scripted {
            rounds: vec![vec![call("c", "rm_rf", "{}")], vec![text("ok")]],
            ..Scripted::default()
        });
        let (result, conversation, events) = turn(&agent(backend), "go").await;
        result.unwrap();
        assert_eq!(
            conversation.messages()[2].content,
            "error: unknown tool: rm_rf"
        );
        assert!(events.contains(&AgentEvent::ToolFinished {
            id: "c".into(),
            name: "rm_rf".into(),
            summary: "unknown tool: rm_rf".into(),
            is_error: true,
        }));
    }

    #[tokio::test]
    async fn gives_up_after_max_iterations() {
        let backend = Arc::new(Scripted {
            rounds: vec![vec![call("c", "list_files", "")]],
            ..Scripted::default()
        });
        let mut agent = agent(backend);
        agent.max_iterations = 3;
        let (result, conversation, events) = turn(&agent, "loop forever").await;

        assert_eq!(
            result.unwrap_err().to_string(),
            "stopped after 3 tool rounds"
        );
        assert_eq!(
            events.last(),
            Some(&AgentEvent::TurnFailed(
                "stopped after 3 tool rounds".into()
            ))
        );
        // The user message, then three complete rounds of call + result.
        assert_eq!(conversation.messages().len(), 1 + 3 * 2);
        assert_eq!(conversation.messages()[6].role, Role::Tool);
    }

    #[tokio::test]
    async fn a_failed_request_keeps_the_text_but_not_the_calls() {
        let backend = Arc::new(Scripted {
            rounds: vec![vec![text("par"), call("c", "read_file", "{}")]],
            fail: Some("connection reset"),
            ..Scripted::default()
        });
        let (result, conversation, events) = turn(&agent(backend), "hi").await;
        assert!(result.is_err());
        assert_eq!(
            conversation.messages()[1],
            Message::new(Role::Assistant, "par")
        );
        assert_eq!(conversation.messages().len(), 2);
        assert_eq!(
            events.last(),
            Some(&AgentEvent::TurnFailed("connection reset".into()))
        );
    }

    #[test]
    fn system_prompt_is_static_and_environment_names_the_directory() {
        let prompt = default_system_prompt();
        assert!(prompt.starts_with("You are loops"));
        assert!(
            !prompt.contains(std::env::consts::OS),
            "no run facts in the prompt"
        );

        let environment = default_environment(Path::new("/work/repo"));
        assert!(environment.contains("working directory: /work/repo"));
        assert!(environment.contains(std::env::consts::OS));
        let shell = environment.lines().last().unwrap();
        if cfg!(windows) {
            assert_eq!(
                shell,
                "shell: bash (Git Bash; use forward slashes in paths)"
            );
        } else {
            assert_eq!(shell, "shell: bash");
        }
    }

    /// The environment block goes on the first user message of the request
    /// only, and the stored conversation is untouched.
    #[tokio::test]
    async fn environment_is_prepended_to_the_first_user_message_only() {
        let backend = Arc::new(Scripted {
            rounds: vec![vec![text("ok")]],
            ..Scripted::default()
        });
        let agent = agent(Arc::clone(&backend)).with_environment("working directory: /w");

        let mut conversation = Conversation::new();
        conversation.push(Message::new(Role::User, "first"));
        conversation.push(Message::new(Role::Assistant, "reply"));
        conversation.push(Message::new(Role::User, "second"));
        let (tx, _rx) = mpsc::channel(256);
        agent
            .run_turn("m", &mut conversation, tx, answers(&[]))
            .await
            .unwrap();

        let sent = backend.requests.lock().unwrap();
        let (request, _tools) = sent.last().unwrap();
        let users: Vec<&str> = request
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            users[0],
            "<environment>\nworking directory: /w\n</environment>\n\nfirst"
        );
        assert_eq!(users[1], "second");
        assert_eq!(
            conversation.messages()[0].content,
            "first",
            "stored copy untouched"
        );
    }

    /// An agent over `backend` with every standard tool, the ones that
    /// change files included, working in `dir`.
    fn editing_agent(backend: Arc<Scripted>, dir: &TempDir) -> Agent {
        Agent::new(backend, Arc::new(ToolSet::standard(&dir.0)), None)
    }

    /// `edit_file` arguments for `notes.txt`.
    fn edit_notes(old: &str, new: &str) -> String {
        json!({ "path": "notes.txt", "old": old, "new": new }).to_string()
    }

    /// A model that asks for `calls` in one round, then says `done`.
    fn asks_for(calls: Vec<ChatDelta>) -> Arc<Scripted> {
        Arc::new(Scripted {
            rounds: vec![calls, vec![text("done")]],
            ..Scripted::default()
        })
    }

    #[tokio::test]
    async fn an_allowed_change_is_previewed_asked_then_made() {
        let dir = TempDir::new("agent-allow");
        dir.write("notes.txt", "helo\n");
        let args = edit_notes("helo", "hello");
        let agent = editing_agent(asks_for(vec![call("c1", "edit_file", &args)]), &dir);
        let (result, conversation, events) =
            turn_answering(&agent, "fix it", &[Decision::Allow]).await;
        result.unwrap();

        let done = "replaced 1 line with 1 at line 1";
        assert_eq!(
            events[..3],
            [
                AgentEvent::ToolStarted {
                    id: "c1".into(),
                    name: "edit_file".into(),
                    arguments: args.clone(),
                },
                AgentEvent::PermissionRequested {
                    id: "c1".into(),
                    name: "edit_file".into(),
                    arguments: args,
                    preview: "edit notes.txt at line 1\n- helo\n+ hello".into(),
                    scope: "edit_file".into(),
                },
                AgentEvent::ToolFinished {
                    id: "c1".into(),
                    name: "edit_file".into(),
                    summary: done.into(),
                    is_error: false,
                },
            ]
        );
        assert_eq!(dir.read("notes.txt"), "hello\n");
        assert_eq!(conversation.messages()[2], Message::tool_result("c1", done));
    }

    #[tokio::test]
    async fn a_denied_change_is_not_made_and_the_model_is_told() {
        let dir = TempDir::new("agent-deny");
        dir.write("notes.txt", "helo\n");
        let args = edit_notes("helo", "hello");
        let agent = editing_agent(asks_for(vec![call("c1", "edit_file", &args)]), &dir);
        let (result, conversation, events) =
            turn_answering(&agent, "fix it", &[Decision::Deny]).await;
        result.unwrap();

        assert_eq!(dir.read("notes.txt"), "helo\n");
        assert_eq!(
            conversation.messages()[2],
            Message::tool_result(
                "c1",
                "declined by the user; do not retry this change, ask what they want instead"
            )
        );
        assert!(events.contains(&AgentEvent::ToolFinished {
            id: "c1".into(),
            name: "edit_file".into(),
            summary: "declined".into(),
            is_error: true,
        }));
        assert_eq!(
            conversation.messages()[3].content,
            "done",
            "the turn goes on"
        );
    }

    #[tokio::test]
    async fn with_nobody_to_ask_a_change_is_declined() {
        let dir = TempDir::new("agent-nobody");
        dir.write("notes.txt", "helo\n");
        let args = edit_notes("helo", "hello");
        let agent = editing_agent(asks_for(vec![call("c1", "edit_file", &args)]), &dir);
        let (result, conversation, events) = turn_answering(&agent, "fix it", &[]).await;
        result.unwrap();

        assert_eq!(dir.read("notes.txt"), "helo\n");
        assert!(
            conversation.messages()[2]
                .content
                .starts_with("declined by the user")
        );
        assert!(events.contains(&AgentEvent::ToolFinished {
            id: "c1".into(),
            name: "edit_file".into(),
            summary: "declined: nobody to ask".into(),
            is_error: true,
        }));
    }

    #[tokio::test]
    async fn a_change_that_cannot_work_is_an_error_without_asking() {
        let dir = TempDir::new("agent-preview-error");
        dir.write("notes.txt", "helo\n");
        let args = edit_notes("nope", "hello");
        let agent = editing_agent(asks_for(vec![call("c1", "edit_file", &args)]), &dir);
        let (result, conversation, events) =
            turn_answering(&agent, "fix it", &[Decision::Allow]).await;
        result.unwrap();

        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::PermissionRequested { .. })),
            "the user is not asked about a change that would fail"
        );
        let error = "old text not found in notes.txt";
        assert_eq!(
            conversation.messages()[2],
            Message::tool_result("c1", format!("error: {error}"))
        );
        assert!(events.contains(&AgentEvent::ToolFinished {
            id: "c1".into(),
            name: "edit_file".into(),
            summary: error.into(),
            is_error: true,
        }));
    }

    #[tokio::test]
    async fn changes_in_one_round_are_asked_one_at_a_time_in_order() {
        let dir = TempDir::new("agent-two-changes");
        dir.write("notes.txt", "one\ntwo\n");
        let agent = editing_agent(
            asks_for(vec![
                call_at(0, "c1", "edit_file", &edit_notes("one", "1")),
                call_at(1, "c2", "edit_file", &edit_notes("two", "2")),
            ]),
            &dir,
        );
        let (result, _conversation, events) =
            turn_answering(&agent, "go", &[Decision::Allow, Decision::Deny]).await;
        result.unwrap();

        let steps: Vec<String> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolStarted { id, .. } => Some(format!("start {id}")),
                AgentEvent::PermissionRequested { id, .. } => Some(format!("ask {id}")),
                AgentEvent::ToolFinished { id, summary, .. } => Some(format!("{id}: {summary}")),
                _ => None,
            })
            .collect();
        assert_eq!(
            steps,
            [
                "start c1",
                "ask c1",
                "c1: replaced 1 line with 1 at line 1",
                "start c2",
                "ask c2",
                "c2: declined",
            ]
        );
        assert_eq!(dir.read("notes.txt"), "1\ntwo\n");
    }

    #[tokio::test]
    async fn read_only_tools_never_ask() {
        let dir = TempDir::new("agent-read-only");
        dir.write("notes.txt", "helo\n");
        let agent = editing_agent(
            asks_for(vec![
                call_at(0, "c1", "read_file", r#"{"path":"notes.txt"}"#),
                call_at(1, "c2", "list_files", ""),
            ]),
            &dir,
        );
        // No answers queued and the channel closed: asking would decline.
        let (result, conversation, events) = turn_answering(&agent, "look", &[]).await;
        result.unwrap();

        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::PermissionRequested { .. }))
        );
        assert_eq!(conversation.messages()[2].content, "1: helo\n");
        assert_eq!(conversation.messages()[3].content, "notes.txt");
    }

    /// A `bash` call is asked about with the exact command as its scope, and
    /// once allowed it runs and the model reads the exit code and output.
    #[tokio::test]
    async fn a_command_is_asked_about_by_its_scope_then_run() {
        let dir = TempDir::new("agent-bash");
        let Some(shell) = find_bash() else {
            assert!(!ToolSet::standard(&dir.0).has("bash"));
            return;
        };
        let args = json!({ "command": "echo hi" }).to_string();
        let agent = Agent::new(
            asks_for(vec![call("c1", "bash", &args)]),
            Arc::new(ToolSet::standard_with_shell(&dir.0, Some(shell))),
            None,
        );
        let (result, conversation, events) =
            turn_answering(&agent, "say hi", &[Decision::Allow]).await;
        result.unwrap();

        assert!(events.contains(&AgentEvent::PermissionRequested {
            id: "c1".into(),
            name: "bash".into(),
            arguments: args,
            preview: "$ echo hi".into(),
            scope: "bash: echo hi".into(),
        }));
        let output = &conversation.messages()[2].content;
        assert!(output.starts_with("exit code: 0"), "{output}");
        assert_eq!(output, "exit code: 0\nhi");
        assert!(events.contains(&AgentEvent::ToolFinished {
            id: "c1".into(),
            name: "bash".into(),
            summary: "exit 0, 1 line".into(),
            is_error: false,
        }));
    }

    /// A command the tool cannot take (blank) is an error without asking.
    #[tokio::test]
    async fn a_blank_command_is_an_error_without_asking() {
        let dir = TempDir::new("agent-bash-blank");
        let args = json!({ "command": "  " }).to_string();
        let agent = Agent::new(
            asks_for(vec![call("c1", "bash", &args)]),
            // The shell is never started, so any path does.
            Arc::new(ToolSet::standard_with_shell(&dir.0, Some("bash".into()))),
            None,
        );
        let (result, conversation, events) = turn_answering(&agent, "go", &[Decision::Allow]).await;
        result.unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::PermissionRequested { .. }))
        );
        assert_eq!(
            conversation.messages()[2],
            Message::tool_result("c1", "error: empty command")
        );
    }

    /// Asks the loaded model about this repository with tools on. Loads nothing.
    #[tokio::test]
    #[ignore = "needs a live LM Studio server with a model loaded"]
    async fn live_agent_reads_the_readme() {
        let lms = LmStudio::default();
        let model = lms
            .loaded_model()
            .await
            .unwrap()
            .expect("load a model in LM Studio first");
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = root.canonicalize().unwrap();
        let agent = Agent::new(
            Arc::new(lms),
            Arc::new(ToolSet::read_only(&root)),
            Some(default_system_prompt()),
        )
        .with_environment(default_environment(&root));

        let mut conversation = Conversation::new();
        conversation.push(Message::new(
            Role::User,
            "What is this project? Read the README before answering.",
        ));
        let (tx, mut rx) = mpsc::channel(256);
        let started = Instant::now();
        let printer = tokio::spawn(async move {
            let mut answer = String::new();
            let mut called = Vec::new();
            while let Some(event) = rx.recv().await {
                match &event {
                    AgentEvent::Text(text) => answer.push_str(text),
                    AgentEvent::ToolStarted { name, .. } => called.push(name.clone()),
                    _ => {}
                }
                if !matches!(event, AgentEvent::Text(_)) {
                    println!("{:>6.1?} {event:?}", started.elapsed());
                }
            }
            (answer, called)
        });
        let result = agent
            .run_turn(&model.id, &mut conversation, tx, answers(&[]))
            .await;
        let (answer, called) = printer.await.unwrap();
        println!("model: {}", model.id);
        println!("tools called: {called:?}");
        println!("answer: {answer}");
        println!("total: {:?}", started.elapsed());
        result.unwrap();
    }

    /// Asks the loaded model to fix a typo in a scratch directory, allowing
    /// every change it asks for. Loads nothing.
    #[tokio::test]
    #[ignore = "needs a live LM Studio server with a model loaded"]
    async fn live_agent_edits_a_file() {
        let lms = LmStudio::default();
        let model = lms
            .loaded_model()
            .await
            .unwrap()
            .expect("load a model in LM Studio first");
        let dir = TempDir::new("live-edit");
        dir.write("notes.txt", "Shopping list\n- milk\n- breaad\n- eggs\n");
        let root = dir.0.canonicalize().unwrap();
        let agent = Agent::new(
            Arc::new(lms),
            Arc::new(ToolSet::standard(&root)),
            Some(default_system_prompt()),
        )
        .with_environment(default_environment(&root));

        let mut conversation = Conversation::new();
        conversation.push(Message::new(
            Role::User,
            "notes.txt has a spelling mistake. Fix it.",
        ));
        let (tx, mut rx) = mpsc::channel(256);
        let (allow, decisions) = mpsc::channel(4);
        let started = Instant::now();
        let printer = tokio::spawn(async move {
            let mut answer = String::new();
            let mut called = Vec::new();
            while let Some(event) = rx.recv().await {
                match &event {
                    AgentEvent::Text(text) => answer.push_str(text),
                    AgentEvent::ToolStarted { name, .. } => called.push(name.clone()),
                    AgentEvent::PermissionRequested { .. } => {
                        let _ = allow.send(Decision::Allow).await;
                    }
                    _ => {}
                }
                if !matches!(event, AgentEvent::Text(_)) {
                    println!("{:>6.1?} {event:?}", started.elapsed());
                }
            }
            (answer, called)
        });
        let result = agent
            .run_turn(&model.id, &mut conversation, tx, decisions)
            .await;
        let (answer, called) = printer.await.unwrap();
        println!("model: {}", model.id);
        println!("tools called: {called:?}");
        println!("answer: {answer}");
        println!("total: {:?}", started.elapsed());
        result.unwrap();
        let notes = dir.read("notes.txt");
        println!("notes.txt now:\n{notes}");
        assert!(notes.contains("- bread\n"), "typo fixed");
        assert!(!notes.contains("breaad"));
    }

    /// Asks the loaded model to count lines with `bash` in a scratch
    /// directory, allowing every command it asks for. Loads nothing.
    #[tokio::test]
    #[ignore = "needs a live LM Studio server with a model loaded, and bash"]
    async fn live_agent_runs_a_command() {
        let lms = LmStudio::default();
        let model = lms
            .loaded_model()
            .await
            .unwrap()
            .expect("load a model in LM Studio first");
        let dir = TempDir::new("live-bash");
        let notes: String = (1..=17).map(|i| format!("note {i}\n")).collect();
        dir.write("notes.txt", notes);
        let root = dir.0.canonicalize().unwrap();
        let tools = ToolSet::standard(&root);
        assert!(tools.has("bash"), "no bash found on this machine");
        let agent = Agent::new(
            Arc::new(lms),
            Arc::new(tools),
            Some(default_system_prompt()),
        )
        .with_environment(default_environment(&root));

        let mut conversation = Conversation::new();
        conversation.push(Message::new(
            Role::User,
            "Use bash to count the lines in notes.txt and tell me the number.",
        ));
        let (tx, mut rx) = mpsc::channel(256);
        let (allow, decisions) = mpsc::channel(4);
        let started = Instant::now();
        let printer = tokio::spawn(async move {
            let mut answer = String::new();
            let mut called = Vec::new();
            while let Some(event) = rx.recv().await {
                match &event {
                    AgentEvent::Text(text) => answer.push_str(text),
                    AgentEvent::ToolStarted { name, .. } => called.push(name.clone()),
                    AgentEvent::PermissionRequested { .. } => {
                        let _ = allow.send(Decision::Allow).await;
                    }
                    _ => {}
                }
                if !matches!(event, AgentEvent::Text(_)) {
                    println!("{:>6.1?} {event:?}", started.elapsed());
                }
            }
            (answer, called)
        });
        let result = agent
            .run_turn(&model.id, &mut conversation, tx, decisions)
            .await;
        let (answer, called) = printer.await.unwrap();
        println!("model: {}", model.id);
        println!("tools called: {called:?}");
        println!("answer: {answer}");
        println!("total: {:?}", started.elapsed());
        result.unwrap();
        assert!(called.iter().any(|name| name == "bash"), "bash was called");
        assert!(answer.contains("17"), "the right count");
    }
}
