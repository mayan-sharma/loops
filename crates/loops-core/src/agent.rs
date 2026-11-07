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
//!
//! # Context control
//!
//! Before each request the agent estimates how full the model's context window
//! will be ([`Conversation::estimated_tokens`], anchored on the server's last
//! token count) and stops the turn with a clear message when there is no room
//! left for a reply, instead of sending a request the server would refuse. The
//! room that is left caps the reply (`max_tokens`). A reply that hits that cap
//! (`finish_reason: "length"`) is continued, not taken as the answer: the
//! model is shown what it wrote so far and asked to go on, up to
//! [`MAX_CONTINUATIONS`] times.
//!
//! # Compaction
//!
//! When the window is nearly full, [`Agent::compact`] asks the model to
//! summarise the conversation for itself and replaces every message with that
//! summary, so the work can go on in a window that has room again. It runs
//! only when the front end asks for it (`/compact`), never on its own.

use std::{path::Path, sync::Arc};

use anyhow::{Result, bail};
use tokio::sync::mpsc;

use crate::{
    Conversation, Message, Role, ToolCall,
    backend::ModelBackend,
    chat::{ChatDelta, RequestOptions, Thinking, ToolCallAssembler, to_openai_tools},
    estimate_tokens,
    tools::{ToolSet, ToolSpec},
};

/// Rounds allowed in one turn before the agent gives up.
pub const DEFAULT_MAX_ITERATIONS: usize = 20;

/// Fewest tokens a reply must have room for; with less left in the window the
/// turn stops before sending.
pub const MIN_REPLY_TOKENS: u64 = 256;

/// How many times one reply cut off at the output limit is continued before
/// the turn gives up.
pub const MAX_CONTINUATIONS: u32 = 2;

/// What the model is told when its reply was cut off. Sent only in the
/// continuation request, never stored in the conversation.
const CONTINUE_NUDGE: &str = "Your reply was cut off at the output limit. Continue from exactly where you stopped; do not repeat what you already wrote.";

/// What the server says when the prompt does not fit the loaded window.
const SERVER_CONTEXT_FULL: &str = "exceeds the available context size";

/// What a turn that finds the window full advises.
const FULL_ADVICE: &str = "/compact to shrink it or /clear to start over";

/// What the model is asked by [`Agent::compact`]. Sent only in that request,
/// never stored in the conversation.
pub const COMPACT_INSTRUCTION: &str = "Summarise this conversation for yourself, so you can continue the work after the earlier messages are removed. Include: the user's goal; what has been done, with file paths and the exact changes made; what was learned from tool results and errors; what is still open or was declined; and any preferences or instructions the user gave. Be specific and compact. Plain text, no preamble.";

/// Most tokens the summary written by [`Agent::compact`] may use.
pub const COMPACT_MAX_TOKENS: u64 = 1024;

/// Fewest tokens the summary must have room for; with less left in the window
/// [`Agent::compact`] stops before sending.
pub const COMPACT_MIN_ROOM: u64 = 128;

/// What the summary is introduced with in the compacted conversation.
const SUMMARY_PREFIX: &str =
    "Summary of the conversation so far, written to continue after earlier messages were removed:";

/// The model's side of the compacted conversation, after the summary. It keeps
/// the next user message from following the summary directly, since some chat
/// templates merge user messages that run back to back.
const SUMMARY_ACKNOWLEDGEMENT: &str = "Understood. Continuing from that summary.";

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
        /// The whole result the model reads: the tool's output, `error: …`
        /// for a failure, or the refusal text for a declined call.
        output: String,
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
    /// The reply stopped at the output limit; the agent is asking the model
    /// to continue it. Its text so far stays: the continuation is appended.
    ReplyCutOff {
        /// Which continuation this is, from 1.
        attempt: u32,
        /// Continuations allowed ([`MAX_CONTINUATIONS`]).
        max: u32,
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

/// Limits for one turn, set by the front end.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnOptions {
    /// The loaded model's context window in tokens, if known. When set, each
    /// request is checked to fit and its reply is capped to the room left.
    pub context_length: Option<u64>,
    /// Most tokens one reply may use, thinking included; `None` for no cap
    /// beyond the room left in the window.
    pub max_output_tokens: Option<u64>,
    /// How much the model may think.
    pub thinking: Thinking,
}

/// What [`Agent::compact`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactReport {
    /// About how many tokens the old conversation held: the server's count
    /// where there was one ([`Conversation::estimated_tokens`] with no overhead).
    pub before: u64,
    /// About how many the compacted conversation holds, guessed from its text.
    pub after: u64,
    /// What the model wrote.
    pub summary: String,
}

/// What the model said in one round.
#[derive(Debug, Default)]
struct Reply {
    text: String,
    calls: Vec<ToolCall>,
    /// Why generation stopped, from the first `Done` that said.
    finish_reason: Option<String>,
    /// Prompt plus completion tokens, if the server reported usage.
    tokens: Option<u64>,
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
    /// `options` bounds every request of the turn (see the module docs on
    /// context control). After each round the conversation's token mark is
    /// set from the server's usage, so the next estimate starts from a real count.
    ///
    /// To cancel, drop the future (abort its task), also while it waits for a
    /// decision. `conversation` stays valid to send again: a round is appended
    /// only once all its tools have run (or been declined), so an interrupted
    /// round leaves nothing, never a tool call without its result. A reply
    /// being continued after a cut-off is not stored until it finishes either.
    pub async fn run_turn(
        &self,
        model: &str,
        conversation: &mut Conversation,
        sink: mpsc::Sender<AgentEvent>,
        mut decisions: mpsc::Receiver<Decision>,
        options: &TurnOptions,
    ) -> Result<()> {
        let mut carry = String::new();
        let result = self
            .run_rounds(
                model,
                conversation,
                &sink,
                &mut decisions,
                options,
                &mut carry,
            )
            .await;
        // A failed turn keeps the text of a reply it was still continuing.
        if result.is_err() && !carry.is_empty() {
            conversation.push(Message::new(Role::Assistant, carry));
        }
        let last = match &result {
            Ok(()) => AgentEvent::TurnDone,
            Err(err) => AgentEvent::TurnFailed(format!("{err:#}")),
        };
        emit(&sink, last).await;
        result
    }

    /// Replaces `conversation` with a summary the model writes of it: a user
    /// message holding the summary, then a short assistant acknowledgement.
    ///
    /// The request carries only the conversation and [`COMPACT_INSTRUCTION`]:
    /// no system prompt, environment or tools, and thinking off. With a known
    /// window the summary is capped to the room left (at most
    /// [`COMPACT_MAX_TOKENS`]), and nothing is sent when less than
    /// [`COMPACT_MIN_ROOM`] is left. A summary cut off at that cap is kept as
    /// it is, just shorter.
    ///
    /// Fails, leaving `conversation` unchanged, when it holds nothing the user
    /// or the model said, when it is too full, when the request fails, or when
    /// the summary is empty. To cancel, drop the future; the conversation is
    /// only replaced once the summary is complete.
    pub async fn compact(
        &self,
        model: &str,
        conversation: &mut Conversation,
        options: &TurnOptions,
    ) -> Result<CompactReport> {
        let said = |m: &Message| matches!(m.role, Role::User | Role::Assistant);
        if !conversation.messages().iter().any(said) {
            bail!("nothing to compact");
        }
        let before = conversation.estimated_tokens(0);
        // With a token mark the estimate holds the system prompt and tools
        // instead of the instruction, which more than covers it.
        let instruction = estimate_tokens(COMPACT_INSTRUCTION.chars().count());
        let max_tokens = match options.context_length {
            Some(window) => {
                let used = conversation.estimated_tokens(instruction);
                if used + COMPACT_MIN_ROOM > window {
                    bail!(
                        "too full to compact: about {used} of {window} tokens used; /clear to start over"
                    );
                }
                COMPACT_MAX_TOKENS.min(window - used)
            }
            None => COMPACT_MAX_TOKENS,
        };
        let request = RequestOptions {
            max_tokens: Some(max_tokens),
            thinking: Thinking::Off,
        };
        let mut messages = conversation.messages().to_vec();
        messages.push(Message::new(Role::User, COMPACT_INSTRUCTION));

        let (tx, mut deltas) = mpsc::channel(64);
        let collect = async {
            let mut summary = String::new();
            while let Some(delta) = deltas.recv().await {
                // Thinking, tool calls (none were offered) and usage are ignored.
                if let ChatDelta::Content(text) = delta {
                    summary.push_str(&text);
                }
            }
            summary
        };
        // `chat` owns `tx` and drops it when it returns, which ends `collect`.
        let (result, summary) = tokio::join!(
            self.backend.chat(model, &messages, &[], &request, tx),
            collect
        );
        result.map_err(|err| context_full(err, "/clear to start over"))?;
        let summary = summary.trim();
        if summary.is_empty() {
            bail!("the model returned an empty summary");
        }

        conversation.replace_all(vec![
            Message::new(Role::User, format!("{SUMMARY_PREFIX}\n\n{summary}")),
            Message::new(Role::Assistant, SUMMARY_ACKNOWLEDGEMENT),
        ]);
        Ok(CompactReport {
            before,
            after: conversation.estimated_tokens(0),
            summary: summary.to_owned(),
        })
    }

    /// The loop itself. `carry` holds the text of a reply cut off at the
    /// output limit while it is being continued; `run_turn` stores it if the
    /// turn fails before the reply finishes.
    async fn run_rounds(
        &self,
        model: &str,
        conversation: &mut Conversation,
        sink: &mpsc::Sender<AgentEvent>,
        decisions: &mut mpsc::Receiver<Decision>,
        options: &TurnOptions,
        carry: &mut String,
    ) -> Result<()> {
        let specs = self.tools.specs();
        let overhead = self.overhead(&specs);
        let mut cut_offs = 0;
        for _ in 0..self.max_iterations {
            let request = request_options(conversation, carry, overhead, options)?;
            let sent = conversation.messages().len();
            let (result, reply) = self
                .ask_model(model, conversation, carry, &specs, &request, sink)
                .await;
            if let Err(err) = result {
                // Keep whatever text arrived; tool calls are dropped since they never ran.
                carry.push_str(&reply.text);
                return Err(context_full(err, FULL_ADVICE));
            }
            if reply.finish_reason.as_deref() == Some("length") {
                // Tool calls of a cut-off reply may be truncated, so none run.
                carry.push_str(&reply.text);
                cut_offs += 1;
                if cut_offs > MAX_CONTINUATIONS {
                    bail!("reply cut off at the output limit {cut_offs} times");
                }
                let cut_off = AgentEvent::ReplyCutOff {
                    attempt: cut_offs,
                    max: MAX_CONTINUATIONS,
                };
                emit(sink, cut_off).await;
                continue;
            }
            let text = std::mem::take(carry) + &reply.text;
            cut_offs = 0;
            if reply.calls.is_empty() {
                conversation.push(Message::new(Role::Assistant, text));
                mark(conversation, sent, reply.tokens);
                return Ok(());
            }

            let mut results = Vec::new();
            for call in &reply.calls {
                results.push(self.run_tool(call, sink, decisions).await);
            }
            // The request and its results go in together (see `run_turn`).
            conversation.push(Message::assistant_with_tool_calls(text, reply.calls));
            mark(conversation, sent, reply.tokens);
            for result in results {
                conversation.push(result);
            }
        }
        bail!("stopped after {} tool rounds", self.max_iterations)
    }

    /// A characters/4 guess at what every request carries besides the
    /// conversation: the system prompt, the environment block and the tool
    /// specs. Used only until the server has counted a request.
    fn overhead(&self, specs: &[ToolSpec]) -> u64 {
        let chars = |text: &Option<String>| text.as_deref().map_or(0, |t| t.chars().count());
        let tools = if specs.is_empty() {
            0
        } else {
            to_openai_tools(specs).to_string().chars().count()
        };
        estimate_tokens(chars(&self.system_prompt) + chars(&self.environment) + tools)
    }

    /// Streams one model reply, forwarding text and thinking to `sink` as it
    /// arrives. Returns the backend's result together with everything that
    /// arrived, so text before a failure is not lost.
    ///
    /// When `carry` holds a reply cut off at the output limit, the request ends
    /// with it as an assistant message and a user message asking the model to
    /// continue; neither is stored.
    async fn ask_model(
        &self,
        model: &str,
        conversation: &Conversation,
        carry: &str,
        specs: &[ToolSpec],
        options: &RequestOptions,
        sink: &mpsc::Sender<AgentEvent>,
    ) -> (Result<()>, Reply) {
        let mut messages = Vec::new();
        if let Some(prompt) = &self.system_prompt {
            messages.push(Message::new(Role::System, prompt.clone()));
        }
        messages.extend_from_slice(conversation.messages());
        if !carry.is_empty() {
            messages.push(Message::new(Role::Assistant, carry));
            messages.push(Message::new(Role::User, CONTINUE_NUDGE));
        }
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
                    ChatDelta::Reasoning(_) => {}
                    ChatDelta::Done { finish_reason } => {
                        reply.finish_reason = reply.finish_reason.or(finish_reason);
                    }
                    ChatDelta::Usage {
                        prompt_tokens,
                        completion_tokens,
                    } => {
                        reply.tokens = Some(prompt_tokens + completion_tokens);
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
        tokio::join!(
            self.backend.chat(model, &messages, specs, options, tx),
            collect
        )
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
                output: output.clone(),
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

/// The limits of the next request: stops the turn if the window has no room
/// for a reply, else caps the reply to the room left (and to
/// `options.max_output_tokens`). The estimate is the conversation's
/// ([`Conversation::estimated_tokens`]) plus, while a cut-off reply is being
/// continued, a guess for its text and the nudge sent with it.
fn request_options(
    conversation: &Conversation,
    carry: &str,
    overhead: u64,
    options: &TurnOptions,
) -> Result<RequestOptions> {
    let max_tokens = match options.context_length {
        Some(window) => {
            let mut used = conversation.estimated_tokens(overhead);
            if !carry.is_empty() {
                used += estimate_tokens(carry.chars().count() + CONTINUE_NUDGE.len());
            }
            if used + MIN_REPLY_TOKENS > window {
                bail!("context full: about {used} of {window} tokens used; {FULL_ADVICE}");
            }
            Some(
                options
                    .max_output_tokens
                    .unwrap_or(u64::MAX)
                    .min(window - used),
            )
        }
        None => options.max_output_tokens,
    };
    Ok(RequestOptions {
        max_tokens,
        thinking: options.thinking,
    })
}

/// Sets the token mark after a round whose request carried the first `sent`
/// messages and whose reply is now message `sent`. Without usage the old mark
/// stays and the estimate guesses the rest.
///
/// After a continued reply the count also holds the nudge that asked for the
/// rest, so it is a few tokens high, which errs on the safe side.
fn mark(conversation: &mut Conversation, sent: usize, tokens: Option<u64>) {
    if let Some(tokens) = tokens {
        conversation.set_token_mark(sent + 1, tokens);
    }
}

/// Turns the server's refusal of a prompt larger than the window into the
/// same `advice` the agent's own check gives; other errors pass unchanged.
fn context_full(err: anyhow::Error, advice: &str) -> anyhow::Error {
    let text = format!("{err:#}");
    if text.contains(SERVER_CONTEXT_FULL) {
        anyhow::anyhow!("context full: the server refused the request ({text}); {advice}")
    } else {
        err
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
        /// The options of each request, in order.
        options: Mutex<Vec<RequestOptions>>,
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
            options: &'a RequestOptions,
            sink: mpsc::Sender<ChatDelta>,
        ) -> BoxFuture<'a, Result<()>> {
            let mut requests = self.requests.lock().unwrap();
            let round = self.rounds[requests.len().min(self.rounds.len() - 1)].clone();
            requests.push((messages.to_vec(), tools.len()));
            self.options.lock().unwrap().push(options.clone());
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
            .run_turn(
                "m",
                &mut conversation,
                tx,
                answers(decisions),
                &TurnOptions::default(),
            )
            .await;
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        (result, conversation, events)
    }

    /// Runs one turn on `conversation` under `options`, with nobody to ask;
    /// returns the result and every event sent.
    async fn turn_with(
        agent: &Agent,
        conversation: &mut Conversation,
        options: &TurnOptions,
    ) -> (Result<()>, Vec<AgentEvent>) {
        let (tx, mut rx) = mpsc::channel(256);
        let result = agent
            .run_turn("m", conversation, tx, answers(&[]), options)
            .await;
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        (result, events)
    }

    /// A conversation of one user message the server already counted as
    /// `tokens` tokens, so the next estimate is exactly that.
    fn counted(tokens: u64) -> Conversation {
        let mut conversation = Conversation::new();
        conversation.push(Message::new(Role::User, "go"));
        conversation.set_token_mark(1, tokens);
        conversation
    }

    fn done(reason: &str) -> ChatDelta {
        ChatDelta::Done {
            finish_reason: Some(reason.into()),
        }
    }

    fn usage(prompt_tokens: u64, completion_tokens: u64) -> ChatDelta {
        ChatDelta::Usage {
            prompt_tokens,
            completion_tokens,
        }
    }

    /// A model that answers `ok`.
    fn says_ok() -> Arc<Scripted> {
        Arc::new(Scripted {
            rounds: vec![vec![text("ok"), done("stop")]],
            ..Scripted::default()
        })
    }

    #[tokio::test]
    async fn thinking_and_the_reply_cap_reach_the_backend() {
        let backend = says_ok();
        let agent = agent(backend.clone());
        let cases = [
            // No window: only the cap the user set.
            (None, Some(300), Thinking::Off, Some(300)),
            (None, None, Thinking::High, None),
            // A window: the room left, or the user's cap if smaller.
            (Some(10_000), None, Thinking::Default, Some(9_000)),
            (Some(10_000), Some(500), Thinking::Low, Some(500)),
        ];
        for (context_length, max_output_tokens, thinking, _) in cases {
            let options = TurnOptions {
                context_length,
                max_output_tokens,
                thinking,
            };
            turn_with(&agent, &mut counted(1000), &options)
                .await
                .0
                .unwrap();
        }
        let sent = backend.options.lock().unwrap();
        let expected: Vec<RequestOptions> = cases
            .iter()
            .map(|&(_, _, thinking, max_tokens)| RequestOptions {
                max_tokens,
                thinking,
            })
            .collect();
        assert_eq!(*sent, expected);
    }

    #[tokio::test]
    async fn a_full_window_stops_the_turn_before_any_request() {
        let backend = says_ok();
        let options = TurnOptions {
            context_length: Some(8192),
            ..TurnOptions::default()
        };
        let mut conversation = counted(7_937);
        let (result, events) =
            turn_with(&agent(backend.clone()), &mut conversation, &options).await;
        let error = "context full: about 7937 of 8192 tokens used; /compact to shrink it or /clear to start over";
        assert_eq!(result.unwrap_err().to_string(), error);
        assert_eq!(events, [AgentEvent::TurnFailed(error.into())]);
        assert!(backend.requests.lock().unwrap().is_empty(), "nothing sent");
        assert_eq!(conversation.messages().len(), 1);

        // One token less leaves exactly the minimum reply room.
        let mut conversation = counted(7_936);
        let (result, _) = turn_with(&agent(backend.clone()), &mut conversation, &options).await;
        result.unwrap();
        assert_eq!(
            backend.options.lock().unwrap()[0].max_tokens,
            Some(MIN_REPLY_TOKENS)
        );
    }

    /// Each round's usage marks the conversation, and the next request's
    /// estimate starts from that count plus a guess for the tool result.
    #[tokio::test]
    async fn usage_marks_the_conversation_and_anchors_the_next_estimate() {
        let backend = Arc::new(Scripted {
            rounds: vec![
                vec![
                    call("c", "list_files", ""),
                    done("tool_calls"),
                    usage(100, 10),
                ],
                vec![text("ok"), done("stop"), usage(150, 2)],
            ],
            ..Scripted::default()
        });
        let options = TurnOptions {
            context_length: Some(100_000),
            ..TurnOptions::default()
        };
        let mut conversation = Conversation::new();
        conversation.push(Message::new(Role::User, "look"));
        let (result, _) = turn_with(&agent(backend.clone()), &mut conversation, &options).await;
        result.unwrap();

        let msgs = conversation.messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(conversation.token_mark(), Some((4, 152)));
        let tool_result = (msgs[2].content.chars().count() as u64).div_ceil(4);
        let sent = backend.options.lock().unwrap();
        assert_eq!(sent[1].max_tokens, Some(100_000 - (110 + tool_result)));
    }

    #[tokio::test]
    async fn a_cut_off_reply_is_continued_and_stored_whole() {
        let backend = Arc::new(Scripted {
            rounds: vec![
                vec![text("Hello, "), done("length")],
                vec![text("world."), done("stop")],
            ],
            ..Scripted::default()
        });
        let (result, conversation, events) = turn(&agent(backend.clone()), "greet").await;
        result.unwrap();

        assert_eq!(
            conversation.messages(),
            [
                Message::new(Role::User, "greet"),
                Message::new(Role::Assistant, "Hello, world."),
            ]
        );
        let cut_offs: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ReplyCutOff { .. }))
            .collect();
        assert_eq!(
            cut_offs,
            [&AgentEvent::ReplyCutOff {
                attempt: 1,
                max: MAX_CONTINUATIONS
            }]
        );

        let requests = backend.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let (second, _) = &requests[1];
        assert_eq!(
            second[second.len() - 2..],
            [
                Message::new(Role::Assistant, "Hello, "),
                Message::new(Role::User, CONTINUE_NUDGE),
            ]
        );
    }

    #[tokio::test]
    async fn a_reply_cut_off_too_often_fails_and_keeps_its_text() {
        let backend = Arc::new(Scripted {
            rounds: vec![vec![text("ab"), done("length")]],
            ..Scripted::default()
        });
        let (result, conversation, events) = turn(&agent(backend.clone()), "go").await;
        let error = "reply cut off at the output limit 3 times";
        assert_eq!(result.unwrap_err().to_string(), error);
        assert_eq!(
            conversation.messages()[1..],
            [Message::new(Role::Assistant, "ababab")]
        );
        let tail: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| !matches!(e, AgentEvent::Text(_)))
            .collect();
        assert_eq!(
            tail,
            [
                &AgentEvent::ReplyCutOff { attempt: 1, max: 2 },
                &AgentEvent::ReplyCutOff { attempt: 2, max: 2 },
                &AgentEvent::TurnFailed(error.into()),
            ]
        );
        assert_eq!(backend.requests.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn tool_calls_of_a_cut_off_reply_are_not_run() {
        let backend = Arc::new(Scripted {
            rounds: vec![
                vec![
                    text("let me "),
                    call("c", "list_files", "{\"pa"),
                    done("length"),
                ],
                vec![text("look."), done("stop")],
            ],
            ..Scripted::default()
        });
        let (result, conversation, events) = turn(&agent(backend), "go").await;
        result.unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolStarted { .. }))
        );
        assert_eq!(
            conversation.messages()[1..],
            [Message::new(Role::Assistant, "let me look.")]
        );
    }

    #[tokio::test]
    async fn the_server_refusing_a_full_window_says_context_full() {
        let backend = Arc::new(Scripted {
            rounds: vec![Vec::new()],
            fail: Some(
                "POST /v1/chat/completions returned 400 Bad Request: \
                 {\"error\":\"the request exceeds the available context size\"}",
            ),
            ..Scripted::default()
        });
        let (result, _, _) = turn(&agent(backend), "go").await;
        let error = result.unwrap_err().to_string();
        assert!(
            error.starts_with("context full: the server refused the request (POST "),
            "{error}"
        );
        assert!(
            error.ends_with("context size\"}); /compact to shrink it or /clear to start over"),
            "{error}"
        );
    }

    /// Compacts `conversation` with a model that replies `round`; returns the
    /// result and the backend, for what it was sent.
    async fn compact_with(
        round: Vec<ChatDelta>,
        conversation: &mut Conversation,
        context_length: Option<u64>,
    ) -> (Result<CompactReport>, Arc<Scripted>) {
        let backend = Arc::new(Scripted {
            rounds: vec![round],
            ..Scripted::default()
        });
        let agent = agent(backend.clone()).with_environment("working directory: /w");
        let options = TurnOptions {
            context_length,
            // Neither applies to the summary.
            max_output_tokens: Some(50),
            thinking: Thinking::High,
        };
        let result = agent.compact("m", conversation, &options).await;
        (result, backend)
    }

    /// A conversation with a tool round and a harness line, the first four
    /// messages counted by the server as 2,000 tokens.
    fn worked_on() -> Conversation {
        let mut conversation = Conversation::new();
        conversation.push(Message::new(Role::User, "fix the typo"));
        let read = ToolCall {
            id: "c1".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"a.txt"}"#.into(),
        };
        conversation.push(Message::assistant_with_tool_calls("", vec![read]));
        conversation.push(Message::tool_result("c1", "1: helo"));
        conversation.push(Message::new(Role::Assistant, "Fixed."));
        conversation.set_token_mark(4, 2000);
        conversation.push(Message::new(Role::Harness, "loaded a model"));
        conversation
    }

    #[tokio::test]
    async fn compact_sends_only_the_conversation_and_replaces_it_with_the_summary() {
        let mut conversation = worked_on();
        let old = conversation.messages().to_vec();
        let round = vec![
            ChatDelta::Reasoning("hm".into()),
            text("\nGoal: fix the typo. "),
            text("Done: a.txt.\n"),
            done("stop"),
            usage(2100, 9),
        ];
        let (result, backend) = compact_with(round, &mut conversation, Some(2_500)).await;
        let report = result.unwrap();

        // No system prompt, environment or tools; the instruction last.
        let requests = backend.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let (sent, tools) = &requests[0];
        assert_eq!(*tools, 0);
        assert_eq!(sent[..old.len()], old[..]);
        assert_eq!(
            sent[old.len()..],
            [Message::new(Role::User, COMPACT_INSTRUCTION)]
        );
        // Capped to the room left (2,500 - 2,000), thinking off.
        assert_eq!(
            *backend.options.lock().unwrap(),
            [RequestOptions {
                max_tokens: Some(500),
                thinking: Thinking::Off,
            }]
        );

        let summary = "Goal: fix the typo. Done: a.txt.";
        assert_eq!(
            conversation.messages(),
            [
                Message::new(
                    Role::User,
                    format!(
                        "Summary of the conversation so far, written to continue after \
                         earlier messages were removed:\n\n{summary}"
                    )
                ),
                Message::new(Role::Assistant, "Understood. Continuing from that summary."),
            ]
        );
        assert_eq!(conversation.token_mark(), None, "the old count is gone");
        assert_eq!(
            report,
            CompactReport {
                before: 2000,
                after: conversation.estimated_tokens(0),
                summary: summary.into(),
            }
        );
        assert!(report.after > 0 && report.after < report.before);
    }

    #[tokio::test]
    async fn the_summary_is_capped_at_its_maximum_and_a_cut_off_one_is_kept() {
        for context_length in [None, Some(100_000)] {
            let mut conversation = worked_on();
            let round = vec![text("Goal: fix"), done("length")];
            let (result, backend) = compact_with(round, &mut conversation, context_length).await;
            assert_eq!(result.unwrap().summary, "Goal: fix");
            assert_eq!(
                backend.options.lock().unwrap()[0].max_tokens,
                Some(COMPACT_MAX_TOKENS)
            );
        }
    }

    #[tokio::test]
    async fn nothing_to_compact_sends_nothing() {
        let mut conversation = Conversation::new();
        conversation.push(Message::new(Role::Harness, "loaded a model"));
        let (result, backend) = compact_with(vec![text("x")], &mut conversation, None).await;
        assert_eq!(result.unwrap_err().to_string(), "nothing to compact");
        assert!(backend.requests.lock().unwrap().is_empty());
        assert_eq!(conversation.messages().len(), 1);
    }

    #[tokio::test]
    async fn too_full_to_compact_sends_nothing() {
        let mut conversation = counted(7_937);
        let (result, backend) = compact_with(vec![text("x")], &mut conversation, Some(8_000)).await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "too full to compact: about 7937 of 8000 tokens used; /clear to start over"
        );
        assert!(backend.requests.lock().unwrap().is_empty());
        assert_eq!(conversation.messages(), [Message::new(Role::User, "go")]);

        // Exactly the minimum room left is enough.
        let mut conversation = counted(7_872);
        let (result, backend) = compact_with(vec![text("x")], &mut conversation, Some(8_000)).await;
        result.unwrap();
        assert_eq!(
            backend.options.lock().unwrap()[0].max_tokens,
            Some(COMPACT_MIN_ROOM)
        );
    }

    #[tokio::test]
    async fn an_empty_summary_or_a_failed_request_leaves_the_conversation() {
        let mut conversation = worked_on();
        let old = conversation.messages().to_vec();
        let blank = vec![ChatDelta::Reasoning("hm".into()), text(" \n"), done("stop")];
        let (result, _) = compact_with(blank, &mut conversation, None).await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "the model returned an empty summary"
        );
        assert_eq!(conversation.messages(), old);
        assert_eq!(conversation.token_mark(), Some((4, 2000)));

        let backend = Arc::new(Scripted {
            rounds: vec![vec![text("Goal: ")]],
            fail: Some("the request exceeds the available context size"),
            ..Scripted::default()
        });
        let result = agent(backend)
            .compact("m", &mut conversation, &TurnOptions::default())
            .await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "context full: the server refused the request \
             (the request exceeds the available context size); /clear to start over"
        );
        assert_eq!(conversation.messages(), old);
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
                    output: msgs[2].content.clone(),
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
            output: "error: unknown tool: rm_rf".into(),
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
            .run_turn(
                "m",
                &mut conversation,
                tx,
                answers(&[]),
                &TurnOptions::default(),
            )
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
                    output: done.into(),
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
            output: DECLINED_RESULT.into(),
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
            output: DECLINED_RESULT.into(),
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
            output: format!("error: {error}"),
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
            output: output.clone(),
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
            .run_turn(
                &model.id,
                &mut conversation,
                tx,
                answers(&[]),
                &TurnOptions::default(),
            )
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
            .run_turn(
                &model.id,
                &mut conversation,
                tx,
                decisions,
                &TurnOptions::default(),
            )
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
            .run_turn(
                &model.id,
                &mut conversation,
                tx,
                decisions,
                &TurnOptions::default(),
            )
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
    /// One live turn with every permission allowed; returns the answer text.
    async fn live_turn(
        agent: &Agent,
        model: &str,
        options: &TurnOptions,
        conversation: &mut Conversation,
    ) -> String {
        let (tx, mut rx) = mpsc::channel(256);
        let (allow, decisions) = mpsc::channel(4);
        let printer = tokio::spawn(async move {
            let mut answer = String::new();
            while let Some(event) = rx.recv().await {
                match &event {
                    AgentEvent::Text(text) => answer.push_str(text),
                    AgentEvent::PermissionRequested { .. } => {
                        let _ = allow.send(Decision::Allow).await;
                    }
                    other => println!("{other:?}"),
                }
            }
            answer
        });
        let result = agent
            .run_turn(model, conversation, tx, decisions, options)
            .await;
        let answer = printer.await.unwrap();
        result.unwrap();
        answer
    }

    /// Live: a turn that edits a file, then `/compact`, then a question only
    /// the summary can answer. Run with `--ignored` and a model loaded.
    #[tokio::test]
    #[ignore]
    async fn live_agent_compacts_a_conversation() {
        let lms = LmStudio::default();
        let model = lms
            .loaded_model()
            .await
            .unwrap()
            .expect("load a model in LM Studio first");
        let options = TurnOptions {
            context_length: model.context_length(),
            ..TurnOptions::default()
        };
        let dir = TempDir::new("live-compact");
        dir.write(
            "notes.txt",
            "Shopping list
- milk
- breaad
- eggs
",
        );
        let root = dir.0.canonicalize().unwrap();
        let agent = Agent::new(
            Arc::new(lms),
            Arc::new(ToolSet::standard(&root)),
            Some(default_system_prompt()),
        )
        .with_environment(default_environment(&root));

        let mut conversation = Conversation::new();
        let ask = |conversation: &mut Conversation, text: &str| {
            conversation.push(Message::new(Role::User, text));
        };

        ask(
            &mut conversation,
            "notes.txt has a spelling mistake. Fix it.",
        );
        let first = live_turn(&agent, &model.id, &options, &mut conversation).await;
        println!("first answer: {first}");
        let messages_before = conversation.messages().len();
        let mark_before = conversation.token_mark();

        let started = Instant::now();
        let report = agent
            .compact(&model.id, &mut conversation, &options)
            .await
            .unwrap();
        println!(
            "compacted {} messages (mark {mark_before:?}) in {:?}: about {} -> {} tokens",
            messages_before,
            started.elapsed(),
            report.before,
            report.after
        );
        println!(
            "summary:
{}",
            report.summary
        );
        assert_eq!(conversation.messages().len(), 2);
        assert!(report.after < report.before, "the summary is smaller");

        ask(
            &mut conversation,
            "Without using any tools: which file did you change, and what was the mistake?",
        );
        let second = live_turn(&agent, &model.id, &options, &mut conversation).await;
        println!("second answer: {second}");
        let lower = second.to_lowercase();
        assert!(lower.contains("notes.txt"), "remembers the file: {second}");
        assert!(lower.contains("brea"), "remembers the typo: {second}");
    }
}
