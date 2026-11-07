//! Application state and the update logic that drives it.
//!
//! The TUI follows a simple Elm-like split:
//! - [`AppEvent`] is everything that can happen (a key, a resize, a timer tick,
//!   a background task finishing).
//! - [`App::update`] is the only place state changes in response to an event.
//! - `ui::draw` reads [`App`] and renders it; it never changes state.
//!
//! Slow work (talking to LM Studio) never runs inside `update`. It is spawned
//! as a task that sends its result back as an [`AppEvent::Model`], or, for a
//! turn of the agent loop, a run of [`AppEvent::Turn`]s, so `update` stays
//! synchronous and returns immediately.
//!
//! # Turns
//!
//! The agent task works on its own copy of the conversation. While it runs, the
//! app mirrors its events into the transcript (text, `thinking…`, tool lines),
//! and when it finishes the task sends its copy back, which replaces the
//! mirror: only the agent's copy holds the tool results the model saw.
//!
//! # Permission
//!
//! When the agent wants to run a tool that changes things, it sends
//! `PermissionRequested` and waits. The app keeps the preview (so the
//! transcript can show the change under its `tool:` line for good) and opens
//! [`Mode::Permission`]; the user picks one of the [`PermissionOption`]s
//! (Up/Down and Enter, or its digit), which sends the [`Decision`] back on the
//! turn's decision channel. "Don't ask again" remembers the request's scope,
//! not its tool: for a file tool that is the tool's name, for `bash` the exact
//! command, so a yes to `cargo test` does not let `rm -rf .` through. Requests
//! whose scope the user always allowed are answered at once.
//!
//! # Compaction
//!
//! `/compact` runs in the background like a turn, on its own copy of the
//! conversation, and blocks sending the same way; Esc cancels it. When it
//! finishes, the compacted copy replaces the transcript, as `/clear` would.
//!
//! # Sessions
//!
//! The conversation is saved as a session (see [`loops_core::session`]) after
//! every turn, finished or cancelled, after `/compact`, when switching to
//! another session, and on quit; a conversation nobody has typed in is never
//! saved. The display state the transcript needs besides the messages (tool
//! outcomes, previews, command output, reply endings) is saved with it, so a
//! resumed session looks as it did. `/clear` (or `/new`) saves the current
//! session and starts another; `/sessions` lists them to resume one.
//!
//! Because `update` takes plain values and touches no terminal, it can be unit
//! tested by feeding it synthetic key events (see the tests at the bottom).

use std::{
    collections::{HashMap, HashSet},
    future::Future,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use loops_core::{
    Conversation, Message, Role, ToolCall,
    agent::{
        Agent, AgentEvent, CompactReport, Decision, INSTRUCTIONS_FILE, Instructions, TurnOptions,
    },
    chat::Thinking,
    lmstudio::{ModelInfo, ModelState},
    session::{Session, SessionStore, SessionSummary, title_from, unix_now},
};
use ratatui::style::{Modifier, Style};
use ratatui_textarea::TextArea;
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, task::AbortHandle};

/// Something the main loop must react to.
///
/// Events come from background tasks through a channel, so new sources
/// (e.g. streamed model output) become new variants here.
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// A key press from the terminal.
    Key(KeyEvent),
    /// The terminal was resized; the next draw picks up the new size.
    Resize,
    /// Periodic timer; advances the loading spinner.
    Tick,
    /// A model backend task finished.
    Model(ModelEvent),
    /// Progress of a turn. `turn` numbers the turns, so events still queued
    /// from a cancelled one can be told apart and dropped.
    Turn {
        /// Which turn this belongs to.
        turn: u64,
        /// What happened.
        event: TurnEvent,
    },
    /// `/compact` finished. `turn` numbers it like a turn, so the result of
    /// a cancelled one can be told apart and dropped.
    Compacted {
        /// Which compaction this is.
        turn: u64,
        /// The compacted conversation and what it saved, or the error
        /// formatted for display.
        result: Result<(Conversation, CompactReport), String>,
    },
    /// A line for the transcript from outside the app, e.g. a startup notice.
    Say(String),
}

/// What a turn task reports.
#[derive(Debug, Clone)]
pub enum TurnEvent {
    /// Progress from the agent loop, forwarded as it happens.
    Agent(AgentEvent),
    /// The task is done (after `TurnDone` or `TurnFailed`); carries the
    /// conversation as the agent left it.
    Finished(Conversation),
}

/// How a finished tool call went, for its `tool:` line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutcome {
    /// One line, e.g. `31 lines`, or the error text.
    pub summary: String,
    /// Drawn red as `error: …` when set.
    pub is_error: bool,
}

/// How a reply ended, when that needs showing after its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplyEnd {
    /// Finished without answer text; drawn as a dim `(empty reply)`.
    Empty,
    /// Stopped with Esc; drawn as a dim `[cancelled]` after the text.
    Cancelled,
}

/// Token counts of the last request, as the backend reported them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    /// Everything the model read: system prompt, tools, the conversation.
    pub prompt_tokens: u64,
    /// What it wrote, thinking included.
    pub completion_tokens: u64,
}

/// Results of background model work. Errors are already formatted for display.
#[derive(Debug, Clone)]
pub enum ModelEvent {
    /// The server answers; carries the model already loaded, if any.
    ServerReady(Option<ModelInfo>),
    /// The server could not be started or reached.
    ServerFailed(String),
    /// `/model` listing finished.
    ModelsListed(Vec<ModelInfo>),
    /// `/model` listing failed.
    ListFailed(String),
    /// A load finished and the model is ready.
    Loaded(ModelInfo),
    /// A load failed.
    LoadFailed(String),
    /// `/unload` finished; this model is no longer loaded.
    Unloaded(String),
    /// `/unload` failed; the model is presumably still loaded.
    UnloadFailed(String),
}

/// What the status bar reports about the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    /// Waiting for LM Studio's server at startup.
    Starting,
    /// The server is up but nothing is loaded.
    NoModel,
    /// This model is being loaded.
    Loading(String),
    /// This model is being unloaded.
    Unloading(String),
    /// This model is loaded and ready.
    Ready(String),
    /// This model is loaded and running a turn.
    Busy(String),
    /// The last model operation failed; a short message (full text is in the transcript).
    Error(String),
}

/// Whether keys go to the input box, a popup, or the permission prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Keys edit the input box.
    Normal,
    /// The model list is open; `selected` indexes `models`.
    Picker {
        /// Chat models, in backend order.
        models: Vec<ModelInfo>,
        /// Highlighted row.
        selected: usize,
    },
    /// The saved sessions of this directory are listed (`/sessions`).
    Sessions {
        /// Newest first, as the store lists them.
        entries: Vec<SessionSummary>,
        /// Highlighted row.
        selected: usize,
    },
    /// Asking `Load <id>? (y/n)` for `models[selected]`. Kept so `n` can return to the list.
    Confirm {
        /// Same list the picker showed.
        models: Vec<ModelInfo>,
        /// The model being confirmed.
        selected: usize,
    },
    /// The running turn waits for an answer on a tool call that changes
    /// things. Its preview is in [`App::tool_previews`] under `id`.
    Permission {
        /// The call's id.
        id: String,
        /// The tool's name, for the prompt's title.
        name: String,
        /// What "don't ask again" covers: `edit_file`, `bash: cargo test`.
        scope: String,
        /// Highlighted row, an index into [`PermissionOption::ALL`].
        selected: usize,
    },
}

/// One answer to the permission prompt, in the order the prompt lists them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionOption {
    /// Run this call.
    Allow,
    /// Run this call, and every later call with the same scope this run
    /// (the same file tool, or the same `bash` command).
    AlwaysAllow,
    /// Do not run this call; the turn goes on.
    Deny,
}

impl PermissionOption {
    /// Every option, top to bottom; digit `n` picks `ALL[n - 1]`.
    pub const ALL: [Self; 3] = [Self::Allow, Self::AlwaysAllow, Self::Deny];

    /// The row's text in a prompt about a call with this `scope`.
    pub fn label(self, scope: &str) -> String {
        match self {
            Self::Allow => "Yes".to_owned(),
            Self::AlwaysAllow => format!("Yes, and don't ask again for {scope} this run"),
            Self::Deny => "No".to_owned(),
        }
    }
}

/// Text pushed into the transcript by `/help`.
const HELP_TEXT: &str = "commands:\n  /help             show this list\n  /model            pick and load a local model\n  /unload           unload the current model\n  /clear            start a new session; the old one stays saved\n  /new              start a new session (same as /clear)\n  /sessions         list this directory's sessions and resume one\n  /context          tokens used and left in the model's window\n  /compact          summarise the chat so far to free the window\n  /think [level]    how much the model thinks: off, low, medium, high, default\n  /max-tokens [n]   cap each reply at n tokens (off for no cap)\n  /quit (or /exit)  exit (also Ctrl+C or Ctrl+D)\nkeys:\n  Esc               cancel the running turn\n  Ctrl+O            show full command output (toggle)\n  PageUp/PageDown/End  scroll the transcript\n  ↑/↓ + Enter       answer a permission prompt (1/2/3 also work)\ntools the model can use (inside this directory):\n  read_file, list_files         read only; never ask\n  edit_file, write_file, bash   change things; ask first\n  \"don't ask again\" for bash is per exact command\n  AGENTS.md in the working directory is read at start and sent to the model";

/// Said when something must wait for the current turn (or `/compact`).
const BUSY_TEXT: &str = "a turn is running; wait for it or press Esc to cancel";

/// Said by `/sessions` and `--resume` when there is no home directory.
const SESSIONS_OFF_TEXT: &str = "sessions are off: no home directory";

/// Said by `/sessions` and `--resume` when nothing was saved here yet.
const NO_SESSIONS_TEXT: &str = "no saved sessions for this directory";

/// Said when something must wait for `/unload` to finish.
const UNLOADING_TEXT: &str = "a model is unloading; wait for it";

/// Longest error text shown in the status bar.
const STATUS_ERROR_CHARS: usize = 40;

/// Lines moved by one PageUp or PageDown.
const SCROLL_STEP: usize = 10;

/// Shown in the empty input box.
const PLACEHOLDER: &str = "Message loops…";

/// All state the TUI needs to draw a frame.
pub struct App {
    /// The in-run message history.
    pub conversation: Conversation,
    /// The multi-line input box.
    pub input: TextArea<'static>,
    /// Model state shown on the right of the status bar.
    pub model_status: ModelStatus,
    /// Normal typing, a model popup, or the permission prompt.
    pub mode: Mode,
    /// Advanced on every tick; picks the spinner frame while loading.
    pub spinner: usize,
    /// Set when the user asks to exit; the main loop stops on the next turn.
    pub should_quit: bool,
    /// Endings to draw after a reply, keyed by its index in `conversation`.
    /// Only replies get entries. Messages are only removed or replaced after
    /// the start of the running turn, which has none yet, so no stored index shifts.
    pub reply_ends: HashMap<usize, ReplyEnd>,
    /// How each finished tool call went, by call id.
    pub tool_outcomes: HashMap<String, ToolOutcome>,
    /// What each call that asked for permission would change, by call id;
    /// kept after the answer so the transcript still shows it.
    pub tool_previews: HashMap<String, String>,
    /// What each `bash` call that ran printed, by call id: the result the
    /// model read, drawn dim under the call. Failed and declined calls have
    /// none (their outcome line says why).
    pub tool_outputs: HashMap<String, String>,
    /// Whether command output is drawn whole rather than folded to its first
    /// and last lines; Ctrl+O toggles it.
    pub full_output: bool,
    /// Permission scopes the user allowed for the rest of the run
    /// ([`PermissionOption::AlwaysAllow`] in the prompt): tool names, and
    /// `bash: <command>` for single commands.
    pub always_allowed: HashSet<String>,
    /// Context window of the loaded model, in tokens, if known.
    pub context_length: Option<u64>,
    /// How much the model may think (`/think`), from the next request on.
    pub thinking: Thinking,
    /// Most tokens one reply may use (`/max-tokens`), from the next request
    /// on; `None` for no cap beyond the room left in the window.
    pub max_output_tokens: Option<u64>,
    /// What the last request used. It carried the whole conversation, so
    /// this is how full the window is; `None` until something is sent.
    pub usage: Option<Usage>,
    /// The directory loops works in, for the banner.
    pub cwd: PathBuf,
    /// What project instructions the model was given, for `/context`, e.g.
    /// `AGENTS.md, about 420 tokens`; `None` when there were none. Set by
    /// `main` after [`App::new`].
    pub instructions: Option<String>,
    /// How many lines the transcript is scrolled up from the bottom; 0
    /// follows new output. Not clamped here: only the draw knows how tall the
    /// wrapped transcript is, so it clamps, and scrolling past the top is harmless.
    pub scroll_up: usize,
    /// Where sessions are saved; `None` when they are off (no home
    /// directory). Set by `main` with [`App::with_sessions`].
    pub sessions: Option<SessionStore>,
    /// The current session's id; its file is written on the first save.
    pub session_id: String,
    /// When the current session started, in unix seconds.
    pub session_created: u64,
    /// The model the current session last ran a turn with.
    pub session_model: Option<String>,
    /// The current session's title, fixed by its first save so that
    /// `/compact` (whose summary becomes the first user message) keeps it.
    session_title: String,
    /// Whether saving the current session failed and that was said, so a
    /// broken disk is reported once per session, not after every turn.
    save_failed: bool,
    /// The turn running, if any.
    turn: Option<Turn>,
    /// The `/compact` running, if any. Never at the same time as a turn.
    compaction: Option<Compaction>,
    /// Number given to the next turn or compaction.
    next_turn: u64,
    /// The agent loop, and through it the backend for model lists and loads.
    agent: Arc<Agent>,
    /// Where spawned tasks report back.
    events: mpsc::Sender<AppEvent>,
}

/// The turn currently running.
struct Turn {
    /// Matches [`AppEvent::Turn`]'s `turn`; events for other numbers are stale.
    id: u64,
    /// Index of the user message that started it.
    start: usize,
    /// Index of the reply being mirrored. Not necessarily last: harness lines
    /// (e.g. `/help`) can be pushed while it runs.
    reply: usize,
    /// When the current request to the model went out: at the start, and
    /// again after each tool result (the next round is sent right after the
    /// last one). Drives the `thinking… 3s` counter.
    round_started: Instant,
    /// Stops the turn task.
    abort: AbortHandle,
    /// Answers the agent's permission requests.
    decisions: mpsc::Sender<Decision>,
}

/// The `/compact` currently running.
struct Compaction {
    /// Matches [`AppEvent::Compacted`]'s `turn`; results for other numbers are stale.
    id: u64,
    /// Length of the transcript when it started; harness lines said after
    /// that (e.g. `/help`) are kept when the compacted conversation replaces it.
    start: usize,
    /// Stops the compaction task.
    abort: AbortHandle,
}

impl App {
    /// Creates an app with an empty conversation, waiting for the backend to start.
    ///
    /// `events` is the main loop's channel; tasks spawned by `update` send their
    /// results there. `cwd` is where loops works, shown in the banner.
    pub fn new(agent: Arc<Agent>, events: mpsc::Sender<AppEvent>, cwd: PathBuf) -> Self {
        let mut input = TextArea::default();
        // The default underlines the cursor line, which looks noisy in a one-line box.
        input.set_cursor_line_style(Style::default());
        input.set_placeholder_text(PLACEHOLDER);
        // The default is dark gray; dim matches every other secondary text.
        input.set_placeholder_style(Style::new().add_modifier(Modifier::DIM));
        Self {
            conversation: Conversation::new(),
            input,
            model_status: ModelStatus::Starting,
            mode: Mode::Normal,
            spinner: 0,
            should_quit: false,
            reply_ends: HashMap::new(),
            tool_outcomes: HashMap::new(),
            tool_previews: HashMap::new(),
            tool_outputs: HashMap::new(),
            full_output: false,
            always_allowed: HashSet::new(),
            context_length: None,
            thinking: Thinking::Default,
            max_output_tokens: None,
            usage: None,
            cwd,
            instructions: None,
            scroll_up: 0,
            sessions: None,
            session_id: SessionStore::new_id(),
            session_created: unix_now(),
            session_model: None,
            session_title: String::new(),
            save_failed: false,
            turn: None,
            compaction: None,
            next_turn: 0,
            agent,
            events,
        }
    }

    /// Saves sessions to `store` from now on; `None` turns them off.
    pub fn with_sessions(mut self, store: Option<SessionStore>) -> Self {
        self.sessions = store;
        self
    }

    /// Whether a turn or `/compact` is running (so Esc would cancel it, and
    /// nothing can be sent).
    pub fn busy(&self) -> bool {
        self.turn.is_some() || self.compaction.is_some()
    }

    /// Whether `/compact` is running.
    pub fn compacting(&self) -> bool {
        self.compaction.is_some()
    }

    /// The reply to draw as `thinking…`, and how long its round has run: the
    /// running turn's reply while it has neither text nor tool calls. That
    /// covers the model reading the prompt as well as reasoning, since both
    /// are the same silent wait from the user's side.
    pub fn waiting_reply(&self) -> Option<(usize, Duration)> {
        let turn = self.turn.as_ref()?;
        let reply = self.conversation.messages().get(turn.reply)?;
        (reply.content.is_empty() && reply.tool_calls.is_empty())
            .then(|| (turn.reply, turn.round_started.elapsed()))
    }

    /// Applies one event to the state.
    pub fn update(&mut self, event: AppEvent) {
        match event {
            AppEvent::Key(key) => self.on_key(key),
            AppEvent::Tick => self.spinner = self.spinner.wrapping_add(1),
            AppEvent::Model(event) => self.on_model(event),
            AppEvent::Turn { turn, event } => self.on_turn(turn, event),
            AppEvent::Compacted { turn, result } => self.on_compacted(turn, result),
            AppEvent::Say(text) => self.say(text),
            // Nothing to change: the main loop redraws after every event.
            AppEvent::Resize => {}
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        // Windows reports both press and release; act on presses (and repeats) only.
        if key.kind == KeyEventKind::Release {
            return;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && matches!(key.code, KeyCode::Char('c' | 'd')) {
            self.quit();
            return;
        }

        match self.mode {
            Mode::Normal => self.on_input_key(key),
            Mode::Picker { .. } => self.on_picker_key(key.code),
            Mode::Sessions { .. } => self.on_sessions_key(key.code),
            Mode::Confirm { .. } => self.on_confirm_key(key.code),
            Mode::Permission { .. } => self.on_permission_key(key.code),
        }
    }

    /// Keys while typing in the input box.
    fn on_input_key(&mut self, key: KeyEvent) {
        let newline_modifier = key
            .modifiers
            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT);

        match key.code {
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.full_output = !self.full_output;
            }
            KeyCode::Enter if newline_modifier => self.input.insert_newline(),
            KeyCode::Enter => self.submit(),
            KeyCode::Esc => {
                self.cancel_turn();
                self.cancel_compaction();
            }
            // Keys, not the wheel: capturing the mouse to scroll would take
            // away the terminal's own text selection and copy.
            KeyCode::PageUp => {
                self.scroll_up = (self.scroll_up + SCROLL_STEP).min(self.transcript_lines());
            }
            KeyCode::PageDown => self.scroll_up = self.scroll_up.saturating_sub(SCROLL_STEP),
            // End scrolls back to the bottom; once there it is the input box's.
            KeyCode::End if self.scroll_up > 0 => self.scroll_up = 0,
            _ => {
                self.input.input(key);
            }
        }
    }

    /// An upper bound on the transcript's height in rows, so PageUp cannot
    /// scroll far past the top and leave PageDown with nothing to do for a
    /// while. Counts every drawn line twice to allow for wrapping; the draw
    /// clamps to the real overflow.
    fn transcript_lines(&self) -> usize {
        let drawn: usize = self
            .conversation
            .messages()
            .iter()
            .map(|m| {
                let previews: usize = m
                    .tool_calls
                    .iter()
                    .filter_map(|c| self.tool_previews.get(&c.id))
                    .map(|p| p.lines().count())
                    .sum();
                let outputs: usize = m
                    .tool_calls
                    .iter()
                    .filter_map(|c| self.tool_outputs.get(&c.id))
                    .map(|o| crate::ui::output_lines(o, self.full_output).len())
                    .sum();
                m.content.lines().count().max(1) + 1 + m.tool_calls.len() * 2 + previews + outputs
            })
            .sum();
        drawn * 2
    }

    /// Keys while the model list is open.
    fn on_picker_key(&mut self, code: KeyCode) {
        let Mode::Picker { models, selected } = &mut self.mode else {
            return;
        };
        match code {
            KeyCode::Up | KeyCode::Char('k') => *selected = selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                if *selected + 1 < models.len() {
                    *selected += 1;
                }
            }
            KeyCode::Enter => {
                let models = std::mem::take(models);
                let selected = *selected;
                self.mode = Mode::Confirm { models, selected };
            }
            KeyCode::Esc => self.mode = Mode::Normal,
            _ => {}
        }
    }

    /// Keys while the session list is open: Up/Down move, Enter resumes the
    /// highlighted session, Esc closes the list.
    fn on_sessions_key(&mut self, code: KeyCode) {
        let Mode::Sessions { entries, selected } = &mut self.mode else {
            return;
        };
        match code {
            KeyCode::Up | KeyCode::Char('k') => *selected = selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                if *selected + 1 < entries.len() {
                    *selected += 1;
                }
            }
            KeyCode::Enter => {
                let id = entries[*selected].id.clone();
                self.mode = Mode::Normal;
                self.switch_session(&id);
            }
            KeyCode::Esc => self.mode = Mode::Normal,
            _ => {}
        }
    }

    /// Keys while asking `Load <id>? (y/n)`. Anything but y/n/Esc is ignored.
    fn on_confirm_key(&mut self, code: KeyCode) {
        let Mode::Confirm { models, selected } = &mut self.mode else {
            return;
        };
        match code {
            KeyCode::Char('y' | 'Y') => {
                let chosen = &models[*selected];
                let id = chosen.id.clone();
                let already_loaded = chosen.state == ModelState::Loaded;
                let context_length = chosen.context_length();
                self.mode = Mode::Normal;
                // `lms load` on a loaded model starts a second copy in memory
                // (`<id>:2`), so a loaded model is reported, not reloaded.
                if already_loaded {
                    self.say(format!("already loaded: {id}"));
                    self.context_length = context_length;
                    self.model_status = ModelStatus::Ready(id);
                } else {
                    self.model_status = ModelStatus::Loading(id.clone());
                    self.spawn_load(id);
                }
            }
            KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                let models = std::mem::take(models);
                let selected = *selected;
                self.mode = Mode::Picker { models, selected };
            }
            _ => {}
        }
    }

    /// Keys while asking whether a tool may change things: Up/Down move (no
    /// wrap), Enter answers with the highlighted option, 1/2/3 answer with
    /// that option directly, Esc cancels the turn. Anything else is ignored,
    /// so typing ahead cannot answer by accident.
    fn on_permission_key(&mut self, code: KeyCode) {
        let Mode::Permission {
            scope, selected, ..
        } = &mut self.mode
        else {
            return;
        };
        let option = match code {
            KeyCode::Up => {
                *selected = selected.saturating_sub(1);
                return;
            }
            KeyCode::Down => {
                if *selected + 1 < PermissionOption::ALL.len() {
                    *selected += 1;
                }
                return;
            }
            KeyCode::Enter => PermissionOption::ALL[*selected],
            KeyCode::Char(digit @ '1'..='3') => {
                PermissionOption::ALL[usize::from(digit as u8 - b'1')]
            }
            // Ending the turn closes the prompt too.
            KeyCode::Esc => {
                self.cancel_turn();
                return;
            }
            _ => return,
        };
        let decision = match option {
            PermissionOption::Allow => Decision::Allow,
            PermissionOption::AlwaysAllow => {
                self.always_allowed.insert(scope.clone());
                Decision::Allow
            }
            PermissionOption::Deny => Decision::Deny,
        };
        self.mode = Mode::Normal;
        if let Some(turn) = &self.turn {
            // `update` must not wait, hence `try_send`. It fails only if the
            // turn already ended, and then nobody is waiting for the answer.
            let _ = turn.decisions.try_send(decision);
        }
    }

    /// Takes the input box contents and either runs a command or sends a message.
    /// A message that cannot be sent yet stays in the box.
    fn submit(&mut self) {
        let text = self.input.lines().join("\n");
        let text = text.trim();
        if text.is_empty() {
            return;
        }

        if let Some(command) = text.strip_prefix('/') {
            self.run_command(command);
        } else if !self.send(text) {
            return;
        }
        self.input.clear();
        // Whatever was just sent or asked for shows at the bottom.
        self.scroll_up = 0;
    }

    /// Records `text` as a user message and starts a turn.
    /// Returns false, sending nothing, if no model is ready or a turn (or
    /// `/compact`) is running.
    fn send(&mut self, text: &str) -> bool {
        if self.busy() {
            return false;
        }
        let Some(model) = self.ready_model() else {
            return false;
        };

        self.conversation.push(Message::new(Role::User, text));
        let start = self.conversation.messages().len() - 1;
        let history = self.conversation.clone();
        self.conversation.push(Message::new(Role::Assistant, ""));
        self.model_status = ModelStatus::Busy(model.clone());
        self.session_model = Some(model.clone());
        self.spawn_turn(model, history, start);
        true
    }

    /// The id of the model ready to be sent a request, or `None` after saying
    /// why there is none (loading, unloading, or nothing loaded).
    fn ready_model(&mut self) -> Option<String> {
        match &self.model_status {
            ModelStatus::Ready(id) => Some(id.clone()),
            ModelStatus::Loading(id) => {
                self.say(format!("{id} is still loading"));
                None
            }
            ModelStatus::Unloading(_) => {
                self.say(UNLOADING_TEXT);
                None
            }
            _ => {
                self.say("no model loaded; use /model");
                None
            }
        }
    }

    /// Runs a slash command. `command` is the text after the `/`.
    fn run_command(&mut self, command: &str) {
        let mut words = command.split_whitespace();
        let name = words.next().unwrap_or("");
        let arg = words.next();
        match name {
            "quit" | "exit" => self.quit(),
            "help" => self.say(HELP_TEXT),
            // Only reads state, so it works while a turn runs.
            "context" => self.say(self.context_report()),
            // Settings for the next request, so they work while a turn runs.
            "think" => self.think(arg),
            "max-tokens" => self.max_tokens(arg),
            "model" | "unload" | "clear" | "new" | "compact" | "sessions" if self.busy() => {
                self.say(BUSY_TEXT);
            }
            "model" if self.is_unloading() => self.say(UNLOADING_TEXT),
            "model" => self.spawn_list(),
            "unload" => self.unload(),
            "clear" | "new" => self.start_new_session(),
            "sessions" => self.open_sessions(),
            "compact" => self.compact(),
            _ => self.say(format!("unknown command: /{name}")),
        }
    }

    /// `/quit`, Ctrl+C: ends a running turn as Esc would (so what is saved
    /// can be sent again), saves the session, and asks the main loop to stop.
    fn quit(&mut self) {
        self.cancel_turn();
        self.end_compaction();
        self.save_session();
        self.should_quit = true;
    }

    /// `/clear` and `/new`: saves the current session and starts another
    /// with an empty conversation. The model stays loaded and tools the user
    /// always-allowed stay allowed. Refused while a turn runs.
    fn start_new_session(&mut self) {
        self.save_session();
        self.clear();
        self.session_id = SessionStore::new_id();
        self.session_created = unix_now();
        self.session_model = None;
        self.session_title.clear();
        self.save_failed = false;
    }

    /// `/sessions`: opens the list of this directory's saved sessions, or
    /// says why there is nothing to list.
    fn open_sessions(&mut self) {
        let Some(store) = &self.sessions else {
            self.say(SESSIONS_OFF_TEXT);
            return;
        };
        match store.list() {
            Ok(entries) if entries.is_empty() => self.say(NO_SESSIONS_TEXT),
            Ok(entries) => {
                self.mode = Mode::Sessions {
                    entries,
                    selected: 0,
                };
            }
            Err(err) => self.say(format!("error: sessions could not be listed: {err}")),
        }
    }

    /// `loops --resume [id]`: switches to session `id`, or to the latest
    /// one of this directory, saying why when there is none.
    pub fn resume(&mut self, id: Option<&str>) {
        if let Some(id) = id {
            self.switch_session(id);
            return;
        }
        let Some(store) = &self.sessions else {
            self.say(SESSIONS_OFF_TEXT);
            return;
        };
        match store.latest() {
            Ok(Some(latest)) => self.switch_session(&latest.id),
            Ok(None) => self.say(NO_SESSIONS_TEXT),
            Err(err) => self.say(format!("error: sessions could not be listed: {err}")),
        }
    }

    /// Saves the current session and makes session `id` current: its
    /// messages and display state replace the transcript. Switching to the
    /// current session changes nothing; a session that cannot be loaded is
    /// reported and the current one stays.
    pub fn switch_session(&mut self, id: &str) {
        if id == self.session_id {
            return;
        }
        let Some(store) = &self.sessions else {
            self.say(SESSIONS_OFF_TEXT);
            return;
        };
        let session = match store.load(id) {
            Ok(Some(session)) => session,
            Ok(None) => {
                self.say(format!("no session {id}"));
                return;
            }
            Err(err) => {
                self.say(format!("error: session {id} could not be loaded: {err}"));
                return;
            }
        };
        self.save_session();
        self.clear();
        let view: View = serde_json::from_value(session.view).unwrap_or_default();
        self.tool_outcomes = view.tool_outcomes;
        self.tool_previews = view.tool_previews;
        self.tool_outputs = view.tool_outputs;
        self.reply_ends = view.reply_ends.into_iter().collect();
        let count = session.messages.len();
        self.conversation.replace_all(session.messages);
        self.session_id = session.id;
        self.session_created = session.created;
        self.session_model = session.model;
        self.session_title = session.title;
        self.save_failed = false;
        self.scroll_up = 0;
        let title = if self.session_title.is_empty() {
            "(untitled)"
        } else {
            &self.session_title
        };
        self.say(format!("resumed \"{title}\" ({count} messages)"));
    }

    /// Writes the current session to the store. Does nothing when sessions
    /// are off or nobody has typed a message yet; a failure is said once
    /// per session.
    fn save_session(&mut self) {
        let Some(store) = &self.sessions else {
            return;
        };
        let messages = self.conversation.messages();
        if !messages.iter().any(|m| m.role == Role::User) {
            return;
        }
        if self.session_title.is_empty() {
            self.session_title = title_from(messages);
        }
        let mut reply_ends: Vec<(usize, ReplyEnd)> =
            self.reply_ends.iter().map(|(i, end)| (*i, *end)).collect();
        reply_ends.sort_by_key(|(i, _)| *i);
        let view = View {
            tool_outcomes: self.tool_outcomes.clone(),
            tool_previews: self.tool_previews.clone(),
            tool_outputs: self.tool_outputs.clone(),
            reply_ends,
        };
        let session = Session {
            id: self.session_id.clone(),
            title: self.session_title.clone(),
            model: self.session_model.clone(),
            cwd: self.cwd.display().to_string(),
            created: self.session_created,
            updated: unix_now(),
            messages: messages.to_vec(),
            view: serde_json::to_value(view).unwrap_or_default(),
        };
        if let Err(err) = store.save(&session)
            && !self.save_failed
        {
            self.save_failed = true;
            self.say(format!("session not saved: {err}"));
        }
    }

    /// Forgets the conversation and its display state, so the next message
    /// starts fresh. Only what was said goes.
    fn clear(&mut self) {
        self.conversation = Conversation::new();
        self.reply_ends.clear();
        self.tool_outcomes.clear();
        self.tool_previews.clear();
        self.tool_outputs.clear();
        // The next request starts from an empty conversation.
        self.usage = None;
    }

    /// `/compact`: starts summarising the conversation in the background (see
    /// the module docs); the result arrives as [`AppEvent::Compacted`].
    /// Refused while no model is ready. A conversation with nothing to
    /// compact fails in the task, which says so.
    fn compact(&mut self) {
        let Some(model) = self.ready_model() else {
            return;
        };
        let id = self.next_turn;
        self.next_turn += 1;
        let agent = Arc::clone(&self.agent);
        let events = self.events.clone();
        let options = self.turn_options();
        let mut conversation = self.conversation.clone();
        self.model_status = ModelStatus::Busy(model.clone());
        let task = tokio::spawn(async move {
            let result = agent
                .compact(&model, &mut conversation, &options)
                .await
                .map(|report| (conversation, report))
                .map_err(|err| format!("{err:#}"));
            // A send error means the app has quit; nobody is left to tell.
            let _ = events.send(AppEvent::Compacted { turn: id, result }).await;
        });
        self.compaction = Some(Compaction {
            id,
            start: self.conversation.messages().len(),
            abort: task.abort_handle(),
        });
    }

    /// Applies the result of `/compact`; results of any compaction but the
    /// running one (e.g. cancelled with Esc) are dropped. On success the
    /// compacted conversation replaces the transcript, as `/clear` would,
    /// keeping harness lines said while it ran.
    fn on_compacted(&mut self, id: u64, result: Result<(Conversation, CompactReport), String>) {
        if self.compaction.as_ref().is_none_or(|c| c.id != id) {
            return;
        }
        let Some(compaction) = self.end_compaction() else {
            return;
        };
        let (conversation, report) = match result {
            Ok(done) => done,
            // The model stays loaded, so the status bar stays as it is.
            Err(err) => {
                self.say(format!("error: {err}"));
                return;
            }
        };
        let said: Vec<Message> = self
            .conversation
            .messages()
            .iter()
            .skip(compaction.start)
            .filter(|m| m.role == Role::Harness)
            .cloned()
            .collect();
        // The summary becomes the first user message; the title stays what
        // the user first asked.
        if self.session_title.is_empty() {
            self.session_title = title_from(self.conversation.messages());
        }
        self.clear();
        self.conversation = conversation;
        self.scroll_up = 0;
        for message in said {
            self.conversation.push(message);
        }
        self.say(format!(
            "compacted: about {} → {} tokens",
            thousands(report.before),
            thousands(report.after)
        ));
        self.save_session();
    }

    /// Esc while `/compact` runs: stop it; the conversation stays as it was.
    fn cancel_compaction(&mut self) {
        if self.end_compaction().is_some() {
            self.say("compaction cancelled");
        }
    }

    /// Stops the compaction task (harmless if it already finished) and puts
    /// the status back to `Ready`. Returns the compaction, if one was running.
    fn end_compaction(&mut self) -> Option<Compaction> {
        let compaction = self.compaction.take()?;
        compaction.abort.abort();
        if let ModelStatus::Busy(id) = &self.model_status {
            self.model_status = ModelStatus::Ready(id.clone());
        }
        Some(compaction)
    }

    /// `/think`: says the thinking level, or sets it from `level`.
    fn think(&mut self, level: Option<&str>) {
        let Some(level) = level else {
            self.say(format!(
                "thinking: {} (off, low, medium, high, or default)",
                self.thinking
            ));
            return;
        };
        match Thinking::parse(level) {
            Some(thinking) => {
                self.thinking = thinking;
                self.say(format!("thinking: {thinking}"));
            }
            None => self.say(format!("unknown thinking level: {level}")),
        }
    }

    /// `/max-tokens`: says the reply cap, or sets it from `value` (`off`
    /// clears it).
    fn max_tokens(&mut self, value: Option<&str>) {
        match value {
            None => {}
            Some(off) if off.eq_ignore_ascii_case("off") => self.max_output_tokens = None,
            Some(n) => match n.parse::<u64>() {
                Ok(n) if n > 0 => self.max_output_tokens = Some(n),
                _ => {
                    self.say(format!(
                        "not a reply cap: {n} (give a whole number above 0, or off)"
                    ));
                    return;
                }
            },
        }
        self.say(self.reply_cap());
    }

    /// `reply cap: none` or `reply cap: 2,048 tokens`.
    fn reply_cap(&self) -> String {
        match self.max_output_tokens {
            Some(n) => format!("reply cap: {} tokens", thousands(n)),
            None => "reply cap: none".to_owned(),
        }
    }

    /// `/context`: how much of the model's window the conversation fills, from
    /// the last request's usage and the loaded model's context length, then
    /// the limits the next request goes out with, then the project
    /// instructions sent with it.
    fn context_report(&self) -> String {
        let instructions = match &self.instructions {
            Some(described) => format!("instructions: {described}"),
            None => format!("instructions: none (no {INSTRUCTIONS_FILE} in the working directory)"),
        };
        let session = match &self.sessions {
            Some(store) => format!(
                "session: {} ({} messages, {})",
                self.session_id,
                self.conversation.messages().len(),
                store.dir().display()
            ),
            None => format!(
                "session: {} (not saved: no home directory)",
                self.session_id
            ),
        };
        format!("{}\n{instructions}\n{session}", self.window_report())
    }

    /// The window and settings lines of [`Self::context_report`].
    fn window_report(&self) -> String {
        let settings = format!("thinking: {}, {}", self.thinking, self.reply_cap());
        let Some(model) = self.loaded_model() else {
            return format!("context: no model loaded\n{settings}");
        };
        let Some(usage) = self.usage else {
            let first = match self.context_length {
                Some(window) => format!(
                    "context: nothing sent yet; the window is {} tokens ({model})",
                    thousands(window)
                ),
                None => format!("context: nothing sent yet; the window size is unknown ({model})"),
            };
            return format!("{first}\n{settings}");
        };
        let used = usage.prompt_tokens + usage.completion_tokens;
        let first = match self.context_length {
            Some(window) if used > window => format!(
                "context: {} of {} tokens used (over the window by {})",
                thousands(used),
                thousands(window),
                thousands(used - window)
            ),
            Some(window) => format!(
                "context: {} of {} tokens used ({}%), {} left",
                thousands(used),
                thousands(window),
                // Rounded to the nearest percent.
                (used * 100 + window / 2) / window.max(1),
                thousands(window - used)
            ),
            None => format!(
                "context: {} tokens used of an unknown window",
                thousands(used)
            ),
        };
        format!(
            "{first}\nlast request: {} prompt + {} reply ({model})\n{settings}",
            thousands(usage.prompt_tokens),
            thousands(usage.completion_tokens)
        )
    }

    /// Id of the model that is loaded (ready, busy, or still unloading).
    fn loaded_model(&self) -> Option<&str> {
        match &self.model_status {
            ModelStatus::Ready(id) | ModelStatus::Busy(id) | ModelStatus::Unloading(id) => Some(id),
            _ => None,
        }
    }

    /// `/unload`: starts unloading the ready model. Refused while a model is
    /// loading or unloading; says so if nothing is loaded.
    fn unload(&mut self) {
        match &self.model_status {
            ModelStatus::Ready(id) => {
                let id = id.clone();
                self.model_status = ModelStatus::Unloading(id.clone());
                self.spawn_unload(id);
            }
            ModelStatus::Loading(id) => self.say(format!("{id} is still loading")),
            ModelStatus::Unloading(_) => self.say(UNLOADING_TEXT),
            _ => self.say("no model loaded"),
        }
    }

    /// True while `/unload` runs.
    fn is_unloading(&self) -> bool {
        matches!(self.model_status, ModelStatus::Unloading(_))
    }

    /// Applies the result of a background model task.
    fn on_model(&mut self, event: ModelEvent) {
        match event {
            ModelEvent::ServerReady(Some(model)) => {
                self.context_length = model.context_length();
                self.model_status = ModelStatus::Ready(model.id);
            }
            ModelEvent::ServerReady(None) => {
                self.context_length = None;
                self.model_status = ModelStatus::NoModel;
            }
            // Asked for before the turn started; loading now would pull the model away.
            ModelEvent::ModelsListed(_) if self.busy() => self.say(BUSY_TEXT),
            // Asked for before `/unload`; a load now would race it.
            ModelEvent::ModelsListed(_) if self.is_unloading() => self.say(UNLOADING_TEXT),
            ModelEvent::ModelsListed(models) if models.is_empty() => {
                self.say("no chat models found in LM Studio");
            }
            ModelEvent::ModelsListed(models) => {
                // Start on the loaded model so Enter on it is a no-op reload, not a surprise.
                let selected = models
                    .iter()
                    .position(|m| m.state == ModelState::Loaded)
                    .unwrap_or(0);
                self.mode = Mode::Picker { models, selected };
            }
            // `usage` stays: the conversation is the same, only the window changes.
            ModelEvent::Loaded(model) => {
                self.say(format!("loaded {}", model.id));
                self.context_length = model.context_length();
                self.model_status = ModelStatus::Ready(model.id);
            }
            ModelEvent::Unloaded(id) => {
                self.say(format!("unloaded {id}"));
                self.context_length = None;
                self.model_status = ModelStatus::NoModel;
            }
            // `lms unload` failed or the server still lists the model, so it
            // is most likely still loaded and usable.
            ModelEvent::UnloadFailed(err) => {
                if let ModelStatus::Unloading(id) = &self.model_status {
                    self.model_status = ModelStatus::Ready(id.clone());
                }
                self.say(format!("error: {err}"));
            }
            // A failed listing changes nothing about what is loaded, so the
            // status bar keeps telling the truth and only the transcript reports it.
            ModelEvent::ListFailed(err) => self.say(format!("error: {err}")),
            ModelEvent::ServerFailed(err) | ModelEvent::LoadFailed(err) => self.fail(err),
        }
    }

    /// Applies progress of a turn. Events for any turn but the current one
    /// (e.g. queued before an Esc) are dropped.
    fn on_turn(&mut self, id: u64, event: TurnEvent) {
        if self.turn.as_ref().is_none_or(|turn| turn.id != id) {
            return;
        }
        match event {
            TurnEvent::Agent(event) => self.mirror(event),
            TurnEvent::Finished(conversation) => self.finish_turn(conversation),
        }
    }

    /// Shows one agent event in the transcript while the turn runs.
    fn mirror(&mut self, event: AgentEvent) {
        if let AgentEvent::ReplyCutOff { attempt, max } = event {
            // The reply keeps its text; the continuation is appended to it.
            self.say(format!(
                "reply cut off at the output limit; asking the model to continue ({attempt}/{max})"
            ));
            return;
        }
        let Some(turn) = self.turn.as_mut() else {
            return;
        };
        // Text after a round that asked for tools belongs to a new reply.
        let asked_for_tools = self
            .conversation
            .messages()
            .get(turn.reply)
            .is_some_and(|reply| !reply.tool_calls.is_empty());
        if asked_for_tools && matches!(event, AgentEvent::Thinking | AgentEvent::Text(_)) {
            self.conversation.push(Message::new(Role::Assistant, ""));
            turn.reply = self.conversation.messages().len() - 1;
        }
        let Some(reply) = self.conversation.get_mut(turn.reply) else {
            return;
        };
        match event {
            // The empty reply already draws as `thinking…` (see `waiting_reply`).
            AgentEvent::Thinking => {}
            AgentEvent::Text(text) => reply.content.push_str(&text),
            AgentEvent::ToolStarted {
                id,
                name,
                arguments,
            } => {
                reply.tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            AgentEvent::PermissionRequested {
                id,
                name,
                preview,
                scope,
                ..
            } => {
                self.tool_previews.insert(id.clone(), preview);
                if self.always_allowed.contains(&scope) {
                    // See `on_permission_key` for why the result is ignored.
                    let _ = turn.decisions.try_send(Decision::Allow);
                } else {
                    self.mode = Mode::Permission {
                        id,
                        name,
                        scope,
                        selected: 0,
                    };
                }
            }
            AgentEvent::ToolFinished {
                id,
                name,
                summary,
                is_error,
                output,
            } => {
                // An error's summary already says what went wrong, and a
                // declined call's output is only the refusal the model reads.
                if name == "bash" && !is_error {
                    self.tool_outputs.insert(id.clone(), output);
                }
                let outcome = ToolOutcome { summary, is_error };
                self.tool_outcomes.insert(id, outcome);
                turn.round_started = Instant::now();
            }
            AgentEvent::Usage {
                prompt_tokens,
                completion_tokens,
            } => {
                self.usage = Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                });
            }
            // Said above, before the reply is borrowed.
            AgentEvent::ReplyCutOff { .. } => {}
            // `Finished` follows, carrying the conversation.
            AgentEvent::TurnDone => {}
            AgentEvent::TurnFailed(err) => self.say(format!("error: {err}")),
        }
    }

    /// The turn task finished: its conversation replaces the mirror. Harness
    /// lines said during the turn (e.g. its error) move to the end.
    fn finish_turn(&mut self, conversation: Conversation) {
        let Some(turn) = self.end_turn() else {
            return;
        };
        let said: Vec<Message> = self.conversation.messages()[turn.start + 1..]
            .iter()
            .filter(|m| m.role == Role::Harness)
            .cloned()
            .collect();
        self.conversation = conversation;

        let last = self.conversation.messages().len() - 1;
        let answer = &self.conversation.messages()[last];
        if last > turn.start
            && answer.role == Role::Assistant
            && answer.content.is_empty()
            && answer.tool_calls.is_empty()
        {
            self.reply_ends.insert(last, ReplyEnd::Empty);
        }
        for message in said {
            self.conversation.push(message);
        }
        self.save_session();
    }

    /// Esc while a turn runs: stop it and keep the text that arrived, marked
    /// cancelled. The turn's tool calls go too: their results were in the
    /// agent's copy, and a call without its result cannot be sent again.
    fn cancel_turn(&mut self) {
        let Some(turn) = self.end_turn() else {
            return;
        };
        let mut reply = turn.reply;
        for i in (turn.start + 1..=turn.reply).rev() {
            let Some(message) = self.conversation.get_mut(i) else {
                continue;
            };
            if message.role != Role::Assistant {
                continue;
            }
            message.tool_calls.clear();
            // Earlier rounds that only asked for tools have nothing left to show.
            if message.content.is_empty() && i != turn.reply {
                self.conversation.remove(i);
                reply -= 1;
            }
        }
        self.reply_ends.insert(reply, ReplyEnd::Cancelled);
        // The task's `Finished` is dropped as stale, so `finish_turn` does
        // not run: save here.
        self.save_session();
    }

    /// Stops the turn's task (harmless if it already finished), closes its
    /// permission prompt if one is open, and puts the status back to `Ready`.
    /// Returns the turn, if one was running.
    fn end_turn(&mut self) -> Option<Turn> {
        let turn = self.turn.take()?;
        turn.abort.abort();
        if matches!(self.mode, Mode::Permission { .. }) {
            self.mode = Mode::Normal;
        }
        // A failed or cancelled turn leaves the model loaded.
        if let ModelStatus::Busy(id) = &self.model_status {
            self.model_status = ModelStatus::Ready(id.clone());
        }
        Some(turn)
    }

    /// Shows a failure: short form in the status bar, full text in the transcript.
    fn fail(&mut self, err: String) {
        let first_line = err.lines().next().unwrap_or_default();
        let short = if first_line.chars().count() > STATUS_ERROR_CHARS {
            let cut: String = first_line.chars().take(STATUS_ERROR_CHARS - 1).collect();
            format!("{cut}…")
        } else {
            first_line.to_owned()
        };
        self.model_status = ModelStatus::Error(short);
        self.say(format!("error: {err}"));
    }

    /// Asks the backend for its models; the answer arrives as `ModelsListed`.
    fn spawn_list(&self) {
        let backend = Arc::clone(&self.agent.backend);
        self.spawn(async move {
            match backend.list_models().await {
                Ok(models) => ModelEvent::ModelsListed(models),
                Err(err) => ModelEvent::ListFailed(format!("{err:#}")),
            }
        });
    }

    /// Asks the backend to load `id`; the answer arrives as `Loaded` or `LoadFailed`.
    fn spawn_load(&self, id: String) {
        let backend = Arc::clone(&self.agent.backend);
        self.spawn(async move {
            match backend.load_model(&id).await {
                Ok(model) => ModelEvent::Loaded(model),
                Err(err) => ModelEvent::LoadFailed(format!("{err:#}")),
            }
        });
    }

    /// Asks the backend to unload `id`; the answer arrives as `Unloaded` or `UnloadFailed`.
    fn spawn_unload(&self, id: String) {
        let backend = Arc::clone(&self.agent.backend);
        self.spawn(async move {
            match backend.unload_model(&id).await {
                Ok(()) => ModelEvent::Unloaded(id),
                Err(err) => ModelEvent::UnloadFailed(format!("{err:#}")),
            }
        });
    }

    /// Runs the agent on `history`, whose message at `start` is the user's, in
    /// the background. Its events arrive as [`AppEvent::Turn`]s, `Finished` last.
    fn spawn_turn(&mut self, model: String, history: Conversation, start: usize) {
        let id = self.next_turn;
        self.next_turn += 1;
        let agent = Arc::clone(&self.agent);
        let events = self.events.clone();
        let (decisions, answers) = mpsc::channel(4);
        let options = self.turn_options();
        let task = tokio::spawn(async move {
            let send = |event| events.send(AppEvent::Turn { turn: id, event });
            let (sink, mut progress) = mpsc::channel(64);
            let forward = async {
                while let Some(event) = progress.recv().await {
                    if send(TurnEvent::Agent(event)).await.is_err() {
                        // The app has quit; nobody is left to tell.
                        break;
                    }
                }
            };
            let mut conversation = history;
            // `run_turn` owns `sink` and drops it when it returns, which ends
            // `forward`. Its error was already sent as `TurnFailed`.
            let turn = agent.run_turn(&model, &mut conversation, sink, answers, &options);
            let (_, ()) = tokio::join!(turn, forward);
            let _ = send(TurnEvent::Finished(conversation)).await;
        });
        self.turn = Some(Turn {
            id,
            start,
            reply: start + 1,
            round_started: Instant::now(),
            abort: task.abort_handle(),
            decisions,
        });
    }

    /// The limits the next request goes out with: the loaded model's window
    /// and the `/think` and `/max-tokens` settings.
    fn turn_options(&self) -> TurnOptions {
        TurnOptions {
            context_length: self.context_length,
            max_output_tokens: self.max_output_tokens,
            thinking: self.thinking,
        }
    }

    /// Runs `task` in the background and sends its result into the main loop.
    fn spawn(&self, task: impl Future<Output = ModelEvent> + Send + 'static) {
        let events = self.events.clone();
        tokio::spawn(async move {
            // A send error means the app has quit; nobody is left to tell.
            let _ = events.send(AppEvent::Model(task.await)).await;
        });
    }

    /// Pushes a harness-authored line into the transcript.
    fn say(&mut self, text: impl Into<String>) {
        self.conversation.push(Message::new(Role::Harness, text));
    }
}

/// The display state saved with a session ([`Session::view`]): the maps the
/// transcript draws from besides the messages.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct View {
    /// [`App::tool_outcomes`].
    tool_outcomes: HashMap<String, ToolOutcome>,
    /// [`App::tool_previews`].
    tool_previews: HashMap<String, String>,
    /// [`App::tool_outputs`].
    tool_outputs: HashMap<String, String>,
    /// [`App::reply_ends`] as `(index, end)` pairs, since JSON object keys
    /// cannot be numbers.
    reply_ends: Vec<(usize, ReplyEnd)>,
}

/// How much of the project instructions file the model gets, after its name:
/// `about 420 tokens`, or `cut to the first 8,000 of 12,345 characters (about
/// 2,000 tokens)` when it was too long.
pub(crate) fn describe_instructions(instructions: &Instructions) -> String {
    let tokens = thousands(instructions.estimated_tokens());
    if instructions.truncated {
        format!(
            "cut to the first {} of {} characters (about {tokens} tokens)",
            thousands(instructions.shown_chars as u64),
            thousands(instructions.total_chars as u64)
        )
    } else {
        format!("about {tokens} tokens")
    }
}

/// `n` with a comma between each group of three digits, e.g. `8,192`.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// A fake backend and app constructor shared by the `app` and `ui` tests.
#[cfg(test)]
pub mod testing {
    use std::{path::PathBuf, sync::Mutex};

    use anyhow::{Result, anyhow};
    use loops_core::{
        backend::{BoxFuture, ModelBackend},
        chat::{ChatDelta, RequestOptions},
        tools::{ToolSet, ToolSpec},
    };

    use super::*;

    /// A model entry with the fields the UI shows. Loaded ones report a
    /// window of 8192 tokens, as LM Studio's default load does.
    pub fn model(id: &str, state: ModelState) -> ModelInfo {
        ModelInfo {
            id: id.to_owned(),
            kind: "llm".to_owned(),
            arch: None,
            quantization: Some("Q4_K_M".to_owned()),
            state,
            max_context_length: Some(262_144),
            loaded_context_length: (state == ModelState::Loaded).then_some(8192),
            capabilities: Vec::new(),
        }
    }

    /// Serves a fixed model list and scripted replies, and records every request.
    #[derive(Default)]
    pub struct FakeBackend {
        /// Returned by `list_models`.
        pub models: Vec<ModelInfo>,
        /// Ids passed to `load_model`, in call order.
        pub loads: Mutex<Vec<String>>,
        /// Ids passed to `unload_model`, in call order.
        pub unloads: Mutex<Vec<String>>,
        /// Chat call `n` sends `script[n]` (nothing once the script runs out).
        pub script: Vec<Vec<ChatDelta>>,
        /// When set, every chat call fails with this after sending its deltas.
        pub fail: Option<&'static str>,
        /// Messages passed to `chat`, one entry per call.
        pub chats: Mutex<Vec<Vec<Message>>>,
        /// Options passed to `chat`, one entry per call.
        pub options: Mutex<Vec<RequestOptions>>,
    }

    impl ModelBackend for FakeBackend {
        fn start(&self) -> BoxFuture<'_, Result<Option<ModelInfo>>> {
            Box::pin(async { Ok(None) })
        }

        fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>> {
            Box::pin(async { Ok(self.models.clone()) })
        }

        fn load_model<'a>(&'a self, id: &'a str) -> BoxFuture<'a, Result<ModelInfo>> {
            self.loads.lock().unwrap().push(id.to_owned());
            Box::pin(async move { Ok(model(id, ModelState::Loaded)) })
        }

        fn unload_model<'a>(&'a self, id: &'a str) -> BoxFuture<'a, Result<()>> {
            self.unloads.lock().unwrap().push(id.to_owned());
            Box::pin(async { Ok(()) })
        }

        fn chat<'a>(
            &'a self,
            _model: &'a str,
            messages: &'a [Message],
            _tools: &'a [ToolSpec],
            options: &'a RequestOptions,
            sink: mpsc::Sender<ChatDelta>,
        ) -> BoxFuture<'a, Result<()>> {
            let mut chats = self.chats.lock().unwrap();
            let round = self.script.get(chats.len()).cloned().unwrap_or_default();
            chats.push(messages.to_vec());
            self.options.lock().unwrap().push(options.clone());
            Box::pin(async move {
                for delta in round {
                    if sink.send(delta).await.is_err() {
                        break;
                    }
                }
                self.fail.map_or(Ok(()), |err| Err(anyhow!(err)))
            })
        }
    }

    /// An app whose agent uses `backend`, no system prompt, and the real
    /// read-only tools rooted at this crate's directory; plus the receiver its
    /// tasks report to.
    pub fn app_with(backend: FakeBackend) -> (App, Arc<FakeBackend>, mpsc::Receiver<AppEvent>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        app_with_tools(backend, ToolSet::read_only(root))
    }

    /// Like [`app_with`], with `tools` instead.
    pub fn app_with_tools(
        backend: FakeBackend,
        tools: ToolSet,
    ) -> (App, Arc<FakeBackend>, mpsc::Receiver<AppEvent>) {
        let backend = Arc::new(backend);
        let agent = Arc::new(Agent::new(backend.clone(), Arc::new(tools), None));
        let (tx, rx) = mpsc::channel(16);
        let cwd = PathBuf::from(r"D:\projects\loops");
        (App::new(agent, tx, cwd), backend, rx)
    }

    /// Makes the running turn's current round look `secs` seconds old.
    pub fn backdate_round(app: &mut App, secs: u64) {
        let turn = app.turn.as_mut().expect("a turn is running");
        turn.round_started = Instant::now() - Duration::from_secs(secs);
    }

    /// Like [`chat_app`], with every standard tool (the ones that change
    /// files too) working in `dir`.
    pub fn editing_app(
        script: Vec<Vec<ChatDelta>>,
        dir: &TempDir,
    ) -> (App, Arc<FakeBackend>, mpsc::Receiver<AppEvent>) {
        let backend = FakeBackend {
            script,
            ..FakeBackend::default()
        };
        ready(app_with_tools(backend, ToolSet::standard(&dir.0)))
    }

    /// A fresh directory under the system temp dir, deleted on drop.
    pub struct TempDir(pub PathBuf);

    impl TempDir {
        /// `loops-tui-<name>-<pid>`, emptied first; `name` must be unique per test.
        pub fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("loops-tui-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        /// Writes `contents` to `rel`.
        pub fn write(&self, rel: &str, contents: &str) {
            std::fs::write(self.0.join(rel), contents).unwrap();
        }

        /// The text of `rel`.
        pub fn read(&self, rel: &str) -> String {
            std::fs::read_to_string(self.0.join(rel)).unwrap()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// An app whose backend lists `models`.
    pub fn test_app(models: Vec<ModelInfo>) -> (App, Arc<FakeBackend>, mpsc::Receiver<AppEvent>) {
        app_with(FakeBackend {
            models,
            ..FakeBackend::default()
        })
    }

    /// An app whose backend has no models; for tests that never touch it.
    pub fn plain_app() -> App {
        test_app(Vec::new()).0
    }

    /// Id of the model [`chat_app`] reports as loaded.
    pub const READY_MODEL: &str = "google/gemma-4-e4b";

    /// An app with [`READY_MODEL`] loaded whose backend replies with `script`.
    pub fn chat_app(
        script: Vec<Vec<ChatDelta>>,
    ) -> (App, Arc<FakeBackend>, mpsc::Receiver<AppEvent>) {
        ready(app_with(FakeBackend {
            script,
            ..FakeBackend::default()
        }))
    }

    /// Like [`chat_app`], but every chat call fails with `err` after its script.
    pub fn failing_app(
        script: Vec<Vec<ChatDelta>>,
        err: &'static str,
    ) -> (App, Arc<FakeBackend>, mpsc::Receiver<AppEvent>) {
        ready(app_with(FakeBackend {
            script,
            fail: Some(err),
            ..FakeBackend::default()
        }))
    }

    fn ready(
        (mut app, backend, rx): (App, Arc<FakeBackend>, mpsc::Receiver<AppEvent>),
    ) -> (App, Arc<FakeBackend>, mpsc::Receiver<AppEvent>) {
        app.model_status = ModelStatus::Ready(READY_MODEL.to_owned());
        (app, backend, rx)
    }

    /// Types `text` into the input box and presses Enter.
    pub fn type_line(app: &mut App, text: &str) {
        for c in text.chars() {
            app.update(AppEvent::Key(KeyEvent::from(KeyCode::Char(c))));
        }
        app.update(AppEvent::Key(KeyEvent::from(KeyCode::Enter)));
    }

    /// An agent event for the turn currently running.
    pub fn agent_event(app: &App, event: AgentEvent) -> AppEvent {
        let turn = app.turn.as_ref().expect("a turn is running").id;
        AppEvent::Turn {
            turn,
            event: TurnEvent::Agent(event),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{
        READY_MODEL, TempDir, agent_event, backdate_round, chat_app, editing_app, failing_app,
        model, plain_app, test_app, type_line,
    };
    use super::*;
    use loops_core::chat::{ChatDelta, RequestOptions};

    fn key(code: KeyCode) -> AppEvent {
        AppEvent::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn content(text: &str) -> ChatDelta {
        ChatDelta::Content(text.to_owned())
    }

    fn text(text: &str) -> AgentEvent {
        AgentEvent::Text(text.to_owned())
    }

    /// Applies channel events until the turn ends.
    async fn run_to_end(app: &mut App, rx: &mut mpsc::Receiver<AppEvent>) {
        while app.turn.is_some() {
            app.update(rx.recv().await.unwrap());
        }
    }

    /// A model round that asks to read this crate's `Cargo.toml`.
    fn read_cargo_toml() -> Vec<ChatDelta> {
        vec![
            ChatDelta::ToolCall {
                index: 0,
                id: Some("call_1".into()),
                name: Some("read_file".into()),
                arguments: r#"{"path":"Cargo.toml"}"#.into(),
            },
            ChatDelta::Done {
                finish_reason: Some("tool_calls".into()),
            },
        ]
    }

    #[tokio::test]
    async fn typing_and_enter_sends_a_user_message() {
        let (mut app, backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "hello");

        let msgs = app.conversation.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0], Message::new(Role::User, "hello"));
        assert_eq!(msgs[1], Message::new(Role::Assistant, ""));
        assert_eq!(app.model_status, ModelStatus::Busy(READY_MODEL.into()));
        assert!(app.input.is_empty(), "input is cleared after submit");
        assert!(!app.should_quit);

        // The task runs on its own; give it a turn, then check what it was sent.
        tokio::task::yield_now().await;
        assert_eq!(
            *backend.chats.lock().unwrap(),
            [vec![Message::new(Role::User, "hello")]],
            "the empty placeholder is not sent"
        );
    }

    #[tokio::test]
    async fn harness_lines_are_never_sent_to_the_model() {
        let (mut app, backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "/help");
        type_line(&mut app, "hello");
        tokio::task::yield_now().await;

        // The `/help` line is in the history handed over, but not in what goes on the wire.
        use loops_core::chat::to_openai_messages;
        let chats = backend.chats.lock().unwrap();
        assert_eq!(chats[0][0].role, Role::Harness);
        assert_eq!(
            to_openai_messages(&chats[0]),
            to_openai_messages(&[Message::new(Role::User, "hello")])
        );
    }

    #[test]
    fn submit_without_a_model_hints_and_keeps_the_text() {
        let mut app = plain_app();
        app.model_status = ModelStatus::NoModel;
        type_line(&mut app, "hello");

        let msgs = app.conversation.messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::Harness);
        assert_eq!(msgs[0].content, "no model loaded; use /model");
        assert_eq!(app.input.lines(), ["hello"]);
        assert_eq!(app.model_status, ModelStatus::NoModel);
    }

    #[tokio::test]
    async fn reply_streams_into_the_placeholder_then_status_is_ready() {
        let script = vec![vec![
            ChatDelta::Reasoning("hm".into()),
            content("hel"),
            content("lo"),
            ChatDelta::Done {
                finish_reason: Some("stop".into()),
            },
        ]];
        let (mut app, _backend, mut rx) = chat_app(script);
        type_line(&mut app, "hi");
        run_to_end(&mut app, &mut rx).await;

        assert_eq!(
            app.conversation.messages(),
            [
                Message::new(Role::User, "hi"),
                Message::new(Role::Assistant, "hello")
            ],
            "the thinking is not kept"
        );
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
        assert!(app.reply_ends.is_empty());

        // An event from the finished turn is stale and changes nothing.
        app.update(AppEvent::Turn {
            turn: 0,
            event: TurnEvent::Agent(text("late")),
        });
        assert_eq!(app.conversation.messages()[1].content, "hello");
    }

    /// The reply waits (and draws as `thinking…`) from the moment the turn
    /// starts, through reasoning, until its first text or tool call.
    #[tokio::test]
    async fn the_reply_waits_until_the_first_text_or_tool_call() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        assert_eq!(app.waiting_reply(), None, "no turn");
        type_line(&mut app, "hi");
        assert_eq!(app.waiting_reply().map(|(i, _)| i), Some(1));
        app.update(agent_event(&app, AgentEvent::Thinking));
        assert_eq!(app.waiting_reply().map(|(i, _)| i), Some(1));
        app.update(agent_event(&app, text("hey")));
        assert_eq!(app.waiting_reply(), None);
        assert_eq!(app.conversation.messages()[1].content, "hey");
    }

    /// A tool result restarts the round's clock: the next request goes out then.
    #[tokio::test]
    async fn a_tool_result_starts_a_new_round() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "hi");
        backdate_round(&mut app, 30);
        assert!(app.waiting_reply().unwrap().1 >= Duration::from_secs(30));

        app.update(agent_event(&app, started("c1", "list_files")));
        assert_eq!(
            app.waiting_reply(),
            None,
            "a tool call is something to show"
        );
        app.update(agent_event(&app, finished("c1")));
        app.update(agent_event(&app, AgentEvent::Thinking));
        let (reply, waited) = app.waiting_reply().expect("round two waits");
        assert_eq!(reply, 2);
        assert!(waited < Duration::from_secs(30));
    }

    #[test]
    fn page_keys_scroll_and_end_follows_the_bottom() {
        let mut app = plain_app();
        app.update(key(KeyCode::PageUp));
        assert_eq!(app.scroll_up, 0, "nothing to scroll in an empty chat");
        for i in 0..6 {
            app.conversation
                .push(Message::new(Role::User, format!("line {i}")));
        }
        // Six two-row messages, counted twice: 24 rows, so 3 PageUps stop at 24.
        for _ in 0..4 {
            app.update(key(KeyCode::PageUp));
        }
        assert_eq!(app.scroll_up, 24, "clamped to the transcript's size");
        app.update(key(KeyCode::PageDown));
        app.update(key(KeyCode::PageDown));
        assert_eq!(app.scroll_up, 4);
        app.update(key(KeyCode::PageUp));
        assert_eq!(app.scroll_up, 14);
        for _ in 0..3 {
            app.update(key(KeyCode::PageDown));
        }
        assert_eq!(app.scroll_up, 0, "no lower than the bottom");

        app.update(key(KeyCode::PageUp));
        app.update(key(KeyCode::End));
        assert_eq!(app.scroll_up, 0);
        assert!(app.input.is_empty(), "the keys do not reach the input box");

        // At the bottom, End is the input box's again: it moves the cursor.
        app.input.insert_str("abc");
        app.input.move_cursor(ratatui_textarea::CursorMove::Head);
        app.update(key(KeyCode::End));
        assert_eq!(app.input.cursor(), (0, 3));
    }

    #[tokio::test]
    async fn sending_a_message_jumps_back_to_the_bottom() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        app.update(key(KeyCode::PageUp));
        type_line(&mut app, "hi");
        assert_eq!(app.scroll_up, 0);
    }

    #[test]
    fn page_keys_are_left_to_the_picker() {
        let mut app = plain_app();
        app.update(AppEvent::Model(ModelEvent::ModelsListed(vec![model(
            "a/one",
            ModelState::NotLoaded,
        )])));
        app.update(key(KeyCode::PageUp));
        assert_eq!(app.scroll_up, 0);
    }

    #[test]
    fn the_input_box_has_a_placeholder() {
        let app = plain_app();
        assert_eq!(app.input.placeholder_text(), PLACEHOLDER);
    }

    #[tokio::test]
    async fn tool_call_then_answer() {
        let script = vec![read_cargo_toml(), vec![content("It is loops-tui.")]];
        let (mut app, backend, mut rx) = chat_app(script);
        type_line(&mut app, "which crate?");

        // Mirrored while running: the call shows before its result exists.
        loop {
            let event = rx.recv().await.unwrap();
            let started = matches!(
                &event,
                AppEvent::Turn {
                    event: TurnEvent::Agent(AgentEvent::ToolStarted { .. }),
                    ..
                }
            );
            app.update(event);
            if started {
                break;
            }
        }
        assert_eq!(
            app.conversation.messages()[1].tool_calls[0].name,
            "read_file"
        );
        assert!(app.tool_outcomes.is_empty());

        run_to_end(&mut app, &mut rx).await;
        let msgs = app.conversation.messages();
        let roles: Vec<Role> = msgs.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            [Role::User, Role::Assistant, Role::Tool, Role::Assistant]
        );
        assert!(msgs[2].content.starts_with("1: [package]"));
        assert_eq!(msgs[3].content, "It is loops-tui.");
        let outcome = &app.tool_outcomes["call_1"];
        assert!(!outcome.is_error);
        assert!(outcome.summary.ends_with(" lines"), "{}", outcome.summary);
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));

        // The second request carried the call and its result.
        assert_eq!(backend.chats.lock().unwrap()[1], msgs[..3]);
    }

    #[tokio::test]
    async fn reply_done_without_text_is_marked_empty() {
        let (mut app, _backend, mut rx) = chat_app(Vec::new());
        type_line(&mut app, "hi");
        run_to_end(&mut app, &mut rx).await;
        assert_eq!(app.reply_ends.get(&1), Some(&ReplyEnd::Empty));
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
    }

    #[tokio::test]
    async fn failure_before_any_text_removes_the_placeholder() {
        let (mut app, _backend, mut rx) = failing_app(Vec::new(), "boom");
        type_line(&mut app, "hi");
        run_to_end(&mut app, &mut rx).await;

        let msgs = app.conversation.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1], Message::new(Role::Harness, "error: boom"));
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
    }

    #[tokio::test]
    async fn failure_after_some_text_keeps_it() {
        let (mut app, _backend, mut rx) = failing_app(vec![vec![content("par")]], "boom");
        type_line(&mut app, "hi");
        run_to_end(&mut app, &mut rx).await;

        let msgs = app.conversation.messages();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1], Message::new(Role::Assistant, "par"));
        assert_eq!(msgs[2].content, "error: boom");
    }

    #[tokio::test]
    async fn esc_cancels_keeps_text_and_drops_late_events() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "hi");
        app.update(agent_event(&app, text("par")));
        let late = agent_event(&app, text("tial"));

        app.update(key(KeyCode::Esc));
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
        assert_eq!(app.reply_ends.get(&1), Some(&ReplyEnd::Cancelled));

        app.update(late);
        assert_eq!(app.conversation.messages()[1].content, "par");

        // Esc with nothing running does nothing.
        app.update(key(KeyCode::Esc));
        assert_eq!(app.conversation.messages().len(), 2);
    }

    /// Only a `bash` call that ran keeps its output for the transcript: a
    /// file tool's result is not shown, nor is a failed or declined command's.
    #[tokio::test]
    async fn only_a_bash_call_that_ran_keeps_its_output() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "hi");
        let finish =
            |id: &str, name: &str, is_error: bool, output: &str| AgentEvent::ToolFinished {
                id: id.into(),
                name: name.into(),
                summary: "s".into(),
                is_error,
                output: output.into(),
            };
        for event in [
            finish("c1", "bash", false, "exit code: 0\nhi"),
            finish("c2", "read_file", false, "1: # loops"),
            finish("c3", "bash", true, "error: command is empty"),
            finish("c4", "bash", true, "declined by the user"),
        ] {
            app.update(agent_event(&app, event));
        }
        assert_eq!(
            app.tool_outputs,
            HashMap::from([("c1".to_owned(), "exit code: 0\nhi".to_owned())])
        );
        assert_eq!(app.tool_outcomes.len(), 4);
    }

    /// Ctrl+O toggles between folded and full command output, and types
    /// nothing into the input box.
    #[test]
    fn ctrl_o_toggles_full_output() {
        let mut app = plain_app();
        let ctrl_o = || AppEvent::Key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(!app.full_output);
        app.update(ctrl_o());
        assert!(app.full_output);
        assert_eq!(app.input.lines(), [""]);
        app.update(ctrl_o());
        assert!(!app.full_output);
    }

    #[tokio::test]
    async fn esc_after_a_tool_round_keeps_text_without_tool_calls() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "hi");
        let started = AgentEvent::ToolStarted {
            id: "c1".into(),
            name: "list_files".into(),
            arguments: "{}".into(),
        };
        let finished = AgentEvent::ToolFinished {
            id: "c1".into(),
            name: "list_files".into(),
            summary: "3 entries".into(),
            is_error: false,
            output: "a\nb\nc".into(),
        };
        for event in [started, finished, AgentEvent::Thinking, text("par")] {
            app.update(agent_event(&app, event));
        }
        // Round two got its own reply after the tool call.
        assert_eq!(app.conversation.messages().len(), 3);

        app.update(key(KeyCode::Esc));
        assert_eq!(
            app.conversation.messages(),
            [
                Message::new(Role::User, "hi"),
                Message::new(Role::Assistant, "par")
            ]
        );
        assert_eq!(app.reply_ends.get(&1), Some(&ReplyEnd::Cancelled));
        assert_eq!(app.waiting_reply(), None);
    }

    /// `/clear` empties the transcript and its per-message side tables, but
    /// keeps the model and the always-allowed tools; the next message starts
    /// a conversation of its own.
    #[tokio::test]
    async fn clear_forgets_the_conversation_but_not_the_model() {
        let (mut app, backend, _rx) = chat_app(vec![vec![ChatDelta::Content("hi".into())]]);
        app.conversation.push(Message::new(Role::User, "first"));
        app.conversation.push(Message::new(Role::Assistant, ""));
        app.reply_ends.insert(1, ReplyEnd::Empty);
        app.tool_outcomes.insert(
            "c1".into(),
            ToolOutcome {
                summary: "declined".into(),
                is_error: true,
            },
        );
        app.tool_previews.insert("c1".into(), "edit a.txt".into());
        app.tool_outputs
            .insert("c2".into(), "exit code: 0\nhi".into());
        app.always_allowed.insert("edit_file".into());

        type_line(&mut app, "/clear");
        assert!(app.conversation.messages().is_empty());
        assert!(app.reply_ends.is_empty());
        assert!(app.tool_outcomes.is_empty());
        assert!(app.tool_previews.is_empty());
        assert!(app.tool_outputs.is_empty());
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
        assert!(app.always_allowed.contains("edit_file"));

        // The next message is sent on its own, without the old history.
        type_line(&mut app, "second");
        tokio::task::yield_now().await;
        let chats = backend.chats.lock().unwrap();
        let users: Vec<&str> = chats
            .last()
            .unwrap()
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(users, ["second"]);
    }

    #[tokio::test]
    async fn clear_is_refused_while_a_turn_runs() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "first");
        app.input.clear();
        type_line(&mut app, "/clear");
        assert_eq!(app.conversation.messages()[2].content, BUSY_TEXT);
        assert!(app.turn.is_some());
    }

    #[tokio::test]
    async fn submit_while_running_is_ignored_and_model_is_blocked() {
        let (mut app, backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "first");
        type_line(&mut app, "second");
        assert_eq!(app.conversation.messages().len(), 2);
        assert_eq!(app.input.lines(), ["second"]);

        app.input.clear();
        type_line(&mut app, "/model");
        assert_eq!(app.conversation.messages()[2].content, BUSY_TEXT);
        type_line(&mut app, "/unload");
        assert_eq!(app.conversation.messages()[3].content, BUSY_TEXT);
        assert!(backend.unloads.lock().unwrap().is_empty());

        // Text still lands in the reply, not in the line pushed after it.
        app.update(agent_event(&app, text("ok")));
        assert_eq!(app.conversation.messages()[1].content, "ok");
        tokio::task::yield_now().await;
        assert_eq!(backend.chats.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn harness_lines_said_during_a_turn_survive_its_end() {
        let (mut app, _backend, mut rx) = chat_app(vec![vec![content("ok")]]);
        type_line(&mut app, "hi");
        type_line(&mut app, "/help");
        run_to_end(&mut app, &mut rx).await;
        let roles: Vec<Role> = app.conversation.messages().iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::User, Role::Assistant, Role::Harness]);
    }

    #[test]
    fn quit_command_sets_should_quit() {
        let mut app = plain_app();
        type_line(&mut app, "/quit");
        assert!(app.should_quit);
        assert!(app.conversation.messages().is_empty());

        let mut app = plain_app();
        type_line(&mut app, "/exit");
        assert!(app.should_quit, "/exit is an alias");
    }

    #[tokio::test]
    async fn unload_command_unloads_the_ready_model() {
        let (mut app, backend, mut rx) = chat_app(Vec::new());
        type_line(&mut app, "/unload");
        assert_eq!(app.model_status, ModelStatus::Unloading(READY_MODEL.into()));

        let unloaded = rx.recv().await.unwrap();
        assert_eq!(*backend.unloads.lock().unwrap(), [READY_MODEL]);
        app.update(unloaded);
        assert_eq!(app.model_status, ModelStatus::NoModel);
        let last = app.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::Harness);
        assert_eq!(last.content, format!("unloaded {READY_MODEL}"));
    }

    #[test]
    fn failed_unload_restores_ready_and_reports() {
        let mut app = plain_app();
        app.model_status = ModelStatus::Unloading(READY_MODEL.into());
        app.update(AppEvent::Model(ModelEvent::UnloadFailed("boom".into())));
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
        assert_eq!(app.conversation.messages()[0].content, "error: boom");
    }

    #[tokio::test]
    async fn unload_without_a_model_says_so_and_spawns_nothing() {
        let (mut app, backend, mut rx) = test_app(Vec::new());
        app.model_status = ModelStatus::NoModel;
        type_line(&mut app, "/unload");
        assert_eq!(app.conversation.messages()[0].content, "no model loaded");
        assert_eq!(app.model_status, ModelStatus::NoModel);

        tokio::task::yield_now().await;
        assert!(rx.try_recv().is_err(), "no task reported back");
        assert!(backend.unloads.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unloading_refuses_model_unload_and_messages() {
        let (mut app, backend, mut rx) = test_app(Vec::new());
        app.model_status = ModelStatus::Unloading(READY_MODEL.into());
        type_line(&mut app, "/model");
        type_line(&mut app, "/unload");
        type_line(&mut app, "hello");

        let said: Vec<&str> = app
            .conversation
            .messages()
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(said, [UNLOADING_TEXT; 3]);
        assert_eq!(app.input.lines(), ["hello"], "the message stays in the box");
        tokio::task::yield_now().await;
        assert!(rx.try_recv().is_err(), "nothing was spawned");
        assert!(backend.chats.lock().unwrap().is_empty());
    }

    #[test]
    fn ctrl_c_quits() {
        let mut app = plain_app();
        app.update(AppEvent::Key(KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        )));
        assert!(app.should_quit);
    }

    #[test]
    fn help_and_unknown_commands_reply_as_harness() {
        let mut app = plain_app();
        type_line(&mut app, "/help");
        type_line(&mut app, "/foo bar");

        let msgs = app.conversation.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::Harness);
        assert!(msgs[0].content.contains("/quit (or /exit)"));
        assert!(msgs[0].content.contains("/model"));
        assert!(msgs[0].content.contains("/unload"));
        assert!(
            msgs[0]
                .content
                .contains("/context          tokens used and left in the model's window")
        );
        assert!(
            msgs[0]
                .content
                .contains("/compact          summarise the chat so far to free the window")
        );
        assert!(msgs[0].content.contains(
            "/think [level]    how much the model thinks: off, low, medium, high, default"
        ));
        assert!(
            msgs[0]
                .content
                .contains("/max-tokens [n]   cap each reply at n tokens (off for no cap)")
        );
        assert!(msgs[0].content.contains("read_file, list_files"));
        assert!(
            msgs[0]
                .content
                .contains("edit_file, write_file, bash   change things; ask first")
        );
        assert!(
            msgs[0]
                .content
                .contains("\"don't ask again\" for bash is per exact command")
        );
        assert!(
            msgs[0]
                .content
                .contains("answer a permission prompt (1/2/3 also work)")
        );
        assert_eq!(msgs[1].content, "unknown command: /foo");
    }

    #[tokio::test]
    async fn alt_enter_inserts_newline_instead_of_submitting() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        app.update(key(KeyCode::Char('a')));
        app.update(AppEvent::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::ALT,
        )));
        app.update(key(KeyCode::Char('b')));
        assert!(app.conversation.messages().is_empty());

        app.update(key(KeyCode::Enter));
        assert_eq!(app.conversation.messages()[0].content, "a\nb");
    }

    #[test]
    fn key_release_is_ignored() {
        let mut app = plain_app();
        let mut release = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        release.kind = KeyEventKind::Release;
        app.update(AppEvent::Key(release));
        assert!(app.input.is_empty());
    }

    #[test]
    fn startup_events_set_the_status() {
        let mut app = plain_app();
        assert_eq!(app.model_status, ModelStatus::Starting);

        app.update(AppEvent::Model(ModelEvent::ServerReady(None)));
        assert_eq!(app.model_status, ModelStatus::NoModel);

        let loaded = model("qwen/qwen3.5-9b", ModelState::Loaded);
        app.update(AppEvent::Model(ModelEvent::ServerReady(Some(loaded))));
        assert_eq!(
            app.model_status,
            ModelStatus::Ready("qwen/qwen3.5-9b".into())
        );
    }

    #[test]
    fn failures_show_short_status_and_full_transcript_line() {
        let mut app = plain_app();
        let err = format!("{}\nsecond line", "x".repeat(100));
        app.update(AppEvent::Model(ModelEvent::LoadFailed(err.clone())));

        let ModelStatus::Error(short) = &app.model_status else {
            panic!("expected Error, got {:?}", app.model_status);
        };
        assert_eq!(short.chars().count(), STATUS_ERROR_CHARS);
        assert!(short.ends_with('…'));
        assert_eq!(
            app.conversation.messages()[0].content,
            format!("error: {err}")
        );
    }

    #[test]
    fn tick_advances_the_spinner() {
        let mut app = plain_app();
        app.update(AppEvent::Tick);
        app.update(AppEvent::Tick);
        assert_eq!(app.spinner, 2);
    }

    /// Confirming the model that is already loaded reports it instead of loading a second copy.
    #[tokio::test]
    async fn confirming_a_loaded_model_does_not_reload_it() {
        let models = vec![
            model("qwen/qwen3.5-9b", ModelState::NotLoaded),
            model("google/gemma-4-e4b", ModelState::Loaded),
        ];
        let (mut app, backend, mut rx) = test_app(models);
        app.model_status = ModelStatus::Ready("google/gemma-4-e4b".into());

        type_line(&mut app, "/model");
        app.update(rx.recv().await.unwrap());
        // The picker opens on the loaded model, so Enter + y is the reflex case.
        assert!(matches!(app.mode, Mode::Picker { selected: 1, .. }));
        app.update(key(KeyCode::Enter));
        app.update(key(KeyCode::Char('y')));

        assert_eq!(app.mode, Mode::Normal);
        assert!(
            backend.loads.lock().unwrap().is_empty(),
            "no load requested"
        );
        assert_eq!(
            app.model_status,
            ModelStatus::Ready("google/gemma-4-e4b".into())
        );
        let last = app.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::Harness);
        assert_eq!(last.content, "already loaded: google/gemma-4-e4b");
        assert_eq!(app.context_length, Some(8192), "its loaded window");
    }

    /// The whole `/model` flow, with the real channel standing in for the main loop.
    #[tokio::test]
    async fn model_command_picks_confirms_and_loads() {
        let models = vec![
            model("qwen/qwen3.5-9b", ModelState::NotLoaded),
            model("google/gemma-4-e4b", ModelState::NotLoaded),
            model("google/gemma-4-12b", ModelState::NotLoaded),
        ];
        let (mut app, backend, mut rx) = test_app(models.clone());

        // `/model` spawns a list request; its answer comes back on the channel.
        type_line(&mut app, "/model");
        assert_eq!(app.mode, Mode::Normal, "picker waits for the list");
        let listed = rx.recv().await.unwrap();
        assert!(matches!(&listed, AppEvent::Model(ModelEvent::ModelsListed(m)) if *m == models));
        app.update(listed);
        assert_eq!(
            app.mode,
            Mode::Picker {
                models: models.clone(),
                selected: 0
            }
        );

        // While the picker is open, typing does not reach the input box.
        app.update(key(KeyCode::Char('x')));
        assert!(app.input.is_empty());

        // Down moves, up moves back, j moves, and the bottom is a wall.
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Up));
        app.update(key(KeyCode::Char('j')));
        assert!(matches!(app.mode, Mode::Picker { selected: 1, .. }));

        // Enter asks; n returns to the list on the same row.
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.mode, Mode::Confirm { selected: 1, .. }));
        app.update(key(KeyCode::Char('q')));
        assert!(
            matches!(app.mode, Mode::Confirm { .. }),
            "other keys ignored"
        );
        app.update(key(KeyCode::Char('n')));
        assert!(matches!(app.mode, Mode::Picker { selected: 1, .. }));

        // Esc closes the picker without loading anything.
        app.update(key(KeyCode::Esc));
        assert_eq!(app.mode, Mode::Normal);
        assert!(backend.loads.lock().unwrap().is_empty());

        // Again, this time saying yes.
        type_line(&mut app, "/model");
        app.update(rx.recv().await.unwrap());
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Enter));
        app.update(key(KeyCode::Char('y')));
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(
            app.model_status,
            ModelStatus::Loading("google/gemma-4-e4b".into())
        );

        let loaded = rx.recv().await.unwrap();
        assert_eq!(*backend.loads.lock().unwrap(), ["google/gemma-4-e4b"]);
        app.update(loaded);
        assert_eq!(
            app.model_status,
            ModelStatus::Ready("google/gemma-4-e4b".into())
        );
        let last = app.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::Harness);
        assert_eq!(last.content, "loaded google/gemma-4-e4b");
    }

    #[test]
    fn picker_starts_on_the_loaded_model() {
        let mut app = plain_app();
        let models = vec![
            model("a/one", ModelState::NotLoaded),
            model("b/two", ModelState::Loaded),
        ];
        app.update(AppEvent::Model(ModelEvent::ModelsListed(models)));
        assert!(matches!(app.mode, Mode::Picker { selected: 1, .. }));
    }

    #[test]
    fn empty_model_list_says_so_and_stays_normal() {
        let mut app = plain_app();
        app.update(AppEvent::Model(ModelEvent::ModelsListed(Vec::new())));
        assert_eq!(app.mode, Mode::Normal);
        assert!(
            app.conversation.messages()[0]
                .content
                .contains("no chat models")
        );
    }

    /// Swaps the running turn's decision channel for one the test can read.
    fn watch_decisions(app: &mut App) -> mpsc::Receiver<Decision> {
        let (tx, rx) = mpsc::channel(4);
        app.turn.as_mut().expect("a turn is running").decisions = tx;
        rx
    }

    /// The agent starting call `id` of tool `name`.
    fn started(id: &str, name: &str) -> AgentEvent {
        AgentEvent::ToolStarted {
            id: id.into(),
            name: name.into(),
            arguments: "{}".into(),
        }
    }

    /// Call `id` finishing without error.
    fn finished(id: &str) -> AgentEvent {
        AgentEvent::ToolFinished {
            id: id.into(),
            name: "list_files".into(),
            summary: "3 entries".into(),
            is_error: false,
            output: "a\nb\nc".into(),
        }
    }

    /// The agent asking to run call `id` of tool `name`, whose scope is
    /// the whole tool (as for the file tools).
    fn asks(id: &str, name: &str) -> AgentEvent {
        asks_scoped(id, name, name)
    }

    /// The agent asking to run call `id` of tool `name` with `scope`.
    fn asks_scoped(id: &str, name: &str, scope: &str) -> AgentEvent {
        AgentEvent::PermissionRequested {
            id: id.into(),
            name: name.into(),
            arguments: "{}".into(),
            preview: format!("preview of {id}"),
            scope: scope.into(),
        }
    }

    /// The prompt for call `id` of tool `name`, with row `selected` highlighted.
    fn prompt_at(id: &str, name: &str, selected: usize) -> Mode {
        Mode::Permission {
            id: id.into(),
            name: name.into(),
            scope: name.into(),
            selected,
        }
    }

    /// The prompt for call `id` of tool `name`, as it opens (top row highlighted).
    fn prompt(id: &str, name: &str) -> Mode {
        prompt_at(id, name, 0)
    }

    #[test]
    fn permission_options_are_listed_in_order() {
        let labels = PermissionOption::ALL.map(|option| option.label("edit_file"));
        assert_eq!(
            labels,
            [
                "Yes",
                "Yes, and don't ask again for edit_file this run",
                "No"
            ]
        );
        assert_eq!(
            PermissionOption::AlwaysAllow.label("bash: cargo test"),
            "Yes, and don't ask again for bash: cargo test this run"
        );
    }

    /// "Don't ask again" for a command covers that exact command only.
    #[tokio::test]
    async fn dont_ask_again_for_a_command_covers_only_that_command() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "test it");
        let mut decisions = watch_decisions(&mut app);
        app.update(agent_event(&app, asks_scoped("c1", "bash", "bash: echo a")));
        assert_eq!(
            app.mode,
            Mode::Permission {
                id: "c1".into(),
                name: "bash".into(),
                scope: "bash: echo a".into(),
                selected: 0,
            }
        );
        app.update(key(KeyCode::Char('2')));
        assert_eq!(decisions.try_recv(), Ok(Decision::Allow));
        assert!(app.always_allowed.contains("bash: echo a"));
        assert!(!app.always_allowed.contains("bash"), "not the whole tool");

        // Another command still asks.
        app.update(agent_event(&app, asks_scoped("c2", "bash", "bash: echo b")));
        assert!(matches!(&app.mode, Mode::Permission { scope, .. } if scope == "bash: echo b"));
        assert!(decisions.try_recv().is_err());
        app.update(key(KeyCode::Char('1')));
        assert_eq!(decisions.try_recv(), Ok(Decision::Allow));
        assert!(!app.always_allowed.contains("bash: echo b"), "a plain yes");

        // The same command again: answered at once.
        app.update(agent_event(&app, asks_scoped("c3", "bash", "bash: echo a")));
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(decisions.try_recv(), Ok(Decision::Allow));
    }

    #[test]
    fn say_events_become_harness_lines() {
        let mut app = plain_app();
        app.update(AppEvent::Say("bash not found; the bash tool is off".into()));
        assert_eq!(
            app.conversation.messages(),
            [Message::new(
                Role::Harness,
                "bash not found; the bash tool is off"
            )]
        );
    }

    #[tokio::test]
    async fn a_permission_request_opens_the_prompt_and_enter_allows() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "fix it");
        let mut decisions = watch_decisions(&mut app);
        app.update(agent_event(&app, started("c1", "edit_file")));
        app.update(agent_event(&app, asks("c1", "edit_file")));
        assert_eq!(app.mode, prompt("c1", "edit_file"));
        assert_eq!(app.tool_previews["c1"], "preview of c1");
        assert_eq!(app.model_status, ModelStatus::Busy(READY_MODEL.into()));
        assert!(decisions.try_recv().is_err(), "nothing answered yet");

        app.update(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(decisions.try_recv(), Ok(Decision::Allow));
        assert!(app.always_allowed.is_empty());
        assert_eq!(
            app.tool_previews["c1"], "preview of c1",
            "kept for the transcript"
        );
    }

    #[tokio::test]
    async fn up_at_the_top_and_down_at_the_bottom_stay_put() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "fix it");
        let mut decisions = watch_decisions(&mut app);
        app.update(agent_event(&app, asks("c1", "edit_file")));

        app.update(key(KeyCode::Up));
        assert_eq!(app.mode, prompt_at("c1", "edit_file", 0));
        app.update(key(KeyCode::Down));
        assert_eq!(app.mode, prompt_at("c1", "edit_file", 1));
        for _ in 0..3 {
            app.update(key(KeyCode::Down));
        }
        assert_eq!(app.mode, prompt_at("c1", "edit_file", 2), "no wrap");
        app.update(key(KeyCode::Up));
        assert_eq!(app.mode, prompt_at("c1", "edit_file", 1));
        assert!(decisions.try_recv().is_err(), "moving answers nothing");
    }

    #[tokio::test]
    async fn enter_on_no_denies() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "fix it");
        let mut decisions = watch_decisions(&mut app);
        app.update(agent_event(&app, asks("c1", "write_file")));
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(decisions.try_recv(), Ok(Decision::Deny));
        assert!(app.always_allowed.is_empty());
        assert!(app.turn.is_some(), "the turn goes on");
    }

    #[tokio::test]
    async fn digits_answer_directly() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "fix it");
        let mut decisions = watch_decisions(&mut app);

        app.update(agent_event(&app, asks("c1", "write_file")));
        app.update(key(KeyCode::Char('3')));
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(decisions.try_recv(), Ok(Decision::Deny));

        // The digit wins over the highlighted row.
        app.update(agent_event(&app, asks("c2", "write_file")));
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Char('1')));
        assert_eq!(decisions.try_recv(), Ok(Decision::Allow));
        assert!(app.always_allowed.is_empty());

        app.update(agent_event(&app, asks("c3", "write_file")));
        app.update(key(KeyCode::Char('2')));
        assert_eq!(decisions.try_recv(), Ok(Decision::Allow));
        assert!(app.always_allowed.contains("write_file"));
    }

    #[tokio::test]
    async fn enter_on_dont_ask_again_always_allows_that_tool_but_not_others() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "fix it");
        let mut decisions = watch_decisions(&mut app);
        app.update(agent_event(&app, asks("c1", "edit_file")));
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(decisions.try_recv(), Ok(Decision::Allow));
        assert!(app.always_allowed.contains("edit_file"));

        // The same tool again: answered at once, no prompt.
        app.update(agent_event(&app, asks("c2", "edit_file")));
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(decisions.try_recv(), Ok(Decision::Allow));
        assert_eq!(app.tool_previews["c2"], "preview of c2");

        // Another tool still asks.
        app.update(agent_event(&app, asks("c3", "write_file")));
        assert_eq!(app.mode, prompt("c3", "write_file"));
        assert!(decisions.try_recv().is_err());
    }

    #[tokio::test]
    async fn esc_in_the_prompt_cancels_the_turn() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "fix it");
        let mut decisions = watch_decisions(&mut app);
        app.update(agent_event(&app, started("c1", "edit_file")));
        app.update(agent_event(&app, asks("c1", "edit_file")));
        app.update(key(KeyCode::Esc));

        assert_eq!(app.mode, Mode::Normal);
        assert!(app.turn.is_none());
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
        assert!(decisions.try_recv().is_err(), "no answer, the turn is gone");
        assert!(app.conversation.messages()[1].tool_calls.is_empty());
    }

    #[tokio::test]
    async fn the_prompt_closes_when_the_turn_ends() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "fix it");
        app.update(agent_event(&app, asks("c1", "edit_file")));
        app.update(agent_event(&app, AgentEvent::TurnFailed("boom".into())));
        let turn = app.turn.as_ref().unwrap().id;
        let mut left = Conversation::new();
        left.push(Message::new(Role::User, "fix it"));
        app.update(AppEvent::Turn {
            turn,
            event: TurnEvent::Finished(left),
        });
        assert_eq!(app.mode, Mode::Normal);
        assert!(app.turn.is_none());
    }

    #[tokio::test]
    async fn a_request_from_a_stale_turn_is_ignored() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "first");
        let old = app.turn.as_ref().unwrap().id;
        app.update(key(KeyCode::Esc));
        let stale = |event| AppEvent::Turn {
            turn: old,
            event: TurnEvent::Agent(event),
        };

        // With no turn running, and with a newer one running.
        app.update(stale(asks("c1", "edit_file")));
        assert_eq!(app.mode, Mode::Normal);
        type_line(&mut app, "second");
        assert_ne!(app.turn.as_ref().unwrap().id, old);
        app.update(stale(asks("c2", "edit_file")));
        assert_eq!(app.mode, Mode::Normal);
        assert!(app.tool_previews.is_empty());
    }

    #[tokio::test]
    async fn typing_in_the_prompt_does_not_reach_the_input_box_or_answer() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "fix it");
        let mut decisions = watch_decisions(&mut app);
        app.update(agent_event(&app, asks("c1", "edit_file")));
        // The old letter shortcuts are gone, and digits past the list do nothing.
        for c in ['y', 'Y', 'n', 'a', 'x', 'q', '0', '4'] {
            app.update(key(KeyCode::Char(c)));
        }
        assert!(app.input.is_empty());
        assert_eq!(app.mode, prompt("c1", "edit_file"));
        assert!(decisions.try_recv().is_err());
    }

    /// The whole flow with the real `edit_file`: asked, allowed, file changed.
    #[tokio::test]
    async fn an_allowed_edit_changes_the_file() {
        let dir = TempDir::new("allowed-edit");
        dir.write("notes.txt", "helo\n");
        let edit = ChatDelta::ToolCall {
            index: 0,
            id: Some("call_1".into()),
            name: Some("edit_file".into()),
            arguments: r#"{"path":"notes.txt","old":"helo","new":"hello"}"#.into(),
        };
        let (mut app, _backend, mut rx) =
            editing_app(vec![vec![edit], vec![content("fixed")]], &dir);
        type_line(&mut app, "fix the typo");
        while !matches!(app.mode, Mode::Permission { .. }) {
            app.update(rx.recv().await.unwrap());
        }
        assert_eq!(
            app.tool_previews["call_1"],
            "edit notes.txt at line 1\n- helo\n+ hello"
        );
        assert_eq!(
            dir.read("notes.txt"),
            "helo\n",
            "nothing changed before the yes"
        );

        app.update(key(KeyCode::Enter));
        run_to_end(&mut app, &mut rx).await;
        assert_eq!(dir.read("notes.txt"), "hello\n");
        let outcome = &app.tool_outcomes["call_1"];
        assert_eq!(outcome.summary, "replaced 1 line with 1 at line 1");
        assert!(!outcome.is_error);
        assert_eq!(app.conversation.messages().last().unwrap().content, "fixed");
    }

    #[test]
    fn thousands_puts_a_comma_every_three_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(8192), "8,192");
        assert_eq!(thousands(131_072), "131,072");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn context_length_follows_the_loaded_model() {
        let mut app = plain_app();
        let gemma = model(READY_MODEL, ModelState::Loaded);
        app.update(AppEvent::Model(ModelEvent::ServerReady(Some(gemma))));
        assert_eq!(
            app.context_length,
            Some(8192),
            "the loaded length, not the max"
        );

        app.usage = Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 2,
        });
        let mut qwen = model("qwen/qwen3.5-9b", ModelState::Loaded);
        qwen.loaded_context_length = Some(32_768);
        app.update(AppEvent::Model(ModelEvent::Loaded(qwen)));
        assert_eq!(app.context_length, Some(32_768));
        assert!(app.usage.is_some(), "same conversation, so usage stays");

        app.update(AppEvent::Model(ModelEvent::Unloaded(
            "qwen/qwen3.5-9b".into(),
        )));
        assert_eq!(app.context_length, None);
    }

    #[tokio::test]
    async fn usage_is_stored_from_the_agent_and_cleared_by_clear() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "hi");
        for (prompt_tokens, completion_tokens) in [(100, 10), (150, 2)] {
            let usage = AgentEvent::Usage {
                prompt_tokens,
                completion_tokens,
            };
            app.update(agent_event(&app, usage));
        }
        assert_eq!(
            app.usage,
            Some(Usage {
                prompt_tokens: 150,
                completion_tokens: 2,
            }),
            "the last request wins"
        );
        assert_eq!(app.conversation.messages()[1].content, "", "nothing drawn");

        app.update(key(KeyCode::Esc));
        type_line(&mut app, "/clear");
        assert_eq!(app.usage, None);
    }

    /// What `/context` says, up to its last line, which names the session
    /// (see `context_names_the_session`).
    fn context_says(app: &mut App) -> String {
        type_line(app, "/context");
        let last = app.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::Harness);
        let (said, session) = last.content.rsplit_once('\n').unwrap();
        assert!(session.starts_with("session: "), "{session}");
        said.to_owned()
    }

    /// An app with [`READY_MODEL`] ready, `window` tokens of context, and
    /// `usage` from the last request.
    fn context_app(window: Option<u64>, usage: Option<(u64, u64)>) -> App {
        let mut app = plain_app();
        app.model_status = ModelStatus::Ready(READY_MODEL.into());
        app.context_length = window;
        app.usage = usage.map(|(prompt_tokens, completion_tokens)| Usage {
            prompt_tokens,
            completion_tokens,
        });
        app
    }

    #[test]
    fn context_without_a_model_says_so() {
        let mut app = plain_app();
        app.model_status = ModelStatus::NoModel;
        assert_eq!(
            context_says(&mut app),
            "context: no model loaded\nthinking: default, reply cap: none\n\
             instructions: none (no AGENTS.md in the working directory)"
        );
    }

    #[test]
    fn context_before_anything_is_sent_gives_the_window() {
        let mut app = context_app(Some(8192), None);
        assert_eq!(
            context_says(&mut app),
            "context: nothing sent yet; the window is 8,192 tokens (google/gemma-4-e4b)\n\
             thinking: default, reply cap: none\n\
             instructions: none (no AGENTS.md in the working directory)"
        );
    }

    #[test]
    fn context_gives_used_and_left() {
        let mut app = context_app(Some(8192), Some((1401, 131)));
        assert_eq!(
            context_says(&mut app),
            "context: 1,532 of 8,192 tokens used (19%), 6,660 left\n\
             last request: 1,401 prompt + 131 reply (google/gemma-4-e4b)\n\
             thinking: default, reply cap: none\n\
             instructions: none (no AGENTS.md in the working directory)"
        );
    }

    #[test]
    fn context_over_the_window_says_by_how_much() {
        let mut app = context_app(Some(8192), Some((8000, 1000)));
        assert_eq!(
            context_says(&mut app),
            "context: 9,000 of 8,192 tokens used (over the window by 808)\n\
             last request: 8,000 prompt + 1,000 reply (google/gemma-4-e4b)\n\
             thinking: default, reply cap: none\n\
             instructions: none (no AGENTS.md in the working directory)"
        );
    }

    #[test]
    fn context_with_an_unknown_window_gives_what_was_used() {
        let mut app = context_app(None, Some((1401, 131)));
        app.thinking = Thinking::Off;
        app.max_output_tokens = Some(2048);
        assert_eq!(
            context_says(&mut app),
            "context: 1,532 tokens used of an unknown window\n\
             last request: 1,401 prompt + 131 reply (google/gemma-4-e4b)\n\
             thinking: off, reply cap: 2,048 tokens\n\
             instructions: none (no AGENTS.md in the working directory)"
        );
    }

    #[test]
    fn context_names_the_instructions_sent() {
        let mut app = context_app(Some(8192), None);
        app.instructions = Some("AGENTS.md, about 420 tokens".into());
        assert_eq!(
            context_says(&mut app),
            "context: nothing sent yet; the window is 8,192 tokens (google/gemma-4-e4b)\n\
             thinking: default, reply cap: none\n\
             instructions: AGENTS.md, about 420 tokens"
        );
    }

    #[test]
    fn instructions_are_described_by_size_and_any_cut() {
        let mut instructions = Instructions {
            text: "x".repeat(1_680),
            total_chars: 1_680,
            shown_chars: 1_680,
            truncated: false,
        };
        assert_eq!(describe_instructions(&instructions), "about 420 tokens");
        instructions.text = "x".repeat(8_000);
        instructions.total_chars = 12_345;
        instructions.shown_chars = 7_950;
        instructions.truncated = true;
        assert_eq!(
            describe_instructions(&instructions),
            "cut to the first 7,950 of 12,345 characters (about 2,000 tokens)"
        );
    }

    #[tokio::test]
    async fn context_works_while_a_turn_runs() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        app.context_length = Some(8192);
        type_line(&mut app, "hi");
        let said = context_says(&mut app);
        assert_eq!(
            said,
            "context: nothing sent yet; the window is 8,192 tokens (google/gemma-4-e4b)\n\
             thinking: default, reply cap: none\n\
             instructions: none (no AGENTS.md in the working directory)"
        );
        assert!(app.turn.is_some());
    }

    /// What the app said last.
    fn last_said(app: &App) -> &str {
        let last = app.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::Harness);
        &last.content
    }

    #[test]
    fn think_says_and_sets_the_level() {
        let mut app = plain_app();
        type_line(&mut app, "/think");
        assert_eq!(
            last_said(&app),
            "thinking: default (off, low, medium, high, or default)"
        );
        type_line(&mut app, "/think OFF");
        assert_eq!(app.thinking, Thinking::Off);
        assert_eq!(last_said(&app), "thinking: off");
        type_line(&mut app, "/think lots");
        assert_eq!(last_said(&app), "unknown thinking level: lots");
        assert_eq!(app.thinking, Thinking::Off, "unchanged");
        type_line(&mut app, "/think");
        assert_eq!(
            last_said(&app),
            "thinking: off (off, low, medium, high, or default)"
        );
    }

    #[test]
    fn max_tokens_says_sets_and_clears_the_cap() {
        let mut app = plain_app();
        type_line(&mut app, "/max-tokens");
        assert_eq!(last_said(&app), "reply cap: none");
        type_line(&mut app, "/max-tokens 4096");
        assert_eq!(app.max_output_tokens, Some(4096));
        assert_eq!(last_said(&app), "reply cap: 4,096 tokens");
        for bad in ["0", "-5", "lots"] {
            type_line(&mut app, &format!("/max-tokens {bad}"));
            assert_eq!(
                last_said(&app),
                format!("not a reply cap: {bad} (give a whole number above 0, or off)")
            );
        }
        assert_eq!(app.max_output_tokens, Some(4096), "unchanged");
        type_line(&mut app, "/max-tokens off");
        assert_eq!(app.max_output_tokens, None);
        assert_eq!(last_said(&app), "reply cap: none");
    }

    /// Both settings work while a turn runs and go out with the next request,
    /// together with the loaded model's window.
    #[tokio::test]
    async fn settings_change_during_a_turn_and_reach_the_next_request() {
        let (mut app, backend, mut rx) = chat_app(vec![vec![content("one")], vec![content("two")]]);
        app.context_length = Some(8192);
        type_line(&mut app, "first");
        type_line(&mut app, "/think low");
        type_line(&mut app, "/max-tokens 100");
        assert_eq!(app.thinking, Thinking::Low, "set while the turn runs");
        run_to_end(&mut app, &mut rx).await;
        type_line(&mut app, "second");
        run_to_end(&mut app, &mut rx).await;

        let options = backend.options.lock().unwrap();
        assert_eq!(options.len(), 2);
        assert_eq!(options[0].thinking, Thinking::Default);
        assert!(
            options[0].max_tokens.is_some_and(|n| n > 100 && n < 8192),
            "the room left in the window: {:?}",
            options[0].max_tokens
        );
        assert_eq!(
            options[1],
            RequestOptions {
                max_tokens: Some(100),
                thinking: Thinking::Low,
            }
        );
    }

    #[tokio::test]
    async fn a_cut_off_reply_is_mirrored_as_a_harness_line() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "hi");
        app.update(agent_event(&app, text("Hello, ")));
        let cut_off = AgentEvent::ReplyCutOff { attempt: 1, max: 2 };
        app.update(agent_event(&app, cut_off));
        app.update(agent_event(&app, text("world.")));

        let msgs = app.conversation.messages();
        assert_eq!(msgs.len(), 3);
        assert_eq!(
            msgs[1],
            Message::new(Role::Assistant, "Hello, world."),
            "the continuation lands in the same reply"
        );
        assert_eq!(
            msgs[2],
            Message::new(
                Role::Harness,
                "reply cut off at the output limit; asking the model to continue (1/2)"
            )
        );
    }

    /// Applies channel events until `/compact` ends.
    async fn compact_to_end(app: &mut App, rx: &mut mpsc::Receiver<AppEvent>) {
        while app.compaction.is_some() {
            app.update(rx.recv().await.unwrap());
        }
    }

    /// A chat of one exchange the server counted as 6,200 tokens.
    fn chatted(app: &mut App) {
        app.conversation.push(Message::new(Role::User, "hi"));
        app.conversation
            .push(Message::new(Role::Assistant, "hello"));
        app.conversation.set_token_mark(2, 6200);
    }

    #[tokio::test]
    async fn compact_replaces_the_conversation_with_the_summary() {
        let (mut app, backend, mut rx) = chat_app(vec![vec![content("Greeted.")]]);
        chatted(&mut app);
        app.reply_ends.insert(1, ReplyEnd::Empty);
        app.tool_previews.insert("c1".into(), "edit a.txt".into());
        app.tool_outputs
            .insert("c2".into(), "exit code: 0\nhi".into());
        app.usage = Some(Usage {
            prompt_tokens: 6000,
            completion_tokens: 200,
        });
        type_line(&mut app, "/compact");
        assert!(app.compacting());
        assert_eq!(app.model_status, ModelStatus::Busy(READY_MODEL.into()));
        // Said while it runs, so kept.
        type_line(&mut app, "/think");
        compact_to_end(&mut app, &mut rx).await;

        let mut compacted = Conversation::new();
        compacted.push(Message::new(
            Role::User,
            "Summary of the conversation so far, written to continue after \
             earlier messages were removed:\n\nGreeted.",
        ));
        compacted.push(Message::new(
            Role::Assistant,
            "Understood. Continuing from that summary.",
        ));
        let after = compacted.estimated_tokens(0);
        let msgs = app.conversation.messages();
        assert_eq!(msgs[..2], *compacted.messages());
        assert_eq!(
            msgs[2].content,
            "thinking: default (off, low, medium, high, or default)"
        );
        assert_eq!(
            msgs[3],
            Message::new(
                Role::Harness,
                format!("compacted: about 6,200 → {after} tokens")
            )
        );
        assert_eq!(msgs.len(), 4);
        assert!(app.reply_ends.is_empty());
        assert!(app.tool_previews.is_empty());
        assert!(app.tool_outputs.is_empty());
        assert_eq!(app.usage, None);
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
        assert!(!app.compacting());
        assert_eq!(backend.options.lock().unwrap()[0].thinking, Thinking::Off);

        // The next message goes out with the summary, not the old chat.
        type_line(&mut app, "go on");
        tokio::task::yield_now().await;
        let chats = backend.chats.lock().unwrap();
        assert_eq!(chats[1][..2], *compacted.messages());
    }

    #[tokio::test]
    async fn compact_is_refused_while_busy() {
        let (mut app, backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "first");
        type_line(&mut app, "/compact");
        assert_eq!(last_said(&app), BUSY_TEXT);
        assert!(!app.compacting());

        app.update(key(KeyCode::Esc));
        type_line(&mut app, "/compact");
        assert!(app.compacting());
        for command in ["/compact", "/clear", "/new", "/sessions", "/model"] {
            type_line(&mut app, command);
            assert_eq!(last_said(&app), BUSY_TEXT);
        }
        type_line(&mut app, "second");
        assert_eq!(
            app.input.lines(),
            ["second"],
            "the message stays in the box"
        );
        tokio::task::yield_now().await;
        // The turn was cancelled before it sent anything.
        assert_eq!(backend.chats.lock().unwrap().len(), 1, "the summary only");
    }

    #[tokio::test]
    async fn compact_without_a_ready_model_says_so() {
        let (mut app, backend, mut rx) = test_app(Vec::new());
        app.model_status = ModelStatus::NoModel;
        type_line(&mut app, "/compact");
        assert_eq!(last_said(&app), "no model loaded; use /model");
        app.model_status = ModelStatus::Loading(READY_MODEL.into());
        type_line(&mut app, "/compact");
        assert_eq!(last_said(&app), format!("{READY_MODEL} is still loading"));
        assert!(!app.compacting());
        tokio::task::yield_now().await;
        assert!(rx.try_recv().is_err(), "nothing was spawned");
        assert!(backend.chats.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn compact_errors_are_said_and_change_nothing() {
        let (mut app, _backend, mut rx) = chat_app(Vec::new());
        type_line(&mut app, "/compact");
        compact_to_end(&mut app, &mut rx).await;
        assert_eq!(last_said(&app), "error: nothing to compact");
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));

        let (mut app, _backend, mut rx) = failing_app(Vec::new(), "boom");
        chatted(&mut app);
        let old = app.conversation.messages().to_vec();
        type_line(&mut app, "/compact");
        compact_to_end(&mut app, &mut rx).await;
        assert_eq!(app.conversation.messages()[..2], old[..]);
        assert_eq!(last_said(&app), "error: boom");
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
    }

    #[tokio::test]
    async fn esc_cancels_compaction_and_keeps_the_conversation() {
        let (mut app, _backend, _rx) = chat_app(vec![vec![content("Greeted.")]]);
        chatted(&mut app);
        let old = app.conversation.messages().to_vec();
        type_line(&mut app, "/compact");
        app.update(key(KeyCode::Esc));

        assert!(!app.compacting());
        assert_eq!(app.model_status, ModelStatus::Ready(READY_MODEL.into()));
        assert_eq!(app.conversation.messages()[..2], old[..]);
        assert_eq!(last_said(&app), "compaction cancelled");
        assert_eq!(app.conversation.token_mark(), Some((2, 6200)));
    }

    #[tokio::test]
    async fn a_result_from_a_stale_compaction_is_ignored() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        chatted(&mut app);
        type_line(&mut app, "/compact");
        let old = app.compaction.as_ref().unwrap().id;
        app.update(key(KeyCode::Esc));
        let before = app.conversation.messages().to_vec();
        let stale = || {
            let report = CompactReport {
                before: 6200,
                after: 10,
                summary: "late".into(),
            };
            AppEvent::Compacted {
                turn: old,
                result: Ok((Conversation::new(), report)),
            }
        };

        // With nothing running, and with a newer compaction running.
        app.update(stale());
        assert_eq!(app.conversation.messages(), before);
        type_line(&mut app, "/compact");
        assert_ne!(app.compaction.as_ref().unwrap().id, old);
        app.update(stale());
        assert_eq!(app.conversation.messages(), before);
        assert!(app.compacting(), "the newer one goes on");
    }

    // --- sessions ---------------------------------------------------------

    /// Saves `app`'s sessions in `dir`.
    fn save_in(app: &mut App, dir: &TempDir) {
        app.sessions = Some(SessionStore::in_dir(dir.0.clone()));
    }

    /// The sessions saved in `dir`, newest first.
    fn saved(dir: &TempDir) -> Vec<SessionSummary> {
        SessionStore::in_dir(dir.0.clone()).list().unwrap()
    }

    /// Session `id` as saved in `dir`.
    fn load(dir: &TempDir, id: &str) -> Session {
        SessionStore::in_dir(dir.0.clone())
            .load(id)
            .unwrap()
            .expect(id)
    }

    #[tokio::test]
    async fn a_finished_turn_saves_the_session() {
        let dir = TempDir::new("session-turn");
        let (mut app, _backend, mut rx) = chat_app(vec![vec![content("done")]]);
        save_in(&mut app, &dir);
        type_line(&mut app, "fix the bug");
        run_to_end(&mut app, &mut rx).await;

        let listed = saved(&dir);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, app.session_id);
        assert_eq!(listed[0].title, "fix the bug");
        let session = load(&dir, &app.session_id);
        assert_eq!(session.messages, app.conversation.messages());
        assert_eq!(session.model.as_deref(), Some(READY_MODEL));
        assert_eq!(session.cwd, r"D:\projects\loops");
        assert_eq!(session.created, app.session_created);
    }

    #[test]
    fn nothing_is_saved_until_someone_types() {
        let dir = TempDir::new("session-unsaid");
        let mut app = plain_app();
        save_in(&mut app, &dir);
        type_line(&mut app, "/help");
        type_line(&mut app, "/clear");
        type_line(&mut app, "/quit");
        assert!(app.should_quit);
        assert!(std::fs::read_dir(&dir.0).unwrap().next().is_none());
    }

    #[test]
    fn quitting_saves_the_session() {
        let dir = TempDir::new("session-quit");
        let mut app = plain_app();
        save_in(&mut app, &dir);
        chatted(&mut app);
        app.update(AppEvent::Key(KeyEvent::new(
            KeyCode::Char('d'),
            KeyModifiers::CONTROL,
        )));
        assert!(app.should_quit);
        assert_eq!(saved(&dir)[0].id, app.session_id);
    }

    #[tokio::test]
    async fn a_cancelled_turn_is_saved_marked_cancelled() {
        let dir = TempDir::new("session-cancel");
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        save_in(&mut app, &dir);
        type_line(&mut app, "hi");
        app.update(agent_event(&app, text("partial")));
        app.update(key(KeyCode::Esc));

        let session = load(&dir, &app.session_id);
        assert_eq!(
            session.messages,
            [
                Message::new(Role::User, "hi"),
                Message::new(Role::Assistant, "partial")
            ]
        );
        assert_eq!(
            session.view["reply_ends"],
            serde_json::json!([[1, "Cancelled"]])
        );
    }

    #[tokio::test]
    async fn the_compacted_conversation_is_saved_under_the_first_title() {
        let dir = TempDir::new("session-compact");
        let (mut app, _backend, mut rx) = chat_app(vec![vec![content("Greeted.")]]);
        save_in(&mut app, &dir);
        chatted(&mut app);
        type_line(&mut app, "/compact");
        compact_to_end(&mut app, &mut rx).await;

        let session = load(&dir, &app.session_id);
        assert_eq!(session.messages, app.conversation.messages());
        assert!(session.messages[0].content.starts_with("Summary of"));
        assert_eq!(session.title, "hi");
    }

    #[test]
    fn clear_and_new_save_the_session_and_start_another() {
        let dir = TempDir::new("session-clear");
        let mut app = plain_app();
        save_in(&mut app, &dir);
        chatted(&mut app);
        app.session_model = Some(READY_MODEL.into());
        let first = app.session_id.clone();
        type_line(&mut app, "/clear");
        assert!(app.conversation.messages().is_empty(), "nothing said");
        assert_ne!(app.session_id, first);
        assert_eq!(app.session_model, None);
        let listed = saved(&dir);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, first);
        assert_eq!(listed[0].title, "hi");

        app.conversation.push(Message::new(Role::User, "second"));
        let second = app.session_id.clone();
        type_line(&mut app, "/new");
        assert_ne!(app.session_id, second);
        assert_eq!(load(&dir, &second).title, "second");
        assert_eq!(saved(&dir).len(), 2);
    }

    #[tokio::test]
    async fn sessions_lists_and_enter_resumes_with_the_display_state() {
        let dir = TempDir::new("session-resume");
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        save_in(&mut app, &dir);

        // Session A, with display state to bring back.
        chatted(&mut app);
        app.reply_ends.insert(1, ReplyEnd::Empty);
        let outcome = ToolOutcome {
            summary: "31 lines".into(),
            is_error: false,
        };
        app.tool_outcomes.insert("c1".into(), outcome);
        app.tool_previews.insert("c1".into(), "edit a.txt".into());
        app.tool_outputs
            .insert("c2".into(), "exit code: 0\nhi".into());
        app.session_model = Some(READY_MODEL.into());
        let a = app.session_id.clone();
        let a_messages = app.conversation.messages().to_vec();
        let (outcomes, previews, outputs) = (
            app.tool_outcomes.clone(),
            app.tool_previews.clone(),
            app.tool_outputs.clone(),
        );
        type_line(&mut app, "/new");

        // Session B, not saved yet.
        app.conversation.push(Message::new(Role::User, "other"));
        app.usage = Some(Usage {
            prompt_tokens: 100,
            completion_tokens: 10,
        });
        app.scroll_up = 5;
        let b = app.session_id.clone();
        type_line(&mut app, "/sessions");
        let Mode::Sessions { entries, selected } = &app.mode else {
            panic!("not listing: {:?}", app.mode);
        };
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, [a.as_str()]);
        assert_eq!(*selected, 0);
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Enter));

        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(app.session_id, a);
        let msgs = app.conversation.messages();
        assert_eq!(msgs[..2], a_messages[..]);
        assert_eq!(
            msgs[2],
            Message::new(Role::Harness, r#"resumed "hi" (2 messages)"#)
        );
        assert_eq!(app.reply_ends, HashMap::from([(1, ReplyEnd::Empty)]));
        assert_eq!(app.tool_outcomes, outcomes);
        assert_eq!(app.tool_previews, previews);
        assert_eq!(app.tool_outputs, outputs);
        assert_eq!(app.usage, None);
        assert_eq!(app.conversation.token_mark(), None);
        assert_eq!(app.session_model.as_deref(), Some(READY_MODEL));
        assert_eq!(app.scroll_up, 0);
        assert_eq!(load(&dir, &b).title, "other", "B was saved on the way out");

        // Esc closes the list; choosing the current session changes nothing.
        type_line(&mut app, "/sessions");
        app.update(key(KeyCode::Esc));
        assert_eq!(app.mode, Mode::Normal);
        type_line(&mut app, "/sessions");
        let Mode::Sessions { entries, .. } = &app.mode else {
            panic!("not listing");
        };
        assert_eq!(entries.len(), 2);
        let current = entries.iter().position(|e| e.id == a).unwrap();
        for _ in 0..current {
            app.update(key(KeyCode::Down));
        }
        let before = app.conversation.messages().len();
        app.update(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Normal);
        assert_eq!(app.session_id, a);
        assert_eq!(app.conversation.messages().len(), before);
    }

    #[test]
    fn sessions_says_when_there_is_nothing_to_list() {
        let mut app = plain_app();
        type_line(&mut app, "/sessions");
        assert_eq!(last_said(&app), SESSIONS_OFF_TEXT);
        let dir = TempDir::new("session-none");
        save_in(&mut app, &dir);
        type_line(&mut app, "/sessions");
        assert_eq!(last_said(&app), NO_SESSIONS_TEXT);
        assert_eq!(app.mode, Mode::Normal);
    }

    #[test]
    fn resume_opens_the_latest_or_the_named_session() {
        let dir = TempDir::new("session-resume-flag");
        let store = SessionStore::in_dir(dir.0.clone());
        let mut app = plain_app().with_sessions(Some(store.clone()));
        app.resume(None);
        assert_eq!(last_said(&app), NO_SESSIONS_TEXT);
        app.resume(Some("nope"));
        assert_eq!(last_said(&app), "no session nope");

        for (id, updated, text) in [("100-000", 100, "old"), ("200-000", 200, "new")] {
            let session = Session {
                id: id.into(),
                title: text.into(),
                model: None,
                cwd: String::new(),
                created: updated,
                updated,
                messages: vec![Message::new(Role::User, text)],
                view: serde_json::Value::Null,
            };
            store.save(&session).unwrap();
        }
        app.resume(None);
        assert_eq!(app.session_id, "200-000");
        assert_eq!(last_said(&app), r#"resumed "new" (1 messages)"#);
        assert!(app.tool_outcomes.is_empty(), "no view, nothing to restore");

        let mut app = plain_app().with_sessions(Some(store));
        app.resume(Some("100-000"));
        assert_eq!(app.session_id, "100-000");
        assert_eq!(app.conversation.messages()[0].content, "old");
    }

    #[test]
    fn context_names_the_session() {
        let mut app = plain_app();
        type_line(&mut app, "/context");
        let off = format!(
            "\nsession: {} (not saved: no home directory)",
            app.session_id
        );
        assert!(last_said(&app).ends_with(&off), "{}", last_said(&app));

        let dir = TempDir::new("session-context");
        save_in(&mut app, &dir);
        chatted(&mut app);
        type_line(&mut app, "/context");
        // The first `/context` line and the chat.
        let on = format!(
            "\nsession: {} (3 messages, {})",
            app.session_id,
            dir.0.display()
        );
        assert!(last_said(&app).ends_with(&on), "{}", last_said(&app));
    }
}
