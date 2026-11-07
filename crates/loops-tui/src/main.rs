//! `loops`: terminal front end for the loops harness.
//!
//! # Event loop
//!
//! One tokio runtime, one main loop, one channel:
//!
//! ```text
//!  terminal ──EventStream──> [reader task] ──mpsc──┐
//!  LM Studio ─────> [model tasks, agent turn] ──mpsc──┤
//!                                                  ├─> select! ─> App::update ─> ui::draw
//!  250ms interval ────────────────────── Tick ─────┘
//! ```
//!
//! Every input source is a task that sends [`AppEvent`]s into the channel. The
//! main loop waits on whichever arrives first, applies it with
//! [`App::update`], then redraws. Model work (startup, `/model` listing and
//! loading, agent turns) is just more senders on the same channel, so the
//! loop itself does not change. A turn sends one event per agent event.
//!
//! With `--print` none of this runs: [`print`] does one headless turn and
//! exits, and the terminal is never put in raw mode. With `--resume` the TUI
//! opens on a saved session (see [`loops_core::session`]).

mod app;
mod print;
mod ui;

use std::{process::ExitCode, sync::Arc, time::Duration};

use anyhow::{Result, anyhow};
use crossterm::event::{Event, EventStream};
use futures::StreamExt;
use loops_core::{
    agent::{Agent, INSTRUCTIONS_FILE, default_system_prompt, environment_with, load_instructions},
    backend::ModelBackend,
    lmstudio::LmStudio,
    session::SessionStore,
    tools::ToolSet,
};
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

use app::{App, AppEvent, ModelEvent, describe_instructions};

/// How often [`AppEvent::Tick`] fires.
const TICK: Duration = Duration::from_millis(250);

#[tokio::main]
async fn main() -> Result<ExitCode> {
    // Arguments first: only the TUI may touch the terminal's mode.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let resume = match print::parse(&args) {
        Ok(print::Command::Tui { resume }) => resume,
        Ok(print::Command::Help) => {
            print!("{}", print::USAGE);
            return Ok(ExitCode::SUCCESS);
        }
        Ok(print::Command::Print(args)) => {
            let code = print::run(args).await.unwrap_or_else(|err| {
                eprintln!("error: {err:#}");
                2
            });
            return Ok(ExitCode::from(code));
        }
        Err(err) => {
            eprintln!(
                "error: {err:#}

{}",
                print::USAGE
            );
            return Ok(ExitCode::from(2));
        }
    };

    // `init` enables raw mode, enters the alternate screen, and installs a
    // panic hook that restores the terminal before the panic message prints.
    let terminal = ratatui::init();
    let result = run(terminal, resume).await;
    // Restore on every exit path that is not a panic (quit or error).
    ratatui::restore();
    result.map(|()| ExitCode::SUCCESS)
}

/// Runs the main loop until the app asks to quit, first opening the session
/// `resume` names, if any.
async fn run(mut terminal: DefaultTerminal, resume: Option<print::Resume>) -> Result<()> {
    let (tx, mut rx) = mpsc::channel(256);
    let mut reader = tokio::spawn(forward_terminal_events(tx.clone()));

    // Start LM Studio in the background so the first frame draws immediately.
    let backend: Arc<dyn ModelBackend> = Arc::new(LmStudio::default());
    tokio::spawn(start_backend(Arc::clone(&backend), tx.clone()));

    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    // Tools work inside the directory loops was started in.
    let cwd = std::env::current_dir()?;
    let tools = Arc::new(ToolSet::standard(&cwd));
    if !tools.has("bash") {
        // Queued before the loop starts, so it shows from the first frames.
        // The receiver is right here, so the send cannot fail.
        let _ = tx
            .send(AppEvent::Say("bash not found; the bash tool is off".into()))
            .await;
    }
    // Project instructions ride in the environment block; say what was sent.
    let (instructions, said) = match load_instructions(&cwd) {
        Ok(Some(found)) => {
            let described = describe_instructions(&found);
            let said = format!("{INSTRUCTIONS_FILE} loaded, {described}");
            (Some((found, described)), Some(said))
        }
        Ok(None) => (None, None),
        Err(err) => (
            None,
            Some(format!("{INSTRUCTIONS_FILE} could not be read: {err}")),
        ),
    };
    if let Some(said) = said {
        let _ = tx.send(AppEvent::Say(said)).await;
    }
    let environment = environment_with(&cwd, instructions.as_ref().map(|(found, _)| found));
    let agent =
        Agent::new(backend, tools, Some(default_system_prompt())).with_environment(environment);

    let store = SessionStore::for_project(&cwd);
    let mut app = App::new(Arc::new(agent), tx, cwd).with_sessions(store);
    app.instructions =
        instructions.map(|(_, described)| format!("{INSTRUCTIONS_FILE}, {described}"));
    match resume {
        Some(print::Resume::Latest) => app.resume(None),
        Some(print::Resume::Id(id)) => app.resume(Some(&id)),
        None => {}
    }
    while !app.should_quit {
        terminal.draw(|frame| ui::draw(frame, &app))?;

        let event = tokio::select! {
            // `app` holds a sender, so the channel never closes while we run.
            Some(event) = rx.recv() => event,
            // The reader task ended, so no more input can arrive. Surface why.
            ended = &mut reader => return Err(match ended {
                Ok(Err(err)) => err.into(),
                _ => anyhow!("terminal input stream closed"),
            }),
            _ = tick.tick() => AppEvent::Tick,
        };
        app.update(event);
    }
    Ok(())
}

/// Makes sure the backend is up and reports what is already loaded.
async fn start_backend(backend: Arc<dyn ModelBackend>, tx: mpsc::Sender<AppEvent>) {
    let event = match backend.start().await {
        Ok(loaded) => ModelEvent::ServerReady(loaded),
        Err(err) => ModelEvent::ServerFailed(format!("{err:#}")),
    };
    // A send error means the app has already quit.
    let _ = tx.send(AppEvent::Model(event)).await;
}

/// Reads terminal events and forwards the ones the app cares about.
///
/// Returns when the stream ends, the stream errors, or the receiver is gone.
async fn forward_terminal_events(tx: mpsc::Sender<AppEvent>) -> std::io::Result<()> {
    let mut events = EventStream::new();
    while let Some(event) = events.next().await {
        let app_event = match event? {
            Event::Key(key) => AppEvent::Key(key),
            Event::Resize(..) => AppEvent::Resize,
            _ => continue,
        };
        if tx.send(app_event).await.is_err() {
            break;
        }
    }
    Ok(())
}
