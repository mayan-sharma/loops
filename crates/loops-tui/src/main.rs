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

mod app;
mod ui;

use std::{sync::Arc, time::Duration};

use anyhow::{Result, anyhow};
use crossterm::event::{Event, EventStream};
use futures::StreamExt;
use loops_core::{
    agent::{Agent, default_environment, default_system_prompt},
    backend::ModelBackend,
    lmstudio::LmStudio,
    tools::ToolSet,
};
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

use app::{App, AppEvent, ModelEvent};

/// How often [`AppEvent::Tick`] fires.
const TICK: Duration = Duration::from_millis(250);

#[tokio::main]
async fn main() -> Result<()> {
    // `init` enables raw mode, enters the alternate screen, and installs a
    // panic hook that restores the terminal before the panic message prints.
    let terminal = ratatui::init();
    let result = run(terminal).await;
    // Restore on every exit path that is not a panic (quit or error).
    ratatui::restore();
    result
}

/// Runs the main loop until the app asks to quit.
async fn run(mut terminal: DefaultTerminal) -> Result<()> {
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
    let agent = Agent::new(backend, tools, Some(default_system_prompt()))
        .with_environment(default_environment(&cwd));

    let mut app = App::new(Arc::new(agent), tx, cwd);
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
