//! Rendering. Everything here reads [`App`] and draws; nothing mutates state.
//!
//! Colors come only from the 16 ANSI colors and no background is ever set, so
//! the screen follows the user's terminal theme, light or dark. One accent per
//! role: cyan is the user and the input box, magenta the model and its tools,
//! yellow anything waiting on the user, red errors, green success. Everything
//! secondary is dim.

use std::time::Duration;

use loops_core::{
    Message, Role, ToolCall,
    agent::{DECLINED, DECLINED_NOBODY_TO_ASK},
    lmstudio::{ModelInfo, ModelState},
    session::{SessionSummary, relative_time, unix_now},
};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, BorderType, Clear, List, ListItem, ListState, Padding, Paragraph, Wrap},
};
use serde_json::Value;

use crate::{
    app::{App, Mode, ModelStatus, PermissionOption, ReplyEnd, ToolOutcome, Usage},
    torus,
};

/// Braille spinner frames, advanced by `AppEvent::Tick`.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Footer of the model picker.
const PICKER_HINT: &str = " ↑↓ move · Enter select · Esc close ";

/// Footer of the permission prompt.
const PERMISSION_HINT: &str = " ↑↓ move   Enter select   Esc cancel turn ";

/// Drawn on the transcript's bottom row while it is scrolled up and a turn
/// is still writing below.
const NEW_OUTPUT: &str = "↓ new output (End)";

/// The selected row of a list: the picker's row, the prompt's option.
const HIGHLIGHT: Style = Style::new().add_modifier(Modifier::REVERSED);

/// Most characters of a `bash` command shown on its tool line.
const COMMAND_CHARS: usize = 60;

/// Height of the bottom area holding the input box: one line and its border.
const INPUT_HEIGHT: u16 = 3;

/// Height of the bottom area while the permission prompt replaces the input
/// box: one row per option and the border.
const PERMISSION_HEIGHT: u16 = PermissionOption::ALL.len() as u16 + 2;

/// Height of the banner: two lines and the border.
const BANNER_HEIGHT: u16 = 4;

/// The splash needs this much room under the banner (and the harness lines
/// there), or the transcript draws as usual.
const SPLASH_MIN_WIDTH: u16 = 24;
const SPLASH_MIN_HEIGHT: u16 = 10;

/// Widest the splash torus is drawn.
const SPLASH_MAX_WIDTH: u16 = 90;

/// The dim line under the splash torus.
const SPLASH_HINT: &str = "/model to pick a model · type a message to begin";

/// Lines of command output kept from each end while it is folded (see
/// [`fold_output`]).
const OUTPUT_KEEP: usize = 4;

/// Indent of a message's later lines, under the text after its prefix.
const INDENT: &str = "  ";

/// Draws one full frame: the transcript (under the banner while the chat is
/// new), the input box, or the permission prompt in its place while one is
/// open, and the status bar on the bottom row. The model picker and the
/// session list are drawn on top of the transcript when open.
pub fn draw(frame: &mut Frame, app: &App) {
    let bottom_height = match app.mode {
        Mode::Permission { .. } => PERMISSION_HEIGHT,
        _ => INPUT_HEIGHT,
    };
    let [transcript, bottom, status] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(bottom_height),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    // The transcript area shrinks with the taller prompt, and the transcript
    // pins its last line to its own bottom row, so the tool line and the
    // preview of the change being asked about stay in view above the prompt.
    draw_transcript(frame, transcript, app);
    draw_status(frame, status, app);

    match &app.mode {
        Mode::Normal => draw_input(frame, bottom, app),
        Mode::Picker { models, selected } => {
            draw_input(frame, bottom, app);
            draw_picker(frame, transcript, models, *selected);
        }
        Mode::Sessions { entries, selected } => {
            draw_input(frame, bottom, app);
            draw_sessions(frame, transcript, entries, *selected, unix_now());
        }
        Mode::Confirm { models, selected } => {
            draw_input(frame, bottom, app);
            draw_confirm(frame, transcript, &models[*selected]);
        }
        Mode::Permission {
            name,
            scope,
            selected,
            ..
        } => {
            draw_permission_prompt(frame, bottom, name, scope, *selected);
        }
    }
}

/// A bordered box with rounded corners, the one box style used everywhere.
fn rounded() -> Block<'static> {
    Block::bordered().border_type(BorderType::Rounded)
}

/// The status bar: the model chip, how full the context window is, then key
/// hints, separated by dim dots. When the row is too narrow the hints go
/// first, then the chip falls back to the short model id, then the context.
fn draw_status(frame: &mut Frame, area: Rect, app: &App) {
    let waiting = matches!(app.mode, Mode::Permission { .. });
    let context = context_span(app.usage, app.context_length);
    let mut hints: Vec<&'static str> = Vec::new();
    if app.busy() {
        hints.push("Esc cancel");
    }
    hints.push("/help");

    let line = |short: bool, hints: &[&'static str], with_context: bool| -> Line<'static> {
        let separator = || Span::styled(" · ", Style::new().dim());
        let mut spans = vec![
            Span::raw(" "),
            chip_span(
                &app.model_status,
                app.spinner,
                short,
                waiting,
                app.compacting(),
            ),
        ];
        if let Some(context) = context.as_ref().filter(|_| with_context) {
            spans.extend([separator(), context.clone()]);
        }
        for hint in hints {
            spans.extend([separator(), Span::styled(*hint, Style::new().dim())]);
        }
        Line::from(spans)
    };

    // Most to least complete; the first that fits wins.
    let room = usize::from(area.width);
    let mut tries: Vec<Line> = (0..=hints.len())
        .rev()
        .map(|kept| line(false, &hints[..kept], true))
        .collect();
    tries.extend([line(true, &[], true), line(true, &[], false)]);
    let last = tries.len() - 1;
    let fits = tries
        .into_iter()
        .enumerate()
        .find(|(i, line)| line.width() <= room || *i == last)
        .map(|(_, line)| line);
    if let Some(line) = fits {
        frame.render_widget(line, area);
    }
}

/// The model status as one styled span, with the short id when `short`.
/// `waiting`: a turn is paused on the user's answer to a permission prompt.
/// `compacting`: the model is busy writing a `/compact` summary, not a reply.
fn chip_span(
    status: &ModelStatus,
    spinner: usize,
    short: bool,
    waiting: bool,
    compacting: bool,
) -> Span<'static> {
    let frame = SPINNER[spinner % SPINNER.len()];
    let (text, style) = match status {
        ModelStatus::Busy(id) if compacting => (
            format!("{frame} {} compacting…", display_id(id, short)),
            Style::new().magenta(),
        ),
        ModelStatus::Busy(id) if waiting => (
            format!("{} waiting for you", display_id(id, short)),
            Style::new().yellow(),
        ),
        ModelStatus::Busy(id) => (
            format!("{frame} {} replying", display_id(id, short)),
            Style::new().magenta(),
        ),
        ModelStatus::Ready(id) => (format!("{} ✓", display_id(id, short)), Style::new().green()),
        ModelStatus::Loading(id) => (
            format!("{frame} loading {}", display_id(id, short)),
            Style::new().yellow(),
        ),
        ModelStatus::Unloading(id) => (
            format!("{frame} unloading {}", display_id(id, short)),
            Style::new().yellow(),
        ),
        ModelStatus::Starting => ("starting lm studio…".to_owned(), Style::new().dim()),
        ModelStatus::NoModel => ("no model".to_owned(), Style::new().dim()),
        ModelStatus::Error(msg) => (format!("error: {msg}"), Style::new().red()),
    };
    Span::styled(text, style)
}

/// `ctx 19%`: the last request's tokens over the loaded model's window, dim
/// while there is room, yellow from 80%, red from 95%. `None` when either is
/// unknown (nothing sent yet, or no model).
fn context_span(usage: Option<Usage>, window: Option<u64>) -> Option<Span<'static>> {
    let used = usage.map(|u| u.prompt_tokens + u.completion_tokens)?;
    let window = window.filter(|&w| w > 0)?;
    // Rounded to the nearest percent, as `/context` does.
    let percent = (used * 100 + window / 2) / window;
    let style = match percent {
        95.. => Style::new().red(),
        80.. => Style::new().yellow(),
        _ => Style::new().dim(),
    };
    Some(Span::styled(format!("ctx {percent}%"), style))
}

/// `publisher/name` as is, or just `name` when `short`.
fn display_id(id: &str, short: bool) -> &str {
    if short {
        id.rsplit('/').next().unwrap_or(id)
    } else {
        id
    }
}

/// Whether the chat is new: nobody has said anything yet. Harness lines
/// (a startup notice, `/help`) do not count, so the banner stays over them.
fn shows_banner(app: &App) -> bool {
    !app.conversation
        .messages()
        .iter()
        .any(|m| matches!(m.role, Role::User | Role::Assistant))
}

/// The banner (while the chat is new) over the splash, if there is room for
/// it; otherwise the message history, wrapped and pinned to the bottom unless
/// scrolled up.
fn draw_transcript(frame: &mut Frame, area: Rect, app: &App) {
    let mut area = area;
    if shows_banner(app) {
        // One blank row between the banner and what follows.
        let [banner, _, rest] = Layout::vertical([
            Constraint::Length(BANNER_HEIGHT),
            Constraint::Length(1),
            Constraint::Fill(1),
        ])
        .areas(area);
        draw_banner(frame, banner, app);
        if draw_splash(frame, rest, app) {
            return;
        }
        area = rest;
    }
    if area.height == 0 {
        return;
    }

    let paragraph = Paragraph::new(transcript_lines(app)).wrap(Wrap { trim: false });
    // Pin the last wrapped line to the bottom row, then back off by however
    // far the user scrolled up, but never past the first line. Recomputed every
    // frame from the content, so a resize or new output needs no bookkeeping.
    let overflow = paragraph
        .line_count(area.width)
        .saturating_sub(usize::from(area.height));
    let scrolled = app.scroll_up.min(overflow);
    let offset = u16::try_from(overflow - scrolled).unwrap_or(u16::MAX);
    frame.render_widget(paragraph.scroll((offset, 0)), area);

    if scrolled > 0 && app.busy() {
        let bottom_row = Rect {
            y: area.bottom() - 1,
            height: 1,
            ..area
        };
        frame.render_widget(Line::from(NEW_OUTPUT.dim()).right_aligned(), bottom_row);
    }
}

/// The rounded box at the top of a new chat: name and version, then where
/// loops works, which model is loaded, and how to get help.
fn draw_banner(frame: &mut Frame, area: Rect, app: &App) {
    let model = match &app.model_status {
        ModelStatus::Ready(id) | ModelStatus::Busy(id) => id.as_str(),
        _ => "no model",
    };
    let lines = vec![
        Line::from(vec![
            "loops".cyan().bold(),
            Span::raw("  "),
            Span::styled(concat!("v", env!("CARGO_PKG_VERSION")), Style::new().dim()),
        ]),
        Line::styled(
            format!("{} · {model} · /help for commands", app.cwd.display()),
            Style::new().dim(),
        ),
    ];
    // Dim, so the banner sits back and the input box is the brightest frame.
    let block = rounded()
        .border_style(Style::new().dim())
        .padding(Padding::new(2, 1, 0, 0));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// The splash under the banner of a new chat: any harness lines at the top,
/// then a spinning torus (turned by `app.torus_angle`) and a dim hint,
/// centered in the room left. Returns false, drawing nothing, when there is
/// not room for it, so the transcript draws as usual.
fn draw_splash(frame: &mut Frame, area: Rect, app: &App) -> bool {
    if area.width < SPLASH_MIN_WIDTH || area.height < SPLASH_MIN_HEIGHT {
        return false;
    }
    let notes = Paragraph::new(transcript_lines(app)).wrap(Wrap { trim: false });
    let used = u16::try_from(notes.line_count(area.width)).unwrap_or(u16::MAX);
    let [notes_area, rest] =
        Layout::vertical([Constraint::Length(used), Constraint::Fill(1)]).areas(area);
    if rest.height < SPLASH_MIN_HEIGHT {
        return false;
    }
    frame.render_widget(notes, notes_area);

    // Twice as wide as tall looks round in terminal cells.
    let width = rest
        .width
        .min(rest.height.saturating_mul(2))
        .min(SPLASH_MAX_WIDTH);
    let height = (rest.height - 2).min(width / 2);
    let hint = rest.width >= u16::try_from(SPLASH_HINT.chars().count()).unwrap_or(u16::MAX);
    let block = height + u16::from(hint);
    // Round the top margin up, so a spare row parts it from the lines above.
    let top = rest.y + (rest.height - block).div_ceil(2);
    let torus_area = Rect {
        x: rest.x + (rest.width - width) / 2,
        y: top,
        width,
        height,
    };
    let (a, b) = app.torus_angle;
    frame.render_widget(
        Paragraph::new(torus_lines(torus::frame(a, b, width, height))),
        torus_area,
    );
    if hint {
        let hint_row = Rect {
            y: top + height,
            height: 1,
            ..rest
        };
        frame.render_widget(Line::from(SPLASH_HINT.dim()).centered(), hint_row);
    }
    true
}

/// Torus rows in cyan, the darkest four shades dim so it has depth.
fn torus_lines(rows: Vec<String>) -> Vec<Line<'static>> {
    let bright = Style::new().cyan();
    let dark = bright.dim();
    rows.into_iter()
        .map(|row| {
            let mut spans = Vec::new();
            let mut run = String::new();
            let mut run_style = Style::new();
            for c in row.chars() {
                let style = if c == ' ' {
                    Style::new()
                } else if torus::SHADES[..4].contains(c) {
                    dark
                } else {
                    bright
                };
                if style != run_style && !run.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut run), run_style));
                }
                run_style = style;
                run.push(c);
            }
            spans.push(Span::styled(run, run_style));
            Line::from(spans)
        })
        .collect()
}

/// Every drawn message, oldest first, with a blank line between messages.
/// The replies of one turn (split by tool rounds) read as one message: only
/// the first gets the bullet, and no blank line separates them.
fn transcript_lines(app: &App) -> Vec<Line<'_>> {
    let mut lines = Vec::new();
    // The role of the last message drawn; tool results and system prompts are not.
    let mut previous = None;
    for (index, message) in app.conversation.messages().iter().enumerate() {
        let continues = message.role == Role::Assistant && previous == Some(Role::Assistant);
        let block = message_lines(app, index, message, continues);
        if block.is_empty() {
            continue;
        }
        if previous.is_some() && !continues {
            lines.push(Line::default());
        }
        lines.extend(block);
        previous = Some(message.role);
    }
    lines
}

/// One message as display lines. `continues`: an assistant message that
/// carries on the reply above it. Tool results and system prompts are not
/// drawn: a call's `⎿` line sums its result up.
fn message_lines<'a>(
    app: &'a App,
    index: usize,
    message: &'a Message,
    continues: bool,
) -> Vec<Line<'a>> {
    match message.role {
        Role::User => prefixed(
            &message.content,
            Span::styled("❯ ", Style::new().cyan().bold()),
            Style::new().bold(),
        ),
        Role::Harness => {
            let style = if message.content.starts_with("error:") {
                Style::new().red()
            } else {
                Style::new().dim()
            };
            prefixed(
                &message.content,
                Span::styled("loops › ", Style::new().dim()),
                style,
            )
        }
        Role::Assistant => assistant_lines(app, index, message, continues),
        Role::Tool | Role::System => Vec::new(),
    }
}

/// `text` in `style`, `prefix` before its first line and [`INDENT`] before the rest.
fn prefixed<'a>(text: &'a str, prefix: Span<'static>, style: Style) -> Vec<Line<'a>> {
    text.lines()
        .enumerate()
        .map(|(i, line)| {
            let lead = if i == 0 {
                prefix.clone()
            } else {
                Span::raw(INDENT)
            };
            Line::from(vec![lead, Span::styled(line, style)])
        })
        .collect()
}

/// A reply: its text after a magenta `●` (none when it `continues` the one
/// above), or a dim spinner and `thinking…` while the model has said nothing
/// yet; its tool calls; then how it ended, if that needs saying.
fn assistant_lines<'a>(
    app: &'a App,
    index: usize,
    message: &'a Message,
    continues: bool,
) -> Vec<Line<'a>> {
    let dim = Style::new().dim();
    let lead = if continues {
        Span::raw(INDENT)
    } else {
        Span::styled("● ", Style::new().magenta())
    };
    let mut lines = reply_text_lines(&message.content, lead.clone());

    let end = app.reply_ends.get(&index);
    if end == Some(&ReplyEnd::Empty) {
        let lead = if continues { INDENT } else { "● " };
        return vec![Line::styled(format!("{lead}(empty reply)"), dim)];
    }
    if lines.is_empty() {
        match app.waiting_reply() {
            Some((reply, waited)) if reply == index => {
                lines.push(thinking_line(app.spinner, waited));
            }
            // A reply with no text at all still shows its bullet, unless
            // its tool calls stand in for it.
            _ if !continues && message.tool_calls.is_empty() => lines.push(Line::from(lead)),
            _ => {}
        }
    }
    for call in &message.tool_calls {
        lines.extend(tool_lines(app, call));
    }

    if end == Some(&ReplyEnd::Cancelled) {
        let marker = if message.content.is_empty() {
            "[cancelled]"
        } else {
            " [cancelled]"
        };
        match lines.last_mut() {
            Some(last) => last.push_span(Span::styled(marker, dim)),
            None => lines.push(Line::from(vec![
                Span::raw(INDENT),
                Span::styled(marker, dim),
            ])),
        }
    }
    lines
}

/// `⠋ thinking… 3s`, all dim; the spinner stands where the bullet goes. The
/// seconds show from the first whole one, so a quick reply does not flash `0s`.
fn thinking_line(spinner: usize, waited: Duration) -> Line<'static> {
    let frame = SPINNER[spinner % SPINNER.len()];
    let secs = waited.as_secs();
    let elapsed = match secs {
        0 => String::new(),
        1..60 => format!(" {secs}s"),
        _ => format!(" {}m {}s", secs / 60, secs % 60),
    };
    Line::styled(format!("{frame} thinking…{elapsed}"), Style::new().dim())
}

/// A reply's text: `lead` before the first line, [`INDENT`] before the rest.
/// Inside a ``` fence the indent becomes a dim `│ ` bar, and the fence lines
/// themselves (language tag and all) are just the bar, so a code block reads
/// as one block. An open fence runs to the end, which is what a reply still
/// streaming its code block needs.
fn reply_text_lines<'a>(text: &'a str, lead: Span<'static>) -> Vec<Line<'a>> {
    let bar = Style::new().dim();
    let mut in_code = false;
    text.lines()
        .enumerate()
        .map(|(i, line)| {
            let lead = (i == 0).then(|| lead.clone());
            if line.trim_start().starts_with("```") {
                in_code = !in_code;
                return Line::from(lead.unwrap_or_else(|| Span::styled("│", bar)));
            }
            let indent = if in_code {
                Span::styled("│ ", bar)
            } else {
                Span::raw(INDENT)
            };
            Line::from(vec![lead.unwrap_or(indent), Span::raw(line)])
        })
        .collect()
}

/// One tool call, indented under its reply: `● name(args)` with the bullet
/// and name magenta, then `⎿` and how it went, then the preview of the change
/// it asked permission for, if any, then a command's output (dim, folded
/// unless [`App::full_output`]).
fn tool_lines(app: &App, call: &ToolCall) -> Vec<Line<'static>> {
    let magenta = Style::new().magenta();
    let mut lines = vec![
        Line::from(vec![
            Span::raw(INDENT),
            Span::styled("● ", magenta),
            Span::styled(call.name.clone(), magenta),
            Span::raw(format!("({})", call_arguments(&call.arguments))),
        ]),
        outcome_line(app.tool_outcomes.get(&call.id), app.spinner),
    ];
    if let Some(preview) = app.tool_previews.get(&call.id) {
        lines.extend(preview_lines(preview, "      "));
    }
    if let Some(output) = app.tool_outputs.get(&call.id) {
        let dim = Style::new().dim();
        lines.extend(
            output_lines(output, app.full_output)
                .into_iter()
                .map(|line| Line::styled(format!("      {line}"), dim)),
        );
    }
    lines
}

/// The lines of a `bash` result drawn under its outcome: all but the first
/// (`exit code: N`, which the summary already gives), through [`fold_output`].
pub fn output_lines(output: &str, full: bool) -> Vec<String> {
    let body: Vec<&str> = output.lines().skip(1).collect();
    fold_output(&body, full)
}

/// `lines` as drawn: all of them when `full` or when they are few; otherwise
/// the first and last [`OUTPUT_KEEP`] with `… N lines omitted …` between.
/// The tool already caps what it returns, so `full` needs no cap of its own.
fn fold_output(lines: &[&str], full: bool) -> Vec<String> {
    if full || lines.len() <= 2 * OUTPUT_KEEP {
        return lines.iter().map(|&line| line.to_owned()).collect();
    }
    let omitted = lines.len() - 2 * OUTPUT_KEEP;
    let head = lines[..OUTPUT_KEEP].iter();
    let tail = lines[lines.len() - OUTPUT_KEEP..].iter();
    head.map(|&line| line.to_owned())
        .chain([format!("… {omitted} lines omitted …")])
        .chain(tail.map(|&line| line.to_owned()))
        .collect()
}

/// `    ⎿ 31 lines`: dim when it worked, yellow if the user said no, red
/// `error: …` (first line only) if it failed, and a dim spinner and `running`
/// while there is no outcome yet.
fn outcome_line(outcome: Option<&ToolOutcome>, spinner: usize) -> Line<'static> {
    let (text, style) = match outcome {
        None => (
            format!("{} running", SPINNER[spinner % SPINNER.len()]),
            Style::new().dim(),
        ),
        Some(outcome) if is_declined(outcome) => (outcome.summary.clone(), Style::new().yellow()),
        Some(outcome) if outcome.is_error => {
            let error = outcome.summary.lines().next().unwrap_or_default();
            (format!("error: {error}"), Style::new().red())
        }
        Some(outcome) => (outcome.summary.clone(), Style::new().dim()),
    };
    Line::styled(format!("    ⎿ {text}"), style)
}

/// Whether the call was declined rather than failed. The agent marks both as
/// errors; only the summary tells them apart.
fn is_declined(outcome: &ToolOutcome) -> bool {
    outcome.is_error && [DECLINED, DECLINED_NOBODY_TO_ASK].contains(&outcome.summary.as_str())
}

/// A tool's preview of a change, one line per line, each after `indent`:
/// `- ` lines (taken out) red, `+ ` lines (put in) green, the rest (the
/// heading, a `…` note) dim.
fn preview_lines(preview: &str, indent: &str) -> Vec<Line<'static>> {
    preview
        .lines()
        .map(|line| {
            let style = if line.starts_with("- ") {
                Style::new().red()
            } else if line.starts_with("+ ") {
                Style::new().green()
            } else {
                Style::new().dim()
            };
            Line::styled(format!("{indent}{line}"), style)
        })
        .collect()
}

/// A call's arguments for display. A call that names a `path` shows just
/// that, unquoted (`notes.txt`): the rest of an edit or write is in its
/// preview, and would flood the line. A `command` (`bash`) shows just that
/// too, as a JSON string so its quotes and newlines stay on one line, cut
/// after [`COMMAND_CHARS`] characters; the preview has all of it. Otherwise
/// the values of the JSON object, as JSON, comma-separated. Anything else is
/// shown as it came.
fn call_arguments(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => match (map.get("path"), map.get("command")) {
            (Some(Value::String(path)), _) => path.clone(),
            (Some(path), _) => path.to_string(),
            (None, Some(Value::String(command))) => {
                let shown = if command.chars().count() > COMMAND_CHARS {
                    let cut: String = command.chars().take(COMMAND_CHARS).collect();
                    format!("{cut}…")
                } else {
                    command.clone()
                };
                Value::String(shown).to_string()
            }
            _ => map
                .values()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        },
        _ => arguments.to_owned(),
    }
}

/// The input box: rounded, ` ❯ ` in its top border, `Message loops…` while
/// empty (the text area's placeholder). Its color tells the state at a glance:
/// red after a model error, magenta while the model replies (or compacts),
/// cyan otherwise.
/// (The permission prompt, which takes its place, is yellow.)
fn draw_input(frame: &mut Frame, area: Rect, app: &App) {
    let color = match app.model_status {
        ModelStatus::Error(_) => Color::Red,
        ModelStatus::Busy(_) => Color::Magenta,
        _ => Color::Cyan,
    };
    let block = rounded()
        .border_style(Style::new().fg(color))
        .title(Span::styled(" ❯ ", Style::new().fg(color).bold()));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(&app.input, inner);
}

/// The model list popup: a dim header row, then one row per model with
/// `loaded` in green; the selected row reversed.
fn draw_picker(frame: &mut Frame, area: Rect, models: &[ModelInfo], selected: usize) {
    const HEADER: [&str; 4] = ["model", "quant", "context", "state"];
    let id_width = models
        .iter()
        .map(|m| m.id.chars().count())
        .chain([HEADER[0].len()])
        .max()
        .unwrap_or(0);
    let quant_width = models
        .iter()
        .filter_map(|m| m.quantization.as_deref())
        .map(|q| q.chars().count())
        .chain([HEADER[1].len()])
        .max()
        .unwrap_or(0);
    // Everything before the state column, which gets its own color.
    let columns = |id: &str, quant: &str, context: &str| {
        format!(" {id:<id_width$}  {quant:<quant_width$}  {context:>7}  ")
    };

    let header = Line::styled(
        format!(
            "{}{:<6} ",
            columns(HEADER[0], HEADER[1], HEADER[2]),
            HEADER[3]
        ),
        Style::new().dim(),
    );
    let rows: Vec<Line> = models
        .iter()
        .map(|m| {
            let quant = m.quantization.as_deref().unwrap_or("-");
            let context = m.context_length().map_or("-".to_owned(), context_label);
            let state = if m.state == ModelState::Loaded {
                Span::styled("loaded", Style::new().green())
            } else {
                Span::raw("      ")
            };
            Line::from(vec![
                Span::raw(columns(&m.id, quant, &context)),
                state,
                Span::raw(" "),
            ])
        })
        .collect();

    let content_width = rows
        .iter()
        .map(Line::width)
        .chain([header.width(), PICKER_HINT.chars().count()])
        .max()
        .unwrap_or(0);
    let popup = centered(area, content_width + 2, rows.len() + 3);

    let block = rounded()
        .title(" load a model ")
        .title_bottom(PICKER_HINT.dim());
    let inner = block.inner(popup);
    let [header_row, list_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(inner);
    let list = List::new(rows.into_iter().map(ListItem::new)).highlight_style(HIGHLIGHT);
    let mut state = ListState::default().with_selected(Some(selected));
    frame.render_widget(Clear, popup);
    frame.render_widget(block, popup);
    frame.render_widget(header, header_row);
    frame.render_stateful_widget(list, list_area, &mut state);
}

/// The `/sessions` popup: one row per saved session, newest first, as
/// `<title>  <how long ago>  <n> messages` in aligned columns (the last two
/// dim), the selected row reversed. `now` is the time ages are measured from.
fn draw_sessions(
    frame: &mut Frame,
    area: Rect,
    entries: &[SessionSummary],
    selected: usize,
    now: u64,
) {
    let titles: Vec<&str> = entries
        .iter()
        .map(|e| {
            if e.title.is_empty() {
                "(untitled)"
            } else {
                e.title.as_str()
            }
        })
        .collect();
    let ages: Vec<String> = entries
        .iter()
        .map(|e| relative_time(e.updated, now))
        .collect();
    let title_width = titles.iter().map(|t| t.chars().count()).max().unwrap_or(0);
    let age_width = ages.iter().map(String::len).max().unwrap_or(0);
    let rows: Vec<Line> = entries
        .iter()
        .zip(titles.iter().zip(&ages))
        .map(|(entry, (title, age))| {
            let count = match entry.messages {
                1 => "1 message".to_owned(),
                n => format!("{n} messages"),
            };
            Line::from(vec![
                Span::raw(format!(" {title:<title_width$}  ")),
                Span::styled(format!("{age:>age_width$}  {count} "), Style::new().dim()),
            ])
        })
        .collect();

    let content_width = rows
        .iter()
        .map(Line::width)
        .chain([PICKER_HINT.chars().count()])
        .max()
        .unwrap_or(0);
    let popup = centered(area, content_width + 2, rows.len() + 2);
    let block = rounded()
        .title(" resume a session ")
        .title_bottom(PICKER_HINT.dim());
    let list = List::new(rows.into_iter().map(ListItem::new)).highlight_style(HIGHLIGHT);
    let mut state = ListState::default().with_selected(Some(selected));
    frame.render_widget(Clear, popup);
    frame.render_stateful_widget(list.block(block), popup, &mut state);
}

/// The `Load <id>? (y/n)` popup.
fn draw_confirm(frame: &mut Frame, area: Rect, model: &ModelInfo) {
    let text = format!("Load {}? (y/n)", model.id);
    let popup = centered(area, text.chars().count() + 4, 3);
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(text).centered().block(rounded()), popup);
}

/// The `allow <tool>?` prompt, drawn where the input box goes, in a yellow
/// box because the turn is waiting on the user: one row per
/// [`PermissionOption`] (worded for the call's `scope`), `❯` and the
/// highlight on the `selected` one, the others dim, and the keys in the
/// bottom border. The change itself is not repeated: the transcript shows it
/// under the call's tool line.
fn draw_permission_prompt(frame: &mut Frame, area: Rect, name: &str, scope: &str, selected: usize) {
    let lines: Vec<Line> = PermissionOption::ALL
        .iter()
        .enumerate()
        .map(|(i, option)| {
            let label = option.label(scope);
            if i == selected {
                Line::styled(format!("❯ {label}"), HIGHLIGHT.bold())
            } else {
                Line::styled(format!("  {label}"), Style::new().dim())
            }
        })
        .collect();
    let block = rounded()
        .border_style(Style::new().yellow())
        .title(Span::styled(
            format!(" allow {name}? "),
            Style::new().yellow().bold(),
        ))
        .title_bottom(PERMISSION_HINT.dim())
        .padding(Padding::horizontal(1));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// A context length in thousands, e.g. 262144 -> `262k`.
fn context_label(tokens: u64) -> String {
    format!("{}k", tokens / 1000)
}

/// A `width` x `height` rect centered in `area`, shrunk to fit.
fn centered(area: Rect, width: usize, height: usize) -> Rect {
    let width = u16::try_from(width).unwrap_or(u16::MAX).min(area.width);
    let height = u16::try_from(height).unwrap_or(u16::MAX).min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::testing::{agent_event, backdate_round, chat_app, model, plain_app, type_line};
    use crossterm::event::{KeyCode, KeyEvent};
    use loops_core::{Message, agent::AgentEvent, chat::ChatDelta};
    use ratatui::{Terminal, backend::TestBackend, buffer::Buffer};

    fn render_buffer(app: &App, width: u16, height: u16) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        terminal.backend().buffer().clone()
    }

    fn rows(buffer: &Buffer) -> Vec<String> {
        buffer
            .content()
            .chunks(usize::from(buffer.area.width))
            .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect()
    }

    fn render(app: &App) -> String {
        rows(&render_buffer(app, 40, 8)).join("\n")
    }

    /// Finds the first row containing `text`, and the column (in chars) where it starts.
    fn find(rows: &[String], text: &str) -> (u16, u16) {
        let y = rows.iter().position(|r| r.contains(text)).expect(text);
        let x = rows[y][..rows[y].find(text).unwrap()].chars().count();
        (u16::try_from(x).unwrap(), u16::try_from(y).unwrap())
    }

    /// The cell where `text` first starts on screen.
    fn cell_at<'a>(buffer: &'a Buffer, text: &str) -> &'a ratatui::buffer::Cell {
        let (x, y) = find(&rows(buffer), text);
        &buffer[(x, y)]
    }

    fn is_dim(cell: &ratatui::buffer::Cell) -> bool {
        cell.modifier.contains(Modifier::DIM)
    }

    fn press(app: &mut App, code: KeyCode) {
        app.update(crate::app::AppEvent::Key(KeyEvent::from(code)));
    }

    // --- banner -------------------------------------------------------------

    #[test]
    fn a_new_chat_shows_the_banner() {
        let mut app = plain_app();
        app.model_status = ModelStatus::Ready("google/gemma-4-e4b".into());
        let buffer = render_buffer(&app, 80, 12);
        let screen = rows(&buffer);
        assert!(screen[0].starts_with('╭'), "{}", screen[0]);
        assert!(screen[1].starts_with("│  loops  v0.1.0"), "{}", screen[1]);
        assert!(
            screen[2].contains(r"D:\projects\loops · google/gemma-4-e4b · /help for commands"),
            "{}",
            screen[2]
        );
        assert!(screen[3].starts_with('╰'));

        let name = cell_at(&buffer, "loops  v");
        assert_eq!(name.fg, Color::Cyan);
        assert!(name.modifier.contains(Modifier::BOLD));
        assert!(is_dim(cell_at(&buffer, "v0.1.0")));
        assert!(is_dim(cell_at(&buffer, r"D:\projects")));

        app.model_status = ModelStatus::NoModel;
        let screen = rows(&render_buffer(&app, 80, 12)).join("\n");
        assert!(screen.contains("· no model ·"));
    }

    #[test]
    fn the_banner_stays_over_harness_lines_and_goes_with_the_first_message() {
        let mut app = plain_app();
        app.conversation.push(Message::new(
            Role::Harness,
            "bash not found; the bash tool is off",
        ));
        let screen = rows(&render_buffer(&app, 60, 12));
        assert!(screen[1].contains("loops  v0.1.0"));
        assert_eq!(screen[4].trim(), "", "a gap under the banner");
        assert!(
            screen[5].starts_with("loops › bash not found"),
            "{}",
            screen[5]
        );

        app.conversation.push(Message::new(Role::User, "hi"));
        let screen = rows(&render_buffer(&app, 60, 12));
        assert!(!screen.join("\n").contains("v0.1.0"));
        assert!(
            screen[0].starts_with("loops › bash not found"),
            "{}",
            screen[0]
        );
        assert_eq!(screen[1].trim(), "");
        assert!(screen[2].starts_with("❯ hi"));
    }

    // --- splash -------------------------------------------------------------

    /// The rows above the input box, where the transcript draws.
    fn transcript_rows(buffer: &Buffer) -> Vec<String> {
        let mut screen = rows(buffer);
        screen.truncate(screen.len() - usize::from(INPUT_HEIGHT) - 1);
        screen
    }

    fn has_torus(rows: &[String]) -> bool {
        rows.iter().any(|row| row.contains(['@', '$', '#']))
    }

    #[test]
    fn a_new_chat_spins_a_torus_under_the_banner() {
        let mut app = plain_app();
        let buffer = render_buffer(&app, 80, 30);
        let screen = transcript_rows(&buffer);
        assert!(screen[1].contains("loops  v0.1.0"));
        assert!(!has_torus(&screen[..5]), "nothing over the banner's gap");
        assert!(has_torus(&screen[5..]));
        let (x, y) = find(&screen, SPLASH_HINT);
        assert_eq!(
            usize::from(x),
            (80 - SPLASH_HINT.chars().count()) / 2,
            "centered"
        );
        assert!(is_dim(&buffer[(x, y)]));
        assert!(
            !has_torus(&screen[usize::from(y)..]),
            "the hint is under the torus"
        );

        // The first cell of each shade under the banner.
        let shade = |c: char| {
            let (x, y) = find(&screen[5..], &c.to_string());
            &buffer[(x, y + 5)]
        };
        assert_eq!(shade('@').fg, Color::Cyan);
        assert!(!is_dim(shade('@')));
        assert_eq!(shade('.').fg, Color::Cyan);
        assert!(is_dim(shade('.')));

        app.update(crate::app::AppEvent::Tick);
        assert_ne!(
            transcript_rows(&render_buffer(&app, 80, 30)),
            screen,
            "it turns"
        );
    }

    #[test]
    fn harness_lines_stay_above_the_torus() {
        let mut app = plain_app();
        app.conversation.push(Message::new(
            Role::Harness,
            "bash not found; the bash tool is off",
        ));
        let screen = transcript_rows(&render_buffer(&app, 80, 30));
        assert!(
            screen[5].starts_with("loops › bash not found"),
            "{}",
            screen[5]
        );
        assert!(has_torus(&screen[6..]));
        assert!(
            screen
                .join(
                    "
"
                )
                .contains(SPLASH_HINT)
        );
    }

    #[test]
    fn the_torus_goes_with_the_first_message() {
        let mut app = plain_app();
        app.conversation.push(Message::new(Role::User, "hi"));
        let screen = transcript_rows(&render_buffer(&app, 80, 30));
        assert!(!has_torus(&screen));
        assert!(
            !screen
                .join(
                    "
"
                )
                .contains(SPLASH_HINT)
        );
        assert!(screen.iter().any(|row| row.starts_with("❯ hi")));
    }

    #[test]
    fn no_torus_without_room_and_no_panic_at_any_size() {
        let app = plain_app();
        let screen = transcript_rows(&render_buffer(&app, 40, 12));
        assert!(!has_torus(&screen));
        assert!(
            !screen
                .join(
                    "
"
                )
                .contains(SPLASH_HINT)
        );
        for width in [1, 10, 23, 24, 25, 40, 49, 60, 100, 200] {
            for height in [1, 5, 18, 19, 20, 25, 50, 80] {
                render_buffer(&app, width, height);
            }
        }
    }

    #[tokio::test]
    async fn clear_brings_the_torus_back() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        app.conversation.push(Message::new(Role::User, "hi"));
        app.conversation
            .push(Message::new(Role::Assistant, "hello"));
        assert!(!has_torus(&transcript_rows(&render_buffer(&app, 80, 30))));
        type_line(&mut app, "/clear");
        let screen = transcript_rows(&render_buffer(&app, 80, 30));
        assert!(has_torus(&screen));
        assert!(
            screen
                .join(
                    "
"
                )
                .contains(SPLASH_HINT)
        );
    }

    // --- messages -----------------------------------------------------------

    #[test]
    fn transcript_sticks_to_the_bottom() {
        let mut app = plain_app();
        for i in 0..20 {
            app.conversation
                .push(Message::new(Role::User, format!("msg {i}")));
        }
        let screen = rows(&render_buffer(&app, 40, 8));
        // 8 rows: transcript 0..4, input box 4..7, status bar 7.
        assert!(screen[3].starts_with("❯ msg 19"), "{}", screen[3]);
        assert_eq!(screen[2].trim(), "", "a blank line between messages");
        assert!(screen[1].starts_with("❯ msg 18"));
        assert!(!screen.join("\n").contains("msg 0 "));
    }

    #[test]
    fn user_reply_and_harness_lines_have_their_own_prefix_and_color() {
        let mut app = plain_app();
        app.conversation
            .push(Message::new(Role::User, "fix it\nplease"));
        app.conversation
            .push(Message::new(Role::Assistant, "line one\nline two"));
        app.conversation
            .push(Message::new(Role::Harness, "loaded google/gemma-4-e4b"));
        app.conversation
            .push(Message::new(Role::Harness, "error: boom"));

        let buffer = render_buffer(&app, 40, 14);
        let screen = rows(&buffer);
        let lines: Vec<&str> = screen[..10].iter().map(|r| r.trim_end()).collect();
        assert_eq!(
            lines,
            [
                "❯ fix it",
                "  please",
                "",
                "● line one",
                "  line two",
                "",
                "loops › loaded google/gemma-4-e4b",
                "",
                "loops › error: boom",
                "",
            ]
        );

        // User: bold cyan arrow, bold text in the default color.
        assert_eq!(buffer[(0, 0)].fg, Color::Cyan);
        assert!(buffer[(0, 0)].modifier.contains(Modifier::BOLD));
        assert_eq!(buffer[(2, 0)].fg, Color::Reset);
        assert!(buffer[(2, 0)].modifier.contains(Modifier::BOLD));
        assert!(buffer[(2, 1)].modifier.contains(Modifier::BOLD));
        // Reply: magenta bullet, plain text.
        assert_eq!(buffer[(0, 3)].fg, Color::Magenta);
        assert_eq!(buffer[(2, 3)].fg, Color::Reset);
        assert!(buffer[(2, 3)].modifier.is_empty());
        // Harness: all dim, but an error is red.
        assert!(is_dim(&buffer[(0, 6)]) && is_dim(&buffer[(8, 6)]));
        assert!(is_dim(&buffer[(0, 8)]));
        assert_eq!(buffer[(8, 8)].fg, Color::Red);
    }

    #[test]
    fn the_replies_of_one_turn_share_one_bullet() {
        let mut app = plain_app();
        app.conversation.push(Message::new(Role::User, "fix it"));
        app.conversation.push(Message::assistant_with_tool_calls(
            "I'll read it first.",
            vec![ToolCall {
                id: "a".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"notes.txt"}"#.into(),
            }],
        ));
        app.conversation.push(Message::tool_result("a", "1: hi"));
        app.conversation
            .push(Message::new(Role::Assistant, "It says hi."));
        app.tool_outcomes.insert(
            "a".into(),
            ToolOutcome {
                summary: "4 lines".into(),
                is_error: false,
            },
        );
        let screen = rows(&render_buffer(&app, 60, 10));
        let lines: Vec<&str> = screen[..6].iter().map(|r| r.trim_end()).collect();
        assert_eq!(
            lines,
            [
                "❯ fix it",
                "",
                "● I'll read it first.",
                "  ● read_file(notes.txt)",
                "    ⎿ 4 lines",
                "  It says hi.",
            ]
        );
    }

    #[tokio::test]
    async fn thinking_shows_a_spinner_and_the_seconds_then_gives_way_to_text() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "hi");
        let buffer = render_buffer(&app, 40, 8);
        let screen = rows(&buffer);
        let (x, y) = find(&screen, "⠋ thinking…");
        assert_eq!(x, 0, "the spinner stands where the bullet goes");
        assert_eq!(screen[usize::from(y)].trim_end(), "⠋ thinking…", "no 0s");
        assert!(is_dim(&buffer[(0, y)]) && is_dim(&buffer[(2, y)]));

        app.update(agent_event(&app, AgentEvent::Thinking));
        backdate_round(&mut app, 3);
        app.update(crate::app::AppEvent::Tick);
        assert!(render(&app).contains("⠙ thinking… 3s"));

        app.update(agent_event(&app, AgentEvent::Text("hey".into())));
        let screen = render(&app);
        assert!(screen.contains("● hey"));
        assert!(!screen.contains("thinking"));
    }

    #[test]
    fn thinking_time_reads_in_minutes_past_one() {
        let line = |secs| thinking_line(0, Duration::from_secs(secs)).to_string();
        assert_eq!(line(0), "⠋ thinking…");
        assert_eq!(line(59), "⠋ thinking… 59s");
        assert_eq!(line(125), "⠋ thinking… 2m 5s");
    }

    #[test]
    fn empty_and_cancelled_replies_say_so_in_dim() {
        let mut app = plain_app();
        app.conversation.push(Message::new(Role::User, "hi"));
        app.conversation
            .push(Message::new(Role::Assistant, "line one\nline two"));
        app.reply_ends.insert(1, ReplyEnd::Cancelled);
        let buffer = render_buffer(&app, 40, 8);
        let screen = rows(&buffer);
        assert_eq!(screen[3].trim_end(), "  line two [cancelled]");
        assert!(is_dim(cell_at(&buffer, "[cancelled]")));

        let mut app = plain_app();
        app.conversation.push(Message::new(Role::User, "hi"));
        app.conversation.push(Message::new(Role::Assistant, ""));
        app.reply_ends.insert(1, ReplyEnd::Empty);
        let buffer = render_buffer(&app, 40, 8);
        // Three lines in four rows: the short transcript starts at the top.
        assert_eq!(rows(&buffer)[2].trim_end(), "● (empty reply)");
        assert!(is_dim(&buffer[(0, 2)]) && is_dim(&buffer[(2, 2)]));

        app.reply_ends.insert(1, ReplyEnd::Cancelled);
        assert_eq!(
            rows(&render_buffer(&app, 40, 8))[2].trim_end(),
            "● [cancelled]"
        );
    }

    #[test]
    fn code_fences_draw_as_a_dim_bar() {
        let mut app = plain_app();
        app.conversation.push(Message::new(
            Role::Assistant,
            "Run this:\n```rust\nfn main() {}\n```\nDone.",
        ));
        let buffer = render_buffer(&app, 40, 10);
        let screen = rows(&buffer);
        let lines: Vec<&str> = screen[..5].iter().map(|r| r.trim_end()).collect();
        assert_eq!(
            lines,
            ["● Run this:", "│", "│ fn main() {}", "│", "  Done."]
        );
        assert!(!screen.join("\n").contains("rust"), "language tag dropped");
        assert!(is_dim(&buffer[(0, 1)]) && is_dim(&buffer[(0, 2)]));
        assert!(!is_dim(&buffer[(2, 2)]), "the code itself is not");
    }

    // --- tool calls ---------------------------------------------------------

    #[test]
    fn tool_calls_show_running_then_their_outcome() {
        let mut app = plain_app();
        let call = |id: &str, args: &str| ToolCall {
            id: id.into(),
            name: "read_file".into(),
            arguments: args.into(),
        };
        app.conversation.push(Message::assistant_with_tool_calls(
            "",
            vec![
                call("a", r#"{"path":"README.md"}"#),
                call("b", r#"{"path":"../x"}"#),
            ],
        ));
        app.conversation
            .push(Message::tool_result("a", "1: # loops"));
        let buffer = render_buffer(&app, 60, 10);
        let screen = rows(&buffer);
        assert_eq!(screen[0].trim_end(), "  ● read_file(README.md)");
        assert_eq!(screen[1].trim_end(), "    ⎿ ⠋ running");
        assert!(is_dim(&buffer[(4, 1)]) && is_dim(&buffer[(6, 1)]));
        assert!(
            !screen.join("\n").contains("# loops"),
            "results are not drawn"
        );
        // Bullet and name magenta, arguments in the default color.
        assert_eq!(buffer[(2, 0)].fg, Color::Magenta);
        assert_eq!(buffer[(4, 0)].fg, Color::Magenta);
        let (x, y) = find(&screen, "README");
        assert_eq!(buffer[(x, y)].fg, Color::Reset);

        app.tool_outcomes.insert(
            "a".into(),
            ToolOutcome {
                summary: "31 lines".into(),
                is_error: false,
            },
        );
        app.tool_outcomes.insert(
            "b".into(),
            ToolOutcome {
                summary: "../x is outside the working directory\nmore".into(),
                is_error: true,
            },
        );
        let buffer = render_buffer(&app, 70, 10);
        let screen = rows(&buffer);
        assert_eq!(screen[1].trim_end(), "    ⎿ 31 lines");
        assert!(is_dim(&buffer[(6, 1)]));
        assert_eq!(screen[2].trim_end(), "  ● read_file(../x)");
        assert_eq!(
            screen[3].trim_end(),
            "    ⎿ error: ../x is outside the working directory"
        );
        assert_eq!(buffer[(4, 3)].fg, Color::Red);
        assert_eq!(buffer[(6, 3)].fg, Color::Red);
    }

    /// An app showing one finished `bash` call whose output is `body` after
    /// its exit code line.
    fn ran_command(body: &[String]) -> App {
        let mut app = plain_app();
        let call = ToolCall {
            id: "c1".into(),
            name: "bash".into(),
            arguments: r#"{"command":"ls"}"#.into(),
        };
        app.conversation
            .push(Message::assistant_with_tool_calls("", vec![call]));
        let summary = format!("exit 0, {} lines", body.len());
        let outcome = ToolOutcome {
            summary,
            is_error: false,
        };
        app.tool_outcomes.insert("c1".into(), outcome);
        let output = std::iter::once("exit code: 0".to_owned())
            .chain(body.iter().cloned())
            .collect::<Vec<_>>()
            .join("\n");
        app.tool_outputs.insert("c1".into(), output);
        app
    }

    fn numbered(n: usize) -> Vec<String> {
        (1..=n).map(|i| format!("line {i}")).collect()
    }

    #[test]
    fn a_command_shows_its_output_dim_under_its_outcome() {
        let app = ran_command(&numbered(3));
        let buffer = render_buffer(&app, 60, 12);
        let screen = rows(&buffer);
        assert_eq!(screen[0].trim_end(), r#"  ● bash("ls")"#);
        assert_eq!(screen[1].trim_end(), "    ⎿ exit 0, 3 lines");
        assert_eq!(screen[2].trim_end(), "      line 1");
        assert_eq!(screen[3].trim_end(), "      line 2");
        assert_eq!(screen[4].trim_end(), "      line 3");
        assert!(is_dim(&buffer[(6, 2)]) && is_dim(&buffer[(6, 4)]));
        assert!(!screen.join("\n").contains("exit code"));
    }

    #[test]
    fn long_output_folds_to_its_ends_until_ctrl_o() {
        let mut app = ran_command(&numbered(20));
        let screen = rows(&render_buffer(&app, 60, 20));
        let shown: Vec<&str> = screen[2..11].iter().map(|r| r.trim()).collect();
        assert_eq!(
            shown,
            [
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "… 12 lines omitted …",
                "line 17",
                "line 18",
                "line 19",
                "line 20",
            ]
        );
        assert_eq!(screen[11].trim(), "");

        press_ctrl_o(&mut app);
        assert!(app.full_output);
        let screen = rows(&render_buffer(&app, 60, 30));
        let shown: Vec<&str> = screen[2..22].iter().map(|r| r.trim()).collect();
        assert_eq!(shown, numbered(20));
        assert!(!screen.join("\n").contains("omitted"));
    }

    fn press_ctrl_o(app: &mut App) {
        let ctrl_o = KeyEvent::new(KeyCode::Char('o'), crossterm::event::KeyModifiers::CONTROL);
        app.update(crate::app::AppEvent::Key(ctrl_o));
    }

    #[test]
    fn output_of_eight_lines_or_fewer_is_not_folded() {
        let eight = numbered(8);
        let lines: Vec<&str> = eight.iter().map(String::as_str).collect();
        assert_eq!(fold_output(&lines, false), eight);
        assert_eq!(output_lines("exit code: 0", false), Vec::<String>::new());
        assert_eq!(fold_output(&[], true), Vec::<String>::new());
    }

    /// A failed command shows its error on the outcome line and nothing more;
    /// one that ran shows what it printed.
    #[tokio::test]
    async fn a_failed_command_shows_no_output() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "go");
        let command = |id: &str, is_error: bool, summary: &str, output: &str| {
            [
                AgentEvent::ToolStarted {
                    id: id.into(),
                    name: "bash".into(),
                    arguments: r#"{"command":"make"}"#.into(),
                },
                AgentEvent::ToolFinished {
                    id: id.into(),
                    name: "bash".into(),
                    summary: summary.into(),
                    is_error,
                    output: output.into(),
                },
            ]
        };
        let events = command("c1", true, "no make here", "error: no make here\nhidden")
            .into_iter()
            .chain(command(
                "c2",
                false,
                "exit 0, 1 line",
                "exit code: 0\nbuilt",
            ));
        for event in events {
            app.update(agent_event(&app, event));
        }
        let screen = rows(&render_buffer(&app, 60, 16)).join("\n");
        assert!(screen.contains("⎿ error: no make here"), "{screen}");
        assert!(!screen.contains("hidden"), "{screen}");
        assert!(screen.contains("      built"), "{screen}");
    }

    #[tokio::test]
    async fn a_turn_with_a_tool_shows_the_tool_line_and_the_answer() {
        let read = ChatDelta::ToolCall {
            index: 0,
            id: Some("call_1".into()),
            name: Some("read_file".into()),
            arguments: r#"{"path":"Cargo.toml"}"#.into(),
        };
        let script = vec![
            vec![read],
            vec![ChatDelta::Content("It is loops-tui.".into())],
        ];
        let (mut app, _backend, mut rx) = chat_app(script);
        type_line(&mut app, "which crate?");
        while matches!(app.model_status, ModelStatus::Busy(_)) {
            app.update(rx.recv().await.unwrap());
        }

        let screen = rows(&render_buffer(&app, 60, 10));
        assert_eq!(screen[0].trim_end(), "❯ which crate?");
        assert_eq!(screen[2].trim_end(), "  ● read_file(Cargo.toml)");
        assert!(screen[3].starts_with("    ⎿ "), "{}", screen[3]);
        assert!(screen[3].trim_end().ends_with(" lines"));
        assert_eq!(screen[4].trim_end(), "  It is loops-tui.");
    }

    #[test]
    fn a_previewed_call_shows_its_change_and_a_declined_one_is_yellow() {
        let mut app = plain_app();
        let call = |id: &str| ToolCall {
            id: id.into(),
            name: "edit_file".into(),
            arguments: r#"{"path":"a.txt"}"#.into(),
        };
        app.conversation.push(Message::assistant_with_tool_calls(
            "",
            vec![call("a"), call("b")],
        ));
        app.tool_previews
            .insert("a".into(), "edit a.txt at line 3\n- old\n+ new".into());
        app.tool_outcomes.insert(
            "a".into(),
            ToolOutcome {
                summary: "replaced 1 line with 1 at line 3".into(),
                is_error: false,
            },
        );
        app.tool_outcomes.insert(
            "b".into(),
            ToolOutcome {
                summary: DECLINED.into(),
                is_error: true,
            },
        );

        let buffer = render_buffer(&app, 60, 12);
        let screen = rows(&buffer);
        let lines: Vec<&str> = screen[..7].iter().map(|r| r.trim_end()).collect();
        assert_eq!(
            lines,
            [
                "  ● edit_file(a.txt)",
                "    ⎿ replaced 1 line with 1 at line 3",
                "      edit a.txt at line 3",
                "      - old",
                "      + new",
                "  ● edit_file(a.txt)",
                "    ⎿ declined",
            ]
        );
        assert!(is_dim(&buffer[(6, 2)]));
        assert_eq!(buffer[(6, 3)].fg, Color::Red);
        assert_eq!(buffer[(6, 4)].fg, Color::Green);
        assert_eq!(buffer[(6, 6)].fg, Color::Yellow);
    }

    #[test]
    fn a_command_shows_on_its_tool_line_cut_when_long() {
        let mut app = plain_app();
        let long = format!("echo {}", "x".repeat(100));
        let call = |id: &str, command: &str| ToolCall {
            id: id.into(),
            name: "bash".into(),
            arguments: serde_json::json!({ "command": command, "timeout_seconds": 30 }).to_string(),
        };
        app.conversation.push(Message::assistant_with_tool_calls(
            "",
            vec![call("a", "cargo test"), call("b", &long)],
        ));
        let screen = rows(&render_buffer(&app, 100, 10));
        assert_eq!(screen[0].trim_end(), r#"  ● bash("cargo test")"#);
        let cut: String = long.chars().take(60).collect();
        assert_eq!(screen[2].trim_end(), format!(r#"  ● bash("{cut}…")"#));
    }

    // --- permission prompt --------------------------------------------------

    /// The permission prompt for call `c1` of `edit_file`, row `selected` highlighted.
    fn permission(selected: usize) -> Mode {
        Mode::Permission {
            id: "c1".into(),
            name: "edit_file".into(),
            scope: "edit_file".into(),
            selected,
        }
    }

    #[test]
    fn a_command_prompt_names_the_exact_command() {
        let mut app = plain_app();
        app.mode = Mode::Permission {
            id: "c1".into(),
            name: "bash".into(),
            scope: "bash: cargo test".into(),
            selected: 0,
        };
        let screen = rows(&render_buffer(&app, 80, 12));
        assert!(screen[6].contains(" allow bash? "), "{}", screen[6]);
        assert!(
            screen[8].contains("Yes, and don't ask again for bash: cargo test this run"),
            "{}",
            screen[8]
        );
    }

    #[test]
    fn the_permission_prompt_replaces_the_input_box() {
        let mut app = plain_app();
        app.model_status = ModelStatus::Busy("google/gemma-4-e4b".into());
        app.input.insert_str("draft");
        app.mode = permission(0);
        let labels = PermissionOption::ALL.map(|option| option.label("edit_file"));

        let buffer = render_buffer(&app, 80, 12);
        let screen = rows(&buffer);
        // Rows 6..11: rounded yellow border with the title, three options, border with the keys.
        assert!(screen[6].starts_with('╭'), "{}", screen[6]);
        assert!(screen[6].contains(" allow edit_file? "), "{}", screen[6]);
        assert_eq!(buffer[(0, 6)].fg, Color::Yellow);
        assert_eq!(cell_at(&buffer, "allow edit_file").fg, Color::Yellow);
        assert!(
            screen[7].contains(&format!("❯ {}", labels[0])),
            "{}",
            screen[7]
        );
        assert!(
            screen[8].contains(&format!("  {}", labels[1])),
            "{}",
            screen[8]
        );
        assert!(
            screen[9].contains(&format!("  {}", labels[2])),
            "{}",
            screen[9]
        );
        assert!(screen[10].starts_with('╰'));
        assert!(screen[10].contains("↑↓ move   Enter select   Esc cancel turn"));
        assert!(
            !screen.join("\n").contains("draft"),
            "the input box is not drawn"
        );
        // The status bar waits on the user, in yellow.
        assert!(screen[11].starts_with(" google/gemma-4-e4b waiting for you"));
        assert_eq!(buffer[(1, 11)].fg, Color::Yellow);

        // The selected row is bold and highlighted like the picker's; the others dim.
        let (x, y) = find(&screen, "❯ Yes");
        let cell = &buffer[(x, y)];
        assert!(cell.modifier.contains(Modifier::BOLD | Modifier::REVERSED));
        let (x, y) = find(&screen, &labels[2]);
        assert!(is_dim(&buffer[(x, y)]));
        assert!(!buffer[(x, y)].modifier.contains(Modifier::REVERSED));

        // The marker follows `selected`.
        app.mode = permission(2);
        let screen = rows(&render_buffer(&app, 80, 12));
        assert!(
            screen[7].contains(&format!("  {}", labels[0])),
            "{}",
            screen[7]
        );
        assert!(
            screen[9].contains(&format!("❯ {}", labels[2])),
            "{}",
            screen[9]
        );
        assert_eq!(screen[6..11].join("\n").matches('❯').count(), 1);

        // Back to normal: the three-row input box, with its text, and no prompt.
        app.mode = Mode::Normal;
        let screen = rows(&render_buffer(&app, 80, 12));
        assert!(screen[8].starts_with('╭'), "{}", screen[8]);
        assert!(screen[9].contains("draft"), "{}", screen[9]);
        assert!(screen[10].starts_with('╰'), "{}", screen[10]);
        assert!(!screen.join("\n").contains("allow edit_file?"));
    }

    #[test]
    fn the_change_asked_about_stays_in_view_above_the_prompt() {
        let mut app = plain_app();
        for i in 0..20 {
            app.conversation
                .push(Message::new(Role::User, format!("msg {i}")));
        }
        app.conversation.push(Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "edit_file".into(),
                arguments: r#"{"path":"a.txt"}"#.into(),
            }],
        ));
        app.tool_previews
            .insert("c1".into(), "edit a.txt at line 3\n- old\n+ new".into());
        app.mode = permission(0);

        // Twelve rows: six of transcript, five of prompt, the status bar.
        let screen = rows(&render_buffer(&app, 80, 12));
        let lines: Vec<&str> = screen[..6].iter().map(|r| r.trim_end()).collect();
        assert_eq!(
            lines,
            [
                "",
                "  ● edit_file(a.txt)",
                "    ⎿ ⠋ running",
                "      edit a.txt at line 3",
                "      - old",
                "      + new",
            ]
        );
        assert!(screen[6].contains("allow edit_file?"));
    }

    // --- status bar ---------------------------------------------------------

    #[test]
    fn the_status_bar_is_the_bottom_row() {
        let mut app = plain_app();
        let buffer = render_buffer(&app, 40, 8);
        assert_eq!(rows(&buffer)[7].trim_end(), " starting lm studio… · /help");
        assert!(is_dim(&buffer[(1, 7)]));
        assert!(is_dim(cell_at(&buffer, "·")), "dim separators");

        app.model_status = ModelStatus::NoModel;
        assert_eq!(
            rows(&render_buffer(&app, 40, 8))[7].trim_end(),
            " no model · /help"
        );

        app.model_status = ModelStatus::Error("lms not found".into());
        let buffer = render_buffer(&app, 40, 8);
        assert!(rows(&buffer)[7].starts_with(" error: lms not found"));
        assert_eq!(buffer[(1, 7)].fg, Color::Red);
    }

    #[test]
    fn ready_status_is_green_and_uses_short_id_when_narrow() {
        let mut app = plain_app();
        app.model_status = ModelStatus::Ready("google/gemma-4-e4b".into());

        let wide = render_buffer(&app, 60, 8);
        assert_eq!(rows(&wide)[7].trim_end(), " google/gemma-4-e4b ✓ · /help");
        assert_eq!(wide[(1, 7)].fg, Color::Green);

        let narrow = rows(&render_buffer(&app, 20, 8));
        assert_eq!(narrow[7].trim_end(), " gemma-4-e4b ✓");
    }

    #[test]
    fn loading_unloading_and_replying_spin() {
        let mut app = plain_app();
        app.model_status = ModelStatus::Unloading("google/gemma-4-e4b".into());
        let buffer = render_buffer(&app, 60, 8);
        assert!(rows(&buffer)[7].starts_with(" ⠋ unloading google/gemma-4-e4b"));
        assert_eq!(buffer[(1, 7)].fg, Color::Yellow);

        app.model_status = ModelStatus::Loading("google/gemma-4-e4b".into());
        let buffer = render_buffer(&app, 60, 8);
        assert!(rows(&buffer)[7].starts_with(" ⠋ loading google/gemma-4-e4b"));
        assert_eq!(buffer[(1, 7)].fg, Color::Yellow);

        app.model_status = ModelStatus::Busy("google/gemma-4-e4b".into());
        let buffer = render_buffer(&app, 60, 8);
        assert!(rows(&buffer)[7].starts_with(" ⠋ google/gemma-4-e4b replying"));
        assert_eq!(buffer[(1, 7)].fg, Color::Magenta);
    }

    /// `/compact` shows in the status bar, the input box is busy, and Esc
    /// is offered.
    #[tokio::test]
    async fn compacting_spins_in_the_status_bar() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        app.conversation.push(Message::new(Role::User, "hi"));
        type_line(&mut app, "/compact");
        let buffer = render_buffer(&app, 60, 8);
        assert_eq!(
            rows(&buffer)[7].trim_end(),
            " ⠋ google/gemma-4-e4b compacting… · Esc cancel · /help"
        );
        assert_eq!(buffer[(1, 7)].fg, Color::Magenta);
        assert_eq!(buffer[(0, 4)].fg, Color::Magenta, "busy input box");
    }

    /// An app with a ready model, a window of 1000 tokens, and `used` of them used.
    fn context_app(used: u64) -> App {
        let mut app = plain_app();
        app.model_status = ModelStatus::Ready("google/gemma-4-e4b".into());
        app.context_length = Some(1000);
        app.usage = Some(Usage {
            prompt_tokens: used - 10,
            completion_tokens: 10,
        });
        app
    }

    #[test]
    fn context_is_dim_then_yellow_then_red() {
        for (used, color, dim) in [
            (190, Color::Reset, true),
            (850, Color::Yellow, false),
            (970, Color::Red, false),
        ] {
            let app = context_app(used);
            let buffer = render_buffer(&app, 60, 8);
            let percent = format!("ctx {}%", used / 10);
            assert!(rows(&buffer)[7].contains(&format!(" · {percent} · /help")));
            let cell = cell_at(&buffer, &percent);
            assert_eq!(cell.fg, color, "{percent}");
            assert_eq!(is_dim(cell), dim, "{percent}");
        }

        // Unknown usage or window: no context at all.
        let mut app = context_app(190);
        app.usage = None;
        assert!(!rows(&render_buffer(&app, 60, 8))[7].contains("ctx"));
        let mut app = context_app(190);
        app.context_length = None;
        assert!(!rows(&render_buffer(&app, 60, 8))[7].contains("ctx"));
    }

    #[tokio::test]
    async fn hints_go_first_when_the_row_is_narrow() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        app.context_length = Some(1000);
        app.usage = Some(Usage {
            prompt_tokens: 190,
            completion_tokens: 0,
        });
        type_line(&mut app, "hi");
        let status = |width| {
            rows(&render_buffer(&app, width, 8))[7]
                .trim_end()
                .to_owned()
        };
        assert_eq!(
            status(80),
            " ⠋ google/gemma-4-e4b replying · ctx 19% · Esc cancel · /help"
        );
        assert_eq!(
            status(55),
            " ⠋ google/gemma-4-e4b replying · ctx 19% · Esc cancel"
        );
        assert_eq!(status(45), " ⠋ google/gemma-4-e4b replying · ctx 19%");
        assert_eq!(status(40), " ⠋ google/gemma-4-e4b replying · ctx 19%");
        assert_eq!(status(39), " ⠋ gemma-4-e4b replying · ctx 19%");
        assert_eq!(status(25), " ⠋ gemma-4-e4b replying");
    }

    // --- scrolling ----------------------------------------------------------

    /// An app with 30 user messages, `msg 0` to `msg 29`.
    fn long_app() -> App {
        let mut app = plain_app();
        for i in 0..30 {
            app.conversation
                .push(Message::new(Role::User, format!("msg {i}")));
        }
        app
    }

    #[test]
    fn page_up_scrolls_clamped_at_the_top_and_end_returns() {
        let mut app = long_app();
        // 40x12: eight transcript rows, so the bottom shows msg 26..29.
        let screen = rows(&render_buffer(&app, 40, 12));
        assert!(screen[7].starts_with("❯ msg 29"));

        press(&mut app, KeyCode::PageUp);
        let screen = rows(&render_buffer(&app, 40, 12));
        assert!(screen[7].starts_with("❯ msg 24"), "{}", screen[7]);

        // Far past the top: the first message on the first row, no blank space.
        for _ in 0..20 {
            press(&mut app, KeyCode::PageUp);
        }
        let screen = rows(&render_buffer(&app, 40, 12));
        assert!(screen[0].starts_with("❯ msg 0"), "{}", screen[0]);
        assert!(screen[6].starts_with("❯ msg 3"), "{}", screen[6]);

        press(&mut app, KeyCode::End);
        let screen = rows(&render_buffer(&app, 40, 12));
        assert!(screen[7].starts_with("❯ msg 29"));
    }

    #[tokio::test]
    async fn new_output_is_flagged_only_when_scrolled_up_during_a_turn() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        for i in 0..30 {
            app.conversation
                .push(Message::new(Role::User, format!("msg {i}")));
        }
        app.scroll_up = 10;
        assert!(!render(&app).contains("new output"), "no turn running");

        type_line(&mut app, "go");
        assert_eq!(app.scroll_up, 0, "sending jumps to the bottom");
        assert!(!render(&app).contains("new output"), "at the bottom");

        press(&mut app, KeyCode::PageUp);
        let buffer = render_buffer(&app, 40, 8);
        let screen = rows(&buffer);
        // Right-aligned on the transcript's bottom row, above the input box.
        assert!(screen[3].ends_with("↓ new output (End)"), "{}", screen[3]);
        assert!(is_dim(cell_at(&buffer, "↓ new output")));

        press(&mut app, KeyCode::End);
        assert!(!render(&app).contains("new output"));
    }

    #[tokio::test]
    async fn no_new_output_flag_when_everything_fits() {
        let (mut app, _backend, _rx) = chat_app(Vec::new());
        type_line(&mut app, "go");
        press(&mut app, KeyCode::PageUp);
        assert!(!render(&app).contains("new output"), "nothing to scroll");
    }

    // --- input box ----------------------------------------------------------

    #[test]
    fn the_input_box_shows_a_placeholder_until_typed_in() {
        let mut app = plain_app();
        app.model_status = ModelStatus::NoModel;
        let buffer = render_buffer(&app, 40, 8);
        let screen = rows(&buffer);
        assert!(screen[4].starts_with("╭ ❯ ─"), "{}", screen[4]);
        assert!(screen[5].contains("Message loops…"), "{}", screen[5]);
        assert!(is_dim(cell_at(&buffer, "Message")));
        assert!(screen[6].starts_with('╰'));

        press(&mut app, KeyCode::Char('h'));
        let screen = rows(&render_buffer(&app, 40, 8));
        assert!(!screen[5].contains("Message loops"));
        assert!(screen[5].starts_with("│h"), "{}", screen[5]);
    }

    #[test]
    fn the_input_border_tells_the_state() {
        let mut app = plain_app();
        let border = |app: &App| render_buffer(app, 40, 8)[(0, 4)].fg;
        assert_eq!(border(&app), Color::Cyan, "starting");
        app.model_status = ModelStatus::Ready("google/gemma-4-e4b".into());
        assert_eq!(border(&app), Color::Cyan);
        app.model_status = ModelStatus::Busy("google/gemma-4-e4b".into());
        assert_eq!(border(&app), Color::Magenta);
        app.model_status = ModelStatus::Error("boom".into());
        assert_eq!(border(&app), Color::Red);
        // The title follows the border.
        assert_eq!(render_buffer(&app, 40, 8)[(2, 4)].fg, Color::Red);
    }

    // --- popups -------------------------------------------------------------

    #[test]
    fn picker_lists_models_and_highlights_the_selection() {
        let mut app = plain_app();
        app.mode = Mode::Picker {
            models: vec![
                model("qwen/qwen3.5-9b", ModelState::Loaded),
                model("google/gemma-4-e4b", ModelState::NotLoaded),
            ],
            selected: 1,
        };
        let buffer = render_buffer(&app, 70, 12);
        let rows = rows(&buffer);
        let screen = rows.join("\n");
        assert!(screen.contains("load a model"));
        assert!(screen.contains("╭"), "rounded");

        // A dim header over the columns.
        let (x, y) = find(&rows, "quant");
        assert!(rows[usize::from(y)].contains(" model "));
        assert!(rows[usize::from(y)].contains("context  state"));
        assert!(is_dim(&buffer[(x, y)]));

        // The context column: loaded size if loaded, the max otherwise.
        let qwen = rows.iter().find(|r| r.contains("qwen/qwen3.5-9b")).unwrap();
        assert!(qwen.contains("Q4_K_M       8k  loaded"), "{qwen}");
        let gemma = rows.iter().find(|r| r.contains("gemma-4-e4b")).unwrap();
        assert!(gemma.contains("262k"), "{gemma}");
        assert_eq!(cell_at(&buffer, "loaded").fg, Color::Green);

        let reversed = |id: &str| cell_at(&buffer, id).modifier.contains(Modifier::REVERSED);
        assert!(
            reversed("google/gemma-4-e4b"),
            "selected row is highlighted"
        );
        assert!(!reversed("qwen/qwen3.5-9b"), "other rows are not");
    }

    #[test]
    fn sessions_list_titles_ages_and_highlights_the_selection() {
        let mut app = plain_app();
        let now = unix_now();
        let entry = |id: &str, title: &str, ago: u64, messages: usize| SessionSummary {
            id: id.into(),
            title: title.into(),
            updated: now - ago,
            messages,
        };
        app.mode = Mode::Sessions {
            entries: vec![
                entry("3", "fix the parser", 10, 4),
                entry("2", "", 7_200, 1),
                entry("1", "add a --resume flag", 3 * 86_400, 12),
            ],
            selected: 1,
        };
        let buffer = render_buffer(&app, 70, 12);
        let rows = rows(&buffer);
        let screen = rows.join("\n");
        assert!(screen.contains("resume a session"));
        assert!(screen.contains("Enter select"), "the picker's footer");

        let row = |text: &str| rows.iter().find(|r| r.contains(text)).unwrap().clone();
        assert!(
            row("fix the parser").contains("fix the parser       just now  4 messages"),
            "{}",
            row("fix the parser")
        );
        assert!(row("(untitled)").contains("2 h ago  1 message "));
        assert!(row("add a --resume").contains("3 d ago  12 messages"));
        assert!(is_dim(cell_at(&buffer, "3 d ago")));

        let reversed = |text: &str| cell_at(&buffer, text).modifier.contains(Modifier::REVERSED);
        assert!(reversed("(untitled)"), "selected row is highlighted");
        assert!(!reversed("fix the parser"), "other rows are not");
    }

    #[test]
    fn confirm_asks_before_loading() {
        let mut app = plain_app();
        app.mode = Mode::Confirm {
            models: vec![model("google/gemma-4-e4b", ModelState::NotLoaded)],
            selected: 0,
        };
        let screen = rows(&render_buffer(&app, 60, 12)).join("\n");
        assert!(screen.contains("Load google/gemma-4-e4b? (y/n)"));
    }

    #[test]
    fn nothing_breaks_at_the_narrowest_width() {
        let mut app = long_app();
        app.model_status = ModelStatus::Busy("google/gemma-4-e4b".into());
        app.conversation.push(Message::new(
            Role::Assistant,
            "a long line that will certainly wrap at forty columns wide",
        ));
        let screen = rows(&render_buffer(&app, 40, 10));
        assert!(screen[9].starts_with(" ⠋"), "{}", screen[9]);
        let app = plain_app();
        let screen = rows(&render_buffer(&app, 40, 10));
        assert!(screen[1].contains("loops  v0.1.0"));
    }
}
