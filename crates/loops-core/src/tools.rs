//! Tools the model can call, and the five the harness ships.
//!
//! A [`Tool`] describes itself with a [`ToolSpec`] (name, description, JSON
//! schema of its arguments) and runs on the raw JSON arguments string the model
//! produced. A [`ToolSet`] is the list handed to the agent loop.
//!
//! Two shipped tools only read (`read_file`, `list_files`); two change files
//! (`edit_file`, `write_file`); one runs shell commands (`bash`). A tool that
//! changes things says so with [`Tool::mutates`] and describes the change
//! before making it with [`Tool::preview`], so the agent can show the user
//! exactly what would happen and run it only after a yes.
//!
//! Every shipped tool works inside one root directory (the directory the
//! harness started in). The file tools refuse paths that resolve outside it.
//! `bash` starts its commands there but cannot be confined: a shell can `cd`
//! anywhere. That is why each command is shown and approved on its own, and
//! why "don't ask again" for it is keyed by the exact command
//! ([`Tool::permission_scope`]), not by the tool.

use std::{
    path::{Component, Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::backend::BoxFuture;

/// Most file lines `read_file` returns.
const READ_MAX_LINES: usize = 400;
/// Most bytes `read_file` returns (line-number prefixes included).
const READ_MAX_BYTES: usize = 64 * 1024;
/// Starts the line `read_file` appends when it cuts a file short.
const TRUNCATED: &str = "[truncated:";
/// Directory entries `list_files` leaves out: big, and noise to the model.
const LIST_SKIP: [&str; 2] = [".git", "target"];
/// Most content lines a `write_file` preview shows.
const PREVIEW_MAX_LINES: usize = 30;
/// Seconds a `bash` command may run when the model does not say.
const BASH_DEFAULT_TIMEOUT: u64 = 120;
/// Most seconds a `bash` command may run, whatever the model asks for.
const BASH_MAX_TIMEOUT: u64 = 600;
/// Lines kept from each end of a `bash` stream longer than twice this.
const BASH_KEEP_LINES: usize = 150;
/// Most bytes of one `bash` stream returned; half from each end.
const BASH_MAX_BYTES: usize = 64 * 1024;
/// The line between a `bash` command's stdout and its stderr.
const STDERR_HEADER: &str = "--- stderr ---";
/// Starts the last line of a `bash` result whose command ran out of time.
const TIMED_OUT: &str = "[timed out after ";
/// Most bytes of one `bash` stream held while it runs (see [`read_all`]).
const BASH_READ_CAP: usize = 4 * 1024 * 1024;

/// What the model is told about a tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    /// The name the model calls it by, e.g. `read_file`.
    pub name: String,
    /// When and why to use it, in a sentence or two.
    pub description: String,
    /// JSON schema of the arguments object.
    pub parameters: Value,
}

/// Something the model can ask the harness to do.
pub trait Tool: Send + Sync {
    /// Name, description and argument schema.
    fn spec(&self) -> ToolSpec;

    /// Runs the tool on `arguments`, the raw JSON string from the model. An
    /// error is reported to the model as the result; it does not end the turn.
    fn run(&self, arguments: &str) -> BoxFuture<'_, Result<String>>;

    /// A few words about a successful `output`, for the transcript.
    fn summarize(&self, output: &str) -> String {
        format!("{} lines", output.lines().count())
    }

    /// Whether running the tool changes anything (a file, the system). The
    /// agent asks the user before running a tool that does.
    fn mutates(&self) -> bool {
        false
    }

    /// For a mutating tool: what `run` would do with `arguments`, exactly and
    /// readably, worked out without doing it. The agent shows this to the user
    /// before asking. An error means the call cannot succeed (bad arguments, a
    /// path outside the root, text not found); the agent hands it to the model
    /// as the result without asking, so nobody approves a change that would
    /// fail anyway. The default shows the arguments as they came.
    fn preview(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let preview = arguments.to_owned();
        Box::pin(async move { Ok(preview) })
    }

    /// What a "yes, don't ask again" to this call covers: a front end that
    /// remembers the answer keys it by this text. The default is the tool's
    /// name, so one yes covers every later call of the tool. A tool whose
    /// calls differ too much for that (`bash`: `cargo test` is not `rm -rf`)
    /// narrows it. An error means the arguments are unusable, like a
    /// [`Tool::preview`] error.
    fn permission_scope(&self, arguments: &str) -> Result<String> {
        let _ = arguments;
        Ok(self.spec().name)
    }
}

/// The tools offered to the model for a turn.
#[derive(Clone, Default)]
pub struct ToolSet {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolSet {
    /// A set of exactly `tools`.
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self { tools }
    }

    /// `read_file` and `list_files`, both confined to `root`.
    pub fn read_only(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self::new(vec![
            Arc::new(ReadFile { root: root.clone() }),
            Arc::new(ListFiles { root }),
        ])
    }

    /// The read-only tools plus `edit_file` and `write_file`, all confined to
    /// `root`, and `bash` starting in `root` if [`find_bash`] finds a shell.
    pub fn standard(root: impl Into<PathBuf>) -> Self {
        Self::standard_with_shell(root, find_bash())
    }

    /// [`ToolSet::standard`] with `bash` running `shell`, or without `bash`
    /// when `shell` is `None`.
    pub fn standard_with_shell(root: impl Into<PathBuf>, shell: Option<PathBuf>) -> Self {
        let root = root.into();
        let mut tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(ReadFile { root: root.clone() }),
            Arc::new(ListFiles { root: root.clone() }),
            Arc::new(EditFile { root: root.clone() }),
            Arc::new(WriteFile { root: root.clone() }),
        ];
        if let Some(shell) = shell {
            tools.push(Arc::new(Bash { root, shell }));
        }
        Self::new(tools)
    }

    /// Whether the set has a tool called `name`.
    pub fn has(&self, name: &str) -> bool {
        self.find(name).is_ok()
    }

    /// The specs of every tool, in order.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|tool| tool.spec()).collect()
    }

    /// Runs the tool called `name`. An unknown name is an error like any other,
    /// so the model reads "unknown tool" as the result instead of the turn failing.
    pub async fn run(&self, name: &str, arguments: &str) -> Result<String> {
        self.find(name)?.run(arguments).await
    }

    /// The tool's summary of a successful `output`.
    pub fn summarize(&self, name: &str, output: &str) -> String {
        match self.find(name) {
            Ok(tool) => tool.summarize(output),
            Err(_) => format!("{} lines", output.lines().count()),
        }
    }

    /// Whether the tool called `name` changes anything. False for an unknown
    /// name: running one only produces the "unknown tool" error.
    pub fn mutates(&self, name: &str) -> bool {
        self.find(name).is_ok_and(|tool| tool.mutates())
    }

    /// The tool's [`Tool::preview`] of a call with `arguments`.
    pub async fn preview(&self, name: &str, arguments: &str) -> Result<String> {
        self.find(name)?.preview(arguments).await
    }

    /// The tool's [`Tool::permission_scope`] of a call with `arguments`.
    pub fn permission_scope(&self, name: &str, arguments: &str) -> Result<String> {
        self.find(name)?.permission_scope(arguments)
    }

    fn find(&self, name: &str) -> Result<&Arc<dyn Tool>> {
        self.tools
            .iter()
            .find(|tool| tool.spec().name == name)
            .ok_or_else(|| anyhow!("unknown tool: {name}"))
    }
}

/// `read_file { path }`: a text file with `N: ` line numbers, capped in size.
pub struct ReadFile {
    root: PathBuf,
}

/// Arguments of `read_file`.
#[derive(Deserialize)]
struct ReadArgs {
    path: String,
}

impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: format!(
                "Read a text file in the working directory. Lines are prefixed with \
                 their numbers; at most {READ_MAX_LINES} lines are returned."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path, relative to the working directory" }
                },
                "required": ["path"]
            }),
        }
    }

    fn run(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let args = parse_args::<ReadArgs>(arguments);
        Box::pin(async move {
            let (_, text) = read_text(&self.root, &args?.path).await?;
            Ok(numbered(&text))
        })
    }

    fn summarize(&self, output: &str) -> String {
        let shown = output.lines().filter(|l| !l.starts_with(TRUNCATED)).count();
        let cut = if output.contains(TRUNCATED) {
            " (truncated)"
        } else {
            ""
        };
        format!("{shown} lines{cut}")
    }
}

/// `text` as `N: line` rows, cut at [`READ_MAX_LINES`] or [`READ_MAX_BYTES`]
/// with a closing note saying how much was shown.
fn numbered(text: &str) -> String {
    let total = text.lines().count();
    let mut out = String::new();
    let mut shown = 0;
    for (i, line) in text.lines().enumerate() {
        let row = format!("{}: {line}\n", i + 1);
        if shown == READ_MAX_LINES || out.len() + row.len() > READ_MAX_BYTES {
            break;
        }
        out.push_str(&row);
        shown += 1;
    }
    if shown < total {
        out.push_str(&format!(
            "{TRUNCATED} showing first {shown} of {total} lines]\n"
        ));
    }
    out
}

/// `list_files { path? }`: one directory level, sorted, directories end in `/`.
pub struct ListFiles {
    root: PathBuf,
}

/// Arguments of `list_files`.
#[derive(Deserialize)]
struct ListArgs {
    #[serde(default = "here")]
    path: String,
}

/// The default `list_files` path: the working directory itself.
fn here() -> String {
    ".".into()
}

impl Tool for ListFiles {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_files".into(),
            description: "List one directory level in the working directory. \
                          Directories end in `/`; .git and target are skipped."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory, relative to the working directory. Default \".\"" }
                }
            }),
        }
    }

    fn run(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let args = parse_args::<ListArgs>(arguments);
        Box::pin(async move {
            let path = args?.path;
            let dir = resolve(&self.root, &path).await?;
            let mut entries = tokio::fs::read_dir(&dir)
                .await
                .with_context(|| format!("listing {path}"))?;
            let mut names = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                let name = entry.file_name().to_string_lossy().into_owned();
                if LIST_SKIP.contains(&name.as_str()) {
                    continue;
                }
                let is_dir = entry.file_type().await?.is_dir();
                names.push(if is_dir { format!("{name}/") } else { name });
            }
            names.sort();
            Ok(names.join("\n"))
        })
    }

    fn summarize(&self, output: &str) -> String {
        format!("{} entries", output.lines().count())
    }
}

/// `edit_file { path, old, new }`: replaces the one occurrence of `old` in a
/// text file with `new`, leaving every other byte as it was (line endings too).
pub struct EditFile {
    root: PathBuf,
}

/// Arguments of `edit_file`.
#[derive(Deserialize)]
struct EditArgs {
    path: String,
    old: String,
    new: String,
}

/// Where an edit lands, found without changing anything.
struct EditPlan {
    /// The resolved file.
    full: PathBuf,
    /// The file as it is now.
    text: String,
    /// Byte offset of `old` in `text`.
    at: usize,
    /// Line number (from 1) where `old` starts.
    line: usize,
    /// The text being replaced: the model's `old`, or that with
    /// [`strip_line_numbers`] applied when only that matched.
    old: String,
    /// The text put in its place: the model's `new`, stripped likewise when
    /// `old` was and `new` has the same `N: ` rows.
    new: String,
}

impl EditFile {
    /// Reads the file and finds the single occurrence of `old`. Both `preview`
    /// and `run` call this, so `run` checks the file as it is when the user
    /// says yes, not as it was when they were asked.
    ///
    /// When `old` is not in the file but is `read_file` rows pasted verbatim
    /// (`3: - bread`), the rows without their numbers are looked for instead,
    /// and used if they occur once; `new` loses its numbers too if it has them.
    async fn plan(&self, args: &EditArgs) -> Result<EditPlan> {
        if args.old.is_empty() {
            bail!("old text is empty; give the exact text to replace");
        }
        let (full, text) = read_text(&self.root, &args.path).await?;
        let path = &args.path;
        let (mut old, mut new) = (args.old.clone(), args.new.clone());
        let mut found = occurrences(&text, &old);
        if found.is_empty()
            && let Some(stripped) = strip_line_numbers(&args.old)
            && !stripped.is_empty()
            && let [at] = occurrences(&text, &stripped)[..]
        {
            found = vec![at];
            old = stripped;
            if let Some(stripped) = strip_line_numbers(&args.new) {
                new = stripped;
            }
        }
        match found[..] {
            [] => bail!("old text not found in {path}"),
            [at] => Ok(EditPlan {
                line: text[..at].matches('\n').count() + 1,
                full,
                text,
                at,
                old,
                new,
            }),
            ref found => bail!(
                "old text found {} times in {path}; include more surrounding lines so it is unique",
                found.len()
            ),
        }
    }
}

impl Tool for EditFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".into(),
            description: "Replace one exact snippet in an existing text file. Read the file \
                          first. `old` must match the file exactly and occur only once, so \
                          include enough surrounding lines to make it unique. The `N: ` line \
                          numbers that read_file shows are not part of the file; do not \
                          include them in `old` or `new`."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path, relative to the working directory" },
                    "old": { "type": "string", "description": "The exact text to replace, as it appears in the file" },
                    "new": { "type": "string", "description": "The text to put in its place" }
                },
                "required": ["path", "old", "new"]
            }),
        }
    }

    fn run(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let args = parse_args::<EditArgs>(arguments);
        Box::pin(async move {
            let args = args?;
            let EditPlan {
                full,
                mut text,
                at,
                line,
                old,
                new,
            } = self.plan(&args).await?;
            text.replace_range(at..at + old.len(), &new);
            tokio::fs::write(&full, text)
                .await
                .with_context(|| format!("writing {}", args.path))?;
            Ok(format!(
                "replaced {} with {} at line {line}",
                count_lines(old.lines().count()),
                new.lines().count()
            ))
        })
    }

    fn summarize(&self, output: &str) -> String {
        output.to_owned()
    }

    fn mutates(&self) -> bool {
        true
    }

    /// `edit <path> at line N`, then one `- ` line per old line and one `+ `
    /// line per new line.
    fn preview(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let args = parse_args::<EditArgs>(arguments);
        Box::pin(async move {
            let args = args?;
            let plan = self.plan(&args).await?;
            let mut out = format!("edit {} at line {}", args.path, plan.line);
            for line in plan.old.lines() {
                out.push_str(&format!("\n- {line}"));
            }
            for line in plan.new.lines() {
                out.push_str(&format!("\n+ {line}"));
            }
            Ok(out)
        })
    }
}

/// Byte offsets of every occurrence of `needle` (not empty) in `haystack`,
/// overlapping ones included: `aa` occurs twice in `aaa`, so it is ambiguous.
fn occurrences(haystack: &str, needle: &str) -> Vec<usize> {
    let step = needle.chars().next().map_or(1, char::len_utf8);
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(i) = haystack[from..].find(needle) {
        found.push(from + i);
        from += i + step;
    }
    found
}

/// `text` without the `N: ` prefixes [`numbered`] puts on `read_file` rows,
/// or `None` if it is not such rows: every line must start with a number and
/// `: ` (or just `:` for a blank line, whose trailing space a model may drop),
/// and the numbers must count up by one. Line endings are kept.
///
/// A model sometimes pastes the rows it read into `edit_file`'s `old`. A file
/// whose lines really start like that is unaffected: `edit_file` tries the
/// text as given first.
fn strip_line_numbers(text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let mut next: Option<usize> = None;
    for row in text.split_inclusive('\n') {
        let digits = row.len() - row.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        let n: usize = row[..digits].parse().ok()?;
        if next.is_some_and(|next| next != n) {
            return None;
        }
        next = Some(n + 1);
        let rest = row[digits..].strip_prefix(':')?;
        let line = match rest.strip_prefix(' ') {
            Some(line) => line,
            None if matches!(rest, "" | "\n" | "\r\n") => rest,
            None => return None,
        };
        out.push_str(line);
    }
    next.map(|_| out)
}

/// `write_file { path, content }`: creates a file, or replaces a whole one.
pub struct WriteFile {
    root: PathBuf,
}

/// Arguments of `write_file`.
#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

impl WriteFile {
    /// The resolved target, and how many lines it has now (`None` if it does
    /// not exist yet). A directory is refused.
    async fn target(&self, path: &str) -> Result<(PathBuf, Option<usize>)> {
        let full = resolve_new(&self.root, path).await?;
        match tokio::fs::metadata(&full).await {
            Ok(meta) if meta.is_dir() => bail!("{path} is a directory"),
            Ok(_) => {
                let bytes = tokio::fs::read(&full)
                    .await
                    .with_context(|| format!("reading {path}"))?;
                let lines = String::from_utf8_lossy(&bytes).lines().count();
                Ok((full, Some(lines)))
            }
            Err(_) => Ok((full, None)),
        }
    }
}

impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Create a new text file, or replace the whole content of one. \
                          Missing parent directories are created. For a small change to an \
                          existing file use edit_file instead."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path, relative to the working directory" },
                    "content": { "type": "string", "description": "The complete new content of the file" }
                },
                "required": ["path", "content"]
            }),
        }
    }

    fn run(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let args = parse_args::<WriteArgs>(arguments);
        Box::pin(async move {
            let WriteArgs { path, content } = args?;
            let (full, _) = self.target(&path).await?;
            if let Some(parent) = full.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .with_context(|| format!("creating the directories of {path}"))?;
            }
            tokio::fs::write(&full, &content)
                .await
                .with_context(|| format!("writing {path}"))?;
            Ok(format!(
                "wrote {path} ({})",
                count_lines(content.lines().count())
            ))
        })
    }

    fn summarize(&self, output: &str) -> String {
        output.to_owned()
    }

    fn mutates(&self) -> bool {
        true
    }

    /// `create <path> (N lines)` or `overwrite <path> (M lines → N lines)`,
    /// then the new content as `+ ` lines, cut after [`PREVIEW_MAX_LINES`].
    fn preview(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let args = parse_args::<WriteArgs>(arguments);
        Box::pin(async move {
            let WriteArgs { path, content } = args?;
            let (_, before) = self.target(&path).await?;
            let total = content.lines().count();
            let after = count_lines(total);
            let mut out = match before {
                None => format!("create {path} ({after})"),
                Some(before) => format!("overwrite {path} ({} → {after})", count_lines(before)),
            };
            for line in content.lines().take(PREVIEW_MAX_LINES) {
                out.push_str(&format!("\n+ {line}"));
            }
            if total > PREVIEW_MAX_LINES {
                let more = total - PREVIEW_MAX_LINES;
                let unit = if more == 1 { "line" } else { "lines" };
                out.push_str(&format!("\n… ({more} more {unit})"));
            }
            Ok(out)
        })
    }
}

/// `bash { command, timeout_seconds? }`: runs `<shell> -c <command>` in the
/// root and returns its exit code, stdout and stderr (see [`Bash::execute`]).
///
/// The command gets no stdin, since nobody would ever type into it, and is
/// killed when it runs out of time or when its future is dropped (the turn was
/// aborted). A non-zero exit is not a tool error: the tool did its job, and the
/// model reads the exit code.
pub struct Bash {
    root: PathBuf,
    shell: PathBuf,
}

/// Arguments of `bash`.
#[derive(Deserialize)]
struct BashArgs {
    command: String,
    timeout_seconds: Option<u64>,
}

impl BashArgs {
    /// The command and its timeout in seconds, from the model's arguments. A
    /// blank command is refused; the timeout defaults to
    /// [`BASH_DEFAULT_TIMEOUT`] and is clamped to 1..=[`BASH_MAX_TIMEOUT`].
    fn parse(arguments: &str) -> Result<(String, u64)> {
        let args = parse_args::<BashArgs>(arguments)?;
        if args.command.trim().is_empty() {
            bail!("empty command");
        }
        let timeout = args
            .timeout_seconds
            .unwrap_or(BASH_DEFAULT_TIMEOUT)
            .clamp(1, BASH_MAX_TIMEOUT);
        Ok((args.command, timeout))
    }
}

impl Bash {
    /// Runs `command` for at most `timeout` seconds and reports on it:
    ///
    /// ```text
    /// exit code: 0
    /// <stdout>
    /// --- stderr ---
    /// <stderr>
    /// ```
    ///
    /// The stderr part is left out when there is none; empty stdout is just
    /// left out, so stderr alone still comes under its header. Each stream is
    /// cut by [`cap_stream`]. A command that runs out of time is killed; it
    /// has `exit code: none`, keeps what it printed, and ends with
    /// `[timed out after N s]`.
    async fn execute(&self, command: &str, timeout: u64) -> Result<String> {
        let mut child = tokio::process::Command::new(&self.shell)
            .arg("-c")
            .arg(command)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // An aborted turn drops this future, and the child with it.
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting {}", self.shell.display()))?;
        let mut stdout_pipe = child.stdout.take().context("no stdout pipe")?;
        let mut stderr_pipe = child.stderr.take().context("no stderr pipe")?;
        // Declared after `child`, so on an abort it is dropped first, while
        // the child is still there to find its descendants by.
        let mut tree = KillTree(child.id());
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        // Both pipes are drained while waiting, so a chatty command never
        // blocks on a full pipe.
        let finished = tokio::time::timeout(Duration::from_secs(timeout), async {
            let (_, _, status) = tokio::join!(
                read_all(&mut stdout_pipe, &mut stdout),
                read_all(&mut stderr_pipe, &mut stderr),
                child.wait()
            );
            status
        })
        .await;
        let (code, timed_out) = match finished {
            Ok(status) => {
                // Exited: its id may be reused, so nothing to kill any more.
                tree.0 = None;
                (status.context("waiting for the command")?.code(), false)
            }
            Err(_) => {
                drop(tree);
                // It may have just exited on its own; either way it is gone.
                let _ = child.kill().await;
                (None, true)
            }
        };

        let code = code.map_or_else(|| "none".to_owned(), |code| code.to_string());
        let mut out = format!("exit code: {code}");
        let stdout = cap_stream(&String::from_utf8_lossy(&stdout));
        let stderr = cap_stream(&String::from_utf8_lossy(&stderr));
        if !stdout.is_empty() {
            out.push_str(&format!("\n{stdout}"));
        }
        if !stderr.is_empty() {
            out.push_str(&format!("\n{STDERR_HEADER}\n{stderr}"));
        }
        if timed_out {
            out.push_str(&format!("\n{TIMED_OUT}{timeout} s]"));
        }
        Ok(out)
    }
}

impl Tool for Bash {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: format!(
                "Run a bash command in the working directory and get its exit code, \
                 stdout and stderr. Use it to build, run tests and check your work; use \
                 read_file and list_files to read. There is no stdin, so interactive \
                 commands cannot work. Long output keeps its first and last \
                 {BASH_KEEP_LINES} lines."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command, run with `bash -c`" },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": format!(
                            "Seconds before the command is killed. Default {BASH_DEFAULT_TIMEOUT}, at most {BASH_MAX_TIMEOUT}"
                        )
                    }
                },
                "required": ["command"]
            }),
        }
    }

    fn run(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let args = BashArgs::parse(arguments);
        Box::pin(async move {
            let (command, timeout) = args?;
            self.execute(&command, timeout).await
        })
    }

    /// `exit 0, 12 lines`, or `timed out, 3 lines`: how it ended, and how many
    /// lines of output came back (the headers and cut markers not counted).
    fn summarize(&self, output: &str) -> String {
        let mut lines = output.lines();
        let code = lines
            .next()
            .and_then(|first| first.strip_prefix("exit code: "));
        let mut rest: Vec<&str> = lines.collect();
        let timed_out = rest.last().is_some_and(|last| last.starts_with(TIMED_OUT));
        if timed_out {
            rest.pop();
        }
        if let Some(header) = rest.iter().position(|line| *line == STDERR_HEADER) {
            rest.remove(header);
        }
        let shown = rest.iter().filter(|line| !is_omitted_marker(line)).count();
        let ended = match code {
            _ if timed_out => "timed out".to_owned(),
            Some(code) => format!("exit {code}"),
            None => "done".to_owned(),
        };
        format!("{ended}, {}", count_lines(shown))
    }

    fn mutates(&self) -> bool {
        true
    }

    /// `$ <command>`, and `(timeout N s)` under it when the model asked for
    /// another timeout than the default.
    fn preview(&self, arguments: &str) -> BoxFuture<'_, Result<String>> {
        let args = BashArgs::parse(arguments);
        Box::pin(async move {
            let (command, timeout) = args?;
            let mut out = format!("$ {command}");
            if timeout != BASH_DEFAULT_TIMEOUT {
                out.push_str(&format!("\n(timeout {timeout} s)"));
            }
            Ok(out)
        })
    }

    /// `bash: <command>`, trimmed and with every run of whitespace made one
    /// space, so a yes to `cargo test` covers `cargo  test` but not `rm -rf .`.
    fn permission_scope(&self, arguments: &str) -> Result<String> {
        let (command, _) = BashArgs::parse(arguments)?;
        let words: Vec<&str> = command.split_whitespace().collect();
        Ok(format!("bash: {}", words.join(" ")))
    }
}

/// On Windows, kills the process with this id and all its descendants when
/// dropped, unless the id has been taken out.
///
/// `kill_on_drop` (and `Child::kill`) end only the process `bash` started,
/// and on Windows that is not enough: Git's `bin\bash.exe` is a launcher that
/// runs the real `usr\bin\bash.exe`, which runs the command, and killing a
/// parent there leaves its children running. A timed-out or cancelled
/// `cargo build` would go on in the background. `taskkill /T` ends the whole
/// tree. It waits for `taskkill` (well under a second), which is acceptable
/// on the paths that drop this: a timeout or an aborted turn.
///
/// Elsewhere it does nothing: killing the child ends the shell, and `bash -c`
/// runs a lone command in place of itself. The other processes of a pipeline
/// or a background job can outlive it there.
struct KillTree(Option<u32>);

impl Drop for KillTree {
    fn drop(&mut self) {
        if cfg!(windows)
            && let Some(pid) = self.0.take()
        {
            let _ = std::process::Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid.to_string()])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

/// Appends everything `pipe` yields to `buf` until it closes or fails. Bytes
/// land in `buf` as they arrive, so a read cut short (the command timed out)
/// keeps what came before. Past [`BASH_READ_CAP`] the middle is dropped as
/// it streams in, so a runaway command cannot fill memory; only the ends are
/// returned anyway (the omitted count then covers only what was held).
async fn read_all(pipe: &mut (impl AsyncRead + Unpin), buf: &mut Vec<u8>) {
    let mut chunk = [0; 8192];
    while let Ok(n @ 1..) = pipe.read(&mut chunk).await {
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > BASH_READ_CAP {
            buf.drain(BASH_MAX_BYTES..buf.len() - BASH_READ_CAP / 2);
        }
    }
}

/// One stream of `bash` output for the model: its lines joined by `\n` (CRLF
/// read as LF, no final newline). More than twice [`BASH_KEEP_LINES`] lines
/// keep that many from each end with `[… N lines omitted …]` between; if that
/// is still over [`BASH_MAX_BYTES`], half of it is kept from each end with
/// `[… N bytes omitted …]` between. The end of a build or test log is where
/// the error is, so both ends are kept rather than just the start.
fn cap_stream(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = if lines.len() > 2 * BASH_KEEP_LINES {
        let omitted = lines.len() - 2 * BASH_KEEP_LINES;
        format!(
            "{}\n[… {omitted} lines omitted …]\n{}",
            lines[..BASH_KEEP_LINES].join("\n"),
            lines[lines.len() - BASH_KEEP_LINES..].join("\n")
        )
    } else {
        lines.join("\n")
    };
    if out.len() > BASH_MAX_BYTES {
        let head = out.floor_char_boundary(BASH_MAX_BYTES / 2);
        let tail = out.ceil_char_boundary(out.len() - BASH_MAX_BYTES / 2);
        out = format!(
            "{}\n[… {} bytes omitted …]\n{}",
            &out[..head],
            tail - head,
            &out[tail..]
        );
    }
    out
}

/// Whether `line` is a marker [`cap_stream`] put in place of what it cut.
fn is_omitted_marker(line: &str) -> bool {
    line.starts_with("[… ") && line.ends_with(" omitted …]")
}

/// The bash that runs `bash` commands, if this machine has one.
///
/// On Windows that must be Git for Windows's bash ("Git Bash"). The
/// `bash.exe` in `System32` (and its alias in `WindowsApps`) starts WSL
/// instead: a Linux VM with its own filesystem and its own tools, where the
/// working directory is a `/mnt/c/...` path and `cargo` is not the user's
/// `cargo`. So Git's bash is looked for first, near the `git` on PATH (whose
/// `--exec-path`, `<git>\mingw64\libexec\git-core`, lies inside its install
/// directory) and then at the default install path; any other `bash.exe` on
/// PATH comes last, and never one of the WSL launchers. `<git>\bin\bash.exe`
/// is preferred to `<git>\usr\bin\bash.exe`: it sets up PATH so the Unix tools
/// are found even when they are not on the user's PATH.
///
/// Elsewhere it is `bash` on PATH, else `/bin/bash`.
pub fn find_bash() -> Option<PathBuf> {
    if cfg!(windows) {
        let program_files = std::env::var_os("ProgramFiles")
            .map_or_else(|| PathBuf::from(r"C:\Program Files"), PathBuf::from);
        let near_git: Vec<PathBuf> = git_exec_path()
            .iter()
            .flat_map(|exec| exec.ancestors().skip(1).take(3))
            .flat_map(|dir| [dir.join(r"bin\bash.exe"), dir.join(r"usr\bin\bash.exe")])
            .collect();
        let installed = program_files.join(r"Git\bin\bash.exe");
        near_git
            .into_iter()
            .chain([installed])
            .find(|path| path.is_file())
            .or_else(|| {
                on_path("bash.exe")
                    .into_iter()
                    .find(|path| !is_wsl_launcher(path))
            })
    } else {
        let fallback = PathBuf::from("/bin/bash");
        on_path("bash")
            .into_iter()
            .next()
            .or_else(|| fallback.is_file().then_some(fallback))
    }
}

/// What `git --exec-path` prints, if `git` runs.
fn git_exec_path() -> Option<PathBuf> {
    let output = std::process::Command::new("git")
        .arg("--exec-path")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (output.status.success() && !path.is_empty()).then(|| PathBuf::from(path))
}

/// Every file called `file` in a directory on PATH, in PATH order.
fn on_path(file: &str) -> Vec<PathBuf> {
    let Some(path) = std::env::var_os("PATH") else {
        return Vec::new();
    };
    std::env::split_paths(&path)
        .map(|dir| dir.join(file))
        .filter(|candidate| candidate.is_file())
        .collect()
}

/// Whether `path` is one of Windows's `bash.exe` launchers for WSL (see
/// [`find_bash`]), going by the directory it is in.
fn is_wsl_launcher(path: &Path) -> bool {
    path.parent().and_then(Path::file_name).is_some_and(|dir| {
        let dir = dir.to_string_lossy().to_ascii_lowercase();
        dir == "system32" || dir == "windowsapps"
    })
}

/// `1 line`, `3 lines`.
fn count_lines(n: usize) -> String {
    if n == 1 {
        "1 line".to_owned()
    } else {
        format!("{n} lines")
    }
}

/// Parses a tool's JSON arguments. Empty text counts as `{}`: some models send
/// that for a call with no arguments.
fn parse_args<T: DeserializeOwned>(arguments: &str) -> Result<T> {
    let arguments = if arguments.trim().is_empty() {
        "{}"
    } else {
        arguments
    };
    serde_json::from_str(arguments).with_context(|| format!("bad arguments {arguments}"))
}

/// The text file at `path` under `root`, with its resolved path. Refused if
/// it is outside the root, missing, a directory, or not UTF-8 text without
/// NUL bytes.
async fn read_text(root: &Path, path: &str) -> Result<(PathBuf, String)> {
    let full = resolve(root, path).await?;
    if full.is_dir() {
        bail!("{path} is a directory; use list_files");
    }
    let bytes = tokio::fs::read(&full)
        .await
        .with_context(|| format!("reading {path}"))?;
    match String::from_utf8(bytes) {
        Ok(text) if !text.contains('\0') => Ok((full, text)),
        _ => bail!("{path} is not a text file"),
    }
}

/// `path` joined to `root`, refused if it lands outside `root` once `..` and
/// links are resolved. The path must exist (canonicalizing needs it to); for
/// one that may not exist yet, see [`resolve_new`].
async fn resolve(root: &Path, path: &str) -> Result<PathBuf> {
    let root = tokio::fs::canonicalize(root)
        .await
        .with_context(|| format!("working directory {}", root.display()))?;
    let full = tokio::fs::canonicalize(root.join(path))
        .await
        .with_context(|| format!("{path} not found"))?;
    if !full.starts_with(&root) {
        bail!("{path} is outside the working directory");
    }
    Ok(full)
}

/// `path` joined to `root`, for a file that may not exist yet (`write_file`).
///
/// [`resolve`] cannot check such a path, since only existing paths can be
/// canonicalized, so this checks in two steps:
/// 1. By the text alone: absolute paths and `..` components are refused
///    outright, so the joined path cannot climb out of the root.
/// 2. On disk: the deepest part of the joined path that exists (the file
///    itself, or an ancestor directory, at worst the root) is canonicalized and
///    must lie inside the root. That catches a symlink, to a directory or to
///    the file, that points out; a dangling link cannot be canonicalized and
///    is refused too. Everything below that part does not exist yet, so
///    creating it puts it inside the root.
async fn resolve_new(root: &Path, path: &str) -> Result<PathBuf> {
    let mut relative = PathBuf::new();
    for part in Path::new(path).components() {
        match part {
            Component::Normal(name) => relative.push(name),
            Component::CurDir => {}
            Component::ParentDir => {
                bail!("{path}: `..` is not allowed; give a path inside the working directory")
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("{path} is absolute; give a path relative to the working directory")
            }
        }
    }
    if relative.as_os_str().is_empty() {
        bail!("{path:?} does not name a file");
    }

    let root = tokio::fs::canonicalize(root)
        .await
        .with_context(|| format!("working directory {}", root.display()))?;
    let full = root.join(relative);
    let mut existing = full.as_path();
    // `symlink_metadata` sees a link itself, dangling or not.
    while tokio::fs::symlink_metadata(existing).await.is_err() {
        existing = existing
            .parent()
            .with_context(|| format!("{path}: no part of it exists"))?;
    }
    let real = tokio::fs::canonicalize(existing)
        .await
        .with_context(|| format!("{path} cannot be resolved"))?;
    if !real.starts_with(&root) {
        bail!("{path} is outside the working directory");
    }
    Ok(full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    fn read_args(path: &str) -> String {
        json!({ "path": path }).to_string()
    }

    #[tokio::test]
    async fn read_file_numbers_lines() {
        let dir = TempDir::new("read");
        dir.write("notes.txt", "alpha\r\nbeta\n");
        let tools = ToolSet::read_only(&dir.0);
        let out = tools
            .run("read_file", &read_args("notes.txt"))
            .await
            .unwrap();
        assert_eq!(out, "1: alpha\n2: beta\n");
        assert_eq!(tools.summarize("read_file", &out), "2 lines");
    }

    #[tokio::test]
    async fn read_file_truncates_by_lines_and_by_bytes() {
        let dir = TempDir::new("truncate");
        let many: String = (1..=1000).map(|i| format!("line {i}\n")).collect();
        dir.write("many.txt", many);
        let wide: String = (0..100)
            .map(|_| format!("{}\n", "x".repeat(1000)))
            .collect();
        dir.write("wide.txt", wide);
        let tools = ToolSet::read_only(&dir.0);

        let out = tools
            .run("read_file", &read_args("many.txt"))
            .await
            .unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), READ_MAX_LINES + 1);
        assert_eq!(lines[399], "400: line 400");
        assert_eq!(lines[400], "[truncated: showing first 400 of 1000 lines]");
        assert_eq!(tools.summarize("read_file", &out), "400 lines (truncated)");

        let out = tools
            .run("read_file", &read_args("wide.txt"))
            .await
            .unwrap();
        assert!(out.len() <= READ_MAX_BYTES + 100);
        assert!(out.ends_with("[truncated: showing first 65 of 100 lines]\n"));
    }

    #[tokio::test]
    async fn read_file_refuses_escapes_binaries_and_missing_files() {
        let dir = TempDir::new("escape");
        dir.write("root/inside.txt", "ok");
        dir.write("root/blob.bin", [0u8, 159, 146, 150]);
        dir.write("secret.txt", "no");
        let tools = ToolSet::read_only(dir.0.join("root"));

        let err = |args: String| {
            let tools = tools.clone();
            async move { format!("{:#}", tools.run("read_file", &args).await.unwrap_err()) }
        };
        assert!(err(read_args("../secret.txt")).await.contains("outside"));
        let absolute = dir.0.join("secret.txt").display().to_string();
        assert!(err(read_args(&absolute)).await.contains("outside"));
        assert!(err(read_args("blob.bin")).await.contains("not a text file"));
        assert!(err(read_args("missing.txt")).await.contains("not found"));
        assert!(err(read_args(".")).await.contains("is a directory"));
        assert!(err("{\"file\":1}".into()).await.contains("bad arguments"));
        assert!(
            tools
                .run("read_file", &read_args("inside.txt"))
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn list_files_sorts_marks_dirs_and_skips_noise() {
        let dir = TempDir::new("list");
        dir.write("b.txt", "");
        dir.write("a.txt", "");
        dir.write("src/main.rs", "");
        dir.write(".git/HEAD", "");
        dir.write("target/x", "");
        let tools = ToolSet::read_only(&dir.0);

        for args in ["", "{}", r#"{"path":"."}"#] {
            let out = tools.run("list_files", args).await.unwrap();
            assert_eq!(out, "a.txt\nb.txt\nsrc/", "args {args:?}");
            assert_eq!(tools.summarize("list_files", &out), "3 entries");
        }
        let sub = tools.run("list_files", &read_args("src")).await.unwrap();
        assert_eq!(sub, "main.rs");
        let up = tools.run("list_files", &read_args("..")).await.unwrap_err();
        assert!(format!("{up:#}").contains("outside"));
    }

    #[tokio::test]
    async fn unknown_tools_are_an_error_result() {
        let tools = ToolSet::read_only(".");
        let names: Vec<String> = tools.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["read_file", "list_files"]);
        let err = tools.run("rm_rf", "{}").await.unwrap_err();
        assert_eq!(err.to_string(), "unknown tool: rm_rf");
    }

    #[tokio::test]
    async fn standard_adds_the_two_tools_that_change_files() {
        let tools = ToolSet::standard_with_shell(".", None);
        let names: Vec<String> = tools.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(
            names,
            ["read_file", "list_files", "edit_file", "write_file"]
        );
        let mutating: Vec<&String> = names.iter().filter(|n| tools.mutates(n)).collect();
        assert_eq!(mutating, ["edit_file", "write_file"]);
        assert!(!tools.mutates("rm_rf"));
        assert!(tools.has("read_file"));
        assert!(!tools.has("bash"), "no shell, no bash");
        assert!(!tools.has("rm_rf"));

        // The default scope of "don't ask again" is the whole tool.
        let edit = r#"{"path":"a.txt","old":"x","new":"y"}"#;
        assert_eq!(
            tools.permission_scope("edit_file", edit).unwrap(),
            "edit_file"
        );
        let err = tools.permission_scope("rm_rf", "{}").unwrap_err();
        assert_eq!(err.to_string(), "unknown tool: rm_rf");

        // A read-only tool previews as its arguments; an unknown one is an error.
        assert_eq!(tools.preview("list_files", "{}").await.unwrap(), "{}");
        let err = tools.preview("rm_rf", "{}").await.unwrap_err();
        assert_eq!(err.to_string(), "unknown tool: rm_rf");
    }

    fn edit_args(path: &str, old: &str, new: &str) -> String {
        json!({ "path": path, "old": old, "new": new }).to_string()
    }

    fn write_args(path: &str, content: &str) -> String {
        json!({ "path": path, "content": content }).to_string()
    }

    #[tokio::test]
    async fn edit_file_previews_then_replaces_the_one_match() {
        let dir = TempDir::new("edit");
        let before = "fn main() {\n    let x = 1;\n    let y = 2;\n}\n";
        dir.write("src/main.rs", before);
        let tools = ToolSet::standard(&dir.0);
        let args = edit_args(
            "src/main.rs",
            "    let x = 1;\n    let y = 2;\n",
            "    let z = 3;\n",
        );

        let preview = tools.preview("edit_file", &args).await.unwrap();
        assert_eq!(
            preview,
            "edit src/main.rs at line 2\n-     let x = 1;\n-     let y = 2;\n+     let z = 3;"
        );
        assert_eq!(dir.read("src/main.rs"), before, "preview changes nothing");

        let out = tools.run("edit_file", &args).await.unwrap();
        assert_eq!(out, "replaced 2 lines with 1 at line 2");
        assert_eq!(tools.summarize("edit_file", &out), out);
        assert_eq!(dir.read("src/main.rs"), "fn main() {\n    let z = 3;\n}\n");
    }

    #[tokio::test]
    async fn edit_file_keeps_crlf_and_does_not_normalize_line_endings() {
        let dir = TempDir::new("edit-crlf");
        dir.write("a.txt", "one\r\ntwo\r\nthree\r\n");
        let tools = ToolSet::standard(&dir.0);
        let out = tools
            .run("edit_file", &edit_args("a.txt", "two", "2"))
            .await
            .unwrap();
        assert_eq!(out, "replaced 1 line with 1 at line 2");
        assert_eq!(dir.read("a.txt"), "one\r\n2\r\nthree\r\n");

        // LF text does not match CRLF lines: that is simply not found.
        let err = tools
            .run("edit_file", &edit_args("a.txt", "one\n2", "x"))
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "old text not found in a.txt");
        assert_eq!(dir.read("a.txt"), "one\r\n2\r\nthree\r\n");
    }

    #[tokio::test]
    async fn edit_file_refuses_missing_ambiguous_and_outside_text() {
        let dir = TempDir::new("edit-refuse");
        let before = "x = 1\nx = 1\ny = 2\naaa\n";
        dir.write("root/a.txt", before);
        dir.write("root/blob.bin", [0u8, 159, 146, 150]);
        dir.write("secret.txt", "no");
        let tools = ToolSet::standard(dir.0.join("root"));

        // Preview and run fail the same way, before anything is written.
        for step in ["preview", "run"] {
            let err = |args: String| {
                let tools = tools.clone();
                async move {
                    let result = if step == "preview" {
                        tools.preview("edit_file", &args).await
                    } else {
                        tools.run("edit_file", &args).await
                    };
                    format!("{:#}", result.unwrap_err())
                }
            };
            assert_eq!(
                err(edit_args("a.txt", "z", "w")).await,
                "old text not found in a.txt"
            );
            assert_eq!(
                err(edit_args("a.txt", "x = 1", "x = 3")).await,
                "old text found 2 times in a.txt; include more surrounding lines so it is unique"
            );
            assert!(err(edit_args("a.txt", "aa", "b")).await.contains("2 times"));
            assert!(err(edit_args("a.txt", "", "w")).await.contains("empty"));
            assert!(
                err(edit_args("gone.txt", "a", "b"))
                    .await
                    .contains("not found")
            );
            assert!(
                err(edit_args("../secret.txt", "no", "yes"))
                    .await
                    .contains("outside")
            );
            assert!(
                err(edit_args("blob.bin", "a", "b"))
                    .await
                    .contains("not a text file")
            );
            assert!(
                err(edit_args(".", "a", "b"))
                    .await
                    .contains("is a directory")
            );
            assert!(
                err(r#"{"path":"a.txt"}"#.into())
                    .await
                    .contains("bad arguments")
            );
        }
        assert_eq!(dir.read("root/a.txt"), before);
        assert_eq!(dir.read("secret.txt"), "no");
    }

    #[tokio::test]
    async fn edit_file_strips_pasted_line_numbers_from_old() {
        let dir = TempDir::new("edit-numbered");
        dir.write("a.txt", "a\nb\nc\n");
        let tools = ToolSet::standard(&dir.0);
        let out = tools
            .run("edit_file", &edit_args("a.txt", "2: b", "B"))
            .await
            .unwrap();
        assert_eq!(out, "replaced 1 line with 1 at line 2");
        assert_eq!(dir.read("a.txt"), "a\nB\nc\n");
    }

    #[tokio::test]
    async fn edit_file_strips_numbers_from_new_too_when_it_has_them() {
        let dir = TempDir::new("edit-numbered-both");
        dir.write("a.txt", "a\nb\nc\n");
        let tools = ToolSet::standard(&dir.0);
        let args = edit_args("a.txt", "2: b\n3: c", "2: B\n3: C");

        let preview = tools.preview("edit_file", &args).await.unwrap();
        assert_eq!(preview, "edit a.txt at line 2\n- b\n- c\n+ B\n+ C");
        assert_eq!(dir.read("a.txt"), "a\nb\nc\n", "preview changes nothing");

        let out = tools.run("edit_file", &args).await.unwrap();
        assert_eq!(out, "replaced 2 lines with 2 at line 2");
        assert_eq!(dir.read("a.txt"), "a\nB\nC\n");
    }

    #[tokio::test]
    async fn edit_file_keeps_new_verbatim_when_only_old_is_numbered() {
        let dir = TempDir::new("edit-numbered-old");
        dir.write("a.txt", "a\nb\nc\n");
        let tools = ToolSet::standard(&dir.0);
        let args = edit_args("a.txt", "1: a\n2: b\n", "x\n7: y\n");

        let preview = tools.preview("edit_file", &args).await.unwrap();
        assert_eq!(preview, "edit a.txt at line 1\n- a\n- b\n+ x\n+ 7: y");

        let out = tools.run("edit_file", &args).await.unwrap();
        assert_eq!(out, "replaced 2 lines with 2 at line 1");
        assert_eq!(dir.read("a.txt"), "x\n7: y\nc\n");
    }

    #[tokio::test]
    async fn edit_file_strips_numbers_only_when_the_text_itself_is_not_found() {
        let dir = TempDir::new("edit-numbered-not");
        dir.write("a.txt", "1: x\nb\nc\n");
        let tools = ToolSet::standard(&dir.0);

        // Not consecutive: not read_file rows, so not stripped.
        let err = tools
            .run("edit_file", &edit_args("a.txt", "2: b\n5: c", "x"))
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "old text not found in a.txt");

        // A line that really reads `1: x` is matched as it is.
        let out = tools
            .run("edit_file", &edit_args("a.txt", "1: x", "1: y"))
            .await
            .unwrap();
        assert_eq!(out, "replaced 1 line with 1 at line 1");
        assert_eq!(dir.read("a.txt"), "1: y\nb\nc\n");
    }

    #[test]
    fn strip_line_numbers_takes_only_consecutive_numbered_rows() {
        assert_eq!(strip_line_numbers("b"), None);
        assert_eq!(strip_line_numbers("let x = 1;\nlet y = 2;"), None);
        assert_eq!(strip_line_numbers(""), None);
        assert_eq!(
            strip_line_numbers("3: - breaad").as_deref(),
            Some("- breaad")
        );
        assert_eq!(
            strip_line_numbers("9: a\n10:\n11:     b\n").as_deref(),
            Some("a\n\n    b\n"),
            "a blank row may lose its trailing space; indentation is kept"
        );
        assert_eq!(strip_line_numbers("4: \n").as_deref(), Some("\n"));
        assert_eq!(
            strip_line_numbers("1: a\nb\n3: c"),
            None,
            "a row without a number"
        );
        assert_eq!(strip_line_numbers("1: a\n3: c"), None, "not consecutive");
        assert_eq!(
            strip_line_numbers("12:30 lunch"),
            None,
            "no space after the colon"
        );
        assert_eq!(strip_line_numbers(": a"), None);
    }

    #[tokio::test]
    async fn write_file_creates_the_file_and_its_directories() {
        let dir = TempDir::new("write-create");
        let tools = ToolSet::standard(&dir.0);
        let args = write_args("notes/todo/list.md", "- a\n- b\n- c\n");

        let preview = tools.preview("write_file", &args).await.unwrap();
        assert_eq!(
            preview,
            "create notes/todo/list.md (3 lines)\n+ - a\n+ - b\n+ - c"
        );
        assert!(!dir.0.join("notes").exists(), "preview creates nothing");

        let out = tools.run("write_file", &args).await.unwrap();
        assert_eq!(out, "wrote notes/todo/list.md (3 lines)");
        assert_eq!(tools.summarize("write_file", &out), out);
        assert_eq!(dir.read("notes/todo/list.md"), "- a\n- b\n- c\n");
    }

    #[tokio::test]
    async fn write_file_overwrites_and_caps_its_preview() {
        let dir = TempDir::new("write-over");
        dir.write("README.md", "a\nb\n");
        let tools = ToolSet::standard(&dir.0);
        let content: String = (1..=40).map(|i| format!("line {i}\n")).collect();
        let args = write_args("./README.md", &content);

        let preview = tools.preview("write_file", &args).await.unwrap();
        let lines: Vec<&str> = preview.lines().collect();
        assert_eq!(lines[0], "overwrite ./README.md (2 lines → 40 lines)");
        assert_eq!(lines.len(), 1 + PREVIEW_MAX_LINES + 1);
        assert_eq!(lines[30], "+ line 30");
        assert_eq!(lines[31], "… (10 more lines)");

        let out = tools.run("write_file", &args).await.unwrap();
        assert_eq!(out, "wrote ./README.md (40 lines)");
        assert_eq!(dir.read("README.md"), content);
    }

    #[tokio::test]
    async fn write_file_refuses_paths_that_could_leave_the_root() {
        let dir = TempDir::new("write-escape");
        dir.write("root/sub/keep.txt", "");
        let tools = ToolSet::standard(dir.0.join("root"));
        let err = |args: String| {
            let tools = tools.clone();
            async move { format!("{:#}", tools.run("write_file", &args).await.unwrap_err()) }
        };

        assert!(
            err(write_args("../x.txt", "no"))
                .await
                .contains("`..` is not allowed")
        );
        assert!(err(write_args("sub/../x.txt", "no")).await.contains("`..`"));
        let absolute = dir.0.join("x.txt").display().to_string();
        assert!(
            err(write_args(&absolute, "no"))
                .await
                .contains("is absolute")
        );
        assert!(
            err(write_args("", "no"))
                .await
                .contains("does not name a file")
        );
        assert!(
            err(write_args("sub", "no"))
                .await
                .contains("is a directory")
        );
        assert!(!dir.0.join("x.txt").exists());
        assert_eq!(dir.read("root/sub/keep.txt"), "");
    }

    #[test]
    fn standard_has_bash_when_this_machine_has_one() {
        let tools = ToolSet::standard(".");
        assert_eq!(tools.has("bash"), find_bash().is_some());
        if tools.has("bash") {
            assert!(tools.mutates("bash"));
            let names: Vec<String> = tools.specs().into_iter().map(|s| s.name).collect();
            assert_eq!(names.last().map(String::as_str), Some("bash"));
        }
    }

    #[test]
    fn find_bash_never_picks_the_wsl_launcher() {
        if let Some(bash) = find_bash() {
            assert!(!is_wsl_launcher(&bash), "{}", bash.display());
            assert!(bash.is_file());
        }
        if cfg!(windows) {
            assert!(is_wsl_launcher(Path::new(r"C:\Windows\System32\bash.exe")));
            assert!(is_wsl_launcher(Path::new(
                r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe"
            )));
            assert!(!is_wsl_launcher(Path::new(
                r"C:\Program Files\Git\bin\bash.exe"
            )));
        }
    }

    /// The standard tools with this machine's bash, working in `dir`. `None`
    /// when there is no bash here, after checking that the tool is then left
    /// out; the test using it passes without running anything.
    fn with_bash(dir: &TempDir) -> Option<ToolSet> {
        let shell = find_bash();
        let tools = ToolSet::standard_with_shell(&dir.0, shell.clone());
        if shell.is_none() {
            assert!(!tools.has("bash"));
            eprintln!("no bash on this machine; skipped");
            return None;
        }
        Some(tools)
    }

    fn bash_args(command: &str) -> String {
        json!({ "command": command }).to_string()
    }

    #[tokio::test]
    async fn bash_runs_a_command_and_reports_the_exit_code() {
        let dir = TempDir::new("bash-echo");
        let Some(tools) = with_bash(&dir) else {
            return;
        };
        let out = tools.run("bash", &bash_args("echo hi")).await.unwrap();
        assert_eq!(out, "exit code: 0\nhi");
        assert_eq!(tools.summarize("bash", &out), "exit 0, 1 line");

        let out = tools.run("bash", &bash_args("true")).await.unwrap();
        assert_eq!(out, "exit code: 0");
        assert_eq!(tools.summarize("bash", &out), "exit 0, 0 lines");
    }

    #[tokio::test]
    async fn bash_shows_stderr_and_a_failure_is_still_a_result() {
        let dir = TempDir::new("bash-stderr");
        let Some(tools) = with_bash(&dir) else {
            return;
        };
        let out = tools
            .run(
                "bash",
                &bash_args(r#"echo "out put"; echo err >&2; exit 3"#),
            )
            .await
            .unwrap();
        assert_eq!(out, "exit code: 3\nout put\n--- stderr ---\nerr");
        assert_eq!(tools.summarize("bash", &out), "exit 3, 2 lines");

        // Only stderr: still under its header, so the model can tell.
        let out = tools
            .run("bash", &bash_args("echo oops >&2; exit 1"))
            .await
            .unwrap();
        assert_eq!(out, "exit code: 1\n--- stderr ---\noops");
        assert_eq!(tools.summarize("bash", &out), "exit 1, 1 line");
    }

    #[tokio::test]
    async fn bash_runs_in_the_root() {
        let dir = TempDir::new("bash-root");
        let Some(tools) = with_bash(&dir) else {
            return;
        };
        dir.write("marker.txt", "");
        let out = tools.run("bash", &bash_args("ls")).await.unwrap();
        assert!(out.lines().any(|line| line == "marker.txt"), "{out}");

        tools
            .run("bash", &bash_args("echo made > made.txt"))
            .await
            .unwrap();
        assert_eq!(dir.read("made.txt"), "made\n");
    }

    #[tokio::test]
    async fn bash_kills_a_command_that_runs_out_of_time() {
        let dir = TempDir::new("bash-timeout");
        let Some(tools) = with_bash(&dir) else {
            return;
        };
        let args = json!({ "command": "echo started; sleep 3", "timeout_seconds": 1 });
        let started = std::time::Instant::now();
        let out = tools.run("bash", &args.to_string()).await.unwrap();
        let took = started.elapsed();
        assert!(took < Duration::from_millis(2500), "took {took:?}");
        assert_eq!(out, "exit code: none\nstarted\n[timed out after 1 s]");
        assert_eq!(tools.summarize("bash", &out), "timed out, 1 line");
    }

    /// Neither a timeout nor an aborted turn leaves the command running
    /// (on Windows that takes killing the whole process tree).
    #[tokio::test]
    async fn bash_leaves_nothing_running_after_a_timeout_or_an_abort() {
        let dir = TempDir::new("bash-kill");
        let Some(tools) = with_bash(&dir) else {
            return;
        };
        let late = |name: &str| json!({ "command": format!("sleep 2; echo late > {name}") });
        let mut timed_out = late("timed-out.txt");
        timed_out["timeout_seconds"] = json!(1);
        let aborted = tokio::spawn({
            let tools = tools.clone();
            async move { tools.run("bash", &late("aborted.txt").to_string()).await }
        });
        let out = tools.run("bash", &timed_out.to_string()).await.unwrap();
        assert!(out.ends_with("[timed out after 1 s]"), "{out}");
        aborted.abort();
        tokio::time::sleep(Duration::from_millis(2500)).await;
        assert!(!dir.0.join("timed-out.txt").exists(), "killed on timeout");
        assert!(!dir.0.join("aborted.txt").exists(), "killed on abort");
    }

    #[tokio::test]
    async fn bash_keeps_both_ends_of_long_output() {
        let dir = TempDir::new("bash-long");
        let Some(tools) = with_bash(&dir) else {
            return;
        };
        let out = tools.run("bash", &bash_args("seq 1 1000")).await.unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 1 + 150 + 1 + 150);
        assert_eq!(lines[1], "1");
        assert_eq!(lines[150], "150");
        assert_eq!(lines[151], "[… 700 lines omitted …]");
        assert_eq!(lines[152], "851");
        assert_eq!(lines[301], "1000");
        assert_eq!(tools.summarize("bash", &out), "exit 0, 300 lines");

        // One huge line: cut by bytes instead.
        let out = tools
            .run("bash", &bash_args("head -c 200000 /dev/zero | tr '\\0' x"))
            .await
            .unwrap();
        let half = "x".repeat(BASH_MAX_BYTES / 2);
        let omitted = 200_000 - BASH_MAX_BYTES;
        assert_eq!(
            out,
            format!("exit code: 0\n{half}\n[… {omitted} bytes omitted …]\n{half}")
        );
        assert_eq!(tools.summarize("bash", &out), "exit 0, 2 lines");
    }

    #[test]
    fn cap_stream_cuts_on_char_boundaries_and_reads_crlf_as_lf() {
        assert_eq!(cap_stream("a\r\nb\r\n"), "a\nb");
        assert_eq!(cap_stream(""), "");
        let wide = "é".repeat(BASH_MAX_BYTES);
        let out = cap_stream(&wide);
        assert!(out.len() <= BASH_MAX_BYTES + 40, "{}", out.len());
        assert!(out.contains(" bytes omitted …]"));
    }

    /// A `bash` that is never started: previews and scopes need no shell.
    fn unstarted_bash() -> ToolSet {
        ToolSet::standard_with_shell(".", Some("bash".into()))
    }

    #[tokio::test]
    async fn bash_previews_the_command_and_any_other_timeout() {
        let tools = unstarted_bash();
        assert!(tools.mutates("bash"));
        let preview = |args: Value| {
            let tools = tools.clone();
            async move { tools.preview("bash", &args.to_string()).await.unwrap() }
        };
        assert_eq!(
            preview(json!({ "command": "cargo test" })).await,
            "$ cargo test"
        );
        assert_eq!(
            preview(json!({ "command": "cargo test", "timeout_seconds": 120 })).await,
            "$ cargo test",
            "the default is not mentioned"
        );
        assert_eq!(
            preview(json!({ "command": "sleep 5", "timeout_seconds": 30 })).await,
            "$ sleep 5\n(timeout 30 s)"
        );
        assert_eq!(
            preview(json!({ "command": "make", "timeout_seconds": 9999 })).await,
            "$ make\n(timeout 600 s)",
            "clamped to the most allowed"
        );
        assert_eq!(
            preview(json!({ "command": "make", "timeout_seconds": 0 })).await,
            "$ make\n(timeout 1 s)"
        );
    }

    #[tokio::test]
    async fn bash_scope_is_the_exact_command_with_whitespace_collapsed() {
        let tools = unstarted_bash();
        let scope = |command: &str| tools.permission_scope("bash", &bash_args(command));
        assert_eq!(scope("cargo test").unwrap(), "bash: cargo test");
        assert_eq!(
            scope("  cargo   test\n  --all ").unwrap(),
            "bash: cargo test --all"
        );
        assert_ne!(scope("cargo test").unwrap(), scope("rm -rf .").unwrap());
    }

    #[tokio::test]
    async fn bash_refuses_an_empty_command() {
        let tools = unstarted_bash();
        for command in ["", "   ", "\n\t"] {
            let args = bash_args(command);
            let err = tools.preview("bash", &args).await.unwrap_err();
            assert_eq!(err.to_string(), "empty command");
            let err = tools.run("bash", &args).await.unwrap_err();
            assert_eq!(err.to_string(), "empty command");
            let err = tools.permission_scope("bash", &args).unwrap_err();
            assert_eq!(err.to_string(), "empty command");
        }
        let err = tools.preview("bash", "{}").await.unwrap_err();
        assert!(err.to_string().contains("bad arguments"));
    }
}
