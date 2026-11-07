//! Saved conversations: one JSON file per session, grouped by the directory
//! loops ran in.
//!
//! ```text
//!  ~/.loops/sessions/<project-key>/<id>.json
//! ```
//!
//! The project key is the working directory's absolute path with every
//! character that is not an ASCII letter or digit replaced by `-`, so
//! `D:\projects\loops` keeps its sessions in `D--projects-loops`. An id is the
//! time it was made, `<unix seconds>-<milliseconds>`, so ids sort in the order
//! they were made. Nothing here ever deletes a file.
//!
//! A [`Session`] holds the messages and a little metadata; what a front end
//! needs to draw the transcript as it was rides along in [`Session::view`],
//! which this module never looks into.

use std::{
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::{Message, Role};

/// Longest title, in characters, before it is cut and `…` appended.
const TITLE_CHARS: usize = 60;

/// One saved conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    /// File name without `.json`; see [`SessionStore::new_id`].
    pub id: String,
    /// What the session is about, for listings: see [`title_from`].
    pub title: String,
    /// The model the session last ran with, if it ever ran one.
    pub model: Option<String>,
    /// The working directory, as the front end saw it.
    pub cwd: String,
    /// When the session started, in unix seconds.
    pub created: u64,
    /// When it was last saved, in unix seconds.
    pub updated: u64,
    /// The whole conversation, harness lines included. The server's token
    /// count is not kept: after a resume the estimate is a guess until the
    /// next request is counted.
    pub messages: Vec<Message>,
    /// The front end's own display state, opaque here; `null` when absent.
    #[serde(default)]
    pub view: serde_json::Value,
}

/// What a listing shows of a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummary {
    /// The session's id.
    pub id: String,
    /// The session's title; may be empty.
    pub title: String,
    /// When it was last saved, in unix seconds.
    pub updated: u64,
    /// How many messages it holds.
    pub messages: usize,
}

impl From<&Session> for SessionSummary {
    fn from(session: &Session) -> Self {
        Self {
            id: session.id.clone(),
            title: session.title.clone(),
            updated: session.updated,
            messages: session.messages.len(),
        }
    }
}

/// The sessions of one working directory: a directory of `<id>.json` files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStore {
    dir: PathBuf,
}

/// Seconds since the unix epoch, now.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// `cwd` with every character that is not an ASCII letter or digit as `-`.
pub fn project_key(cwd: &Path) -> String {
    cwd.to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

impl SessionStore {
    /// The store for sessions started in `cwd`, under the home directory;
    /// `None` when there is no home directory.
    pub fn for_project(cwd: &Path) -> Option<SessionStore> {
        let home = std::env::home_dir()?;
        Some(Self::in_dir(
            home.join(".loops").join("sessions").join(project_key(cwd)),
        ))
    }

    /// A store kept in `dir` (created on the first save).
    pub fn in_dir(dir: PathBuf) -> SessionStore {
        SessionStore { dir }
    }

    /// Where the session files go.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// A fresh id: the time in milliseconds as `<seconds>-<millis>`, e.g.
    /// `1794067200-042`. Ids made in one process always increase, even
    /// within one millisecond, and sort as strings in the order they were made.
    pub fn new_id() -> String {
        static LAST: AtomicU64 = AtomicU64::new(0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_millis() as u64);
        let mut last = LAST.load(Ordering::Relaxed);
        let millis = loop {
            let next = now.max(last + 1);
            match LAST.compare_exchange_weak(last, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break next,
                Err(seen) => last = seen,
            }
        };
        format!("{}-{:03}", millis / 1000, millis % 1000)
    }

    /// Writes `session` to `<id>.json` as pretty JSON, through a temporary
    /// file renamed over it so a crash never leaves half a file.
    pub fn save(&self, session: &Session) -> io::Result<()> {
        let path = self.path(&session.id)?;
        std::fs::create_dir_all(&self.dir)?;
        let json = serde_json::to_vec_pretty(session)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, &path)
    }

    /// The session `id`, or `None` if there is none. An id that could reach
    /// outside the store (a path separator, `..`) is an error.
    pub fn load(&self, id: &str) -> io::Result<Option<Session>> {
        let path = self.path(id)?;
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };
        Ok(Some(serde_json::from_str(&text)?))
    }

    /// Every session that can be read, newest first (by `updated`, then by
    /// id). Files that cannot be read or parsed are skipped; a store with no
    /// directory yet is empty.
    pub fn list(&self) -> io::Result<Vec<SessionSummary>> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut sessions: Vec<SessionSummary> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
            .filter_map(|path| std::fs::read_to_string(path).ok())
            .filter_map(|text| serde_json::from_str::<Session>(&text).ok())
            .map(|session| SessionSummary::from(&session))
            .collect();
        sessions.sort_by(|a, b| b.updated.cmp(&a.updated).then_with(|| b.id.cmp(&a.id)));
        Ok(sessions)
    }

    /// The most recently saved session, if any.
    pub fn latest(&self) -> io::Result<Option<SessionSummary>> {
        Ok(self.list()?.into_iter().next())
    }

    /// `<dir>/<id>.json`, refusing ids that are not a plain file name.
    fn path(&self, id: &str) -> io::Result<PathBuf> {
        if id.is_empty() || id.contains(['/', '\\', ':']) || id.contains("..") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("not a session id: {id:?}"),
            ));
        }
        Ok(self.dir.join(format!("{id}.json")))
    }
}

/// The first line of the first user message, trimmed and cut to
/// [`TITLE_CHARS`] characters with `…` after it when cut; empty when no user
/// has spoken.
pub fn title_from(messages: &[Message]) -> String {
    let Some(first) = messages.iter().find(|m| m.role == Role::User) else {
        return String::new();
    };
    let line = first.content.trim().lines().next().unwrap_or("").trim();
    if line.chars().count() <= TITLE_CHARS {
        return line.to_owned();
    }
    let cut: String = line.chars().take(TITLE_CHARS).collect();
    format!("{}…", cut.trim_end())
}

/// How long ago `then` was at `now` (both unix seconds), roughly: `just
/// now`, `5 min ago`, `3 h ago`, `2 d ago`.
pub fn relative_time(then: u64, now: u64) -> String {
    let secs = now.saturating_sub(then);
    match secs {
        0..60 => "just now".to_owned(),
        60..3_600 => format!("{} min ago", secs / 60),
        3_600..86_400 => format!("{} h ago", secs / 3_600),
        _ => format!("{} d ago", secs / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolCall, test_support::TempDir};

    fn session(id: &str, updated: u64, first: &str) -> Session {
        Session {
            id: id.into(),
            title: title_from(&[Message::new(Role::User, first)]),
            model: Some("google/gemma-4-e4b".into()),
            cwd: r"D:\projects\loops".into(),
            created: updated - 10,
            updated,
            messages: vec![
                Message::new(Role::User, first),
                Message::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "c1".into(),
                        name: "read_file".into(),
                        arguments: r#"{"path":"a.txt"}"#.into(),
                    }],
                ),
                Message::tool_result("c1", "1: hi"),
                Message::new(Role::Harness, "loaded"),
            ],
            view: serde_json::json!({"reply_ends": [[1, "Empty"]]}),
        }
    }

    #[test]
    fn a_saved_session_loads_back_the_same() {
        let dir = TempDir::new("session-roundtrip");
        let store = SessionStore::in_dir(dir.0.join("nested"));
        let saved = session("1700000000-001", 1_700_000_000, "fix the bug");
        store.save(&saved).unwrap();
        assert_eq!(store.load(&saved.id).unwrap(), Some(saved.clone()));
        assert!(!dir.0.join("nested/1700000000-001.json.tmp").exists());

        // Saving again replaces the file.
        let mut later = saved.clone();
        later.messages.push(Message::new(Role::User, "more"));
        store.save(&later).unwrap();
        assert_eq!(store.load(&saved.id).unwrap(), Some(later));
        assert_eq!(store.load("1700000000-999").unwrap(), None);
    }

    #[test]
    fn a_session_without_a_view_loads_with_null() {
        let dir = TempDir::new("session-no-view");
        let mut text = serde_json::to_value(session("a", 100, "hi")).unwrap();
        text.as_object_mut().unwrap().remove("view");
        dir.write("a.json", text.to_string());
        let loaded = SessionStore::in_dir(dir.0.clone()).load("a").unwrap();
        assert_eq!(loaded.unwrap().view, serde_json::Value::Null);
    }

    #[test]
    fn list_is_newest_first_and_skips_what_it_cannot_read() {
        let dir = TempDir::new("session-list");
        let store = SessionStore::in_dir(dir.0.clone());
        store.save(&session("100-000", 100, "old")).unwrap();
        store.save(&session("300-000", 300, "new")).unwrap();
        store.save(&session("200-000", 200, "tie a")).unwrap();
        store.save(&session("200-500", 200, "tie b")).unwrap();
        dir.write("broken.json", "{ not json");
        dir.write("notes.txt", "not a session");

        let listed = store.list().unwrap();
        let ids: Vec<&str> = listed.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["300-000", "200-500", "200-000", "100-000"]);
        assert_eq!(
            listed[0],
            SessionSummary {
                id: "300-000".into(),
                title: "new".into(),
                updated: 300,
                messages: 4,
            }
        );
        assert_eq!(store.latest().unwrap().unwrap().id, "300-000");
    }

    #[test]
    fn an_empty_or_absent_store_has_no_latest() {
        let dir = TempDir::new("session-empty");
        assert_eq!(SessionStore::in_dir(dir.0.clone()).latest().unwrap(), None);
        let absent = SessionStore::in_dir(dir.0.join("missing"));
        assert!(absent.list().unwrap().is_empty());
        assert_eq!(absent.latest().unwrap(), None);
    }

    #[test]
    fn ids_that_leave_the_store_are_refused() {
        let store = SessionStore::in_dir(std::env::temp_dir().join("loops-session-bad-ids"));
        for id in ["", "../x", "a/b", r"a\b", "..", "C:x"] {
            let err = store.load(id).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{id:?}");
        }
    }

    #[test]
    fn new_ids_increase_and_have_the_time_shape() {
        let ids: Vec<String> = (0..50).map(|_| SessionStore::new_id()).collect();
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "{ids:?}");
        let (secs, millis) = ids[0].split_once('-').unwrap();
        assert!(secs.parse::<u64>().unwrap() >= unix_now() - 5);
        assert_eq!(millis.len(), 3);
    }

    #[test]
    fn the_project_key_keeps_only_letters_and_digits() {
        assert_eq!(
            project_key(Path::new(r"D:\projects\loops")),
            "D--projects-loops"
        );
        assert_eq!(project_key(Path::new("/home/me/my app")), "-home-me-my-app");
        assert_eq!(project_key(Path::new("/tmp/ü")), "-tmp--");
    }

    #[test]
    fn the_title_is_the_first_user_line_cut_at_sixty() {
        assert_eq!(title_from(&[]), "");
        assert_eq!(title_from(&[Message::new(Role::Harness, "hi")]), "");
        let messages = [
            Message::new(Role::Harness, "AGENTS.md loaded"),
            Message::new(Role::User, "  fix the bug\nin main.rs"),
            Message::new(Role::User, "later"),
        ];
        assert_eq!(title_from(&messages), "fix the bug");

        let exact = "x".repeat(60);
        assert_eq!(title_from(&[Message::new(Role::User, &exact)]), exact);
        let long = format!("{}y", "x".repeat(60));
        assert_eq!(
            title_from(&[Message::new(Role::User, long)]),
            format!("{}…", "x".repeat(60))
        );
    }

    #[test]
    fn relative_time_steps_at_each_unit() {
        let at = |secs: u64| relative_time(1_000_000 - secs, 1_000_000);
        assert_eq!(at(0), "just now");
        assert_eq!(at(59), "just now");
        assert_eq!(at(60), "1 min ago");
        assert_eq!(at(3_599), "59 min ago");
        assert_eq!(at(3_600), "1 h ago");
        assert_eq!(at(86_399), "23 h ago");
        assert_eq!(at(86_400), "1 d ago");
        assert_eq!(at(10 * 86_400), "10 d ago");
        assert_eq!(
            relative_time(200, 100),
            "just now",
            "a clock that went back"
        );
    }
}
