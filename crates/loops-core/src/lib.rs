//! Core types for the loops harness. No UI code lives here (see D5).
//!
//! The in-memory conversation model lives here; LM Studio control is in
//! [`lmstudio`], behind the [`backend::ModelBackend`] seam, and streamed chat
//! parsing is in [`chat`]. The agent loop is in [`agent`], and the tools it can
//! run are in [`tools`].

pub mod agent;
pub mod backend;
pub mod chat;
pub mod lmstudio;
pub mod tools;

/// Who authored a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The person typing at the keyboard.
    User,
    /// The model.
    Assistant,
    /// Instructions to the model that are not shown as a turn.
    System,
    /// The result of a tool the model asked to run.
    Tool,
    /// The harness itself speaking to the user; shown in the transcript, never sent to a model.
    Harness,
}

/// A tool the model asked to run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCall {
    /// Pairs the call with its result ([`Message::tool_call_id`]).
    pub id: String,
    /// Which tool, e.g. `read_file`.
    pub name: String,
    /// The arguments as the raw JSON string the model produced.
    pub arguments: String,
}

/// One turn in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Author of the message.
    pub role: Role,
    /// Plain-text body (for [`Role::Tool`], the tool's output).
    pub content: String,
    /// Tools the model asked for. Non-empty only on [`Role::Assistant`] messages.
    pub tool_calls: Vec<ToolCall>,
    /// Which call this answers. Set only on [`Role::Tool`] messages.
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Builds a message with the given role and content.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// The output of the tool call `call_id`.
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(call_id.into()),
            ..Self::new(Role::Tool, content)
        }
    }

    /// A model reply that asks for `calls` (with any text it said first).
    pub fn assistant_with_tool_calls(text: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            tool_calls: calls,
            ..Self::new(Role::Assistant, text)
        }
    }
}

/// An ordered list of messages for the current run. It changes only at the end.
#[derive(Debug, Clone, Default)]
pub struct Conversation {
    messages: Vec<Message>,
}

impl Conversation {
    /// Creates an empty conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a message to the end of the conversation.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// The newest message, for appending a streamed reply in place.
    pub fn last_mut(&mut self) -> Option<&mut Message> {
        self.messages.last_mut()
    }

    /// The message at `index`, for a streamed reply that is no longer the newest.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Message> {
        self.messages.get_mut(index)
    }

    /// Removes and returns the message at `index` (e.g. a reply that never
    /// arrived), shifting later ones down. Panics if `index` is out of range.
    pub fn remove(&mut self, index: usize) -> Message {
        self.messages.remove(index)
    }

    /// All messages, oldest first.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}

/// Helpers shared by the tests of several modules.
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::PathBuf;

    /// A fresh directory under the system temp dir, deleted on drop.
    pub(crate) struct TempDir(pub(crate) PathBuf);

    impl TempDir {
        /// `loops-<name>-<pid>`, emptied first. Tests run in parallel in one
        /// process, so `name` must be unique per test.
        pub(crate) fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("loops-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        /// Writes `contents` to `rel`, creating its directories.
        pub(crate) fn write(&self, rel: &str, contents: impl AsRef<[u8]>) {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }

        /// The text of `rel`, byte for byte.
        pub(crate) fn read(&self, rel: &str) -> String {
            std::fs::read_to_string(self.0.join(rel)).unwrap()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_conversation_is_empty() {
        assert!(Conversation::new().messages().is_empty());
    }

    #[test]
    fn push_appends_in_order() {
        let mut convo = Conversation::new();
        convo.push(Message::new(Role::User, "hi"));
        convo.push(Message::new(Role::Assistant, "hello"));

        let msgs = convo.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0], Message::new(Role::User, "hi"));
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[1].content, "hello");
    }

    #[test]
    fn messages_can_be_edited_in_place_and_removed() {
        let mut convo = Conversation::new();
        assert!(convo.last_mut().is_none());
        convo.push(Message::new(Role::Assistant, "he"));
        convo.push(Message::new(Role::User, "x"));
        convo.last_mut().unwrap().content.push('y');
        convo.get_mut(0).unwrap().content.push_str("llo");
        assert!(convo.get_mut(2).is_none());
        assert_eq!(convo.messages()[0].content, "hello");
        assert_eq!(convo.remove(0).content, "hello");
        assert_eq!(convo.messages(), [Message::new(Role::User, "xy")]);
    }

    #[test]
    fn tool_constructors_set_the_tool_fields() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"README.md"}"#.into(),
        };
        let ask = Message::assistant_with_tool_calls("", vec![call.clone()]);
        assert_eq!(ask.role, Role::Assistant);
        assert_eq!(ask.tool_calls, [call]);

        let result = Message::tool_result("call_1", "1: # loops");
        assert_eq!(result.role, Role::Tool);
        assert_eq!(result.tool_call_id.as_deref(), Some("call_1"));
        assert!(result.tool_calls.is_empty());
        assert_eq!(Message::new(Role::User, "hi").tool_call_id, None);
    }
}
