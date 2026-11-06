//! Streaming chat over the OpenAI-compatible `/v1/chat/completions` endpoint.
//!
//! The response is server-sent events: `data: {json}` lines separated by blank
//! lines, ending with `data: [DONE]`. [`SseParser`] turns raw bytes into those
//! `data` payloads, and [`parse_chunk`] turns one payload into [`ChatDelta`]s.
//! Tool calls stream in fragments; [`ToolCallAssembler`] puts them back together.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{Message, Role, ToolCall, tools::ToolSpec};

/// One piece of a streamed reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatDelta {
    /// More answer text.
    Content(String),
    /// More of the model's thinking (LM Studio's `delta.reasoning_content`).
    Reasoning(String),
    /// A fragment of a tool call (one entry of `delta.tool_calls`). The first
    /// fragment of a call carries its `id` and `name`; `arguments` is a piece of
    /// the JSON string, possibly all of it. Feed these to a [`ToolCallAssembler`].
    ToolCall {
        /// Which call of this reply the fragment belongs to.
        index: usize,
        /// The call's id, on its first fragment.
        id: Option<String>,
        /// The tool's name, on its first fragment.
        name: Option<String>,
        /// More of the arguments JSON (may be empty).
        arguments: String,
    },
    /// The reply is complete. `finish_reason` is e.g. `stop`, `length` or
    /// `tool_calls`, or
    /// `None` when only the closing `[DONE]` said so.
    Done {
        /// Why generation stopped, as the server reported it.
        finish_reason: Option<String>,
    },
    /// Tokens the request used, as the server counted them. Sent once, after
    /// `Done`, since the request asks for usage (see [`request_body`]).
    Usage {
        /// Everything the model read: system prompt, tools, the conversation.
        prompt_tokens: u64,
        /// What it wrote, thinking included.
        completion_tokens: u64,
    },
}

/// Splits a byte stream of server-sent events into `data:` payloads.
///
/// Each `data:` line is returned as one payload. That is all the chat endpoint
/// sends (one JSON object per line); multi-line events, `event:` and `id:`
/// fields, and `:` comments are not needed and are skipped.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Bytes after the last complete line, kept until the rest arrives.
    pending: Vec<u8>,
}

impl SseParser {
    /// Creates a parser with nothing buffered.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds the next bytes and returns every payload whose line is now complete.
    ///
    /// Lines may end in `\n` or `\r\n`. Lines are decoded only once complete, so
    /// a multi-byte character split across two calls is not mangled.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(bytes);
        let Some(last_newline) = self.pending.iter().rposition(|&b| b == b'\n') else {
            return Vec::new();
        };
        let rest = self.pending.split_off(last_newline + 1);
        let complete = std::mem::replace(&mut self.pending, rest);

        complete
            .split(|&b| b == b'\n')
            .filter_map(|line| {
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                let data = line.strip_prefix(b"data:")?;
                let data = data.strip_prefix(b" ").unwrap_or(data);
                Some(String::from_utf8_lossy(data).into_owned())
            })
            .collect()
    }
}

/// One `data:` JSON object. Fields the harness does not use are ignored.
#[derive(Debug, Deserialize)]
struct Chunk {
    /// Empty on the trailing usage-only chunk.
    #[serde(default)]
    choices: Vec<Choice>,
    /// Set instead of `choices` when the server fails mid-stream.
    error: Option<Value>,
    /// Token counts; only on the usage chunk.
    usage: Option<UsageCounts>,
}

/// The `usage` object of a chunk.
#[derive(Debug, Deserialize)]
struct UsageCounts {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// One entry of `choices`; the endpoint sends exactly one per chunk.
#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    delta: Delta,
    finish_reason: Option<String>,
}

/// New text in a chunk. Any field may be absent.
#[derive(Debug, Default, Deserialize)]
struct Delta {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallFragment>,
}

/// One entry of `delta.tool_calls`.
#[derive(Debug, Deserialize)]
struct ToolCallFragment {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    #[serde(default)]
    function: FunctionFragment,
}

/// The `function` part of a tool call fragment.
#[derive(Debug, Default, Deserialize)]
struct FunctionFragment {
    name: Option<String>,
    arguments: Option<String>,
}

/// Turns one `data:` payload into deltas, in the order reasoning, content,
/// tool calls, done, usage.
///
/// `[DONE]` becomes `Done { finish_reason: None }`; a chunk with a
/// `finish_reason` becomes `Done` with it. Both usually arrive, so callers keep
/// only the first `Done`. A `usage` object becomes `Usage`. Empty strings
/// yield nothing.
pub fn parse_chunk(payload: &str) -> Result<Vec<ChatDelta>> {
    if payload.trim() == "[DONE]" {
        return Ok(vec![ChatDelta::Done {
            finish_reason: None,
        }]);
    }
    let chunk: Chunk =
        serde_json::from_str(payload).with_context(|| format!("bad stream chunk: {payload}"))?;
    if let Some(error) = chunk.error {
        bail!("server error mid-stream: {error}");
    }

    let mut deltas = Vec::new();
    for choice in chunk.choices {
        let non_empty = |s: &String| !s.is_empty();
        if let Some(text) = choice.delta.reasoning_content.filter(non_empty) {
            deltas.push(ChatDelta::Reasoning(text));
        }
        if let Some(text) = choice.delta.content.filter(non_empty) {
            deltas.push(ChatDelta::Content(text));
        }
        for call in choice.delta.tool_calls {
            deltas.push(ChatDelta::ToolCall {
                index: call.index,
                id: call.id.filter(non_empty),
                name: call.function.name.filter(non_empty),
                arguments: call.function.arguments.unwrap_or_default(),
            });
        }
        if let Some(finish_reason) = choice.finish_reason {
            deltas.push(ChatDelta::Done {
                finish_reason: Some(finish_reason),
            });
        }
    }
    if let Some(usage) = chunk.usage {
        deltas.push(ChatDelta::Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
        });
    }
    Ok(deltas)
}

/// Puts streamed [`ChatDelta::ToolCall`] fragments back together, by index.
#[derive(Debug, Default)]
pub struct ToolCallAssembler {
    calls: Vec<ToolCall>,
}

impl ToolCallAssembler {
    /// Adds one fragment: `id` and `name` are set when present, `arguments` is appended.
    pub fn push(
        &mut self,
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: &str,
    ) {
        if self.calls.len() <= index {
            self.calls.resize_with(index + 1, ToolCall::default);
        }
        let call = &mut self.calls[index];
        if let Some(id) = id {
            call.id = id;
        }
        if let Some(name) = name {
            call.name = name;
        }
        call.arguments.push_str(arguments);
    }

    /// The complete calls, in index order. Indexes that never got a name are
    /// dropped, and a call without an id gets a made-up unique one.
    pub fn finish(self) -> Vec<ToolCall> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        self.calls
            .into_iter()
            .filter(|call| !call.name.is_empty())
            .map(|mut call| {
                if call.id.is_empty() {
                    call.id = format!("loops_call_{}", NEXT_ID.fetch_add(1, Ordering::Relaxed));
                }
                call
            })
            .collect()
    }
}

/// The `messages` array of a chat request. [`Role::Harness`] lines are left out;
/// an assistant message with tool calls and a tool result take the shapes the
/// OpenAI API expects.
pub fn to_openai_messages(messages: &[Message]) -> Value {
    messages
        .iter()
        .filter_map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant if !m.tool_calls.is_empty() => return Some(tool_request(m)),
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => {
                    return Some(json!({
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.content,
                    }));
                }
                Role::Harness => return None,
            };
            Some(json!({ "role": role, "content": m.content }))
        })
        .collect()
}

/// An assistant message that asks for tools. Empty text is sent as `null`.
fn tool_request(message: &Message) -> Value {
    let calls: Vec<Value> = message
        .tool_calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": { "name": call.name, "arguments": call.arguments },
            })
        })
        .collect();
    let content = (!message.content.is_empty()).then_some(&message.content);
    json!({ "role": "assistant", "content": content, "tool_calls": calls })
}

/// The `tools` array of a chat request.
pub fn to_openai_tools(tools: &[ToolSpec]) -> Value {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        })
        .collect()
}

/// A streaming chat request body. `tools` and `tool_choice` are only included
/// when there are tools to offer. Usage is always asked for, so the stream
/// ends with the token counts ([`ChatDelta::Usage`]).
pub fn request_body(model: &str, messages: &[Message], tools: &[ToolSpec]) -> Value {
    let mut body = json!({
        "model": model,
        "messages": to_openai_messages(messages),
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if !tools.is_empty() {
        body["tools"] = to_openai_tools(tools);
        body["tool_choice"] = json!("auto");
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real lines from LM Studio 2026-11 serving `google/gemma-4-e4b` (ids
    /// shortened), plus one invented content chunk with a multi-byte emoji so
    /// byte-level splits cross a UTF-8 boundary.
    const FIXTURE: &str = concat!(
        r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1793902852,"model":"google/gemma-4-e4b","system_fingerprint":"google/gemma-4-e4b","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"The"},"logprobs":null,"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1793902852,"model":"google/gemma-4-e4b","system_fingerprint":"google/gemma-4-e4b","choices":[{"index":0,"delta":{"content":" from"},"logprobs":null,"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"content":" 👋"},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1793902852,"model":"google/gemma-4-e4b","system_fingerprint":"google/gemma-4-e4b","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}"#,
        "\n\n",
        r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1793902853,"model":"google/gemma-4-e4b","system_fingerprint":"google/gemma-4-e4b","choices":[],"usage":{"prompt_tokens":23,"completion_tokens":4,"total_tokens":27,"completion_tokens_details":{"reasoning_tokens":0}}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    fn payloads(text: &str) -> Vec<String> {
        SseParser::new().push(text.as_bytes())
    }

    #[test]
    fn parser_returns_one_payload_per_data_line() {
        let out = payloads(FIXTURE);
        assert_eq!(out.len(), 6);
        assert!(out[0].starts_with(r#"{"id":"chatcmpl-1""#));
        assert_eq!(out[5], "[DONE]");
    }

    #[test]
    fn parser_output_is_the_same_for_every_split_point() {
        for input in [FIXTURE.to_owned(), FIXTURE.replace('\n', "\r\n")] {
            let whole = payloads(&input);
            let bytes = input.as_bytes();
            for split in 0..=bytes.len() {
                let mut parser = SseParser::new();
                let mut out = parser.push(&bytes[..split]);
                out.extend(parser.push(&bytes[split..]));
                assert_eq!(out, whole, "split at byte {split}");
            }
        }
    }

    #[test]
    fn parser_holds_a_partial_line_and_skips_other_fields() {
        let mut parser = SseParser::new();
        assert!(parser.push(b": comment\nevent: x\ndata: {\"a\"").is_empty());
        assert_eq!(parser.push(b":1}\r\n\r\n"), [r#"{"a":1}"#]);
        assert_eq!(parser.push(b"data:no-space\n"), ["no-space"]);
    }

    #[test]
    fn chunks_map_to_deltas() {
        let deltas: Vec<ChatDelta> = payloads(FIXTURE)
            .iter()
            .flat_map(|p| parse_chunk(p).unwrap())
            .collect();
        assert_eq!(
            deltas,
            [
                ChatDelta::Reasoning("The".into()),
                ChatDelta::Content(" from".into()),
                ChatDelta::Content(" 👋".into()),
                ChatDelta::Done {
                    finish_reason: Some("stop".into())
                },
                ChatDelta::Usage {
                    prompt_tokens: 23,
                    completion_tokens: 4
                },
                ChatDelta::Done {
                    finish_reason: None
                },
            ]
        );
    }

    #[test]
    fn choices_come_before_usage_in_one_chunk() {
        let both = r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}"#;
        assert_eq!(
            parse_chunk(both).unwrap(),
            [
                ChatDelta::Content("hi".into()),
                ChatDelta::Done {
                    finish_reason: Some("stop".into())
                },
                ChatDelta::Usage {
                    prompt_tokens: 5,
                    completion_tokens: 1
                },
            ]
        );
    }

    #[test]
    fn empty_text_is_skipped_and_errors_surface() {
        let empty =
            r#"{"choices":[{"delta":{"content":"","reasoning_content":""},"finish_reason":null}]}"#;
        assert!(parse_chunk(empty).unwrap().is_empty());
        let err = parse_chunk(r#"{"error":"model crashed"}"#).unwrap_err();
        assert!(format!("{err:#}").contains("model crashed"));
        assert!(parse_chunk("not json").is_err());
    }

    #[test]
    fn roles_map_to_openai_names() {
        let messages = [
            Message::new(Role::System, "be brief"),
            Message::new(Role::User, "hi"),
            Message::new(Role::Assistant, "hello"),
        ];
        assert_eq!(
            to_openai_messages(&messages),
            json!([
                { "role": "system", "content": "be brief" },
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": "hello" },
            ])
        );
    }

    #[test]
    fn harness_lines_are_not_sent() {
        let messages = [
            Message::new(Role::Harness, "loaded google/gemma-4-e4b"),
            Message::new(Role::User, "hi"),
            Message::new(Role::Harness, "error: boom"),
        ];
        assert_eq!(
            to_openai_messages(&messages),
            json!([{ "role": "user", "content": "hi" }])
        );
    }

    /// Tool-call fragments in the OpenAI streaming shape: the first fragment of
    /// a call has `id`, `type` and `name`, later ones only more `arguments`.
    /// Call 1 arrives whole in one fragment, as some local servers send it.
    const TOOL_FIXTURE: &str = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"README.md\"}"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"list_files","arguments":"{}"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    #[test]
    fn tool_call_fragments_parse_and_assemble() {
        let deltas: Vec<ChatDelta> = payloads(TOOL_FIXTURE)
            .iter()
            .flat_map(|p| parse_chunk(p).unwrap())
            .collect();
        assert_eq!(
            deltas[0],
            ChatDelta::ToolCall {
                index: 0,
                id: Some("call_a".into()),
                name: Some("read_file".into()),
                arguments: String::new(),
            }
        );
        assert_eq!(
            deltas[4],
            ChatDelta::Done {
                finish_reason: Some("tool_calls".into())
            }
        );

        let mut calls = ToolCallAssembler::default();
        for delta in deltas {
            if let ChatDelta::ToolCall {
                index,
                id,
                name,
                arguments,
            } = delta
            {
                calls.push(index, id, name, &arguments);
            }
        }
        assert_eq!(
            calls.finish(),
            [
                ToolCall {
                    id: "call_a".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"README.md"}"#.into(),
                },
                ToolCall {
                    id: "call_b".into(),
                    name: "list_files".into(),
                    arguments: "{}".into(),
                },
            ]
        );
    }

    #[test]
    fn assembler_fills_missing_ids_and_drops_nameless_calls() {
        let mut calls = ToolCallAssembler::default();
        calls.push(1, None, Some("list_files".into()), "");
        let calls = calls.finish();
        assert_eq!(calls.len(), 1, "index 0 never got a name");
        assert!(calls[0].id.starts_with("loops_call_"));
    }

    #[test]
    fn tool_messages_take_the_openai_shapes() {
        let call = ToolCall {
            id: "call_a".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"README.md"}"#.into(),
        };
        let messages = [
            Message::assistant_with_tool_calls("", vec![call.clone()]),
            Message::tool_result("call_a", "1: # loops"),
            Message::assistant_with_tool_calls("looking", vec![call]),
        ];
        let call_json = json!([{
            "id": "call_a",
            "type": "function",
            "function": { "name": "read_file", "arguments": r#"{"path":"README.md"}"# },
        }]);
        assert_eq!(
            to_openai_messages(&messages),
            json!([
                { "role": "assistant", "content": null, "tool_calls": call_json },
                { "role": "tool", "tool_call_id": "call_a", "content": "1: # loops" },
                { "role": "assistant", "content": "looking", "tool_calls": call_json },
            ])
        );
    }

    #[test]
    fn tools_are_only_offered_when_there_are_some() {
        let messages = [Message::new(Role::User, "hi")];
        let plain = request_body("m", &messages, &[]);
        assert_eq!(plain.get("tools"), None);
        assert_eq!(plain.get("tool_choice"), None);
        assert_eq!(plain["stream"], true);
        assert_eq!(plain["stream_options"], json!({ "include_usage": true }));

        let spec = ToolSpec {
            name: "read_file".into(),
            description: "Read a file".into(),
            parameters: json!({ "type": "object" }),
        };
        let body = request_body("m", &messages, &[spec]);
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(
            body["tools"],
            json!([{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": { "type": "object" },
                },
            }])
        );
    }
}
