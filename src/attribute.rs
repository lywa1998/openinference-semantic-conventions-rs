use bon::builder;
use serde::Serialize;
use serde_json::Value;
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[derive(Debug, Serialize)]
#[serde(rename_all = "UPPERCASE", tag = "openinference.span.kind")]
pub enum OpenInferenceSpan {
    /// Use this to wrap any preparation step where a prompt template is combined with prompt variables before being sent to an LLM
    Chain(ChainAttributes),
    /// Use this to wrap agentic flows that may involve multiple LLM and/or tool calls
    Agent(AgentAttributes),
    /// Use this to wrap the LLM call
    Llm(LlmAttributes),
    /// Use this to wrap any tool calls
    Tool(ToolCallAttributes),
    /// Use this to wrap any step that retrieves documents before sending it to the LLM
    Retriever(RetrieverAttributes),
    /// Use this to wrap the embedding step used by the retriever
    Embedding(EmbeddingAttributes),
    /// Use this to wrap any reranker step used after the retriever
    Reranker(RerankerAttributes),
}

impl OpenInferenceSpan {
    pub fn emit(self, span: &tracing::Span) -> Result<(), String> {
        let values = serde_json::to_value(self).map_err(|err| err.to_string())?;
        // values must be object
        let openinference_span = values.as_object().unwrap();
        // value have type: Boolean, Integer, Float, String, JSON String, List of floats, List of objects
        for (key, value) in openinference_span {
            let key = key.clone();
            if value.is_boolean() {
                span.set_attribute(key, value.as_bool().unwrap());
            } else if value.is_i64() {
                span.set_attribute(key, value.as_i64().unwrap());
            } else if value.is_f64() {
                span.set_attribute(key, value.as_f64().unwrap());
            } else if value.is_string() {
                let value = value.as_str().unwrap().to_string();
                span.set_attribute(key, value);
            } else if value.is_object() {
                let value = value.to_string().clone();
                span.set_attribute(key, value);
            } else if value.is_array() {
                let value = value.as_array().unwrap().clone();
                if value.len() == 0 {
                    continue;
                }
                if value[0].is_f64() {
                    let embedding: Vec<f64> =
                        value.into_iter().map(|v| v.as_f64().unwrap()).collect();
                    span.set_attribute(
                        key,
                        opentelemetry::Value::Array(opentelemetry::Array::F64(embedding)),
                    );
                } else if value[0].is_object() {
                    let objects: Vec<opentelemetry::StringValue> = value
                        .into_iter()
                        .map(|v| v.as_str().unwrap().to_string().into())
                        .collect();
                    span.set_attribute(
                        key,
                        opentelemetry::Value::Array(opentelemetry::Array::String(objects)),
                    );
                }
            }
        }
        Ok(())
    }
}

macro_rules! into_span {
    ($(($kind:ident, $t:ty),)+) => {
        $(
            impl From<$t> for OpenInferenceSpan {
                fn from(value: $t) -> Self {
                    Self::$kind(value)
                }
            }
        )+
    }
}

into_span!(
    (Chain, ChainAttributes),
    (Agent, AgentAttributes),
    (Llm, LlmAttributes),
    (Tool, ToolCallAttributes),
    (Retriever, RetrieverAttributes),
    (Embedding, EmbeddingAttributes),
    (Reranker, RerankerAttributes),
);

#[derive(Debug, Serialize)]
#[builder]
pub struct AgentAttributes {
    input: Input,
    output: Output,
}

#[derive(Debug, Clone, Serialize)]
#[builder]
pub struct Input {
    #[serde(rename = "input.value")]
    value: String,
    #[serde(rename = "input.mime_type")]
    mime_type: MimeType,
}

#[derive(Debug, Clone, Serialize)]
#[builder]
pub struct Output {
    #[serde(rename = "output.value")]
    value: String,
    #[serde(rename = "output.mime_type")]
    mime_type: MimeType,
}

#[derive(Debug, Clone, Serialize)]
pub enum MimeType {
    #[serde(rename = "text/plain")]
    Text,
    #[serde(rename = "application/json")]
    Json,
}

#[derive(Debug, Serialize)]
#[builder]
pub struct LlmAttributes {
    #[serde(rename = "llm.model_name")]
    model_name: String,
    #[serde(rename = "llm.input_messages")]
    input_messages: Vec<Message>,
    prompts: ChainAttributes,
    #[serde(rename = "llm.output_messsages")]
    output_messages: Vec<Message>,
    #[serde(rename = "llm.function_call")]
    function_call: Value,
    #[serde(rename = "llm.onvocation_parameters")]
    invocation_parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
#[builder]
pub struct ChainAttributes {
    #[serde(rename = "llm.prompt_template.template")]
    template: String,
    #[serde(rename = "llm.prompt_template.varables")]
    variables: Value,
    #[serde(rename = "llm.prompt_template.version")]
    version: String,
}

#[derive(Debug, Serialize)]
#[builder]
pub struct TokenCount {
    completion: i32,
    prompt: i32,
    total: i32,
}

#[derive(Debug, Serialize)]
#[builder]
pub struct Tool {
    #[serde(rename = "tool.name")]
    pub name: String,
    #[serde(rename = "tool.description")]
    pub description: String,
    #[serde(rename = "tool.parameters")]
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
#[builder]
pub struct Message {
    #[serde(rename = "message.role")]
    role: MessageRole,
    #[serde(rename = "message.content")]
    content: String,
    #[serde(rename = "message.name")]
    name: String,
    #[serde(rename = "message.tool_calls")]
    tool_calls: Vec<ToolCallAttributes>,
    #[serde(rename = "message.function_call_name")]
    function_call_name: String,
    #[serde(rename = "message.function_call_arguments_json")]
    function_call_arguments: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    System,
}

#[derive(Debug, Serialize)]
#[builder]
pub struct RetrieverAttributes {
    #[serde(rename = "retriever.documents")]
    documents: Vec<Document>,
}

#[derive(Debug, Serialize)]
#[builder]
pub struct RerankerAttributes {
    #[serde(rename = "reranker.input_documents")]
    input_documents: Vec<Document>,
    #[serde(rename = "reranker.output_documents")]
    output_documents: Vec<Document>,
    #[serde(rename = "reranker.query")]
    query: String,
    #[serde(rename = "reranker.model_name")]
    model_name: String,
    #[serde(rename = "reranker.top_k")]
    top_k: i32,
}

#[derive(Debug, Clone, Serialize)]
#[builder]
pub struct Document {
    id: DocumentId,
    content: String,
    metadata: Value,
    score: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum DocumentId {
    Number(i64),
    Uuid(String),
}

#[derive(Debug, Serialize)]
#[builder]
pub struct EmbeddingAttributes {
    #[serde(rename = "embedding.model_name")]
    pub model_name: String,
    #[serde(rename = "embedding.embeddings")]
    pub embeddings: Vec<Embedding>,
}

#[derive(Debug, Clone, Serialize)]
#[builder]
pub struct Embedding {
    #[serde(rename = "embedding.text")]
    pub text: String,
    #[serde(rename = "embedding.vector")]
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[builder]
pub struct ToolCallAttributes {
    #[serde(rename = "tool_call.function.name")]
    pub name: String,
    #[serde(rename = "tool_call.function.arguments")]
    pub arguments: Value,
}
