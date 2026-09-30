//! Chat with a GGUF model through llama.cpp.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use serde::{Deserialize, Serialize};

/// llama.cpp is set up once per process and never taken down; it can't be
/// set up twice.
static BACKEND: LazyLock<LlamaBackend> = LazyLock::new(|| {
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    LlamaBackend::init().expect("setting up llama.cpp")
});

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct Reply {
    pub content: String,
    pub finish: Finish,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// It reached `max_tokens`.
    Length,
}

pub struct Chat {
    model: LlamaModel,
}

impl Chat {
    pub fn load(path: &Path) -> Result<Chat> {
        let model = LlamaModel::load_from_file(&BACKEND, path, &LlamaModelParams::default())?;
        Ok(Chat { model })
    }

    /// Writes the model's next turn. A temperature of 0 always picks the
    /// likeliest token.
    pub fn reply(&self, messages: &[Message], max_tokens: u32, temperature: f32) -> Result<Reply> {
        let tokens = self
            .model
            .str_to_token(&prompt(messages)?, AddBos::Always)?;
        let n_prompt = tokens.len();
        let n_ctx = n_prompt + max_tokens as usize;
        let fits = self.model.n_ctx_train() as usize;
        if n_ctx > fits {
            bail!("the messages and max_tokens come to {n_ctx} tokens; {fits} fit");
        }

        // About one a core: more only fight each other and whatever else runs.
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32 / 2);
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx as u32))
            .with_n_batch(n_prompt as u32)
            .with_n_threads(threads)
            .with_n_threads_batch(threads);
        let mut ctx = self.model.new_context(&BACKEND, params)?;

        let mut batch = LlamaBatch::new(n_prompt, 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == n_prompt - 1)?;
        }
        ctx.decode(&mut batch)?;

        // Gemma's recommended sampling; u32::MAX seeds it randomly.
        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_k(64),
                LlamaSampler::top_p(0.95, 1),
                LlamaSampler::temp(temperature),
                LlamaSampler::dist(u32::MAX),
            ])
        };
        let mut text = Vec::new();
        let mut finish = Finish::Length;
        let mut completion_tokens = 0;
        while completion_tokens < max_tokens as usize {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if self.model.is_eog_token(token) {
                finish = Finish::Stop;
                break;
            }
            // Control tokens, like an empty thinking channel, come out as nothing.
            text.extend(self.model.token_to_piece_bytes(token, 256, false, None)?);
            batch.clear();
            batch.add(token, (n_prompt + completion_tokens) as i32, &[0], true)?;
            ctx.decode(&mut batch)?;
            completion_tokens += 1;
        }

        Ok(Reply {
            content: String::from_utf8_lossy(&text).into_owned(),
            finish,
            prompt_tokens: n_prompt,
            completion_tokens,
        })
    }
}

/// Gemma 4's turns, as its chat template writes plain text with thinking off.
/// llama.cpp's built-in templates don't know them.
fn prompt(messages: &[Message]) -> Result<String> {
    let mut prompt = String::new();
    for (i, message) in messages.iter().enumerate() {
        let role = match message.role.as_str() {
            "system" | "developer" if i == 0 => "system",
            "user" => "user",
            "assistant" => "model",
            role => bail!("can't take a {role} message there"),
        };
        prompt += &format!("<|turn>{role}\n{}<turn|>\n", message.content.trim());
    }
    Ok(prompt + "<|turn>model\n")
}
