//! Each model runs on a thread of its own, which loads it on the first request
//! and unloads it once no request has come for a while.

use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use gliner2_rs::{InferenceParams, Precision, SchemaTask, SpanConfig, SpanEngine, privacy::Group};
use serde::Serialize;
use tokio::sync::oneshot;

use crate::catalog::{CATALOG, Source, Spec};
use crate::chat::{Chat, Message, Reply};

pub struct Registry {
    slots: Vec<Slot>,
}

struct Slot {
    spec: &'static Spec,
    jobs: mpsc::Sender<Job>,
    status: Arc<Mutex<Status>>,
}

#[derive(Clone, Copy, Default)]
struct Status {
    loaded_at: Option<Instant>,
    last_used: Option<Instant>,
    requests: u64,
    /// Loading or working on a request.
    busy: bool,
}

#[derive(Serialize)]
pub struct ModelInfo {
    id: &'static str,
    description: &'static str,
    installed: bool,
    loaded: bool,
    loaded_secs: Option<u64>,
    idle_secs: Option<u64>,
    requests: u64,
    busy: bool,
}

pub struct Span {
    pub start: usize,
    pub end: usize,
    pub label: String,
    pub score: f32,
}

enum Job {
    Load(oneshot::Sender<Result<()>>),
    Unload(oneshot::Sender<Result<()>>),
    Detect {
        texts: Vec<String>,
        tasks: Vec<SchemaTask>,
        params: InferenceParams,
        reply: oneshot::Sender<Result<Vec<Vec<Span>>>>,
    },
    Chat {
        messages: Vec<Message>,
        max_tokens: u32,
        temperature: f32,
        reply: oneshot::Sender<Result<Reply>>,
    },
}

enum Engine {
    Spans(Box<SpanEngine>),
    Chat(Chat),
}

impl Registry {
    pub fn new(idle: Duration) -> Registry {
        let slots = CATALOG
            .iter()
            .map(|spec| {
                let (jobs, queue) = mpsc::channel();
                let status = Arc::new(Mutex::new(Status::default()));
                let worker_status = status.clone();
                // The engine isn't Send, so it's made on the thread that uses it.
                std::thread::Builder::new()
                    .name(spec.id.into())
                    .spawn(move || {
                        let worker = Worker {
                            spec,
                            status: worker_status,
                            engine: None,
                        };
                        worker.run(queue, idle)
                    })
                    .expect("spawning a model thread");
                Slot { spec, jobs, status }
            })
            .collect();
        Registry { slots }
    }

    pub fn list(&self) -> Vec<ModelInfo> {
        let now = Instant::now();
        let secs = |t: Option<Instant>| t.map(|t| now.duration_since(t).as_secs());
        self.slots
            .iter()
            .map(|slot| {
                let status = *slot.status.lock().unwrap();
                ModelInfo {
                    id: slot.spec.id,
                    description: slot.spec.description,
                    installed: slot.spec.installed(),
                    loaded: status.loaded_at.is_some(),
                    loaded_secs: secs(status.loaded_at),
                    idle_secs: secs(status.last_used),
                    requests: status.requests,
                    busy: status.busy,
                }
            })
            .collect()
    }

    pub async fn load(&self, id: &str) -> Result<()> {
        self.send(id, Job::Load).await?
    }

    pub async fn unload(&self, id: &str) -> Result<()> {
        self.send(id, Job::Unload).await?
    }

    /// Finds PII in each text. Without labels, it looks for every type the
    /// model knows, one group at a time so the groups don't compete.
    pub async fn detect(
        &self,
        id: &str,
        texts: Vec<String>,
        labels: Option<Vec<String>>,
        threshold: f32,
    ) -> Result<Vec<Vec<Span>>> {
        let tasks = match labels {
            Some(labels) => vec![SchemaTask::Entities(labels)],
            None => Group::ALL.map(Group::task).to_vec(),
        };
        let params = InferenceParams {
            threshold,
            ..InferenceParams::default()
        };
        self.send(id, |reply| Job::Detect {
            texts,
            tasks,
            params,
            reply,
        })
        .await?
    }

    pub async fn chat(
        &self,
        id: &str,
        messages: Vec<Message>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Reply> {
        self.send(id, |reply| Job::Chat {
            messages,
            max_tokens,
            temperature,
            reply,
        })
        .await?
    }

    async fn send<T>(&self, id: &str, job: impl FnOnce(oneshot::Sender<T>) -> Job) -> Result<T> {
        let Some(slot) = self.slots.iter().find(|s| s.spec.id == id) else {
            bail!("no model named {id}");
        };
        let (reply, answer) = oneshot::channel();
        slot.jobs
            .send(job(reply))
            .map_err(|_| anyhow!("{id}'s thread has stopped"))?;
        answer
            .await
            .map_err(|_| anyhow!("{id}'s thread has stopped"))
    }
}

struct Worker {
    spec: &'static Spec,
    status: Arc<Mutex<Status>>,
    engine: Option<Engine>,
}

impl Worker {
    fn run(mut self, queue: mpsc::Receiver<Job>, idle: Duration) {
        loop {
            let job = if self.engine.is_some() {
                match queue.recv_timeout(idle) {
                    Ok(job) => job,
                    Err(RecvTimeoutError::Timeout) => {
                        self.unload();
                        eprintln!("unloaded {}, idle", self.spec.id);
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            } else {
                match queue.recv() {
                    Ok(job) => job,
                    Err(_) => return,
                }
            };
            self.status.lock().unwrap().busy = true;
            match job {
                Job::Load(reply) => {
                    let _ = reply.send(self.load().map(|_| ()));
                }
                Job::Unload(reply) => {
                    self.unload();
                    let _ = reply.send(Ok(()));
                }
                Job::Detect {
                    texts,
                    tasks,
                    params,
                    reply,
                } => {
                    let _ = reply.send(self.detect(&texts, &tasks, &params));
                }
                Job::Chat {
                    messages,
                    max_tokens,
                    temperature,
                    reply,
                } => {
                    let _ = reply.send(self.chat(&messages, max_tokens, temperature));
                }
            }
            self.status.lock().unwrap().busy = false;
        }
    }

    fn load(&mut self) -> Result<&mut Engine> {
        if self.engine.is_none() {
            if !self.spec.installed() {
                bail!(
                    "{} isn't installed; run `model-runtime pull {}`",
                    self.spec.id,
                    self.spec.id
                );
            }
            let started = Instant::now();
            self.engine = Some(match self.spec.source {
                Source::Gliner(_) => {
                    let config = SpanConfig::new(self.spec.dir()).with_precision(Precision::Fp32);
                    Engine::Spans(Box::new(SpanEngine::new(config)?))
                }
                Source::Gguf { file, .. } => Engine::Chat(Chat::load(&self.spec.dir().join(file))?),
            });
            eprintln!("loaded {} in {:.1?}", self.spec.id, started.elapsed());
            self.status.lock().unwrap().loaded_at = Some(Instant::now());
        }
        Ok(self.engine.as_mut().unwrap())
    }

    fn unload(&mut self) {
        self.engine = None;
        // glibc keeps freed memory for reuse; give the model's back to the system.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
        self.status.lock().unwrap().loaded_at = None;
    }

    fn detect(
        &mut self,
        texts: &[String],
        tasks: &[SchemaTask],
        params: &InferenceParams,
    ) -> Result<Vec<Vec<Span>>> {
        // Checked before loading, so asking the wrong model loads nothing.
        if !matches!(self.spec.source, Source::Gliner(_)) {
            bail!("{} doesn't find spans", self.spec.id);
        }
        let Engine::Spans(engine) = self.load()? else {
            unreachable!("a GLiNER2 model loads as spans");
        };
        let found = texts
            .iter()
            .map(|text| {
                let out = engine.extract_long_with(text, tasks, params, Default::default())?;
                Ok(out
                    .entities
                    .into_iter()
                    .map(|e| Span {
                        start: e.char_start,
                        end: e.char_end,
                        label: e.label,
                        score: e.score,
                    })
                    .collect())
            })
            .collect::<Result<Vec<_>>>()?;
        self.used();
        Ok(found)
    }

    fn chat(&mut self, messages: &[Message], max_tokens: u32, temperature: f32) -> Result<Reply> {
        if !matches!(self.spec.source, Source::Gguf { .. }) {
            bail!("{} doesn't chat", self.spec.id);
        }
        let Engine::Chat(chat) = self.load()? else {
            unreachable!("a GGUF model loads as a chat");
        };
        let reply = chat.reply(messages, max_tokens, temperature)?;
        self.used();
        Ok(reply)
    }

    fn used(&self) {
        let mut status = self.status.lock().unwrap();
        status.last_used = Some(Instant::now());
        status.requests += 1;
    }
}
