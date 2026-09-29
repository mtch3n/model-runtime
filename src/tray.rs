//! A tray icon showing whether the service is running and what it has loaded,
//! from the same API other programs use.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use ksni::blocking::TrayMethods;
use ksni::menu::{MenuItem, StandardItem};
use serde::Deserialize;

const POLL: Duration = Duration::from_secs(2);

/// Lucide's sparkles, filled so it reads at tray size, with the status as a dot
/// in the corner.
const SPARKLES: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round"><g fill="#e5e7eb" stroke="#e5e7eb" stroke-width="2"><path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/><path d="M20 2v4"/><path d="M22 4h-4"/></g><circle cx="19" cy="19" r="4.25" fill="DOT" stroke="#1f2937" stroke-width="1.5"/></svg>"##;

#[derive(Deserialize)]
struct ModelList {
    models: Vec<Model>,
    rss_mb: Option<u64>,
}

#[derive(Deserialize)]
struct Model {
    id: String,
    installed: bool,
    loaded: bool,
    idle_secs: Option<u64>,
    busy: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Stopped,
    Running,
    Busy,
}

impl State {
    /// Red, green or yellow.
    fn dot(self) -> &'static str {
        match self {
            State::Stopped => "#ef4444",
            State::Running => "#22c55e",
            State::Busy => "#eab308",
        }
    }
}

struct Tray {
    socket: PathBuf,
    /// What the service last said, or `None` when it isn't running.
    list: Option<ModelList>,
}

impl Tray {
    fn state(&self) -> State {
        match &self.list {
            None => State::Stopped,
            Some(list) if list.models.iter().any(|m| m.busy) => State::Busy,
            Some(_) => State::Running,
        }
    }

    fn summary(&self) -> String {
        match &self.list {
            None => "Not running".into(),
            Some(list) => match list.rss_mb {
                Some(mb) => format!("Running, using {}", memory(mb)),
                None => "Running".into(),
            },
        }
    }
}

impl ksni::Tray for Tray {
    fn id(&self) -> String {
        "model-runtime".into()
    }

    fn title(&self) -> String {
        "model-runtime".into()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        [22, 32, 48]
            .into_iter()
            .map(|size| icon(self.state().dot(), size))
            .collect()
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            title: "model-runtime".into(),
            description: self.summary(),
            ..Default::default()
        }
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        let mut items = vec![label(self.summary())];
        match &self.list {
            None => items.push(action("Start the service", |_| systemctl("start"))),
            Some(list) => {
                for model in &list.models {
                    items.push(label(describe(model)));
                    if model.loaded && !model.busy {
                        let id = model.id.clone();
                        items.push(action(&format!("Unload {id}"), move |tray| {
                            if let Err(err) =
                                request(&tray.socket, "POST", &format!("/models/{id}/unload"))
                            {
                                eprintln!("unloading {id}: {err:#}");
                            }
                        }));
                    }
                }
                items.push(MenuItem::Separator);
                items.push(action("Stop the service", |_| systemctl("stop")));
            }
        }
        items.push(MenuItem::Separator);
        items.push(action("Quit", |_| std::process::exit(0)));
        items
    }
}

fn describe(model: &Model) -> String {
    let state = if !model.installed {
        "not installed".into()
    } else if model.busy {
        "working".into()
    } else if !model.loaded {
        "not loaded".into()
    } else {
        match model.idle_secs {
            Some(s) if s >= 60 => format!("loaded, idle {} min", s / 60),
            _ => "loaded".into(),
        }
    };
    format!("{}: {state}", model.id)
}

fn memory(mb: u64) -> String {
    if mb >= 1000 {
        format!("{:.1} GB", mb as f64 / 1000.0)
    } else {
        format!("{mb} MB")
    }
}

fn label(text: String) -> MenuItem<Tray> {
    StandardItem {
        label: text,
        enabled: false,
        ..Default::default()
    }
    .into()
}

fn action(text: &str, activate: impl Fn(&mut Tray) + Send + 'static) -> MenuItem<Tray> {
    StandardItem {
        label: text.into(),
        activate: Box::new(activate),
        ..Default::default()
    }
    .into()
}

fn systemctl(verb: &str) {
    if let Err(err) = Command::new("systemctl")
        .args(["--user", verb, "model-runtime"])
        .spawn()
    {
        eprintln!("systemctl {verb}: {err}");
    }
}

/// The sparkles with a dot in `dot`, as ARGB32 in network byte order.
fn icon(dot: &str, size: u32) -> ksni::Icon {
    let svg = SPARKLES.replace("DOT", dot);
    let tree = resvg::usvg::Tree::from_str(&svg, &Default::default()).expect("the icon is valid");
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size, size).expect("a non-zero size");
    let scale = size as f32 / 24.0;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    let data = pixmap
        .pixels()
        .iter()
        .flat_map(|p| {
            let c = p.demultiply();
            [c.alpha(), c.red(), c.green(), c.blue()]
        })
        .collect();
    ksni::Icon {
        width: size as i32,
        height: size as i32,
        data,
    }
}

/// Sends an HTTP request over the socket and returns the body of a 2xx answer.
fn request(socket: &PathBuf, method: &str, path: &str) -> Result<String> {
    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("connecting to {}", socket.display()))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    let (head, body) = response
        .split_once("\r\n\r\n")
        .context("a cut-short answer")?;
    if !head.starts_with("HTTP/1.1 2") {
        bail!("{body}");
    }
    Ok(body.to_owned())
}

fn fetch(socket: &PathBuf) -> Option<ModelList> {
    let body = request(socket, "GET", "/models").ok()?;
    serde_json::from_str(&body).ok()
}

pub fn run(socket: PathBuf) -> Result<()> {
    let list = fetch(&socket);
    let handle = Tray {
        socket: socket.clone(),
        list,
    }
    .spawn()
    .context("showing the tray icon; is a StatusNotifierItem host running?")?;
    while !handle.is_closed() {
        std::thread::sleep(POLL);
        let list = fetch(&socket);
        handle.update(|tray| tray.list = list);
    }
    Ok(())
}
