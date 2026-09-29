# model-runtime

Runs local models for other programs on this computer, over a Unix socket. A
model is loaded on its first request and unloaded after a few idle minutes.

```sh
model-runtime pull pii     # download a model; the only command that goes online
model-runtime list         # what it knows and what's installed
model-runtime serve        # serve on $XDG_RUNTIME_DIR/model-runtime.sock
```

`serve` needs ONNX Runtime 1.23 or newer: `onnxruntime-cpu` on Arch, or set
`ORT_DYLIB_PATH` to a `libonnxruntime.so`.

On Arch, `model-runtime-git` from [mtch3n/PKGBUILDS](https://github.com/mtch3n/PKGBUILDS)
installs it with a systemd user service:

```sh
sudo pacman -S model-runtime-git
model-runtime pull pii
systemctl --user enable --now model-runtime
```

The service has no network access, since it's sent the private text it looks
through; [`dist/model-runtime.service`](dist/model-runtime.service) is the unit.

## Models

| id | model | RAM when loaded |
|---|---|---|
| `pii` | [GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi), 42 PII types in 7 languages, via [gliner2-rs](https://github.com/dariofinardi/gliner2-rs) | about 1.9 GB |

## API

HTTP on the socket, which only its owner can open:
`curl --unix-socket $XDG_RUNTIME_DIR/model-runtime.sock http://x/models`.

- `GET /models`: each model, whether it's installed and loaded, how long it's
  been idle, and the memory the runtime is using.
- `POST /models/{id}/load` and `/unload`.
- `POST /models/pii/detect`:

```json
{ "fields": [{ "field_id": "a", "text": "Mail maria@example.dk" }],
  "labels": ["email"], "threshold": 0.5 }
```

```json
{ "fields": [{ "field_id": "a",
    "spans": [{ "start": 5, "end": 21, "type": "email", "score": 1.0 }] }] }
```

`labels` and `threshold` are optional; without labels, every type the model
knows is looked for. `start` and `end` are byte offsets, end exclusive. Spans
can overlap: a name is found as `full_name`, `first_name` and `last_name`.
