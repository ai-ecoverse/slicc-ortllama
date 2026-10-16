# slicc-ortllama

A local model server: ONNX models on ONNX Runtime, served over OpenAI-style HTTP APIs. The same code runs in two places:

- **In the browser**, as a [slicc-kernel](https://github.com/ai-ecoverse/slicc-kernel) program on [jsh](https://github.com/ai-ecoverse/slicc-jsh)'s `node`, on `onnxruntime-web` (WebGPU, single-threaded WASM as the fallback), listening on the kernel's loopback. The page and the slicc-agent worker reach it as `http://<port>.kernel.localhost`.
- **On a host**, on Node 24+ with `onnxruntime-node`.

The plan and milestones are in [#2](https://github.com/ai-ecoverse/slicc-ortllama/issues/2).

## Use

```sh
npm install -g @ai-ecoverse/slicc-ortllama onnxruntime-node @ai-ecoverse/kev.js
hf download ai-ecoverse/kev.js --include 'kev-0.8b/*'
hf download mistralai/Ministral-3-3B-Instruct-2512-ONNX --include '*.json' chat_template.jinja 'onnx/decoder_model_merged_q4f16.onnx*' 'onnx/embed_tokens_q4f16.onnx*' 'onnx/vision_encoder_q4f16.onnx*'
ortllama serve
```

```sh
curl -s localhost:11435/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "chat",
  "messages": [{ "role": "user", "content": "What is the capital of France?" }]
}'
```

```sh
curl -s localhost:11435/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice. Please fix this ASAP.",
  "questions": { "billing": { "type": "noul", "instructions": "Is this ticket about billing?" } }
}'
```

```sh
hf download onnx-community/whisper-tiny --include '*.json' 'onnx/encoder_model.onnx' 'onnx/decoder_model_merged.onnx'
curl -s localhost:11435/v1/audio/transcriptions -F file=@clip.wav -F response_format=verbose_json
```

In slicc-kernel, `onnxruntime-node` is not needed: the `slicc` export condition picks `onnxruntime-web`. It needs slicc-jsh 0.29.10 or later, which keeps a program alive while WebGPU work is pending.

| Command | |
| --- | --- |
| `ortllama serve [--port N] [--host H] [--detach]` | start the server (default `127.0.0.1:11435`); `--detach` runs it in the background |
| `ortllama ls [--json]` | models in the Hugging Face cache that ortllama can run |
| `ortllama ps [--json]` | models the running server has loaded |
| `ortllama alias [role [model]] [--unset role]` | show or set role aliases |
| `ortllama check <model> [--fixtures file] [--json]` | load a model on this device, time a request, and compare with reference fixtures |

## APIs

| Route | |
| --- | --- |
| `GET /v1/models` | models and role aliases; System One models list `decisions` in `architecture.output_modalities` |
| `POST /v1/chat/completions` | OpenAI chat completions, streamed as `chat.completion.chunk`s with `stream: true`, on the decoder engine |
| `POST /v1/systemone` | TypeSafe's System One protocol, on [kev.js](https://github.com/ai-ecoverse/kev.js) |
| `POST /v1/audio/transcriptions` | OpenAI's transcription API on Whisper: multipart `file`, `model`, `language`, `prompt`, `response_format` (`json`, `verbose_json`, `text`, `srt`, `vtt`); WAV or 16-bit `audio/pcm` (`;rate=N`, default 24 kHz) |
| `GET /v1/realtime?intent=transcription` | OpenAI's realtime transcription over WebSocket (subprotocol `realtime`): `session.update`, `input_audio_buffer.append`/`commit`/`clear` in; `session.created`/`updated`, `input_audio_buffer.speech_started`/`speech_stopped`/`committed`/`cleared`, `conversation.item.input_audio_transcription.delta`/`completed` and `error` out. `audio/pcm` at 8–48 kHz (24 kHz by default), `server_vad` on Silero VAD or manual commits (`turn_detection: null`) |
| `GET /ortllama/ps` | loaded models |

Errors use OpenAI's envelope, `{ "error": { "message", "type", "param", "code" } }`. Speech follows in the milestones of #2.

Chat completions take `messages` (`system`, `developer`, `user`, `assistant` with `tool_calls`, `tool`), `tools`, `tool_choice`, `stream`, `stream_options.include_usage`, `max_completion_tokens` or `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop` and `reasoning_effort`, and ignore `store`. Streams carry `delta.content`, `delta.reasoning_content` and `delta.tool_calls`; the last chunk has `finish_reason` (`stop`, `length` or `tool_calls`) and llama.cpp-style `timings`, followed by a usage chunk when asked. `usage.prompt_tokens_details.cached_tokens` counts the prompt tokens served from the prefix cache, which keeps the attention state of recent conversations, so an agent that resends its history pays only for the new turn. A prompt longer than the context window (the model's, at most 32768 tokens) is refused with `400` and `code: "context_length_exceeded"`, never truncated. A closed connection stops generation.

User messages may carry images as OpenAI `image_url` parts (`{ "type": "image_url", "image_url": { "url": "data:image/png;base64,…", "detail": "auto" } }`), which is what pi-ai sends for a model with `input: ["text", "image"]`. Images must be inline `data:` URLs of PNG, JPEG or WebP files; ortllama fetches nothing, so `http(s)` URLs are refused with `400`. A request takes at most 8 images of at most 10 MiB and 40 megapixels each. They are decoded the same way on both tracks (pure-JS PNG and JPEG, libwebp in WebAssembly), with transparency dropped as transformers does, and resized as the model's own preprocessing asks: for Ministral 3, Pixtral's, to fit 1540 pixels on the longest edge, or 512 with `"detail": "low"`. The vision encoder loads on the first image. The prefix cache keys image tokens on the image's SHA-256, so a conversation that resends the same image reuses it without encoding it again. A model whose vision encoder is not cached lists only `text` in `architecture.input_modalities` and answers images with `400`.

System One requests to a kev.js vision bundle (such as `ai-ecoverse/kev.js/kev-0.8b-vision`) take an `image` field, a `data:` URL string or `{ "url": … }`, decoded to RGBA for kev.js; text-only bundles answer it with `400`. pi-ai's System One client does not send images yet.

Transcriptions answer `{ text, language, duration, segments, usage: { type: "duration", seconds } }`. Audio longer than 30 seconds is decoded in windows that follow Whisper's timestamps; the language is detected when the request names none. Other audio formats answer `400` for now.

A realtime session re-decodes the growing turn about every 500 ms of audio and sends the words two decodes agree on as deltas, then the full transcript when the turn ends: on `silence_duration_ms` of silence (500 ms by default), when no audio has arrived for that long, on `commit`, after 30 seconds of speech, or when the socket closes. `threshold` is kept between 0.2 and 0.95, `prefix_padding_ms` between 0 and 2000 and `silence_duration_ms` between 100 and 10000. `clear` drops transcripts that are still being decoded. A socket that sends nothing for 60 seconds is closed. The WebSocket takes the API key as `Authorization: Bearer` or as the `openai-insecure-api-key.<key>` subprotocol that browsers use, and answers only loopback, `*.localhost` and `ORTLLAMA_ORIGINS` origins. The Whisper model and the one Silero VAD session that all sockets share stay loaded while a session is open, so open realtime sessions can push the number of loaded models past `ORTLLAMA_MAX_LOADED`.

`@ai-ecoverse/slicc-ortllama/pi-audio` builds [pi-audio](https://github.com/ai-ecoverse/pi-audio) voice models for ortllama on pi-audio's OpenAI APIs: `transcriptionModel({ baseUrl, id })` returns an `openai-transcriptions` model for `transcribe()`, and `realtimeTranscriptionModel({ baseUrl, id })` an `openai-realtime-transcription` model with server VAD for `listen()`.

## Models

ortllama downloads nothing. It reads the Hugging Face cache that `hf download` fills (`HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`), through its `refs/` and `snapshots/`. A model ID is a Hub ID: `org/repo[/subfolder][@revision][:variant]`, such as `ai-ecoverse/kev.js/kev-0.8b:q8f32`. A model that isn't cached, or a variant with missing files, is answered with `404` and the `hf download` command that fetches it.

Clients can ask for a role instead of a model: `chat`, `chat-reasoning`, `decisions`, `transcribe`, `realtime-transcribe`, `speech` or `vad`. `ortllama alias decisions <model>` changes what a role means, so swapping a model changes no client. `chat` defaults to `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16`, `decisions` to `ai-ecoverse/kev.js/kev-0.8b:q8f32`, `transcribe` and `realtime-transcribe` to `onnx-community/whisper-tiny`, and `vad` to `onnx-community/silero-vad` (fetch it with `hf download onnx-community/silero-vad --include onnx/model.onnx`).

Whisper models are the transformers.js ONNX exports (`config.json` with `model_type: "whisper"`, `onnx/encoder_model*.onnx` and `onnx/decoder_model_merged*.onnx`, external `.onnx_data` included). Variants follow the file suffixes: `fp32`, `fp16`, `q8` (`_quantized`), `int8`, `uint8`, `q4`, `q4f16`, `bnb4`; the default is the first complete one in that order.

Engines are picked by the files a model has, not by its name. The decoder engine runs the transformers.js layout (`onnx/decoder_model_merged*.onnx` with `embed_tokens*.onnx`, or `onnx/model*.onnx`; variants are its quantizations, such as `q4f16`) and the onnxruntime-genai layout (`genai_config.json`, one variant per folder), with the model's own chat template and, when the export has one, its `vision_encoder*.onnx`. A small profile per family, chosen from the template, parses tool calls and reasoning; Mistral's `[TOOL_CALLS]name[ARGS]{…}` and `[THINK]…[/THINK]` are built in, and control-token names in client text are escaped so they cannot inject one. Each engine in `src/engines/` gets ONNX Runtime from the caller and provides `detect`, `missing`, `include`, `load`, `checkOverrides` and `check`.

Model-specific libraries are never installed with ortllama. They are devDependencies here, and engines load them only when a model needs them, such as `@ai-ecoverse/kev.js` for System One. ortllama looks for each library next to itself, then in `ORTLLAMA_MODULE_PATH` (project folders that contain a `node_modules`), then in pnpm's global installs (`$PNPM_HOME/global/…`, including slicc-kernel's `/usr/local/share/pnpm`). A model whose library is missing is still listed, and requests for it answer `501` with code `engine_unavailable` and the package to install.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `ORTLLAMA_HOME` | `~/.ortllama` | `config.json` (aliases, per-model hints) and `run/` |
| `ORTLLAMA_PORT` | `11435` | |
| `ORTLLAMA_API_KEY` | none | require `Authorization: Bearer <key>` |
| `ORTLLAMA_ORIGINS` | none | more CORS origins; loopback and `*.localhost` are always allowed |
| `ORTLLAMA_HOSTS` | none | more `Host` names to accept; loopback names, `*.localhost` and `host.slicc.internal` are always accepted, and any host is when the server listens on `0.0.0.0` or `::` without this list. Other hosts get `403`, which stops DNS rebinding |
| `ORTLLAMA_KEEP_ALIVE` | `5m` | unload an idle model after this long (`-1` keeps it) |
| `ORTLLAMA_MAX_LOADED` | `1` | models kept loaded at once |
| `ORTLLAMA_EP` | `webgpu,cpu` on Node, `webgpu,wasm` in the browser | execution providers, in order |
| `ORTLLAMA_MODEL_PATH` | none | extra plain folders, listed as `local/<name>` |
| `ORTLLAMA_MODULE_PATH` | none | project folders searched for model libraries such as `@ai-ecoverse/kev.js` |
| `ORTLLAMA_ORT_LOG_LEVEL` | `error` | ONNX Runtime's log level |

## Develop

Node ≥ 24. `npm run lint` runs `slicc-lint`. `npm test` runs the integration tests; the Kev, Whisper, realtime, chat and image tests run when `ai-ecoverse/kev.js/kev-0.8b`, `onnx-community/whisper-tiny`, `onnx-community/silero-vad`, `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16` (with its vision encoder) and `ai-ecoverse/kev.js/kev-0.8b-vision` are in the Hugging Face cache (`ORTLLAMA_REQUIRE_MODEL=1` makes a missing model a failure). Unit tests stay local in `test/unit/` (`npm run test:unit`). Releases use semantic-release on `main`.
