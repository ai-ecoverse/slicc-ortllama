# slicc-ortllama

[![npm](https://img.shields.io/npm/v/@ai-ecoverse/slicc-ortllama)](https://www.npmjs.com/package/@ai-ecoverse/slicc-ortllama)
[![CI](https://github.com/ai-ecoverse/slicc-ortllama/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-ecoverse/slicc-ortllama/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ai-ecoverse/slicc-ortllama)](LICENSE)

A local model server: ONNX models on [ONNX Runtime](https://onnxruntime.ai), served over OpenAI-style HTTP APIs. It does chat with tools, reasoning and images, decisions, transcription (batch and realtime) and text to speech. The same code runs in two places:

- **In the browser**, as a [slicc-kernel](https://github.com/ai-ecoverse/slicc-kernel) program on [slicc-jsh](https://github.com/ai-ecoverse/slicc-jsh)'s `node`, on `onnxruntime-web` (WebGPU, single-threaded WASM as the fallback), listening on the kernel's loopback. The page and the [slicc-agent](https://github.com/ai-ecoverse/slicc-agent) worker reach it as `http://<port>.kernel.localhost`.
- **On a host**, on Node 24+ with `onnxruntime-node`.

Models come only from the Hugging Face cache; ortllama downloads nothing. The plan and milestones are in [#2](https://github.com/ai-ecoverse/slicc-ortllama/issues/2).

## Quick start

```sh
npm install -g @ai-ecoverse/slicc-ortllama onnxruntime-node @ai-ecoverse/kev.js
hf download ai-ecoverse/kev.js --include 'kev-0.8b/*'
hf download mistralai/Ministral-3-3B-Instruct-2512-ONNX --include '*.json' chat_template.jinja 'onnx/decoder_model_merged_q4f16.onnx*' 'onnx/embed_tokens_q4f16.onnx*' 'onnx/vision_encoder_fp16.onnx*'
ortllama serve
```

Chat:

```sh
curl -s localhost:11435/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "chat",
  "messages": [{ "role": "user", "content": "What is the capital of France?" }]
}'
```

A System One decision:

```sh
curl -s localhost:11435/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice. Please fix this ASAP.",
  "questions": { "billing": { "type": "noul", "instructions": "Is this ticket about billing?" } }
}'
```

A transcription:

```sh
hf download onnx-community/whisper-tiny --include '*.json' 'onnx/encoder_model.onnx' 'onnx/decoder_model_merged.onnx'
curl -s localhost:11435/v1/audio/transcriptions -F file=@clip.wav -F response_format=verbose_json
```

### In slicc-kernel

`onnxruntime-node` is not needed: the `slicc` export condition picks `onnxruntime-web`. ortllama needs slicc-jsh 0.31.0 or later, which keeps a program alive while WebGPU work is pending and has the `tls` module that `ws`, which serves `/v1/realtime`, loads.

## CLI

| Command | |
| --- | --- |
| `ortllama serve [--port N] [--host H] [--detach]` | start the server (default `127.0.0.1:11435`); `--detach` runs it in the background |
| `ortllama ls [--json]` | models in the Hugging Face cache that ortllama can run; cached repos and `ORTLLAMA_MODEL_PATH` folders with ONNX files in a layout it does not recognize are noted on stderr |
| `ortllama ps [--json]` | models the running server has loaded |
| `ortllama alias [role [model]] [--unset role]` | show or set [role aliases](#roles) |
| `ortllama check <model> [--fixtures file] [--json]` | load a model on this device, time a request, and compare it with [reference fixtures](#check): the ones shipped for the model, or `--fixtures file` |

## APIs

| Route | |
| --- | --- |
| `GET /v1/models` | models and role aliases; System One models list `decisions` in `architecture.output_modalities` |
| `POST /v1/chat/completions` | [OpenAI chat completions](#chat-completions) on the decoder engine, and on [kev-ministral's base weights](#kev-ministrals-base-weights) |
| `POST /v1/systemone` | TypeSafe's [System One](#system-one) protocol, on kev.js and jev-omni.js |
| `POST /v1/audio/transcriptions` | OpenAI's [transcription](#transcription) API, on Whisper |
| `GET /v1/realtime?intent=transcription` | OpenAI's [realtime transcription](#realtime-transcription) over WebSocket |
| `POST /v1/audio/speech` | OpenAI's [text to speech](#speech), on Kokoro-82M and [Phonon 31m English](#phonon) |
| `GET /ortllama/ps` | loaded models, their [execution providers](#execution-providers) and size, and the process's memory |

Errors use OpenAI's envelope, `{ "error": { "message", "type", "param", "code" } }`.

### Chat completions

Requests take `messages` (`system`, `developer`, `user`, `assistant` with `tool_calls`, `tool`), `tools`, `tool_choice`, `stream`, `stream_options.include_usage`, `max_completion_tokens` or `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop` and `reasoning_effort`, and ignore `store`.

- **Streaming.** With `stream: true`, the response is a stream of `chat.completion.chunk`s carrying `delta.content`, `delta.reasoning_content` and `delta.tool_calls`. The last chunk has `finish_reason` (`stop`, `length` or `tool_calls`) and llama.cpp-style `timings`, followed by a usage chunk when asked. The next token is generated only once the chunks before it have been written to the socket, or after 250 ms if the client is not reading, so on single-threaded WASM, where a decoding step blocks the program, each token still arrives as it is generated. While it waits, the generation gives up the runtime, so more urgent work such as a realtime decode runs, and another chat request waits. A closed connection stops generation.
- **Prefix cache.** The server keeps the attention state of recent conversations, so an agent that resends its history pays only for the new turn. `usage.prompt_tokens_details.cached_tokens` counts the prompt tokens served from it.
- **Context window.** A prompt longer than the context window (the model's, at most 32768 tokens) is refused with `400` and `code: "context_length_exceeded"`, never truncated.
- **Token limit.** A request without `max_tokens` or `max_completion_tokens` (pi-ai sends neither unless its caller sets one) stops after `ORTLLAMA_MAX_TOKENS` tokens, 8192 by default, and always within the context window, with `finish_reason: "length"`, so a looping model cannot run for minutes.
- **Sampling.** A request that omits `temperature`, `top_p` or `top_k` gets the model's own values: those in its `generation_config.json` if it has them (`do_sample: false` means greedy), otherwise the family's recommendation (Ministral 3 Instruct 0.1; Ministral 3 Reasoning 0.7 with `top_p` 0.95). A value in the request always wins, and `temperature: 0` is greedy. The response and the last stream chunk report what was used in `generation_settings` (`temperature`, `top_p`, `top_k`, `max_tokens`).
- **Reasoning.** The Reasoning model thinks, streamed as `reasoning_content`, when a request brings no system message, because its template's default system prompt asks for a `[THINK]` block. A request with a system message of its own gets no thinking unless it appends the repo's `SYSTEM_PROMPT.txt`. `reasoning_effort: "none"` turns thinking off: it keeps `[THINK]` out, and a request without a system message is rendered with a direct-answer instruction in place of that default prompt.

#### Images

User messages may carry images as OpenAI `image_url` parts, which is what pi-ai sends for a model with `input: ["text", "image"]`:

```json
{ "type": "image_url", "image_url": { "url": "data:image/png;base64,…", "detail": "auto" } }
```

- **Input.** Images must be inline `data:` URLs of PNG, JPEG or WebP files; ortllama fetches nothing, so `http(s)` URLs are refused with `400`. A request takes at most 8 images of at most 10 MiB and 40 megapixels each. A PNG whose data inflates past the size its header declares is refused before it is decoded, and its compressed side chunks (such as `iCCP`) are skipped. An image too thin for the encoder (an aspect ratio past 1540:1 for Ministral 3, 200:1 for kev.js) answers `400`.
- **Decoding.** Images are decoded the same way on both tracks (pure-JS PNG and JPEG, libwebp in WebAssembly), with EXIF orientation applied and transparency dropped as transformers does, and resized as the model's own preprocessing asks: for Ministral 3, Pixtral's, to fit 1540 pixels on the longest edge, or 512 with `"detail": "low"`.
- **Vision encoder.** It loads on the first image. ortllama picks the most precise one the export has, `fp16`, else `fp32`, else the decoder variant's own quantization, because the 4-bit encoders drift far from fp32 (Mistral's q4f16 encoder reaches a per-patch cosine of only 0.16–0.37 against fp16 on ordinary images). Among those it prefers one already in the cache, and the `hf download` hint names the most precise. A per-model hint in `config.json` overrides the choice: `{ "hints": { "mistralai/Ministral-3-3B-Instruct-2512-ONNX": { "vision": "q4f16" } } }`. A hinted encoder that is not in the cache leaves the model text-only, so image requests get `400` with the `hf download` command, and the next request after the download loads a new session.
- **Reporting.** `/v1/models` names the encoder in use as `ortllama.vision_encoder`, with `vision_encoder_preferred` when a cached fallback stands in for the preferred one, and `generation_settings` reports the same. A model whose vision encoder is not cached lists only `text` in `architecture.input_modalities` and answers images with `400`.
- **Caching.** The prefix cache keys image tokens on the image's SHA-256, so a conversation that resends the same image reuses it without encoding it again.

#### kev-ministral's base weights

`ai-ecoverse/kev-ministral/kev-ministral-3b:base` is [kev-ministral](#system-one) with its decision LoRA off (`lora_scale = 0`), which leaves the stock Ministral-3-3B-Base, and generates through kev.js's `generate`. It runs on the same kev.js load as `/v1/systemone`'s `ai-ecoverse/kev-ministral/kev-ministral-3b`: to the runtime both IDs are one model, listed once by `ortllama ps`, counted once against `ORTLLAMA_MAX_LOADED`, loaded and unloaded together. `/v1/models` lists the `:base` ID with `text` output. Each ID answers only on its own route; the other route answers `400` and names the ID that does.

- **Prompt.** The base model has no chat template, so the messages become a transcript, one `System:`, `User:` or `Assistant:` line each, ending in `Assistant:`. Generation stops when the model starts another such line, at `stop`, at `</s>` or at the token limit. Control token names in client text are escaped, as for decisions.
- **Images.** One per request, as an `image_url` part, decoded as above and placed where the part is, resized as Pixtral does to fit 1540 pixels (`detail` is not applied). A second image answers `400`.
- **Sampling.** `temperature` defaults to 0 (greedy), `top_k` to 40; `seed` works, `top_p` is not applied and reported as 1. Tools, tool calls and `tool` messages answer `400`; `reasoning_effort` is ignored. There is no prefix cache: every request prefills its whole prompt. The context window is 73,728 tokens.
- **Streaming.** kev.js generates in one call, so ortllama gates kev-ministral's ONNX sessions: after each token, the next run waits until its chunk is written (at most 250 ms) and gives up the runtime meanwhile, as for the decoder. kev.js runs one request at a time, so a System One request for kev-ministral, or a second `:base` request, waits until the generation ends.

On an M4 Max, with kev-ministral loaded (7–9 s on host WebGPU, 6 s on four CPU threads), it prefills 65–110 tokens per second of text and generates 15–17 on host WebGPU, 31–42 on the CPU. Describing a 600 × 400 photo in 96 tokens takes 9.3 s on WebGPU and 8 s on the CPU. The Instruct model generates about 95 tokens per second on WebGPU and 20 on the CPU.

On eight chat prompts and seven images (a photo, a chart, a receipt, a screenshot, shapes), the base model answered what the Instruct model did, in plainer text: the capital, a haiku, a 404 error, a translation, a summary, arithmetic, a JavaScript function, the chart's best seller, the receipt's total, the screen's text. It is worth it for a client that already uses kev-ministral for decisions and needs text or image descriptions too: one process then peaks at 8.5 GB on host WebGPU (8.1 GB on the CPU), against 12.5 GB (12.6 GB) with the Instruct model loaded next to kev-ministral, and with the default `ORTLLAMA_MAX_LOADED=1` it never swaps models, where each switch between kev-ministral and the Instruct model reloads for 5.6–6.8 s. A client that only chats is better served by `chat`, a 4.1 GB download against 6.6 GB and six times as fast on WebGPU.

### System One

`POST /v1/systemone` speaks TypeSafe's System One protocol on [kev.js](https://github.com/ai-ecoverse/kev.js). Requests to a kev.js vision bundle (such as `ai-ecoverse/kev.js/kev-0.8b-vision`) or to kev-ministral take an `image` field, a `data:` URL string or `{ "url": … }`, decoded to RGBA for kev.js; text-only bundles answer it with `400`. pi-ai's System One client does not send images yet. Bundles known to work, with the load time and a warm three-question request (an image request for vision bundles) on an M4 Max:

| Model | Download | Host CPU | Host WebGPU | Kernel WebGPU load |
| --- | --- | --- | --- | --- |
| `ai-ecoverse/kev.js/kev-0.8b` (default), `kev-0.8b-vision` | 0.84 GB, +0.2 GB | 0.8 s, 70 ms | 0.9 s, 75 ms (image 50 ms) | 1.9–2.4 s |
| `ai-ecoverse/kev.js/kev-4b`, `kev-4b-vision` | 4.7 GB, +0.7 GB | 5.7 s, 0.5–1 s (image 0.35 s) | 2.4 s, 235 ms (image 155 ms) | 13–23 s |
| `ai-ecoverse/kev.js/kev-9b` | 8.9 GB | 7.8 s, 0.9–1.5 s | 4.4 s, 330 ms | 27 s |

Loading a bundle takes about its size in memory. On a host, onnxruntime-node opens the graph by path and reads the weights itself: `kev-9b` peaks at 9.0 GB, `kev-4b` at 5.2 GB. In the kernel, onnxruntime-web asks for each weight's bytes while it builds the session, and ortllama reads them then, holding at most 1 GiB of weight files: for `kev-9b`, Chrome's page process peaks at 3.7 GiB and its GPU process at 8.1 GiB. jsh reads a whole file for each ranged read ([slicc-jsh#401](https://github.com/ai-ecoverse/slicc-jsh/issues/401)), so the larger bundles load about twice as slowly in the kernel as they would with every weight held at once. Both need kev.js 1.1 or later; with 1.0, every weight is read into memory first, and `kev-9b` peaks at about 20 GB.

[kev-ministral](https://huggingface.co/ai-ecoverse/kev-ministral) (`ai-ecoverse/kev-ministral/kev-ministral-3b`, variant `q8f32`, 6.6 GB) is the decision model from the web-browsing VLM work. It is Ministral-3-3B-Base with a Kev decision LoRA, left unmerged behind the decoder's `lora_scale` input, plus a Kev pointer head and Pixtral's vision encoder. kev.js 1.2 or later runs it through `loadKevMinistral`, and the System One engine finds it by its `manifest.json` (graphs in `onnx/`, no `variants`). It takes an `image` like the kev.js vision bundles: Pixtral's resize, so a 1024 × 576 screenshot becomes 798 tokens after `<state>`. Fetch it with `hf download ai-ecoverse/kev-ministral --include 'kev-ministral-3b/*'`.

On an M4 Max it loads in 5.1 s on the host CPU, 7.7 s on host WebGPU and 7.3–7.6 s in the kernel. The three-question request takes 0.71 s, 0.50 s and 0.50 s. A webrunner step from its demo (500 to 2,150 tokens of page and menu, plus 798 for a 1024 × 576 screenshot) takes a median of 22 s on the CPU, 4.2 s on host WebGPU and 4.2 s in the kernel; without the screenshot, 10 s, 1.8 s and 1.8 s. Loading peaks at a memory footprint of 6.6 GB on the host CPU and 12.6 GB on host WebGPU. In the kernel, Chrome's page process peaks at 11.1 GiB and its GPU process at 10 GiB.

Its adapter was trained for webrunner's menus, and that is where it is strongest. On Kev's 300 held-out transfer-v4 questions (raw logits) it lands between Kev-0.8B and Kev-4B:

| Model | Download | transfer-v4 accuracy / Brier | Demo webrunner steps right, with / without the screenshot | Host WebGPU load, three-question request |
| --- | --- | --- | --- | --- |
| `ai-ecoverse/kev.js/kev-0.8b` (`decisions`), `kev-0.8b-vision` | 0.84 GB, 1.0 GB | 0.640 / 0.492 | 4 / 6 of 10 | 0.8 s, 114 ms |
| `ai-ecoverse/kev-ministral/kev-ministral-3b` | 6.6 GB | 0.733 / 0.414 | 7 / 7 of 10 | 7.7 s, 501 ms |
| `ai-ecoverse/kev.js/kev-4b`, `kev-4b-vision` | 4.7 GB, 5.4 GB | 0.800 / 0.281 | 6 / 8 of 10 | 4.0 s, 353 ms |

`decisions` stays on Kev-0.8B: kev-ministral is eight times its size, takes ten times as long to load and four times as long per request, and Kev-4B is the better general model at a smaller size. Point a client at kev-ministral for browser-agent steps. Its model card reports 69 % on Mind2Web's unseen websites, against 57.6 % for `kev-4b-vision`. The same weights with `lora_scale = 0` are the stock base model, which `/v1/chat/completions` serves from the same load as [`ai-ecoverse/kev-ministral/kev-ministral-3b:base`](#kev-ministrals-base-weights).

[Jev-Omni](https://github.com/ai-ecoverse/jev-omni.js) (`ai-ecoverse/jev-omni.js/jev-omni`, Gemma 4 12B with a 256-way decision head, 13.6 GB) answers the same requests through [jev-omni.js](https://github.com/ai-ecoverse/jev-omni.js), with `@ai-ecoverse/jev-omni.js` installed. Each question is one forward pass over the state, its instructions and its options: a choice's criteria, `no` and `yes` for a noul, a score's levels. Answers, confidences and `usage` follow the same rules as Kev's. Jev-Omni takes 2 to 256 options per question, so a choice with one criterion answers `400`. An `image` is embedded once per request. On WebGPU, onnxruntime-web limits a prompt to about 8,000 tokens; a longer one answers `422` with code `context_length_exceeded`. On an M4 Max, it loads in 7.3 s on the host CPU and answers the three-question request in 3 s; on host WebGPU, 2.6–5.8 s and 0.9 s; in the kernel, 13–23 s and 1–1.2 s. Loading peaks at 13.3 GB on host WebGPU (15.9 GB on the CPU), and in the kernel at 3.3 GiB in Chrome's page process and 13 GiB in its GPU process. That needs jev-omni.js 0.2 or later: 0.1 reads every weight into memory first, which peaks at 30 GB on the host and fails to allocate in the kernel.

### Transcription

`POST /v1/audio/transcriptions` takes a multipart `file`, `model`, `language`, `prompt` and `response_format` (`json`, `verbose_json`, `text`, `srt`, `vtt`). Audio is WAV, mp3, FLAC, M4A or MP4 (also fragmented) with AAC, ADTS `.aac`, Ogg Opus or Vorbis, WebM Opus, or 16-bit `audio/pcm` (`;rate=N`, default 24 kHz). The format comes from the file's first bytes, else its content type or file name, and is decoded the same way on both tracks: Opus on libopus in WebAssembly, mp3 (MPEG-1, 2 and 2.5 Layer III, also free format), FLAC and Vorbis in JavaScript. A leading ID3 tag is skipped before the format is read. ortllama reads the MP4 and ADTS containers itself but ships no AAC codec, so AAC (AAC-LC, the format of iPhone voice memos and most phone recordings, and HE-AAC at its full rate) is decoded by the browser's WebCodecs or an optional peer, as [AAC](#aac) explains. MP4 edit lists and iTunes gapless data trim the encoder delay, and Vorbis ends at the last granule position. The first audio track of an MP4 is used, and every format is mixed down to mono. Audio longer than 30 minutes, or more samples than 30 minutes at 48 kHz, FLAC above 192 kHz, a stream that changes its sample rate, other MP4 codecs (ALAC, Opus, mp3), AAC with no codec to decode it or that the codec refuses, ADTS frames with several raw data blocks, Vorbis floor 0, MPEG audio Layer I and II, and other formats, answer `400`.

It answers `{ text, language, duration, segments, usage: { type: "duration", seconds } }`. Audio longer than 30 seconds is decoded in windows that follow Whisper's timestamps; the language is detected when the request names none.

#### AAC

AAC input and `aac` output use the first of these that can do it. Which ones are there is settled once per process; WebCodecs is still asked about each stream's configuration, and a stream it refuses goes to the peer.

1. **WebCodecs**, `AudioDecoder` and `AudioEncoder` with `mp4a.40.2`, when `isConfigSupported` says yes: slicc-kernel in Chrome. Chrome and Chrome for Testing decode AAC-LC and HE-AAC on every platform; Chromium builds without Google's codecs don't. Chrome encodes AAC only on macOS, Windows and Android, through the system's encoder, and only at 44.1 or 48 kHz (on Windows also only at 96 to 192 kbit/s), so 24 kHz speech is resampled to 48 kHz first. Chrome on Linux has no AAC encoder.
2. **An optional peer**, found like the [phonemizers](#phonemizers): next to ortllama, in pnpm's global directory (`pnpm add -g` in slicc-kernel) or in `ORTLLAMA_MODULE_PATH`. ortllama keeps only a few lines of glue for each.

| Peer | Does | Get it | Licence | |
| --- | --- | --- | --- | --- |
| [`@audio/decode-aac`](https://github.com/audiojs/decode/tree/master/packages/decode-aac) | decoding | `npm install @audio/decode-aac` (0.4 MB) | GPL-2.0, for the FAAD2 decoder it runs in WebAssembly | the only decoding peer; host Node and jsh |
| [`@audio/encode-aac`](https://github.com/audiojs/encode/tree/main/packages/encode-aac) | encoding | `npm install @audio/encode-aac` (0.5 MB) | MIT, and the Fraunhofer FDK AAC licence for the encoder it runs in WebAssembly, which grants no patent licence | tried first: the fox at 64 kbit/s comes back at 36.5 dB SNR; host Node and jsh |
| [`mediabunny`](https://github.com/Vanilagy/mediabunny) with [`@mediabunny/aac-encoder`](https://www.npmjs.com/package/@mediabunny/aac-encoder) | encoding | `npm install mediabunny @mediabunny/aac-encoder` (11 MB and 4.6 MB) | MPL-2.0; the encoder is FFmpeg's, in WebAssembly | tried second: 20.8 dB; host Node only, since `@mediabunny/aac-encoder` does not load under jsh |

3. **Neither**: AAC, M4A and MP4 input and `aac` output answer `400` with a message that names these packages. WAV, mp3, FLAC, Ogg and WebM need none of them.

The decoders' own delays are taken out, so the edit lists and gapless data of an MP4 trim the same way on each. mediabunny has no AAC decoder of its own: it decodes through WebCodecs, which ortllama already asks first. [slicc-media](https://github.com/ai-ecoverse/slicc-media) offers the `media` and `mediaprobe` commands but no library API, so ortllama calls WebCodecs directly.

### Realtime transcription

`GET /v1/realtime?intent=transcription` is OpenAI's realtime transcription over WebSocket (subprotocol `realtime`).

| Direction | Events |
| --- | --- |
| in | `session.update`, `input_audio_buffer.append`, `input_audio_buffer.commit`, `input_audio_buffer.clear` |
| out | `session.created`, `session.updated`, `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped`, `input_audio_buffer.committed`, `input_audio_buffer.cleared`, `conversation.item.input_audio_transcription.delta`, `conversation.item.input_audio_transcription.completed`, `error` |

Audio is `audio/pcm` at 8–48 kHz (24 kHz by default). Turns are detected by `server_vad` on Silero VAD, or committed manually with `turn_detection: null`.

- **Decoding.** A session re-decodes the growing turn about every 500 ms of audio and sends the words two decodes agree on as deltas, then the full transcript when the turn ends: on `silence_duration_ms` of silence (500 ms by default), when no audio has arrived for that long, on `commit`, after 30 seconds of speech, or when the socket closes. `clear` drops transcripts that are still being decoded.
- **Limits.** `threshold` is kept between 0.2 and 0.95, `prefix_padding_ms` between 0 and 2000 and `silence_duration_ms` between 100 and 10000. A socket that sends nothing for 60 seconds is closed.
- **Security.** The WebSocket takes the API key as `Authorization: Bearer` or as the `openai-insecure-api-key.<key>` subprotocol that browsers use, and answers only loopback, `*.localhost` and `ORTLLAMA_ORIGINS` origins.
- **Memory.** The Whisper model and the one Silero VAD session that all sockets share stay loaded while a session is open, so open realtime sessions can push the number of loaded models past `ORTLLAMA_MAX_LOADED`.

### Speech

`POST /v1/audio/speech` takes `{ model, input, voice, response_format, speed, instructions }`, on [Kokoro-82M](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) and on [Phonon 31m English](#phonon).

- `model` defaults to the `speech` role, `voice` to `af_heart`, and `instructions` is accepted and ignored.
- `response_format` is `wav` (the default), `pcm` (24 kHz mono 16-bit little-endian), `mp3` (`audio/mpeg`, 64 kbit/s), `opus` (Ogg Opus, `audio/opus`, 32 kbit/s), `flac` (`audio/flac`) or `aac` (ADTS, `audio/aac`, AAC-LC, mono, 64 kbit/s, from the codec [AAC](#aac) describes, at 96 kbit/s on Windows' WebCodecs encoder; `400` without one). ADTS has no gapless data, so a decoder starts with the encoder's delay, 85 ms with `@audio/encode-aac`. mp3 is encoded in JavaScript as MPEG-1, 2 or 2.5 Layer III, mono, at the nearest mp3 sample rate, with constant bitrate and long blocks only, no psychoacoustic model and no gapless header, so a decoder adds about 44 ms of silence at the start at 24 kHz.
- `input` takes up to 4096 characters. It is split into sentences, long ones at clause marks or spaces (Phonon cuts it itself, [as ptts does](#phonon)), and each piece's audio is sent as soon as it is ready. A piece is synthesized only once the one before it has been written to the socket, or after 250 ms if the client is not reading, so on single-threaded WASM, where a synthesis blocks the program, the first sentence still arrives while the rest is synthesized.

```sh
hf download onnx-community/Kokoro-82M-v1.0-ONNX --include config.json tokenizer.json onnx/model.onnx 'voices/*'
hf download hexgrad/misaki --repo-type dataset --include 'us_*.json' 'gb_*.json'
npm install phonemize
curl -s localhost:11435/v1/audio/speech -H 'content-type: application/json' \
  -d '{ "input": "Hello from the kernel.", "voice": "bf_emma" }' > hello.wav
```

Voices are Kokoro's American (`af_*`, `am_*`) and British English (`bf_*`, `bm_*`), Spanish (`e`), French (`f`), Hindi (`h`), Italian (`i`) and Brazilian Portuguese (`p`) voices; Japanese and Chinese are left out, since they have no phonemizer here. Variants are the repo's quantizations that are in the cache (`fp32`, `fp16`, `q8`, `q8f16`, `q4`, `q4f16`, `uint8`, `uint8f16`). `fp32` is the default when it is in the cache, else the first other quantization in that order, with `fp16` last: `fp32` is the fastest on CPU, and `fp16`, though faster on WebGPU, comes out garbled there.

#### Phonemizers

Kokoro needs a phonemizer, and ortllama ships none. npm phonemizers are imported at runtime wherever ortllama can find them, the misaki lexicon comes from the Hugging Face cache, and speech answers `501` with `code: "engine_unavailable"`, naming what to install, until one is there.

**The recommended setup is misaki plus `phonemize`:** it has no GPL code, and in listening tests it was the best of the combinations. It's not perfect, but usable, and English only. Add `espeak-ng` for the Spanish, French, Hindi, Italian and Portuguese voices.

| Phonemizer | Get it | Languages | Licence | |
| --- | --- | --- | --- | --- |
| [`misaki`](https://github.com/hexgrad/misaki) | `hf download hexgrad/misaki --repo-type dataset` (6–7 MB per accent) | English | Apache-2.0 | the lexicons of the G2P Kokoro was trained with; words it does not know, and verbs whose reading differs from the noun's (`lives`, `record`, `tear`), go with their sentence to the next English phonemizer, and when there is none, numbers, ordinals and symbols are read out and unknown words spelled |
| [`phonemize`](https://github.com/hans00/phonemize) | `npm install phonemize` (3 MB) | English | MIT | pure JavaScript rules and lexicon; British vowels are mapped onto American output, so words like `bath` keep the short vowel |
| [`espeak-ng`](https://github.com/ianmarmour/espeak-ng.js) | `npm install espeak-ng` (18 MB) | all of the above | GPL-3.0-or-later | optional; the only one for Spanish, French, Hindi, Italian and Portuguese |
| [`phonemizer`](https://github.com/xenova/phonemizer.js) | `npm install phonemizer` (2.6 MB) | English | labelled Apache-2.0, but bundles espeak-ng (GPL-3.0-or-later) | the same phonemes as `espeak-ng` |

Phonemizers are tried in the table's order, and the first that knows a voice's language speaks it, so English stays with misaki and `phonemize` even when `espeak-ng` is installed. `ORTLLAMA_PHONEMIZERS` (such as `phonemize,espeak-ng`) changes the order or leaves some out. A phonemizer that fails while speaking is skipped for a minute, and the next one speaks. Other heteronyms, such as past-tense `read` or `bass`, take the lexicon's usual reading until there is a part-of-speech tagger.

#### Phonon

[Phonon 31m English](https://huggingface.co/gradium/phonon-31m-en) (Gradium, Apache-2.0, built on Kyutai's Pocket TTS and the Mimi codec) speaks through the `phonon` engine without a phonemizer. A flow-matching transformer turns the tokens into one Mimi latent every 80 ms, and the Mimi decoder turns those into 24 kHz audio. The engine reads an ONNX export laid out as:

- `config.json`, the checkpoint's own, with an `onnx` section that lists the voices and the decoder's state,
- `tokenizer.json`,
- `onnx/text_conditioner.onnx`, `onnx/flow_lm_main.onnx` (KV cache in and out), `onnx/flow_lm_flow.onnx` and `onnx/mimi_decoder.onnx` (streaming state in and out), with `_quantized`, `_q4` and `_fp16` variants,
- `voices/<name>.bin`, voice prompts already projected for the flow LM.

No such export is published yet. Issue #102's feasibility run made one from the safetensors weights and put it into a local cache as `ai-ecoverse/phonon-31m-en-ONNX`; the [models](https://github.com/ai-ecoverse/models) repo will own the recipe.

- **Text.** Gradium's runtime, `ptts --lang en`, normalizes English before it tokenizes it. ortllama does the same in JavaScript (`src/phonon/`) and gets the same token IDs on 361 test inputs: numbers, currency, dates, emails, URLs, abbreviations, symbols, chunking and fitting. Numbers stay as digits unless they have scale words, as in ptts: `$45.99` reads "45 dollars and 99 cents", `1,234` reads "1 thousand 234", and `https://docs.example.org` is spelled out piece by piece.
- **Voices.** The 37 English voices of [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices), named in lowercase with dashes: `easton` (the default), `freya`, `zoey`, `elodie-rose`, `captain-silas-blackwater` and so on. Any capitalization works.
- **Sampling.** Temperature 0.3, as in ptts. The noise comes from a seeded generator, so a request always sounds the same. It is not ptts's ChaCha12 stream, though, so the same seed does not give ptts's audio. Given the same noise, the export's latents and audio match ptts's.
- **Chunks.** Phonon gets the whole input and cuts it as ptts does: at ptts's sentence ends, into chunks of up to 50 tokens. Each chunk starts again from the voice prompt and is sent as soon as it is ready. One noise stream runs through the request, as ptts's random generator does. `speed` other than 1 answers `400`: Phonon has no speed control.
- **Variants.** `fp32` is the default. `q8` is 8-bit `MatMulNBits` on the flow LM only, as ptts quantizes it, and keeps the Mimi decoder in fp32. `fp16` is the whole graph in half precision. `q4` sounds noticeably worse and is there for comparison.

Speed for four sentences (20 s of audio) on an M4 Max, as real-time factor and time until the first chunk is ready. Host numbers use 4 threads. Kernel numbers are `ortllama check`'s two sentences (4.7 s), on single-threaded WASM; Chrome's page process peaks at 1 GiB. For comparison, ptts itself (f32, 4 threads, streaming each frame) has a real-time factor of 0.023 and its first audio after 16 ms, and Kokoro fp32 on the host CPU 0.17 and 785 ms:

| Variant | Host CPU | Host WebGPU | Kernel WASM | Kernel WebGPU |
| --- | --- | --- | --- | --- |
| `fp32` | 0.044, 169 ms | 0.12, 469 ms | 0.25 | 0.49 |
| `q8` | 0.035, 130 ms | 0.11, 431 ms | 0.56 | 0.20 |
| `fp16` | 0.057, 224 ms | 0.13, 452 ms | 0.28 | 0.47 |

Phonon loads on the CPU on the host and on WASM in the kernel. Each 80 ms frame takes three small graph runs, so per-run overhead decides the speed, and WebGPU pays the most of it. Set `ORTLLAMA_EP` to use WebGPU anyway. Whisper large-v3-turbo hears 20 test sentences with a word error rate of 7.4 % from Phonon `fp32`, 6.5 % from `q8` and 7.4 % from ptts itself. It hears 5.3 % from Kokoro's `af_heart`. Every difference in Phonon's transcripts is the way a number or symbol is written, such as "9" for "nine", and none is a misread word.

### pi-audio

`@ai-ecoverse/slicc-ortllama/pi-audio` builds [pi-audio](https://github.com/ai-ecoverse/pi-audio) voice models for ortllama on pi-audio's OpenAI APIs, without loading the server or pi-audio:

| Function | Returns | For |
| --- | --- | --- |
| `transcriptionModel({ baseUrl, id })` | an `openai-transcriptions` model | `transcribe()` |
| `realtimeTranscriptionModel({ baseUrl, id })` | an `openai-realtime-transcription` model with server VAD | `listen()` |
| `speechModel({ baseUrl, id, voices })` | an `openai-speech` model | `speak()` |
| `voiceModels(baseUrl)` | all three | |

`baseUrl` defaults to `http://127.0.0.1:11435/v1`.

```js
import { speak } from '@ai-ecoverse/pi-audio';
import { registerOpenAI } from '@ai-ecoverse/pi-audio/openai';
import { speechModel } from '@ai-ecoverse/slicc-ortllama/pi-audio';

registerOpenAI();
const model = speechModel();
for await (const event of speak(model, { text: 'Hello.', voice: 'af_heart' }, { apiKey: 'ortllama' }))
  if (event.type === 'audio') play(event.chunk);
```

## Models

ortllama downloads nothing. It reads the Hugging Face cache that [`hf download`](https://huggingface.co/docs/huggingface_hub/guides/cli) fills (`HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`), through its `refs/` and `snapshots/`. A model ID is a Hub ID: `org/repo[/subfolder][@revision][:variant]`, such as `ai-ecoverse/kev.js/kev-0.8b:q8f32`. A model that isn't cached, or a cached variant with missing files, is answered with `404` and the `hf download` command that fetches it. Missing files include the external data files a decoder's or Whisper's `.onnx` graphs name, read from the graphs themselves up to 16 MB each (graphs that keep weights outside are a few hundred KB; larger ones hold their weights inline), so an interrupted download is caught before the model loads, and the `hf download` command fetches them even when they are not named after their graph; a variant that isn't cached at all is answered with `404` and the variants that are, and a cached repo with no ONNX model ortllama can run, such as an MLX or PyTorch checkpoint, with `404` and no download command. When such a repo, subfolder or `ORTLLAMA_MODEL_PATH` folder does have `.onnx` files at its top, in `onnx/`, or in a subfolder that is not itself a model ortllama runs (or that subfolder's `onnx/`), such as a new export with a quantization suffix no engine knows (`onnx/decoder_model_merged_int4.onnx`) or a lone `model.onnx`, the `404` names up to three of them and points at the layouts under [Engines](#engines). `ortllama ls` notes such repos and `ORTLLAMA_MODEL_PATH` folders on stderr, but not the subfolders of one ortllama runs.

Before a model loads, ortllama checks each of its cached files, including the external data its `.onnx` graphs name and a cached fallback vision encoder, against the hub's ETag, which `hf download` names its blob after: the SHA-256 for LFS files and the git blob SHA-1 for the others, since a stalled download can leave a damaged blob under its final name without an error. A file that does not match is refused with `500`, code `model_corrupt`, its name, both hashes and the `hf download --force-download` command that fetches it again. Each blob is checked once per process while its size and modification time stay the same. On a host it is streamed through `node:crypto`; in the kernel, where jsh reads a whole file for each ranged read, it is read once and hashed with WebCrypto. On an M4 Max, the host check takes about 1.2 s for the 3.3 GB Ministral 3 q4f16 files and 4.4 s for `kev-9b`'s 8.9 GB. `ORTLLAMA_MODEL_PATH` folders have no ETags and are not checked.

### Roles

Clients can ask for a role instead of a model. `ortllama alias decisions <model>` changes what a role means, so swapping a model changes no client.

| Role | Default |
| --- | --- |
| `chat` | [`mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX) |
| `chat-reasoning` | [`ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX:q4f16`](https://huggingface.co/ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX), Mistral's Reasoning checkpoint in the Instruct export's layout |
| `decisions` | [`ai-ecoverse/kev.js/kev-0.8b:q8f32`](https://huggingface.co/ai-ecoverse/kev.js) |
| `transcribe`, `realtime-transcribe` | [`onnx-community/whisper-tiny`](https://huggingface.co/onnx-community/whisper-tiny) |
| `vad` | [`onnx-community/silero-vad`](https://huggingface.co/onnx-community/silero-vad) |
| `speech` | [`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) |

The defaults that the [quick start](#quick-start) and [speech](#speech) examples do not fetch:

```sh
hf download ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX --include '*.json' chat_template.jinja 'onnx/*_q4f16.onnx*' 'onnx/vision_encoder_fp16.onnx*'
hf download onnx-community/silero-vad --include onnx/model.onnx
```

### Engines

Engines are picked by the files a model has, not by its name. A model is listed only with the variants whose ONNX graphs are in the cache (both for Whisper), and a `variant` hint in `config.json` that names another one falls back to the engine's default, so MLX, PyTorch and safetensors checkpoints that share a config with an ONNX export, such as `mlx-community/whisper-large-v3-turbo`, stay out of `ortllama ls` and `/v1/models`. Each engine in `src/engines/` gets ONNX Runtime from the caller and provides `detect`, `missing`, `include`, `load`, `checkOverrides` and `check`.

- **Decoder.** Runs the transformers.js layout (`onnx/decoder_model_merged*.onnx` with `embed_tokens*.onnx`, or `onnx/model*.onnx`; variants are its quantizations, such as `q4f16`) and the onnxruntime-genai layout (`genai_config.json`, one variant per folder), with the model's own chat template and, when the export has one, its `vision_encoder*.onnx`. A small profile per family, chosen from the template, parses tool calls and reasoning; Mistral's `[TOOL_CALLS]name[ARGS]{…}` and `[THINK]…[/THINK]` are built in, and control-token names in client text are escaped so they cannot inject one. Ministral 3 models known to work, all with image input, at `q4f16` with the `fp16` vision encoder, with the load time and then prefill / decode tokens per second for a 1,550-token prompt and 128 new tokens on an M4 Max:

| Model | Download | Host CPU | Host WebGPU | Kernel WebGPU |
| --- | --- | --- | --- | --- |
| `mistralai/Ministral-3-3B-Instruct-2512-ONNX` (`chat`) | 3.3 GB | 1.0 s, 179 / 15 | 5.0 s, 926 / 75 | 2.4 s, 782 / 62 |
| `ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX` (`chat-reasoning`) | 3.3 GB | 1.0 s, 187 / 18 | 4.8 s, 1016 / 84 | 2.7 s, 856 / 74 |
| `ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX` | 6.5 GB | 2.2 s, 80 / 7.3 | 5.6 s, 498 / 40 | 4.7 s, 422 / 39 |
| `ai-ecoverse/Ministral-3-8B-Reasoning-2512-ONNX` | 6.5 GB | 2.1 s, 91 / 11 | 5.1 s, 543 / 48 | 7.3 s, 458 / 46 |
| `ai-ecoverse/Ministral-3-14B-Instruct-2512-ONNX` | 9.5 GB | 3.2 s, 55 / 6.0 | 6.2 s, 336 / 31 | 8.0 s, 283 / 30 |
| `ai-ecoverse/Ministral-3-14B-Reasoning-2512-ONNX` | 9.5 GB | 2.7 s, 62 / 9.1 | 4.6 s, 361 / 37 | 7.8 s, 303 / 34 |

The 8B and 14B exports are ai-ecoverse's, made by the [models](https://github.com/ai-ecoverse/models) recipes from onnxruntime-genai's builder (the decoder) and Mistral's 3B vision and embedding graphs, and pick PyTorch's next token 93–97 % of the time (Mistral's 3B: 95 %). Their decoder takes no `position_ids` and leaves the KV head size symbolic; ortllama feeds `position_ids` only to graphs that ask for them and sizes the cache from `config.json`. Fetch one with `hf download ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX --include '*.json' chat_template.jinja 'onnx/decoder_model_merged_q4f16.onnx*' 'onnx/embed_tokens_fp16.onnx*' 'onnx/vision_encoder_fp16.onnx*'` and make it a role's model with `ortllama alias chat ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX`. All of them load in slicc-kernel; the 14B models peak at about 11 GiB in the page and 12 GiB in Chrome's GPU process. Mistral's 3B Instruct graphs never apply Ministral 3's query scaling, which starts at position 16,384 (they divide by 262,144 instead of 16,384, which the `fp16` graph rounds to infinity), so past 16k tokens it drifts from PyTorch; ai-ecoverse's 3B Reasoning export scales queries as transformers does since revision `c592447`, like the 8B and 14B. Both 3B models still recalled every fact asked for after 23,000 tokens of code.
- **Whisper.** Runs the transformers.js ONNX exports (`config.json` with `model_type: "whisper"`, `onnx/encoder_model*.onnx` and `onnx/decoder_model_merged*.onnx`, external `.onnx_data` included). Variants follow the file suffixes: `fp32`, `fp16`, `q8` (`_quantized`), `int8`, `uint8`, `q4`, `q4f16`, `bnb4`; the default is the first complete one in that order. Mel bins, special tokens, languages and the timestamp grid come from the model's configs and tokenizer, so large-v3 models (128 mel bins, the extra `yue` token) work as well. Models known to work, with the mean word error rate on the `check` clips and real-time factors for a 2.7-second and a long clip (44 s on the host, 32 s in the kernel) on an M4 Max:

| Model | Download | WER | Host CPU | Host WebGPU | Kernel WebGPU |
| --- | --- | --- | --- | --- | --- |
| `onnx-community/whisper-tiny` (default) | 151 MB | 0.092 | 0.06 / 0.013 | 0.13 / 0.077 | 0.09 / 0.061 |
| `onnx-community/whisper-base` | 291 MB | 0.046 | 0.09 / 0.020 | 0.11 / 0.096 | 0.13 / 0.096 |
| `onnx-community/whisper-large-v3-turbo:q4` | 759 MB | 0.005 | 1.3 / 0.20 | 0.27 / 0.070 | 0.31 / 0.118 |

`whisper-tiny` stays the default: it is the smallest, and the fastest on the host CPU, which is where Whisper runs best on host Node. `whisper-large-v3-turbo` makes almost no mistakes, in French and Spanish too, but wants WebGPU, where its 32-layer encoder runs five times faster than on the CPU, so it loads on the host's default order rather than Whisper's `cpu`; fetch it with `hf download onnx-community/whisper-large-v3-turbo --include '*.json' onnx/encoder_model_q4.onnx onnx/decoder_model_merged_q4.onnx`. `whisper-base` sits in between. In a realtime session the final transcript follows the end of speech after 0.2 s with `whisper-tiny` on the CPU, 0.4 s with `whisper-base` on WebGPU, and 1 s with `whisper-large-v3-turbo` on WebGPU (5 s on the CPU).

### Model libraries

Model-specific libraries are never installed with ortllama. They are devDependencies here, and engines load them only when a model needs them, such as `@ai-ecoverse/kev.js` and `@ai-ecoverse/jev-omni.js` for System One and `espeak-ng`, `phonemize` or `phonemizer` for Kokoro. ortllama looks for each library:

1. next to itself,
2. in `ORTLLAMA_MODULE_PATH` (project folders that contain a `node_modules`),
3. in pnpm's global installs (`$PNPM_HOME/global/…`, including slicc-kernel's `/usr/local/share/pnpm`).

A model whose library is missing is still listed, and requests for it answer `501` with code `engine_unavailable` and the package to install.

### Check

`ortllama check` tells whether a model computes on this device what it computes on the reference, a host CPU. ortllama ships fixtures in `fixtures/` for every default model and the other models below that work; `check` picks them by model ID, so `ortllama check chat` compares the Instruct model. A fixture file is `{ "fixtures": [...] }`; audio may be inline base64 (`audio`) or a WAV path relative to the file (`file`). Each engine compares what stays stable across execution providers:

| Engine | Fixtures | Passes when |
| --- | --- | --- |
| decoder | 3 prompts, the reference's greedy tokens with their log-probabilities and the margin to the runner-up | the tokens, fed back one by one, keep a mean log-probability within 0.15 nats, and no step with a margin over 1 nat picks another token |
| systemone | 12 System One requests (plus an image for vision bundles) with their raw decision probabilities | mean change ≤ 0.02, at most 0.25, and no clear answer flips |
| systemone, kev-ministral `:base` | 3 prompts and an image, with the reference's 24 greedy tokens each (`kev-ministral-3b.base.json`) | fed the reference's tokens wherever they differ, at most 10 % of the tokens change |
| whisper | 4 clips: `fox.wav`, a 32-second English story, French and Spanish (Kokoro voices, 8 kHz 8-bit) | the language is detected and the word error rate stays within 0.1 of the reference's |
| kokoro | 3 sentences as token IDs, so the phonemizer does not matter, with a 40-band log-mel summary | the mel spectrogram stays within 2 dB and the length within 5 %; phonemes are compared and reported |
| phonon | 3 sentences as text and token IDs, synthesized without sampling noise, with a 40-band log-mel summary | the token IDs match, and the mel spectrogram stays within 2 dB and the length within 5 % |
| silero | speech probabilities for every 32 ms frame of two clips | mean change ≤ 0.01, at most 0.1, and no clear speech/silence flip |

The tolerances leave room over what WebGPU differs from CPU, measured on host Node and in slicc-kernel (both alike): decoder mean 0.026–0.063 nats (max 0.48, flips only below a 0.55 margin; Ministral 3 3B to 14B), System One mean 0.006 (max 0.027 for Kev, 0.051 for kev-ministral), kev-ministral's base weights 2 of 96 tokens on host WebGPU, Kokoro 0.15–0.24 dB, Phonon 0.00 dB (host WebGPU, kernel WASM and WebGPU), Whisper and Silero none. Kokoro's `q8` variant lands at 2–5 dB and its `fp16` on WebGPU, which sounds garbled, at 25 dB. Phonon's `q8` lands at 1.2–1.8 dB, and its `fp16` on kernel WebGPU at 8–9 dB. Fixtures hold the default variant; other variants can fall outside them, and `check` says so. `node scripts/fixtures.mjs <model>…` writes them on the host CPU (`node scripts/fixtures.mjs audio` makes the clips with Kokoro).

## Execution providers

A model loads on one execution provider at a time, with `cpu` (`wasm` in the browser) for the operators that provider lacks. When a session cannot be created, such as CUDA without its libraries, the model loads on the next provider in the order, down to `cpu` alone, and each reason is logged once. Providers this ONNX Runtime does not have are skipped with a warning. The order is the first of:

1. `ORTLLAMA_EP`, such as `cuda,cpu`. It beats every hint because it is the operator's switch for one process: `ORTLLAMA_EP=cpu` keeps every model off the GPU, whatever `config.json` says.
2. The model's hint in `config.json`, such as `{ "hints": { "onnx-community/whisper-tiny": { "executionProviders": ["webgpu", "cpu"] } } }`. A model loaded with a hint is a separate loaded model.
3. The engine's preference: Whisper models with up to 6 encoder layers (`tiny`, `base`) and Silero VAD use `cpu` on Node, which leaves the GPU to chat and System One (small Whisper decoder steps are as fast there as on WebGPU, and Silero's frames much faster), and Silero uses `wasm` in the browser. Larger Whisper models, such as `large-v3-turbo`, take the host's default, since their encoder runs five times faster on WebGPU.
4. The host's default, chosen from what `onnxruntime-node` 1.30 ships there:

| Host | onnxruntime-node 1.30 has | Default |
| --- | --- | --- |
| macOS arm64 | `cpu`, `webgpu`, `coreml` | `webgpu,cpu`; CoreML runs only 147 of Kev's 675 nodes |
| Linux x64 | `cpu`, `webgpu`, `cuda` and `tensorrt` (CUDA 12 libraries fetched by its install script) | `cuda,webgpu,cpu` |
| Linux arm64 | `cpu` | `cpu` |
| Windows x64, arm64 | `cpu`, `webgpu`, `dml` | `webgpu,dml,cpu` |
| macOS x64 | no binary | |
| browser (`onnxruntime-web`) | `webgpu`, `wasm` | `webgpu,wasm`, or `wasm` without WebGPU |

`ortllama ps` and `GET /ortllama/ps` report, per model, the `executionProviders` it loaded on, where the order came from (`providerSource`: `env`, `hint`, `engine` or `default`) and `modelBytes`, the size on disk of the variant's files; and, as `memory`, the process's `rss`, `heapUsed`, `external` and `arrayBuffers` on Node (`null` in slicc-kernel, where `process.memoryUsage()` answers zeros). ONNX Runtime 1.30 reports no memory of its own: `onnxruntime-node` has no arena statistics and `onnxruntime-web` no GPU buffer usage, so ortllama shows none.

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
| `ORTLLAMA_MAX_TOKENS` | `8192` | completion tokens for a chat request that sets no limit |
| `ORTLLAMA_EP` | per host, see [Execution providers](#execution-providers) | execution providers, in order, for every model |
| `ORTLLAMA_MODEL_PATH` | none | extra plain folders, listed as `local/<name>` |
| `ORTLLAMA_MODULE_PATH` | none | project folders searched for model libraries such as `@ai-ecoverse/kev.js` and `espeak-ng` |
| `ORTLLAMA_ORT_LOG_LEVEL` | `error` | ONNX Runtime's log level |
| `ORT_DISABLE_TELEMETRY` | `1` | onnxruntime-node on macOS and Linux uploads usage telemetry to Microsoft unless this is set, and its uploader can abort the process at exit, so ortllama sets it; `0` turns the telemetry back on |
| `ORTLLAMA_PHONEMIZERS` | `misaki,phonemize,espeak-ng,phonemizer` | Kokoro's phonemizers, in order |

## Develop

Node ≥ 24.

| Command | |
| --- | --- |
| `npm run lint` | `slicc-lint`, from [slicc-shared-web](https://github.com/ai-ecoverse/slicc-shared-web) |
| `npm test` | integration tests in `test/integration/` |
| `npm run test:unit` | unit tests, which stay local in `test/unit/` |

Each integration test runs when its models are in the Hugging Face cache; `ORTLLAMA_REQUIRE_MODEL=1` makes a missing model a failure, and `ORTLLAMA_TEST_WHISPER` runs the Whisper and realtime tests on another Whisper model, such as `onnx-community/whisper-base`. Likewise `ORTLLAMA_CHAT_MODEL` runs the chat and image tests, and `ORTLLAMA_REASONING_MODEL` the reasoning tests, on another model, such as `ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX:q4f16` and `ai-ecoverse/Ministral-3-8B-Reasoning-2512-ONNX:q4f16`.

| Tests | Models |
| --- | --- |
| Kev | `ai-ecoverse/kev.js/kev-0.8b` |
| kev-ministral, its base weights | `ai-ecoverse/kev-ministral/kev-ministral-3b` |
| Whisper | `onnx-community/whisper-tiny` |
| Realtime | `onnx-community/whisper-tiny`, `onnx-community/silero-vad` |
| Chat | `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16` |
| Reasoning | `ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX:q4f16` |
| Images | `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16` with its vision encoder, `ai-ecoverse/kev.js/kev-0.8b-vision` |
| Kokoro | `onnx-community/Kokoro-82M-v1.0-ONNX` |
| Phonon | `ai-ecoverse/phonon-31m-en-ONNX` |

Releases are published to npm from `main` by [semantic-release](https://semantic-release.gitbook.io), so commit messages follow [Conventional Commits](https://www.conventionalcommits.org).

## Related projects

### In ai-ecoverse

- [slicc](https://github.com/ai-ecoverse/slicc): the browser-native AI agent ortllama serves models to.
- [slicc-kernel](https://github.com/ai-ecoverse/slicc-kernel): the wasm process kernel ortllama runs in, in the browser.
- [slicc-jsh](https://github.com/ai-ecoverse/slicc-jsh): the kernel's `node` shim that runs ortllama there.
- [slicc-agent](https://github.com/ai-ecoverse/slicc-agent): the agent core that talks to ortllama from its worker.
- [pi-audio](https://github.com/ai-ecoverse/pi-audio): voice bindings for pi that `@ai-ecoverse/slicc-ortllama/pi-audio` builds models for.
- [kev.js](https://github.com/ai-ecoverse/kev.js): Kev decision models on onnxruntime-web, behind `/v1/systemone`.
- [jev-omni.js](https://github.com/ai-ecoverse/jev-omni.js): the Jev-Omni decision classifier on onnxruntime-web, behind `/v1/systemone` too.
- [models](https://github.com/ai-ecoverse/models): conversion recipes for the ONNX models ai-ecoverse publishes on [Hugging Face](https://huggingface.co/ai-ecoverse), including the Ministral 3 Reasoning export.
- [cua-s1.js](https://github.com/ai-ecoverse/cua-s1.js): Cua's form-filling and next-action models on onnxruntime-web, compared with Kev and Jev-Omni in [decision-vision-bench](https://github.com/ai-ecoverse/decision-vision-bench).
- [slicc-shared-web](https://github.com/ai-ecoverse/slicc-shared-web): the shared lint, Renovate and CI configuration this repo uses.

### Elsewhere

- [ONNX Runtime](https://github.com/microsoft/onnxruntime): `onnxruntime-web` and `onnxruntime-node`, the runtimes ortllama injects into its engines.
- [Ollama](https://github.com/ollama/ollama) and [llama.cpp](https://github.com/ggml-org/llama.cpp): the local model servers ortllama's name, port and `timings` take after; ortllama serves only OpenAI-style APIs.
- [pi](https://github.com/earendil-works/pi): pi-ai is the reference client for ortllama's chat completions.
- [transformers.js](https://github.com/huggingface/transformers.js): the ONNX export layout the decoder and Whisper engines read, and the preprocessing they match.
- [huggingface.js](https://github.com/huggingface/huggingface.js) and [tokenizers.js](https://github.com/huggingface/tokenizers.js): chat templates and tokenizers.
- [Kev](https://github.com/jaredpalmer/kev): the decision models kev.js runs.
- [Ministral 3](https://huggingface.co/collections/mistralai/ministral-3), [Whisper](https://github.com/openai/whisper), [Silero VAD](https://github.com/snakers4/silero-vad) and [Kokoro](https://github.com/hexgrad/kokoro): the default models.
- [Phonon](https://github.com/gradium-ai/xn-ptts) (Gradium, built on Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)): the `phonon` speech engine's model, and `ptts`, its reference runtime.
- [jSquash](https://github.com/jamsinclair/jSquash): libwebp in WebAssembly for WebP images.

## License

[Apache-2.0](LICENSE)
