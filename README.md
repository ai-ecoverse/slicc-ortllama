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
| `POST /v1/chat/completions` | [OpenAI chat completions](#chat-completions) on the decoder engine |
| `POST /v1/systemone` | TypeSafe's [System One](#system-one) protocol, on kev.js |
| `POST /v1/audio/transcriptions` | OpenAI's [transcription](#transcription) API, on Whisper |
| `GET /v1/realtime?intent=transcription` | OpenAI's [realtime transcription](#realtime-transcription) over WebSocket |
| `POST /v1/audio/speech` | OpenAI's [text to speech](#speech), on Kokoro-82M |
| `GET /ortllama/ps` | loaded models, their [execution providers](#execution-providers) and size, and the process's memory |

Errors use OpenAI's envelope, `{ "error": { "message", "type", "param", "code" } }`.

### Chat completions

Requests take `messages` (`system`, `developer`, `user`, `assistant` with `tool_calls`, `tool`), `tools`, `tool_choice`, `stream`, `stream_options.include_usage`, `max_completion_tokens` or `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop` and `reasoning_effort`, and ignore `store`.

- **Streaming.** With `stream: true`, the response is a stream of `chat.completion.chunk`s carrying `delta.content`, `delta.reasoning_content` and `delta.tool_calls`. The last chunk has `finish_reason` (`stop`, `length` or `tool_calls`) and llama.cpp-style `timings`, followed by a usage chunk when asked. A closed connection stops generation.
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

### System One

`POST /v1/systemone` speaks TypeSafe's System One protocol on [kev.js](https://github.com/ai-ecoverse/kev.js). Requests to a kev.js vision bundle (such as `ai-ecoverse/kev.js/kev-0.8b-vision`) take an `image` field, a `data:` URL string or `{ "url": … }`, decoded to RGBA for kev.js; text-only bundles answer it with `400`. pi-ai's System One client does not send images yet. Bundles known to work, with the load time and a warm three-question request (an image request for vision bundles) on an M4 Max:

| Model | Download | Host CPU | Host WebGPU | Kernel WebGPU load |
| --- | --- | --- | --- | --- |
| `ai-ecoverse/kev.js/kev-0.8b` (default), `kev-0.8b-vision` | 0.84 GB, +0.2 GB | 0.8 s, 70 ms | 0.9 s, 75 ms (image 50 ms) | 1.9–2.4 s |
| `ai-ecoverse/kev.js/kev-4b`, `kev-4b-vision` | 4.7 GB, +0.7 GB | 5.7 s, 0.5–1 s (image 0.35 s) | 2.4 s, 235 ms (image 155 ms) | 13–23 s |
| `ai-ecoverse/kev.js/kev-9b` | 8.9 GB | 7.8 s, 0.9–1.5 s | 4.4 s, 330 ms | 27 s |

Loading a bundle takes about its size in memory. On a host, onnxruntime-node opens the graph by path and reads the weights itself: `kev-9b` peaks at 9.0 GB, `kev-4b` at 5.2 GB. In the kernel, onnxruntime-web asks for each weight's bytes while it builds the session, and ortllama reads them then, holding at most 1 GiB of weight files: for `kev-9b`, Chrome's page process peaks at 3.7 GiB and its GPU process at 8.1 GiB. jsh reads a whole file for each ranged read ([slicc-jsh#401](https://github.com/ai-ecoverse/slicc-jsh/issues/401)), so the larger bundles load about twice as slowly in the kernel as they would with every weight held at once. Both need kev.js 1.1 or later; with 1.0, every weight is read into memory first, and `kev-9b` peaks at about 20 GB.

### Transcription

`POST /v1/audio/transcriptions` takes a multipart `file`, `model`, `language`, `prompt` and `response_format` (`json`, `verbose_json`, `text`, `srt`, `vtt`). Audio is WAV, mp3, FLAC, Ogg or WebM Opus, or 16-bit `audio/pcm` (`;rate=N`, default 24 kHz). The format comes from the file's first bytes, else its content type or file name, and is decoded the same way on both tracks: mp3 on [minimp3](https://github.com/lieff/minimp3) and Opus on libopus, both in WebAssembly, FLAC in JavaScript. A leading ID3 tag is skipped before the format is read. Audio longer than 30 minutes, or more samples than 30 minutes at 48 kHz, FLAC above 192 kHz, a stream that changes its sample rate, and other formats (AAC, M4A, MP4, Vorbis), answer `400`.

It answers `{ text, language, duration, segments, usage: { type: "duration", seconds } }`. Audio longer than 30 seconds is decoded in windows that follow Whisper's timestamps; the language is detected when the request names none.

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

`POST /v1/audio/speech` takes `{ model, input, voice, response_format, speed, instructions }`, on [Kokoro-82M](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).

- `model` defaults to the `speech` role, `voice` to `af_heart`, and `instructions` is accepted and ignored.
- `response_format` is `wav` (the default), `pcm` (24 kHz mono 16-bit little-endian), `mp3` (`audio/mpeg`, 64 kbit/s), `opus` (Ogg Opus, `audio/opus`, 32 kbit/s) or `flac` (`audio/flac`); `aac` answers `400`. mp3 is encoded by `@breezystack/lamejs` (LGPL-3.0), which ortllama does not ship and finds like a phonemizer (`npm install @breezystack/lamejs`); without it, `mp3` answers `501`.
- `input` takes up to 4096 characters. It is split into sentences, long ones at clause marks or spaces, and each piece's audio is sent as soon as it is ready.

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

- **Decoder.** Runs the transformers.js layout (`onnx/decoder_model_merged*.onnx` with `embed_tokens*.onnx`, or `onnx/model*.onnx`; variants are its quantizations, such as `q4f16`) and the onnxruntime-genai layout (`genai_config.json`, one variant per folder), with the model's own chat template and, when the export has one, its `vision_encoder*.onnx`. A small profile per family, chosen from the template, parses tool calls and reasoning; Mistral's `[TOOL_CALLS]name[ARGS]{…}` and `[THINK]…[/THINK]` are built in, and control-token names in client text are escaped so they cannot inject one.
- **Whisper.** Runs the transformers.js ONNX exports (`config.json` with `model_type: "whisper"`, `onnx/encoder_model*.onnx` and `onnx/decoder_model_merged*.onnx`, external `.onnx_data` included). Variants follow the file suffixes: `fp32`, `fp16`, `q8` (`_quantized`), `int8`, `uint8`, `q4`, `q4f16`, `bnb4`; the default is the first complete one in that order. Mel bins, special tokens, languages and the timestamp grid come from the model's configs and tokenizer, so large-v3 models (128 mel bins, the extra `yue` token) work as well. Models known to work, with the mean word error rate on the `check` clips and real-time factors for a 2.7-second and a long clip (44 s on the host, 32 s in the kernel) on an M4 Max:

| Model | Download | WER | Host CPU | Host WebGPU | Kernel WebGPU |
| --- | --- | --- | --- | --- | --- |
| `onnx-community/whisper-tiny` (default) | 151 MB | 0.092 | 0.06 / 0.013 | 0.13 / 0.077 | 0.09 / 0.061 |
| `onnx-community/whisper-base` | 291 MB | 0.046 | 0.09 / 0.020 | 0.11 / 0.096 | 0.13 / 0.096 |
| `onnx-community/whisper-large-v3-turbo:q4` | 759 MB | 0.005 | 1.3 / 0.20 | 0.27 / 0.070 | 0.31 / 0.118 |

`whisper-tiny` stays the default: it is the smallest, and the fastest on the host CPU, which is where Whisper runs best on host Node. `whisper-large-v3-turbo` makes almost no mistakes, in French and Spanish too, but wants WebGPU, where its 32-layer encoder runs five times faster than on the CPU, so it loads on the host's default order rather than Whisper's `cpu`; fetch it with `hf download onnx-community/whisper-large-v3-turbo --include '*.json' onnx/encoder_model_q4.onnx onnx/decoder_model_merged_q4.onnx`. `whisper-base` sits in between. In a realtime session the final transcript follows the end of speech after 0.2 s with `whisper-tiny` on the CPU, 0.4 s with `whisper-base` on WebGPU, and 1 s with `whisper-large-v3-turbo` on WebGPU (5 s on the CPU).

### Model libraries

Model-specific libraries are never installed with ortllama. They are devDependencies here, and engines load them only when a model needs them, such as `@ai-ecoverse/kev.js` for System One and `espeak-ng`, `phonemize` or `phonemizer` for Kokoro. ortllama looks for each library:

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
| whisper | 4 clips: `fox.wav`, a 32-second English story, French and Spanish (Kokoro voices, 8 kHz 8-bit) | the language is detected and the word error rate stays within 0.1 of the reference's |
| kokoro | 3 sentences as token IDs, so the phonemizer does not matter, with a 40-band log-mel summary | the mel spectrogram stays within 2 dB and the length within 5 %; phonemes are compared and reported |
| silero | speech probabilities for every 32 ms frame of two clips | mean change ≤ 0.01, at most 0.1, and no clear speech/silence flip |

The tolerances leave room over what WebGPU differs from CPU, measured on host Node and in slicc-kernel (both alike): decoder mean 0.026–0.053 nats (max 0.4, flips only below a 0.2 margin), System One mean 0.006 (max 0.027), Kokoro 0.15–0.24 dB, Whisper and Silero none. Kokoro's `q8` variant lands at 2–5 dB and its `fp16` on WebGPU, which sounds garbled, at 25 dB. Fixtures hold the default variant; other variants can fall outside them, and `check` says so. `node scripts/fixtures.mjs <model>…` writes them on the host CPU (`node scripts/fixtures.mjs audio` makes the clips with Kokoro).

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

Each integration test runs when its models are in the Hugging Face cache; `ORTLLAMA_REQUIRE_MODEL=1` makes a missing model a failure, and `ORTLLAMA_TEST_WHISPER` runs the Whisper and realtime tests on another Whisper model, such as `onnx-community/whisper-base`.

| Tests | Models |
| --- | --- |
| Kev | `ai-ecoverse/kev.js/kev-0.8b` |
| Whisper | `onnx-community/whisper-tiny` |
| Realtime | `onnx-community/whisper-tiny`, `onnx-community/silero-vad` |
| Chat | `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16` |
| Reasoning | `ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX:q4f16` |
| Images | `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16` with its vision encoder, `ai-ecoverse/kev.js/kev-0.8b-vision` |
| Kokoro | `onnx-community/Kokoro-82M-v1.0-ONNX` |

Releases are published to npm from `main` by [semantic-release](https://semantic-release.gitbook.io), so commit messages follow [Conventional Commits](https://www.conventionalcommits.org).

## Related projects

### In ai-ecoverse

- [slicc](https://github.com/ai-ecoverse/slicc): the browser-native AI agent ortllama serves models to.
- [slicc-kernel](https://github.com/ai-ecoverse/slicc-kernel): the wasm process kernel ortllama runs in, in the browser.
- [slicc-jsh](https://github.com/ai-ecoverse/slicc-jsh): the kernel's `node` shim that runs ortllama there.
- [slicc-agent](https://github.com/ai-ecoverse/slicc-agent): the agent core that talks to ortllama from its worker.
- [pi-audio](https://github.com/ai-ecoverse/pi-audio): voice bindings for pi that `@ai-ecoverse/slicc-ortllama/pi-audio` builds models for.
- [kev.js](https://github.com/ai-ecoverse/kev.js): Kev decision models on onnxruntime-web, behind `/v1/systemone`.
- [models](https://github.com/ai-ecoverse/models): conversion recipes for the ONNX models ai-ecoverse publishes on [Hugging Face](https://huggingface.co/ai-ecoverse), including the Ministral 3 Reasoning export.
- [jev-omni.js](https://github.com/ai-ecoverse/jev-omni.js) and [cua-s1.js](https://github.com/ai-ecoverse/cua-s1.js): more decision models in the browser on onnxruntime-web, compared with Kev in [decision-vision-bench](https://github.com/ai-ecoverse/decision-vision-bench).
- [slicc-shared-web](https://github.com/ai-ecoverse/slicc-shared-web): the shared lint, Renovate and CI configuration this repo uses.

### Elsewhere

- [ONNX Runtime](https://github.com/microsoft/onnxruntime): `onnxruntime-web` and `onnxruntime-node`, the runtimes ortllama injects into its engines.
- [Ollama](https://github.com/ollama/ollama) and [llama.cpp](https://github.com/ggml-org/llama.cpp): the local model servers ortllama's name, port and `timings` take after; ortllama serves only OpenAI-style APIs.
- [pi](https://github.com/earendil-works/pi): pi-ai is the reference client for ortllama's chat completions.
- [transformers.js](https://github.com/huggingface/transformers.js): the ONNX export layout the decoder and Whisper engines read, and the preprocessing they match.
- [huggingface.js](https://github.com/huggingface/huggingface.js) and [tokenizers.js](https://github.com/huggingface/tokenizers.js): chat templates and tokenizers.
- [Kev](https://github.com/jaredpalmer/kev): the decision models kev.js runs.
- [Ministral 3](https://huggingface.co/collections/mistralai/ministral-3), [Whisper](https://github.com/openai/whisper), [Silero VAD](https://github.com/snakers4/silero-vad) and [Kokoro](https://github.com/hexgrad/kokoro): the default models.
- [jSquash](https://github.com/jamsinclair/jSquash): libwebp in WebAssembly for WebP images.

## License

[Apache-2.0](LICENSE)
