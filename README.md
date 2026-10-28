# slicc-ortllama

[![npm](https://img.shields.io/npm/v/@ai-ecoverse/slicc-ortllama)](https://www.npmjs.com/package/@ai-ecoverse/slicc-ortllama)
[![CI](https://github.com/ai-ecoverse/slicc-ortllama/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-ecoverse/slicc-ortllama/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ai-ecoverse/slicc-ortllama)](LICENSE)

A local model server: ONNX models on [ONNX Runtime](https://onnxruntime.ai), served over OpenAI-style HTTP APIs. It does chat with tools, reasoning and images, decisions, embeddings, transcription (batch and realtime) and text to speech. The same code runs in two places:
- **In the browser**, as a [slicc-kernel](https://github.com/ai-ecoverse/slicc-kernel) program on [slicc-jsh](https://github.com/ai-ecoverse/slicc-jsh)'s `node`, on `onnxruntime-web` (WebGPU, single-threaded WASM as the fallback), listening on the kernel's loopback. The page and the [slicc-agent](https://github.com/ai-ecoverse/slicc-agent) worker reach it as `http://<port>.kernel.localhost`.
- **On a host**, on Node 24+ with `onnxruntime-node`. Without it, ortllama runs on `onnxruntime-web`'s single-threaded WASM, CPU only, and says so once on stderr.

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

### Without onnxruntime-node

When `onnxruntime-node` cannot be imported, because it is not installed or has no binary for the host, ortllama on Node falls back to `onnxruntime-web`'s single-threaded WASM backend and writes one notice to stderr with the reason. Only `wasm` (also called `cpu`) is available there: other providers in the [order](#execution-providers) are skipped, and models load as they do in slicc-kernel, reading their weights and external data into memory, so large models take more memory and run slower than on `onnxruntime-node`. `ortllama ps` shows `onnxruntime-web` as the backend.

### As a library

`createRuntime({ backend })` takes the function that loads the ONNX Runtime backend, and `startServer({ runtime })` serves that runtime. `@ai-ecoverse/slicc-ortllama/ort/node` and `@ai-ecoverse/slicc-ortllama/ort/web` export the two loaders, each as `loadBackend({ env })`: `ort/node` loads `onnxruntime-node`, falling back to `onnxruntime-web` as above, and `ort/web` loads `onnxruntime-web` (WebGPU when `navigator.gpu` is there, else WASM), on Node as in the browser.

```js
import { createRuntime } from '@ai-ecoverse/slicc-ortllama';
import { loadBackend } from '@ai-ecoverse/slicc-ortllama/ort/web';

const runtime = createRuntime({ backend: () => loadBackend() });
```

### In slicc-kernel

`onnxruntime-node` is not needed: the `slicc` export condition picks `onnxruntime-web`. ortllama needs slicc-jsh 0.31.0 or later, which keeps a program alive while WebGPU work is pending and has the `tls` module that `ws`, which serves `/v1/realtime`, loads.

## CLI

| Command                                             |                                                                                                                                                                              |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ortllama serve [--port N] [--host H] [--detach]`   | start the server (default `127.0.0.1:11435`); `--detach` runs it in the background                                                                                           |
| `ortllama ls [--json]`                              | models in the Hugging Face cache that ortllama can run; cached repos and `ORTLLAMA_MODEL_PATH` folders with ONNX files in a layout it does not recognize are noted on stderr |
| `ortllama ps [--json]`                              | models the running server has loaded                                                                                                                                         |
| `ortllama alias [role [model]] [--unset role]`      | show or set [role aliases](#roles)                                                                                                                                           |
| `ortllama check <model> [--fixtures file] [--json]` | load a model on this device, time a request, and compare it with [reference fixtures](#check): the ones shipped for the model, or `--fixtures file`                          |

## APIs

| Route                                   |                                                                                                                                                                                                      |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /v1/models`                        | models and role aliases; System One models list `decisions` in `architecture.output_modalities`, and embedding models `embeddings`, with the sizes they can cut a vector to in `ortllama.dimensions` |
| `POST /v1/chat/completions`             | [OpenAI chat completions](#chat-completions) on the decoder engine, on [kev-ministral's base weights](#kev-ministrals-base-weights) and, experimentally, on [GGUF models](#gguf-experimental)        |
| `POST /v1/systemone`                    | TypeSafe's [System One](#system-one) protocol, on kev.js, jev-omni.js and, experimentally, [GGUF decision models](#gguf-experimental)                                                                |
| `POST /v1/embeddings`                   | OpenAI's [embeddings](#embeddings) API, on EmbeddingGemma 2's text model                                                                                                                             |
| `POST /v1/audio/transcriptions`         | OpenAI's [transcription](#transcription) API, on Whisper and Parakeet TDT                                                                                                                            |
| `GET /v1/realtime?intent=transcription` | OpenAI's [realtime transcription](#realtime-transcription) over WebSocket                                                                                                                            |
| `POST /v1/audio/speech`                 | OpenAI's [text to speech](#speech), on Kokoro-82M and [Phonon 31m English](#phonon)                                                                                                                  |
| `GET /v1/audio/voices[?model=<id>]`     | the [voices](#voices) of the cached speech models, with engine, language and gender                                                                                                                  |
| `GET /ortllama/ps`                      | loaded models, their [execution providers](#execution-providers) and size, and the process's memory                                                                                                  |

Errors use OpenAI's envelope, `{ "error": { "message", "type", "param", "code" } }`. A model sent to a route it does not answer gets `400` naming the route that does, such as `model onnx-community/embeddinggemma-2-ONNX does not answer chat completions; it answers POST /v1/embeddings`.

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

| Model                                                      | Download         | Host CPU                      | Host WebGPU                  | Kernel WebGPU load |
| ---------------------------------------------------------- | ---------------- | ----------------------------- | ---------------------------- | ------------------ |
| `ai-ecoverse/kev.js/kev-0.8b` (default), `kev-0.8b-vision` | 0.84 GB, +0.2 GB | 0.8 s, 70 ms                  | 0.9 s, 75 ms (image 50 ms)   | 1.9–2.4 s          |
| `ai-ecoverse/kev.js/kev-4b`, `kev-4b-vision`               | 4.7 GB, +0.7 GB  | 5.7 s, 0.5–1 s (image 0.35 s) | 2.4 s, 235 ms (image 155 ms) | 13–23 s            |
| `ai-ecoverse/kev.js/kev-9b`                                | 8.9 GB           | 7.8 s, 0.9–1.5 s              | 4.4 s, 330 ms                | 27 s               |

Loading a bundle takes about its size in memory. On a host, onnxruntime-node opens the graph by path and reads the weights itself: `kev-9b` peaks at 9.0 GB, `kev-4b` at 5.2 GB. In the kernel, onnxruntime-web asks for each weight's bytes while it builds the session, and ortllama reads them then, holding at most 1 GiB of weight files: for `kev-9b`, Chrome's page process peaks at 3.7 GiB and its GPU process at 8.1 GiB. jsh reads a whole file for each ranged read ([slicc-jsh#401](https://github.com/ai-ecoverse/slicc-jsh/issues/401)), so the larger bundles load about twice as slowly in the kernel as they would with every weight held at once. Both need kev.js 1.1 or later; with 1.0, every weight is read into memory first, and `kev-9b` peaks at about 20 GB.

[kev-ministral](https://huggingface.co/ai-ecoverse/kev-ministral) (`ai-ecoverse/kev-ministral/kev-ministral-3b`, variant `q8f32`, 6.6 GB) is the decision model from the web-browsing VLM work. It is Ministral-3-3B-Base with a Kev decision LoRA, left unmerged behind the decoder's `lora_scale` input, plus a Kev pointer head and Pixtral's vision encoder. kev.js 1.2 or later runs it through `loadKevMinistral`, and the System One engine finds it by its `manifest.json` (graphs in `onnx/`, no `variants`). It takes an `image` like the kev.js vision bundles: Pixtral's resize, so a 1024 × 576 screenshot becomes 798 tokens after `<state>`. Fetch it with `hf download ai-ecoverse/kev-ministral --include 'kev-ministral-3b/*'`.

On an M4 Max it loads in 5.1 s on the host CPU, 7.7 s on host WebGPU and 7.3–7.6 s in the kernel. The three-question request takes 0.71 s, 0.50 s and 0.50 s. A webrunner step from its demo (500 to 2,150 tokens of page and menu, plus 798 for a 1024 × 576 screenshot) takes a median of 22 s on the CPU, 4.2 s on host WebGPU and 4.2 s in the kernel; without the screenshot, 10 s, 1.8 s and 1.8 s. Loading peaks at a memory footprint of 6.6 GB on the host CPU and 12.6 GB on host WebGPU. In the kernel, Chrome's page process peaks at 11.1 GiB and its GPU process at 10 GiB.

Its adapter was trained for webrunner's menus, and that is where it is strongest. On Kev's 300 held-out transfer-v4 questions (raw logits) it lands between Kev-0.8B and Kev-4B:

| Model                                                          | Download        | transfer-v4 accuracy / Brier | Demo webrunner steps right, with / without the screenshot | Host WebGPU load, three-question request |
| -------------------------------------------------------------- | --------------- | ---------------------------- | --------------------------------------------------------- | ---------------------------------------- |
| `ai-ecoverse/kev.js/kev-0.8b` (`decisions`), `kev-0.8b-vision` | 0.84 GB, 1.0 GB | 0.640 / 0.492                | 4 / 6 of 10                                               | 0.8 s, 114 ms                            |
| `ai-ecoverse/kev-ministral/kev-ministral-3b`                   | 6.6 GB          | 0.733 / 0.414                | 7 / 7 of 10                                               | 7.7 s, 501 ms                            |
| `ai-ecoverse/kev.js/kev-4b`, `kev-4b-vision`                   | 4.7 GB, 5.4 GB  | 0.800 / 0.281                | 6 / 8 of 10                                               | 4.0 s, 353 ms                            |

`decisions` stays on Kev-0.8B: kev-ministral is eight times its size, takes ten times as long to load and four times as long per request, and Kev-4B is the better general model at a smaller size. Point a client at kev-ministral for browser-agent steps. Its model card reports 69 % on Mind2Web's unseen websites, against 57.6 % for `kev-4b-vision`. The same weights with `lora_scale = 0` are the stock base model, which `/v1/chat/completions` serves from the same load as [`ai-ecoverse/kev-ministral/kev-ministral-3b:base`](#kev-ministrals-base-weights).

[Jev-Omni](https://github.com/ai-ecoverse/jev-omni.js) (`ai-ecoverse/jev-omni.js/jev-omni`, Gemma 4 12B with a 256-way decision head, 13.6 GB) answers the same requests through [jev-omni.js](https://github.com/ai-ecoverse/jev-omni.js), with `@ai-ecoverse/jev-omni.js` installed. Each question is one forward pass over the state, its instructions and its options: a choice's criteria, `no` and `yes` for a noul, a score's levels. Answers, confidences and `usage` follow the same rules as Kev's. Jev-Omni takes 2 to 256 options per question, so a choice with one criterion answers `400`. An `image` is embedded once per request. On WebGPU, onnxruntime-web limits a prompt to about 8,000 tokens; a longer one answers `422` with code `context_length_exceeded`. On an M4 Max, it loads in 7.3 s on the host CPU and answers the three-question request in 3 s; on host WebGPU, 2.6–5.8 s and 0.9 s; in the kernel, 13–23 s and 1–1.2 s. Loading peaks at 13.3 GB on host WebGPU (15.9 GB on the CPU), and in the kernel at 3.3 GiB in Chrome's page process and 13 GiB in its GPU process. That needs jev-omni.js 0.2 or later: 0.1 reads every weight into memory first, which peaks at 30 GB on the host and fails to allocate in the kernel.

### Embeddings

`POST /v1/embeddings` is OpenAI's embeddings API, on the text model of Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) in [onnx-community's export](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX). It takes `model`, `input`, `dimensions`, `encoding_format` and `user`, which is ignored, and answers `{ "object": "list", "data": [{ "object": "embedding", "index", "embedding" }], "model", "usage": { "prompt_tokens", "total_tokens" } }`.

```sh
hf download onnx-community/embeddinggemma-2-ONNX --include '*.json' 'onnx/model_quantized.onnx*'
curl -s localhost:11435/v1/embeddings -H 'content-type: application/json' -d '{
  "input": ["Which planet is known as the Red Planet?"], "task": "search result", "dimensions": 256
}'
```

- **Input.** `model` defaults to the `embed` role. `input` is a string or an array of up to 2048 non-empty strings. Token IDs, as an array of numbers or of arrays of them, answer `400`: IDs belong to the tokenizer that made them, and the tiktoken IDs that OpenAI clients send, such as LangChain's `OpenAIEmbeddings` unless it has `check_embedding_ctx_length=False`, would embed as noise.
- **Context.** An input may have 8,192 tokens, counting its prefix and the `<bos>` and `<eos>` the tokenizer adds; a longer one answers `400` with `code: "context_length_exceeded"` and its index, and is never truncated. A request takes at most 300,000 tokens in all.
- **Dimensions.** `dimensions` is 768 (the default), 512, 256 or 128, the sizes EmbeddingGemma 2 was trained to be cut to (Matryoshka). ortllama keeps the leading dimensions and L2-normalizes them again, so the dot product stays the cosine; other values answer `400`. Queries and documents must be embedded at the same size.
- **Encoding.** `encoding_format` is `float` (the default) or `base64`, the vector's float32 little-endian bytes, as with OpenAI.
- **Batching.** The inputs of a request are sorted by length and run in batches of at most 32 inputs and 2,048 tokens with padding, below the 2,700 tokens a WebGPU batch takes before some kernels exceed a dispatch limit, as the model card warns. An input longer than that runs alone; 8,192 tokens ran on WebGPU on the host and on onnxruntime-web with Dawn. Between batches the request gives up the runtime, so more urgent work, such as a realtime decode, runs first, and a closed connection stops it.

EmbeddingGemma 2 is trained with a task prefix in front of each text. Input that already starts with one, `task: … | query: ` or `title: … | text: `, is embedded as it is sent, whatever the fields below say. For other input, three ortllama fields build the prefix, so a client need not:

| Field        | Values                                                                                                                          | Prefix                                                                                                                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`       | `search result`, `question answering`, `fact checking`, `code retrieval`, `classification`, `clustering`, `sentence similarity` | `task: {task} \| query: {text}`; `search result` when only `input_type` is set                                                                                                                                      |
| `input_type` | `query` (the default when `task` is set), `document`                                                                            | a document of the four retrieval tasks, or of none, becomes `title: {title} \| text: {text}`; `classification`, `clustering` and `sentence similarity` put their task prefix on every input, as the model card says |
| `title`      | a non-empty string, or an array with a string or `null` for each input                                                          | the document's title, or its file name for code; `none` without one. Only with `input_type: "document"`                                                                                                             |

Without these fields, text is embedded unchanged, which still works but loses some precision. Other values answer `400`. `task: "search result"` with `input_type` `query` and `document` gives sentence-transformers' `Retrieval-query` and `Retrieval-document` prompts, and `code retrieval` its `CodeRetrieval`.

`/v1/models` lists the model with `embeddings` in `architecture.output_modalities`, a `context_length` of 8192 and `ortllama.dimensions`. Only the text graph, `onnx/model{suffix}.onnx` with its `.onnx_data`, is loaded and fetched; the export's vision and audio encoders are neither needed nor read. Variants are its quantizations. The default is `q8` when it is cached, else the first cached of `q4`, `fp16`, `q4f16` and `fp32`. Measured on an M4 Max, against sentence-transformers on the PyTorch model in fp32, over the 12 fixture texts at 768, 256 and 128 dimensions, with the time for a query, for 32 documents (1,575 tokens) in one request and for one 7,009-token document, warm:

| Variant        | Download | Cosine, worst / mean | Host CPU             | Host WebGPU          | onnxruntime-web WebGPU (Dawn) | onnxruntime-web WASM                |
| -------------- | -------- | -------------------- | -------------------- | -------------------- | ----------------------------- | ----------------------------------- |
| `q8` (default) | 314 MB   | 0.99991 / 0.99996    | 30 ms, 0.42 s, 4.7 s | 16 ms, 0.14 s, 2.4 s | 25 ms, 0.17 s, 2.3 s          | does not load                       |
| `q4`           | 175 MB   | 0.97801 / 0.99027    | 17 ms, 0.78 s, 6.4 s | 19 ms, 0.15 s, 2.5 s | 24 ms, 0.18 s, 2.3 s          | does not load                       |
| `q4f16`        | 157 MB   | 0.97791 / 0.99025    | 20 ms, 0.97 s, 11 s  | 17 ms, 0.10 s, 1.6 s | 24 ms, 0.13 s, 1.5 s          | does not load                       |
| `fp16`         | 543 MB   | 0.99993 / 0.99997    | 19 ms, 0.86 s, 10 s  | 20 ms, 0.09 s, 1.6 s | 25 ms, 0.13 s, 1.5 s          | 117 ms, 11 s, 17 s for 1,759 tokens |
| `fp32`         | 1,085 MB | 1.00000 / 1.00000    | 11 ms, 0.39 s, 4.7 s | 20 ms, 0.13 s, 2.4 s | 21 ms, 0.14 s, 2.3 s          | 110 ms, 10 s, 16 s for 1,759 tokens |

Plus the 32 MB `tokenizer.json`. Every variant ranked every fixture query's documents as the reference did, at every size, and the Red Planet query finds Mars first. `q8` is the default rather than the `q4` that Google recommends for the browser: both run their matrix products as 4- or 8-bit `MatMulNBits` on the GPU, so `q8` is as fast on WebGPU and faster on the host CPU for longer inputs, while its vectors stay within a cosine of 0.9999 of PyTorch's where `q4`'s drift to 0.978 (the model card reports 0.988 for text), for 139 MB more. `q4f16` and `fp16` are the fastest on WebGPU for long inputs. A hint picks another default, such as `{ "hints": { "onnx-community/embeddinggemma-2-ONNX": { "variant": "q4" } } }`. Vectors from different variants are close but not the same, so an index should be built and queried with one variant.

onnxruntime-web's WASM backend, which runs in slicc-kernel without WebGPU and on Node [without onnxruntime-node](#without-onnxruntime-node), has no kernel for the quantized token embedding (`GatherBlockQuantized`) that `q8`, `q4` and `q4f16` start with, so these variants answer `422` with code `model_unsupported` there, naming `fp16` and `fp32` when they are cached. Those two run, at about 5 ms a token on one thread. The onnxruntime-web WebGPU column is onnxruntime-web 1.30 on Node with the `webgpu` package (Dawn), the kernel's runtime without the browser; the kernel itself was not measured. Loading takes 0.5–1 s on the host and 0.9–1.5 s on onnxruntime-web, most of it checking the files against their ETags.

### Transcription

`POST /v1/audio/transcriptions` takes a multipart `file`, `model`, `language`, `prompt` and `response_format` (`json`, `verbose_json`, `text`, `srt`, `vtt`). Audio is WAV, mp3, FLAC, M4A or MP4 (also fragmented) with AAC, ADTS `.aac`, Ogg Opus or Vorbis, WebM Opus, or 16-bit `audio/pcm` (`;rate=N`, default 24 kHz). The format comes from the file's first bytes, else its content type or file name, and is decoded the same way on both tracks: Opus on libopus in WebAssembly, mp3 (MPEG-1, 2 and 2.5 Layer III, also free format), FLAC and Vorbis in JavaScript. A leading ID3 tag is skipped before the format is read. ortllama reads the MP4 and ADTS containers itself but ships no AAC codec, so AAC (AAC-LC, the format of iPhone voice memos and most phone recordings, and HE-AAC at its full rate) is decoded by the browser's WebCodecs or an optional peer, as [AAC](#aac) explains. MP4 edit lists and iTunes gapless data trim the encoder delay, and Vorbis ends at the last granule position. The first audio track of an MP4 is used, and every format is mixed down to mono. Audio longer than 30 minutes, or more samples than 30 minutes at 48 kHz, FLAC above 192 kHz, a stream that changes its sample rate, other MP4 codecs (ALAC, Opus, mp3), AAC with no codec to decode it or that the codec refuses, ADTS frames with several raw data blocks, Vorbis floor 0, MPEG audio Layer I and II, and other formats, answer `400`.

It answers `{ text, language, duration, segments, usage: { type: "duration", seconds } }`. Audio longer than 30 seconds is decoded in windows that follow Whisper's timestamps; the language is detected when the request names none. [Parakeet TDT](#engines) models cut it at the quietest 20 ms of the window's last 10 seconds instead, and transcribe English only.

#### AAC

AAC input and `aac` output use the first of these that can do it. Which ones are there is settled once per process; WebCodecs is still asked about each stream's configuration, and a stream it refuses goes to the peer.
1. **WebCodecs**, `AudioDecoder` and `AudioEncoder` with `mp4a.40.2`, when `isConfigSupported` says yes: slicc-kernel in Chrome. Chrome and Chrome for Testing decode AAC-LC and HE-AAC on every platform; Chromium builds without Google's codecs don't. Chrome encodes AAC only on macOS, Windows and Android, through the system's encoder, and only at 44.1 or 48 kHz (on Windows also only at 96 to 192 kbit/s), so 24 kHz speech is resampled to 48 kHz first. Chrome on Linux has no AAC encoder.
2. **An optional peer**, found like the [phonemizers](#phonemizers): next to ortllama, in pnpm's global directory (`pnpm add -g` in slicc-kernel) or in `ORTLLAMA_MODULE_PATH`. ortllama keeps only a few lines of glue for each.

| Peer                                                                                                                                           | Does     | Get it                                                              | Licence                                                                                                        |                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [`@audio/decode-aac`](https://github.com/audiojs/decode/tree/master/packages/decode-aac)                                                       | decoding | `npm install @audio/decode-aac` (0.4 MB)                            | GPL-2.0, for the FAAD2 decoder it runs in WebAssembly                                                          | the only decoding peer; host Node and jsh                                                      |
| [`@audio/encode-aac`](https://github.com/audiojs/encode/tree/main/packages/encode-aac)                                                         | encoding | `npm install @audio/encode-aac` (0.5 MB)                            | MIT, and the Fraunhofer FDK AAC licence for the encoder it runs in WebAssembly, which grants no patent licence | tried first: the fox at 64 kbit/s comes back at 36.5 dB SNR; host Node and jsh                 |
| [`mediabunny`](https://github.com/Vanilagy/mediabunny) with [`@mediabunny/aac-encoder`](https://www.npmjs.com/package/@mediabunny/aac-encoder) | encoding | `npm install mediabunny @mediabunny/aac-encoder` (11 MB and 4.6 MB) | MPL-2.0; the encoder is FFmpeg's, in WebAssembly                                                               | tried second: 20.8 dB; host Node only, since `@mediabunny/aac-encoder` does not load under jsh |
3. **Neither**: AAC, M4A and MP4 input and `aac` output answer `400` with a message that names these packages. WAV, mp3, FLAC, Ogg and WebM need none of them.

The decoders' own delays are taken out, so the edit lists and gapless data of an MP4 trim the same way on each. mediabunny has no AAC decoder of its own: it decodes through WebCodecs, which ortllama already asks first. [slicc-media](https://github.com/ai-ecoverse/slicc-media) offers the `media` and `mediaprobe` commands but no library API, so ortllama calls WebCodecs directly.

### Realtime transcription

`GET /v1/realtime?intent=transcription` is OpenAI's realtime transcription over WebSocket (subprotocol `realtime`).

| Direction | Events                                                                                                                                                                                                                                                                                              |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| in        | `session.update`, `input_audio_buffer.append`, `input_audio_buffer.commit`, `input_audio_buffer.clear`                                                                                                                                                                                              |
| out       | `session.created`, `session.updated`, `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped`, `input_audio_buffer.committed`, `input_audio_buffer.cleared`, `conversation.item.input_audio_transcription.delta`, `conversation.item.input_audio_transcription.completed`, `error` |

Audio is `audio/pcm` at 8–48 kHz (24 kHz by default). Turns are detected by `server_vad` on Silero VAD, or committed manually with `turn_detection: null`.
- **Decoding.** A session re-decodes the growing turn about every 500 ms of audio and sends the words two decodes agree on as deltas, then the full transcript when the turn ends: on `silence_duration_ms` of silence (500 ms by default), when no audio has arrived for that long, on `commit`, after 30 seconds of speech, or when the socket closes. `clear` drops transcripts that are still being decoded.
- **Limits.** `threshold` is kept between 0.2 and 0.95, `prefix_padding_ms` between 0 and 2000 and `silence_duration_ms` between 100 and 10000. A socket that sends nothing for 60 seconds is closed.
- **Security.** The WebSocket takes the API key as `Authorization: Bearer` or as the `openai-insecure-api-key.<key>` subprotocol that browsers use, and answers only loopback, `*.localhost` and `ORTLLAMA_ORIGINS` origins.
- **Models.** `realtime-transcribe` is Whisper; a session's `transcription.model` may name any transcription model, such as a Parakeet TDT export, whose final transcript follows the end of speech after 0.1 s on the host CPU.
- **Memory.** The transcription model and the one Silero VAD session that all sockets share stay loaded while a session is open, so open realtime sessions can push the number of loaded models past `ORTLLAMA_MAX_LOADED`.

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

#### Voices

`GET /v1/audio/voices` lists the voices the cached speech models can speak, as `{ "object": "list", "data": [{ "id", "model", "engine", "language", "gender" }] }`, from the cache alone: nothing is loaded to answer. It takes the same API key, Host and CORS checks as the other routes.
- **Kokoro.** The `voices/*.bin` in the cache that the speech route accepts, so no Japanese or Chinese. The prefix gives the language (`a` en-US, `b` en-GB, `e` es-ES, `f` fr-FR, `h` hi-IN, `i` it-IT, `p` pt-BR) and the second letter the gender (`f` female, `m` male).
- **Phonon.** The voices that `config.json`'s `onnx` section lists and that have a `voices/<name>.bin`. The language is the voice's locale (`en-US`, `en-GB`), or the export's first language (`en`) for the character voices, and the gender is the export's. A voice the export gives no locale or gender takes them from [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices)' `voices.jsonl` when that is in the cache, else `null`.
- **Models.** Without `model`, every cached speech model is listed once under its ID, along with any role alias that points at a speech model the list does not have (such as a pinned revision), listed under the alias. With `?model=`, that model is listed under the name asked for, and a model that is not cached, incomplete or not a speech model answers `404` or `400` as `POST /v1/audio/speech` does.

```sh
curl -s 'localhost:11435/v1/audio/voices?model=speech'
```

#### Phonemizers

Kokoro needs a phonemizer, and ortllama ships none. npm phonemizers are imported at runtime wherever ortllama can find them, the misaki lexicon comes from the Hugging Face cache, and speech answers `501` with `code: "engine_unavailable"`, naming what to install, until one is there.

**The recommended setup is misaki plus `phonemize`:** it has no GPL code, and in listening tests it was the best of the combinations. It's not perfect, but usable, and English only. Add `espeak-ng` for the Spanish, French, Hindi, Italian and Portuguese voices.

| Phonemizer                                                | Get it                                                               | Languages        | Licence                                                       |                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------- | -------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`misaki`](https://github.com/hexgrad/misaki)             | `hf download hexgrad/misaki --repo-type dataset` (6–7 MB per accent) | English          | Apache-2.0                                                    | the lexicons of the G2P Kokoro was trained with; words it does not know, and verbs whose reading differs from the noun's (`lives`, `record`, `tear`), go with their sentence to the next English phonemizer, and when there is none, numbers, ordinals and symbols are read out and unknown words spelled |
| [`phonemize`](https://github.com/hans00/phonemize)        | `npm install phonemize` (3 MB)                                       | English          | MIT                                                           | pure JavaScript rules and lexicon; British vowels are mapped onto American output, so words like `bath` keep the short vowel                                                                                                                                                                              |
| [`espeak-ng`](https://github.com/ianmarmour/espeak-ng.js) | `npm install espeak-ng` (18 MB)                                      | all of the above | GPL-3.0-or-later                                              | optional; the only one for Spanish, French, Hindi, Italian and Portuguese                                                                                                                                                                                                                                 |
| [`phonemizer`](https://github.com/xenova/phonemizer.js)   | `npm install phonemizer` (2.6 MB)                                    | English          | labelled Apache-2.0, but bundles espeak-ng (GPL-3.0-or-later) | the same phonemes as `espeak-ng`                                                                                                                                                                                                                                                                          |

Phonemizers are tried in the table's order, and the first that knows a voice's language speaks it, so English stays with misaki and `phonemize` even when `espeak-ng` is installed. `ORTLLAMA_PHONEMIZERS` (such as `phonemize,espeak-ng`) changes the order or leaves some out. A phonemizer that fails while speaking is skipped for a minute, and the next one speaks. Other heteronyms, such as past-tense `read` or `bass`, take the lexicon's usual reading until there is a part-of-speech tagger.

#### Phonon

[Phonon 31m English](https://huggingface.co/gradium/phonon-31m-en) (Gradium, Apache-2.0, built on Kyutai's Pocket TTS and the Mimi codec) speaks through the `phonon` engine without a phonemizer. A flow-matching transformer turns the tokens into one Mimi latent every 80 ms, and the Mimi decoder turns those into 24 kHz audio. The engine reads an ONNX export laid out as:
- `config.json`, the checkpoint's own, with an `onnx` section that lists the voices and the decoder's state,
- `tokenizer.json`,
- `onnx/text_conditioner.onnx`, `onnx/flow_lm_main.onnx` (KV cache in and out), `onnx/flow_lm_flow.onnx` and `onnx/mimi_decoder.onnx` (streaming state in and out), with `_quantized`, `_q4` and `_fp16` variants,
- `voices/<name>.bin`, voice prompts already projected for the flow LM.

The [models](https://github.com/ai-ecoverse/models) repo's `recipes/phonon-31m-en` publishes it as [ai-ecoverse/phonon-31m-en-ONNX](https://huggingface.co/ai-ecoverse/phonon-31m-en-ONNX). Since revision `1305012`, its `flow_lm_flow` and `text_conditioner` take a `duration_delta` input (float32, shape `[1]`, default 0 in `config.json`'s `onnx.inputs`). The engine feeds it to both graphs whenever they declare it, 0 at speed 1.
- **Text.** Gradium's runtime, `ptts --lang en`, normalizes English before it tokenizes it. ortllama does the same in JavaScript (`src/phonon/`) and gets the same token IDs on 361 test inputs: numbers, currency, dates, emails, URLs, abbreviations, symbols, chunking and fitting. Numbers stay as digits unless they have scale words, as in ptts: `$45.99` reads "45 dollars and 99 cents", `1,234` reads "1 thousand 234", and `https://docs.example.org` is spelled out piece by piece.
- **Voices.** The 37 English voices of [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices), named in lowercase with dashes: `easton` (the default), `freya`, `zoey`, `elodie-rose`, `captain-silas-blackwater` and so on. Any capitalization works.
- **Sampling.** Temperature 0.3, as in ptts. The noise comes from a seeded generator, so a request always sounds the same. It is not ptts's ChaCha12 stream, though, so the same seed does not give ptts's audio. Given the same noise, the export's latents and audio match ptts's.
- **Chunks.** Phonon gets the whole input and cuts it as ptts does: at ptts's sentence ends, into chunks of up to 50 tokens. Each chunk starts again from the voice prompt. One noise stream runs through the request, as ptts's random generator does.
- **Streaming.** A chunk is sent frame by frame: the Mimi decoder turns the first latent into audio on its own, then groups of 2, 4, 8 and 16, which costs about 3 % in speed against decoding 16 at a time. `wav` and `pcm` send each group as it is decoded; `mp3`, `opus`, `flac` and `aac` send whatever whole frames of their own it completes. The next frame is generated once the last group has been written, as between Kokoro's sentences.
- **Speed.** `speed` sets the checkpoint's `duration_delta` conditioner to −2 ln(`speed`), held between −0.6 and 0.5, where negative speaks faster, as with ptts's `--condition duration_delta`. The published export takes it from revision `1305012`. The conditioner only reaches about 0.85× to 1.25× the normal rate: over six sentences, `speed` 1.25 speaks 1.23 times as fast and 0.85 0.89 times, and 1.35 to 4 and 0.78 to 0.25 stay at the ends. Past −0.6 it speaks slower again, which is why the value is held. `speed` 1 sounds as before. Whisper small gets 1 of 39 words wrong at 1 and at 0.25, and 2 at 4 ("100" for "a hundred"). An export without the input, such as revisions before `1305012`, answers `400` for any `speed` but 1; values outside 0.25 to 4 always do.
- **Variants.** `fp32` is the default. `q8` is 8-bit `MatMulNBits` on the flow LM only, as ptts quantizes it, and keeps the Mimi decoder in fp32. `fp16` is the whole graph in half precision. `q4` sounds noticeably worse and is there for comparison.

Speed for four sentences (20 s of audio) on an M4 Max, as real-time factor and time until the first chunk was ready before frame streaming. With it, `fp32`'s first audio reaches a client after 16 ms on the host CPU (was 165 ms) and after 185 ms in the kernel on WASM (was 830 ms for `pcm`, and 2.3 s, the whole response, for `opus`). Host numbers use 4 threads. Kernel numbers are `ortllama check`'s two sentences (4.7 s), on single-threaded WASM; Chrome's page process peaks at 1 GiB. For comparison, ptts itself (f32, 4 threads, streaming each frame) has a real-time factor of 0.023 and its first audio after 16 ms, and Kokoro fp32 on the host CPU 0.17 and 785 ms:

| Variant | Host CPU      | Host WebGPU  | Kernel WASM | Kernel WebGPU |
| ------- | ------------- | ------------ | ----------- | ------------- |
| `fp32`  | 0.044, 169 ms | 0.12, 469 ms | 0.24        | 0.23          |
| `q8`    | 0.035, 130 ms | 0.11, 431 ms | 0.55        | 0.18          |
| `fp16`  | 0.057, 224 ms | 0.13, 452 ms | 0.27        | 0.22          |

Phonon loads on the CPU on the host and on WASM in the kernel. Each 80 ms frame takes three small graph runs, so per-run overhead decides the speed. On the host, WebGPU pays the most of it; in the kernel, WebGPU is about as fast as WASM, and only `q8` gains from it. Set `ORTLLAMA_EP` to use WebGPU anyway. Whisper large-v3-turbo hears 20 test sentences with a word error rate of 7.4 % from Phonon `fp32`, 6.5 % from `q8` and 7.4 % from ptts itself. It hears 5.3 % from Kokoro's `af_heart`. Every difference in Phonon's transcripts is the way a number or symbol is written, such as "9" for "nine", and none is a misread word.

### pi-audio

`@ai-ecoverse/slicc-ortllama/pi-audio` builds [pi-audio](https://github.com/ai-ecoverse/pi-audio) voice models for ortllama on pi-audio's OpenAI APIs, without loading the server or pi-audio:

| Function                                      | Returns                                                  | For            |
| --------------------------------------------- | -------------------------------------------------------- | -------------- |
| `transcriptionModel({ baseUrl, id })`         | an `openai-transcriptions` model                         | `transcribe()` |
| `realtimeTranscriptionModel({ baseUrl, id })` | an `openai-realtime-transcription` model with server VAD | `listen()`     |
| `speechModel({ baseUrl, id, voices })`        | an `openai-speech` model                                 | `speak()`      |
| `voiceModels(baseUrl)`                        | all three                                                |                |

`baseUrl` defaults to `http://127.0.0.1:11435/v1`.

```js
import { speak } from '@ai-ecoverse/pi-audio';
import { registerOpenAI } from '@ai-ecoverse/pi-audio/openai';
import { speechModel } from '@ai-ecoverse/slicc-ortllama/pi-audio';

registerOpenAI();
const model = speechModel();
for await (const event of speak(
  model,
  { text: 'Hello.', voice: 'af_heart' },
  { apiKey: 'ortllama' }
))
  if (event.type === 'audio') play(event.chunk);
```

## Models

ortllama downloads nothing. It reads the Hugging Face cache that [`hf download`](https://huggingface.co/docs/huggingface_hub/guides/cli) fills (`HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`), through its `refs/` and `snapshots/`. A model ID is a Hub ID: `org/repo[/subfolder][@revision][:variant]`, such as `ai-ecoverse/kev.js/kev-0.8b:q8f32`. A model that isn't cached, or a cached variant with missing files, is answered with `404` and the `hf download` command that fetches it. Missing files include the external data files a decoder's, Whisper's, Parakeet's or embedding model's `.onnx` graphs name, read from the graphs themselves up to 16 MB each (graphs that keep weights outside are a few hundred KB; larger ones hold their weights inline), so an interrupted download is caught before the model loads, and the `hf download` command fetches them even when they are not named after their graph; a variant that isn't cached at all is answered with `404` and the variants that are, and a cached repo with no ONNX model ortllama can run, such as an MLX or PyTorch checkpoint, with `404` and no download command. When such a repo, subfolder or `ORTLLAMA_MODEL_PATH` folder does have `.onnx` files at its top, in `onnx/`, or in a subfolder that is not itself a model ortllama runs (or that subfolder's `onnx/`), such as a new export with a quantization suffix no engine knows (`onnx/decoder_model_merged_int4.onnx`) or a lone `model.onnx`, the `404` names up to three of them and points at the layouts under [Engines](#engines). `ortllama ls` notes such repos and `ORTLLAMA_MODEL_PATH` folders on stderr, but not the subfolders of one ortllama runs.

huggingface_hub 2.x (hf-xet) keeps large files in a shared store, `<cache>/blobs/<2 hex>/<64 hex>`, and links the repo's `blobs/<etag>` there. ortllama reads a snapshot's files only from the snapshot, its repo's `blobs/` and that store, following symlinks; a link anywhere else, or to a store file not named like that, counts as missing. onnxruntime-node refuses external data outside its graph's real folder, and in the shared store a graph and its data sit in different folders. On a host, ortllama then hard links the checked store files into a fresh folder under the system temp folder (`TMPDIR`), with graph and data side by side, opens the session from there and removes the folder once the session exists, so ONNX Runtime reads the weights from disk as with 1.x caches and local folders, at the same memory cost. kev.js and jev-omni.js models get paths into such a folder too. Hard links need the temp folder on the same volume as the cache; where linking fails, ortllama reads the data into memory and hands it to the session, which costs about three times the weights in extra memory.

Before a model loads, ortllama checks each of its cached files, including the external data its `.onnx` graphs name and a cached fallback vision encoder, against the hub's ETag, which `hf download` names its blob after: the SHA-256 for LFS files and the git blob SHA-1 for the others, since a stalled download can leave a damaged blob under its final name without an error. A file that does not match is refused with `500`, code `model_corrupt`, its name, both hashes and the `hf download --force-download` command that fetches it again. Each blob is checked once per process while its size and modification time stay the same. On a host it is streamed through `node:crypto`; in the kernel, where jsh reads a whole file for each ranged read, it is read once and hashed with WebCrypto. On an M4 Max, the host check takes about 1.2 s for the 3.3 GB Ministral 3 q4f16 files and 4.4 s for `kev-9b`'s 8.9 GB. `ORTLLAMA_MODEL_PATH` folders have no ETags and are not checked.

### Roles

Clients can ask for a role instead of a model. `ortllama alias decisions <model>` changes what a role means, so swapping a model changes no client.

| Role                                | Default                                                                                                                                                                                         |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chat`                              | [`mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX)                                                                       |
| `chat-reasoning`                    | [`ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX:q4f16`](https://huggingface.co/ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX), Mistral's Reasoning checkpoint in the Instruct export's layout |
| `decisions`                         | [`ai-ecoverse/kev.js/kev-0.8b:q8f32`](https://huggingface.co/ai-ecoverse/kev.js)                                                                                                                |
| `transcribe`, `realtime-transcribe` | [`onnx-community/whisper-tiny`](https://huggingface.co/onnx-community/whisper-tiny)                                                                                                             |
| `vad`                               | [`onnx-community/silero-vad`](https://huggingface.co/onnx-community/silero-vad)                                                                                                                 |
| `speech`                            | [`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)                                                                                             |
| `embed`                             | [`onnx-community/embeddinggemma-2-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX), `q8` when it is cached                                                                   |

The defaults that the [quick start](#quick-start) and [speech](#speech) examples do not fetch:

```sh
hf download ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX --include '*.json' chat_template.jinja 'onnx/*_q4f16.onnx*' 'onnx/vision_encoder_fp16.onnx*'
hf download onnx-community/silero-vad --include onnx/model.onnx
hf download onnx-community/embeddinggemma-2-ONNX --include '*.json' 'onnx/model_quantized.onnx*'
```

### Engines

Engines are picked by the files a model has, not by its name. A model is listed only with the variants whose ONNX graphs are in the cache (both for Whisper and Parakeet TDT), and a `variant` hint in `config.json` that names another one falls back to the engine's default, so MLX, PyTorch and safetensors checkpoints that share a config with an ONNX export, such as `mlx-community/whisper-large-v3-turbo`, stay out of `ortllama ls` and `/v1/models`. Each engine in `src/engines/` gets ONNX Runtime from the caller and provides `detect`, `missing`, `include`, `load`, `checkOverrides` and `check`.
- **Decoder.** Runs the transformers.js layout (`onnx/decoder_model_merged*.onnx` with `embed_tokens*.onnx`, or `onnx/model*.onnx`; variants are its quantizations, such as `q4f16`) and the onnxruntime-genai layout (`genai_config.json`, one variant per folder), with the model's own chat template and, when the export has one, its `vision_encoder*.onnx`. A small profile per family, chosen from the template, parses tool calls and reasoning; Mistral's `[TOOL_CALLS]name[ARGS]{…}` and `[THINK]…[/THINK]` are built in, and control-token names in client text are escaped so they cannot inject one. Ministral 3 models known to work, all with image input, at `q4f16` with the `fp16` vision encoder, with the load time and then prefill / decode tokens per second for a 1,550-token prompt and 128 new tokens on an M4 Max:

| Model                                                               | Download | Host CPU        | Host WebGPU      | Kernel WebGPU   |
| ------------------------------------------------------------------- | -------- | --------------- | ---------------- | --------------- |
| `mistralai/Ministral-3-3B-Instruct-2512-ONNX` (`chat`)              | 3.3 GB   | 1.0 s, 179 / 15 | 5.0 s, 926 / 75  | 2.4 s, 782 / 62 |
| `ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX` (`chat-reasoning`) | 3.3 GB   | 1.0 s, 187 / 18 | 4.8 s, 1016 / 84 | 2.7 s, 856 / 74 |
| `ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX`                     | 6.5 GB   | 2.2 s, 80 / 7.3 | 5.6 s, 498 / 40  | 4.7 s, 422 / 39 |
| `ai-ecoverse/Ministral-3-8B-Reasoning-2512-ONNX`                    | 6.5 GB   | 2.1 s, 91 / 11  | 5.1 s, 543 / 48  | 7.3 s, 458 / 46 |
| `ai-ecoverse/Ministral-3-14B-Instruct-2512-ONNX`                    | 9.5 GB   | 3.2 s, 55 / 6.0 | 6.2 s, 336 / 31  | 8.0 s, 283 / 30 |
| `ai-ecoverse/Ministral-3-14B-Reasoning-2512-ONNX`                   | 9.5 GB   | 2.7 s, 62 / 9.1 | 4.6 s, 361 / 37  | 7.8 s, 303 / 34 |

The 8B and 14B exports are ai-ecoverse's, made by the [models](https://github.com/ai-ecoverse/models) recipes from onnxruntime-genai's builder (the decoder) and Mistral's 3B vision and embedding graphs, and pick PyTorch's next token 93–97 % of the time (Mistral's 3B: 95 %). Their decoder takes no `position_ids` and leaves the KV head size symbolic; ortllama feeds `position_ids` only to graphs that ask for them and sizes the cache from `config.json`. Fetch one with `hf download ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX --include '*.json' chat_template.jinja 'onnx/decoder_model_merged_q4f16.onnx*' 'onnx/embed_tokens_fp16.onnx*' 'onnx/vision_encoder_fp16.onnx*'` and make it a role's model with `ortllama alias chat ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX`. All of them load in slicc-kernel; the 14B models peak at about 11 GiB in the page and 12 GiB in Chrome's GPU process. Mistral's 3B Instruct graphs never apply Ministral 3's query scaling, which starts at position 16,384 (they divide by 262,144 instead of 16,384, which the `fp16` graph rounds to infinity), so past 16k tokens it drifts from PyTorch; ai-ecoverse's 3B Reasoning export scales queries as transformers does since revision `c592447`, like the 8B and 14B. Both 3B models still recalled every fact asked for after 23,000 tokens of code.
- **Whisper.** Runs the transformers.js ONNX exports (`config.json` with `model_type: "whisper"`, `onnx/encoder_model*.onnx` and `onnx/decoder_model_merged*.onnx`, external `.onnx_data` included). Variants follow the file suffixes: `fp32`, `fp16`, `q8` (`_quantized`), `int8`, `uint8`, `q4`, `q4f16`, `bnb4`; the default is the first complete one in that order. Mel bins, special tokens, languages and the timestamp grid come from the model's configs and tokenizer, so large-v3 models (128 mel bins, the extra `yue` token) work as well. Models known to work, with the mean word error rate on the `check` clips and real-time factors for a 2.7-second and a long clip (44 s on the host, 32 s in the kernel) on an M4 Max:

| Model                                      | Download | WER   | Host CPU     | Host WebGPU  | Kernel WebGPU |
| ------------------------------------------ | -------- | ----- | ------------ | ------------ | ------------- |
| `onnx-community/whisper-tiny` (default)    | 151 MB   | 0.092 | 0.06 / 0.013 | 0.13 / 0.077 | 0.09 / 0.061  |
| `onnx-community/whisper-base`              | 291 MB   | 0.046 | 0.09 / 0.020 | 0.11 / 0.096 | 0.13 / 0.096  |
| `onnx-community/whisper-large-v3-turbo:q4` | 759 MB   | 0.005 | 1.3 / 0.20   | 0.27 / 0.070 | 0.31 / 0.118  |

`whisper-tiny` stays the default: it is the smallest, and the fastest on the host CPU, which is where Whisper runs best on host Node. `whisper-large-v3-turbo` makes almost no mistakes, in French and Spanish too, but wants WebGPU, where its 32-layer encoder runs five times faster than on the CPU, so it loads on the host's default order rather than Whisper's `cpu`; fetch it with `hf download onnx-community/whisper-large-v3-turbo --include '*.json' onnx/encoder_model_q4.onnx onnx/decoder_model_merged_q4.onnx`. `whisper-base` sits in between. In a realtime session the final transcript follows the end of speech after 0.2 s with `whisper-tiny` on the CPU, 0.4 s with `whisper-base` on WebGPU, and 1 s with `whisper-large-v3-turbo` on WebGPU (5 s on the CPU).
- **Parakeet TDT.** Runs NVIDIA Parakeet TDT exports split into an encoder and one greedy decoding step (`config.json` with `model_type: "parakeet_tdt"`, and a variant wherever both `onnx/encoder_model{suffix}.onnx` and `onnx/decoder_joint_model{suffix}.onnx` are there, with Whisper's suffixes). The default is `q8` when it is complete, else the first complete one in Whisper's order. The log-mel features are computed in JavaScript as transformers' `ParakeetFeatureExtractor` does, with the leading silence `preprocessor_config.json` asks for, and match the export's NumPy reference to 3e-7, with 99.997 % of the values and all of the filters bit for bit (2 ms per second of audio on the host). Decoding is transformers' greedy TDT loop: each step predicts a token and how many 80 ms encoder frames to skip, so a token's start and end come from the frames it was emitted on, and segments end at a `.`, `?` or `!` or a pause of 0.5 s. The language is always English: a request for another answers `400`, and `prompt` is ignored. The decoding step is a 70 MB graph run once per token or skipped frame, and it runs on `cpu` (`wasm` in the browser) whatever the encoder's provider, since a step takes 0.4 ms there and 2.3 ms on WebGPU. Audio over 30 seconds is cut at its quietest 20 ms in the window's last 10 seconds. A variant's files and `NOTICE` are fetched with `hf download ai-ecoverse/Underdog-Bark-3.0-ONNX --include '*.json' NOTICE 'onnx/encoder_model_quantized.onnx*' 'onnx/decoder_joint_model_quantized.onnx'`.

[Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0), Conway Research's English fine-tune of NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and Moondream's [parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra), exported by the models recipe `underdog-bark-3.0` as `ai-ecoverse/Underdog-Bark-3.0-ONNX`, is the model known to work. It is licensed CC-BY-4.0, credited to Conway Research (Underdog Bark 3.0), NVIDIA (parakeet-tdt-0.6b-v3) and Moondream (parakeet-ultra), and its `NOTICE` comes with every variant. On the export's 20 parity clips (macOS `say`), `fp32` takes every step, token and duration, that its own ONNX Runtime reference and transformers' PyTorch model took, and `fp16` emits the same tokens. Only `fp32`'s steps are the same on every CPU. `q8`, quantized to 7 bits per channel since revision `31f3aba` because full-range int8 broke on x86 CPUs without VNNI, and `q4f16` have word error rates of 1.7 % and 2.1 % against PyTorch on an M4 Max (1.4 % and 2.1 % on the reference's x86 runner). With the mean word error rate on the `check` clips, the load time, real-time factors for the 2.7-second fox and the 32-second story, and the peak memory footprint on an M4 Max (four CPU threads):

| Variant        | Download | WER   | Host CPU                     | Host WebGPU                  | Kernel WebGPU                                                   |
| -------------- | -------- | ----- | ---------------------------- | ---------------------------- | --------------------------------------------------------------- |
| `q8` (default) | 912 MB   | 0.009 | 1.2 s, 0.024 / 0.017, 1.1 GB | 1.2 s, 0.045 / 0.023, 2.2 GB | 1.7 s, 0.36 / 0.28, page 3.6 GiB, GPU 0.9 GiB, before `31f3aba` |
| `q4f16`        | 524 MB   | 0.009 | 1.1 s, 0.064 / 0.048, 1.1 GB | 1.3 s, 0.022 / 0.010, 1.2 GB | 1.4 s, 0.025 / 0.013, page 1.4 GiB, GPU 0.8 GiB                 |
| `fp16`         | 1,291 MB | 0.009 | 1.9 s, 0.046 / 0.030, 2.7 GB | 3.0 s, 0.024 / 0.010, 2.0 GB | 2.0 s, 0.028 / 0.013, page 5.0 GiB, GPU 2.3 GiB                 |
| `fp32`         | 2,509 MB | 0.009 | 1.8 s, 0.032 / 0.018, 2.7 GB | 8.3 s, 0.026 / 0.011, 3.5 GB | does not load                                                   |

The memory footprints hold for huggingface_hub 2.x's shared blob store too, as long as the temp folder is on the cache's volume; when ortllama has to read the external data into memory, `q8` peaks at 3.4 GB and `q4f16` at 2.3 GB on the host CPU. `q8` is fastest on the CPU, so it stays there on Node unless `ORTLLAMA_EP` says otherwise (the Host WebGPU column was measured with `ORTLLAMA_EP=webgpu`). On WebGPU its integer operators run in WebAssembly, which in the kernel, on one thread, makes it twenty times slower than `q4f16`: there, pick `q4f16` with a hint such as `{ "hints": { "ai-ecoverse/Underdog-Bark-3.0-ONNX": { "variant": "q4f16" } } }`. `fp32`'s 2.4 GB weights file is more than the kernel reads at once. `ortllama alias transcribe ai-ecoverse/Underdog-Bark-3.0-ONNX` makes it the transcription role's model; on the two English `check` clips it gets 2 of 119 words wrong, as against 4 for `whisper-tiny`.
- **Embedding.** Runs EmbeddingGemma 2 exports in the transformers.js layout (`config.json` with `model_type: "embedding_gemma2"` or `architectures: ["EmbeddingGemma2Model"]`, `tokenizer.json`, and `onnx/model{suffix}.onnx` with the external data `transformers.js_config.use_external_data_format` names) for [`/v1/embeddings`](#embeddings). Variants are `q8`, `q4`, `fp16`, `q4f16` and `fp32` (`_quantized`, `_q4`, `_fp16`, `_q4f16`, no suffix), the ones whose text graph is cached. The graph's `sentence_embedding` output, mean-pooled, projected to 768 dimensions and normalized inside the export, is the vector; its image, video and audio feature inputs get empty tensors, so a text-only download runs and the vision and audio encoders are never loaded. The decoder does not claim the export, whose `model*.onnx` is not a causal language model. A family is a small table in `src/engines/embedding.js` (detection, context window, dimensions, variants, prefixes, `check` thresholds), so EmbeddingGemma 1 or another text embedding export is one more entry.

### GGUF (experimental)

ortllama also runs GGUF models from the Hugging Face cache, through the released [`@wllama/wllama`](https://github.com/ngxson/wllama) 3.8.1, the WebAssembly build of llama.cpp `b11364-46ca246` (2026-10-02). It is experimental: ONNX stays the default for every role, and a GGUF model answers only when a request, or an alias you set, names it by ID, such as `mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M`.
- **Libraries.** Install `@wllama/wllama@3.8.1`, and on a host also `webgpu@0.6.2` (Dawn) for WebGPU; both are optional peers, found as [model libraries](#model-libraries) are. wllama needs WebAssembly JSPI: Node 25 or later, Node 24.22 or later, or an older Node 24 with `--experimental-wasm-jspi`; Chrome 137 or later in slicc-kernel. Without it, requests answer `501` with code `engine_unavailable`.
- **In one thread.** wllama starts its own Worker, which jsh does not allow, so ortllama runs wllama's worker code in the calling thread, with a `Worker` stand-in during the load and the model files read in ranges from the cache as wllama asks for them; nothing is copied whole. Reads of 4 MiB and more go straight to the file. Smaller ones that follow the last one, or that have already read half of their aligned 16 MiB chunk, read that whole chunk instead, and ortllama keeps the last four chunks until the load ends: slicc-kernel reads 16 MiB at a time 1.5–1.7× as fast as 1 MiB. On an M4 Max, SmolLM2 loads in the kernel in 2.6 s instead of 3.0 s on WebGPU; Ministral 3 asks for most of its weights in reads of 4 MiB and more, and loads as fast as before. Engine `gguf` serves chat; engine `gguf-decisions` serves the GGUF decision models, those with a `<architecture>.decision.type` key.
- **Devices.** The first execution provider in the [order](#execution-providers) picks the device: `webgpu` offloads every layer to WebGPU (Dawn on Node, the browser's in slicc-kernel), `cpu` or `wasm` keeps them on wllama's single-threaded CPU path. When WebGPU cannot start (no `webgpu` package, no adapter), the model loads on the CPU, with the reason logged once. Other providers, such as `cuda`, are skipped the same way. On the CPU the weights, the projector and the context window share WebAssembly's 4 GB heap; a model that does not fit answers `507` with code `out_of_memory`.
- **Detection.** ortllama reads each `*.gguf` file's header. Files without a `general.architecture`, such as Phonon's bare tensor container, are not models; `clip` projectors (`mmproj`) with a vision encoder are paired with the models in the same folder (the first of `Q8_0`, `F16`, `BF16`, `F32`). Variants are the quantizations in the file names (`Q4_K_M`, `IQ4_XS`, `UD-Q4_K_XL`), else the header's `general.file_type`; the default is the first cached of `Q4_K_M`, `Q4_K_S`, `Q4_0`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, then the rest. A split model (`-00001-of-00003.gguf`) is one variant, and a missing part answers `404` with the `hf download` command. A repo that has ONNX graphs ortllama runs stays on its ONNX engine.
- **Chat.** llama.cpp's own server code renders the model's chat template and parses tool calls and reasoning, so ortllama's family profiles do not apply; streaming, tools, `tool_choice`, `stop`, `seed` and the sampling fields pass through, and llama.cpp's slot cache plays the prefix cache. Streaming is paced as on the decoder engine: wllama generates in steps that ortllama asks for, and it asks for the next one only once the client has taken the last chunk and the runtime has had its turn, so a slow client or a more urgent job holds generation back. A GGUF model serves one chat at a time; a second request waits. Sampling fields a request leaves out take llama.cpp's defaults, and `generation_settings` reports only the fields the request set. The context window is the model's, at most 8192 tokens; a hint raises or lowers it: `{ "hints": { "mistralai/Ministral-3-3B-Instruct-2512-GGUF": { "contextLength": 16384 } } }`.
- **Images.** They are decoded and checked as for the decoder engine (the same limits, EXIF orientation, transparency dropped), sent to llama.cpp as PNG and resized by its projector; `detail` is not applied. A model whose projector is not cached lists only `text` and answers images with `400` and the `hf download` command for `*mmproj*.gguf`.
- **System One.** `gguf-decisions` models answer `/v1/systemone` through wllama's `createSystemOne`, rounded as Kev's answers are. They take no `image`. `ggml-org/Laya-GGUF` works; LiquidAI's d1 models need a llama.cpp newer than wllama 3.8.1 bundles and do not load (`422`, code `model_unsupported`, as for any architecture wllama cannot load).
- **Unloading.** wllama's `exit()` frees nothing in one thread, so ortllama destroys the WebGPU device the model asked for, drops the error handler llama.cpp's WebGPU glue leaves on it, and keeps the glue's never-settling `Atomics.waitAsync` out, so the WebAssembly memory can be collected. On host WebGPU three load, chat and unload cycles of Ministral 3 3B left the process at 1.0 GB each time (5.9 GB loaded, 0.06 GB before the first).
- **Limits.** Loading in slicc-kernel reads the file in ranges, which jsh serves at a fraction of whole-file speed ([slicc-jsh#825](https://github.com/ai-ecoverse/slicc-jsh/issues/825)). ortllama checks a GGUF blob over 2 GiB − 4 MiB against its ETag in 16 MiB ranges with `node:crypto`, because Chrome before 156 cannot hold it in one buffer. Decision models get no images yet, and the `check` contract is weaker than the decoder's (below).

Ministral 3 3B Instruct, `Q4_K_M` (2.15 GB) with Mistral's `BF16` projector (0.84 GB), next to the ONNX export above, with the load time and then prefill / decode tokens per second for the same 1,550-token prompt and 128 new tokens, on an M4 Max:

| Model                                                        | Download | Host CPU                                                                                                          | Host WebGPU         | Kernel WebGPU                                                                 |
| ------------------------------------------------------------ | -------- | ----------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------- |
| `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16` (`chat`) | 3.3 GB   | 1.0 s, 179 / 15                                                                                                   | 5.0 s, 926 / 75     | 2.4 s, 782 / 62                                                               |
| `mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M`         | 3.0 GB   | 3.9 s, `check` only (the 8192-token context does not fit the 4 GB heap next to the projector; at 4096 it answers) | 2.8 s, 258–264 / 67 | 11–15 s, 262–280 / 57–61 (the #117 spike, on a wllama build of the same glue) |
| `bartowski/SmolLM2-135M-Instruct-GGUF:Q4_0`                  | 92 MB    | 0.5 s                                                                                                             | 0.4 s               | 3.5 s (wasm 2.6 s)                                                            |
| `ggml-org/Laya-GGUF:Q8_0` (System One, three questions)      | 449 MB   | 0.6 s, 5.5 s                                                                                                      | 0.5 s, 82 ms        | 3.0–4.8 s, 63 ms (the #117 spike)                                             |

GGUF decodes as fast as ONNX on WebGPU and prefills at about a third of its speed, so long prompts and images wait longer. On host WebGPU the process peaks at 6.0 GB.

`ortllama check` against fixtures made on the host CPU (the `gguf` and `gguf-decisions` rows of [Check](#check)): Ministral 3 `Q4_K_M` on host WebGPU follows 10 greedy tokens up to the first close margin with a mean change of 0.007 nats (max 0.031), SmolLM2 9 tokens with 0.017 (0.057), and Laya's 18 System One questions move by 0.0041 on average (max 0.0164), with no answer changed. The ONNX export's decoder `check` on the same hardware lands at 0.026–0.063 nats. Measured in slicc-kernel 1.146.0 with slicc-jsh 0.53.4 in headless Chrome for Testing 146, SmolLM2 loads, answers, and returns Chrome's GPU process to its baseline after unloading; the Ministral and Laya kernel numbers above are the #117 spike's, since writing their files into the test page's storage took longer than the harness allows on the shared machine.

### Model libraries

Model-specific libraries are never installed with ortllama. They are devDependencies here, and engines load them only when a model needs them, such as `@ai-ecoverse/kev.js` and `@ai-ecoverse/jev-omni.js` for System One, `espeak-ng`, `phonemize` or `phonemizer` for Kokoro, and `@wllama/wllama` (with `webgpu` for WebGPU on Node) for GGUF models. ortllama looks for each library:
1. next to itself,
2. in `ORTLLAMA_MODULE_PATH` (project folders that contain a `node_modules`),
3. in pnpm's global installs (`$PNPM_HOME/global/…`, including slicc-kernel's `/usr/local/share/pnpm`).

A model whose library is missing is still listed, and requests for it answer `501` with code `engine_unavailable` and the package to install.

### Check

`ortllama check` tells whether a model computes on this device what it computes on the reference, a host CPU. ortllama ships fixtures in `fixtures/` for every default model and the other models below that work; `check` picks them by model ID, so `ortllama check chat` compares the Instruct model. A fixture file is `{ "fixtures": [...] }`; audio may be inline base64 (`audio`) or a WAV path relative to the file (`file`). Each engine compares what stays stable across execution providers:

| Engine                           | Fixtures                                                                                                                                                                                                                                                | Passes when                                                                                                                                                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| decoder                          | 3 prompts, the reference's greedy tokens with their log-probabilities and the margin to the runner-up                                                                                                                                                   | the tokens, fed back one by one, keep a mean log-probability within 0.15 nats, and no step with a margin over 1 nat picks another token                                                                                                                  |
| systemone                        | 12 System One requests (plus an image for vision bundles) with their raw decision probabilities                                                                                                                                                         | mean change ≤ 0.02, at most 0.25, and no clear answer flips                                                                                                                                                                                              |
| systemone, kev-ministral `:base` | 3 prompts and an image, with the reference's 24 greedy tokens each (`kev-ministral-3b.base.json`)                                                                                                                                                       | fed the reference's tokens wherever they differ, at most 10 % of the tokens change                                                                                                                                                                       |
| whisper                          | 4 clips: `fox.wav`, a 32-second English story, French and Spanish (Kokoro voices, 8 kHz 8-bit)                                                                                                                                                          | the language is detected and the word error rate stays within 0.1 of the reference's                                                                                                                                                                     |
| parakeet                         | the 2 English clips of Whisper's                                                                                                                                                                                                                        | the transcript stays within a word error rate of 0.05 of the reference's own                                                                                                                                                                             |
| kokoro                           | 3 sentences as token IDs, so the phonemizer does not matter, with a 40-band log-mel summary                                                                                                                                                             | the mel spectrogram stays within 2 dB and the length within 5 %; phonemes are compared and reported                                                                                                                                                      |
| phonon                           | 3 sentences as text and token IDs, synthesized without sampling noise, with a 40-band log-mel summary                                                                                                                                                   | the token IDs match, and the mel spectrogram stays within 2 dB and the length within 5 %                                                                                                                                                                 |
| silero                           | speech probabilities for every 32 ms frame of two clips                                                                                                                                                                                                 | mean change ≤ 0.01, at most 0.1, and no clear speech/silence flip                                                                                                                                                                                        |
| embedding                        | 12 texts (4 queries and 8 documents, 3 of them code, one German) with their prompts and sentence-transformers vectors from the PyTorch model in fp32, at 768, 256 and 128 dimensions, and the reference's ranking of the documents for 2 of the queries | the prompts and token counts are the reference's, every vector keeps a cosine to the reference of at least the variant's threshold (`fp32` 0.9999, `fp16` and `q8` 0.999, `q4` and `q4f16` 0.97), and every ranking keeps its top document at every size |
| gguf                             | the decoder's 3 prompts, with the reference's greedy tokens (as text), their log-probabilities and the margin to the runner-up among the top 5                                                                                                          | generated greedily (wllama has no teacher forcing), every token up to and including the first step whose reference margin is 1 nat or less matches, with a mean log-probability within 0.15 nats                                                         |
| gguf-decisions                   | the 12 System One requests, with llama.cpp's decision probabilities                                                                                                                                                                                     | as for systemone                                                                                                                                                                                                                                         |

The tolerances leave room over what WebGPU differs from CPU, measured on host Node and in slicc-kernel (both alike): decoder mean 0.026–0.063 nats (max 0.48, flips only below a 0.55 margin; Ministral 3 3B to 14B), System One mean 0.006 (max 0.027 for Kev, 0.051 for kev-ministral), kev-ministral's base weights 2 of 96 tokens on host WebGPU, Kokoro 0.15–0.24 dB, Phonon 0.00 dB (host WebGPU, kernel WASM and WebGPU), Whisper and Silero none. Kokoro's `q8` variant lands at 2–5 dB and its `fp16` on WebGPU, which sounds garbled, at 25 dB. Phonon's `q8` lands at 1.2–1.8 dB on kernel WASM; on the host CPU it lands at 0.8 dB but ends `museum` one frame early, which counts as 10 dB. Its `fp16` lands at 8–9 dB on kernel WebGPU. Fixtures hold the default variant; other variants can fall outside them, and `check` says so. `node scripts/fixtures.mjs <model>…` writes them on the host CPU (`node scripts/fixtures.mjs audio` makes the clips with Kokoro). The embedding fixtures come from PyTorch instead: `uv run --python 3.12 --with torch --with torchvision --with pillow --with 'sentence-transformers>=6.1.0' scripts/embedding-fixtures.py` writes them with `google/embeddinggemma-2` in fp32 on the CPU, and `npx biome format --write fixtures/onnx-community/embeddinggemma-2-ONNX.json` formats them.

## Execution providers

A model loads on one execution provider at a time, with `cpu` (`wasm` in the browser) for the operators that provider lacks. When a session cannot be created, such as CUDA without its libraries, the model loads on the next provider in the order, down to `cpu` alone, and each reason is logged once. Providers this ONNX Runtime does not have are skipped with a warning. The order is the first of:
1. `ORTLLAMA_EP`, such as `cuda,cpu`. It beats every hint because it is the operator's switch for one process: `ORTLLAMA_EP=cpu` keeps every model off the GPU, whatever `config.json` says.
2. The model's hint in `config.json`, such as `{ "hints": { "onnx-community/whisper-tiny": { "executionProviders": ["webgpu", "cpu"] } } }`. A model loaded with a hint is a separate loaded model.
3. The engine's preference: Whisper models with up to 6 encoder layers (`tiny`, `base`) and Silero VAD use `cpu` on Node, which leaves the GPU to chat and System One (small Whisper decoder steps are as fast there as on WebGPU, and Silero's frames much faster), and Silero uses `wasm` in the browser. Parakeet TDT's integer variants (`q8`, `int8`, `uint8`) use `cpu` on Node, where they run faster than on WebGPU; its decoding step always runs on `cpu` (`wasm` in the browser). Larger Whisper models, such as `large-v3-turbo`, take the host's default, since their encoder runs five times faster on WebGPU.
4. The host's default, chosen from what `onnxruntime-node` 1.30 ships there:

| Host                            | onnxruntime-node 1.30 has                                                                | Default                                               |
| ------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| macOS arm64                     | `cpu`, `webgpu`, `coreml`                                                                | `webgpu,cpu`; CoreML runs only 147 of Kev's 675 nodes |
| Linux x64                       | `cpu`, `webgpu`, `cuda` and `tensorrt` (CUDA 12 libraries fetched by its install script) | `cuda,webgpu,cpu`                                     |
| Linux arm64                     | `cpu`                                                                                    | `cpu`                                                 |
| Windows x64, arm64              | `cpu`, `webgpu`, `dml`                                                                   | `webgpu,dml,cpu`                                      |
| macOS x64                       | no binary                                                                                |                                                       |
| browser (`onnxruntime-web`)     | `webgpu`, `wasm`                                                                         | `webgpu,wasm`, or `wasm` without WebGPU               |
| Node without `onnxruntime-node` | `wasm`                                                                                   | `wasm`                                                |

`ortllama ps` and `GET /ortllama/ps` report, per model, the `backend` it runs on (`onnxruntime-node` or `onnxruntime-web`), the `executionProviders` it loaded on, where the order came from (`providerSource`: `env`, `hint`, `engine` or `default`) and `modelBytes`, the size on disk of the variant's files; and, as `memory`, the process's `rss`, `heapUsed`, `external` and `arrayBuffers` on Node (`null` in slicc-kernel, where `process.memoryUsage()` answers zeros). ONNX Runtime 1.30 reports no memory of its own: `onnxruntime-node` has no arena statistics and `onnxruntime-web` no GPU buffer usage, so ortllama shows none.

## Configuration

| Variable                 | Default                                                   |                                                                                                                                                                                                                                            |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ORTLLAMA_HOME`          | `~/.ortllama`                                             | `config.json` (aliases, per-model hints) and `run/`                                                                                                                                                                                        |
| `ORTLLAMA_PORT`          | `11435`                                                   |                                                                                                                                                                                                                                            |
| `ORTLLAMA_API_KEY`       | none                                                      | require `Authorization: Bearer <key>`                                                                                                                                                                                                      |
| `ORTLLAMA_ORIGINS`       | none                                                      | more CORS origins; loopback and `*.localhost` are always allowed                                                                                                                                                                           |
| `ORTLLAMA_HOSTS`         | none                                                      | more `Host` names to accept; loopback names, `*.localhost` and `host.slicc.internal` are always accepted, and any host is when the server listens on `0.0.0.0` or `::` without this list. Other hosts get `403`, which stops DNS rebinding |
| `ORTLLAMA_KEEP_ALIVE`    | `5m`                                                      | unload an idle model after this long (`-1` keeps it)                                                                                                                                                                                       |
| `ORTLLAMA_MAX_LOADED`    | `1`                                                       | models kept loaded at once                                                                                                                                                                                                                 |
| `ORTLLAMA_MAX_TOKENS`    | `8192`                                                    | completion tokens for a chat request that sets no limit                                                                                                                                                                                    |
| `ORTLLAMA_EP`            | per host, see [Execution providers](#execution-providers) | execution providers, in order, for every model                                                                                                                                                                                             |
| `ORTLLAMA_MODEL_PATH`    | none                                                      | extra plain folders, listed as `local/<name>`                                                                                                                                                                                              |
| `ORTLLAMA_MODULE_PATH`   | none                                                      | project folders searched for model libraries such as `@ai-ecoverse/kev.js` and `espeak-ng`                                                                                                                                                 |
| `ORTLLAMA_ORT_LOG_LEVEL` | `error`                                                   | ONNX Runtime's log level                                                                                                                                                                                                                   |
| `ORT_DISABLE_TELEMETRY`  | `1`                                                       | onnxruntime-node on macOS and Linux uploads usage telemetry to Microsoft unless this is set, and its uploader can abort the process at exit, so ortllama sets it; `0` turns the telemetry back on                                          |
| `ORTLLAMA_PHONEMIZERS`   | `misaki,phonemize,espeak-ng,phonemizer`                   | Kokoro's phonemizers, in order                                                                                                                                                                                                             |

## Develop

Node ≥ 24.

| Command             |                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------- |
| `npm run lint`      | `slicc-lint`, from [slicc-shared-web](https://github.com/ai-ecoverse/slicc-shared-web) |
| `npm test`          | integration tests in `test/integration/`                                               |
| `npm run test:unit` | unit tests, which stay local in `test/unit/`                                           |

Each integration test runs when its models are in the Hugging Face cache; `ORTLLAMA_REQUIRE_MODEL=1` makes a missing model a failure, and `ORTLLAMA_TEST_WHISPER` runs the Whisper and realtime tests on another Whisper model, such as `onnx-community/whisper-base`, and `ORTLLAMA_TEST_PARAKEET` the Parakeet tests on another Parakeet TDT export. Likewise `ORTLLAMA_CHAT_MODEL` runs the chat and image tests, and `ORTLLAMA_REASONING_MODEL` the reasoning tests, on another model, such as `ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX:q4f16` and `ai-ecoverse/Ministral-3-8B-Reasoning-2512-ONNX:q4f16`.

| Tests                           | Models                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Kev                             | `ai-ecoverse/kev.js/kev-0.8b`                                                                                                                                                              |
| kev-ministral, its base weights | `ai-ecoverse/kev-ministral/kev-ministral-3b`                                                                                                                                               |
| Whisper                         | `onnx-community/whisper-tiny`                                                                                                                                                              |
| Realtime                        | `onnx-community/whisper-tiny`, `onnx-community/silero-vad`                                                                                                                                 |
| Parakeet TDT                    | `ai-ecoverse/Underdog-Bark-3.0-ONNX` (`fp32`, `fp16`, `q8`, `q4f16`; the mel, encoder and parity clip tests read the export's `fixture/` and `export/clips/`), `onnx-community/silero-vad` |
| Chat                            | `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16`                                                                                                                                        |
| Reasoning                       | `ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX:q4f16`                                                                                                                                     |
| Images                          | `mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16` with its vision encoder, `ai-ecoverse/kev.js/kev-0.8b-vision`                                                                          |
| Kokoro                          | `onnx-community/Kokoro-82M-v1.0-ONNX`                                                                                                                                                      |
| Phonon                          | `ai-ecoverse/phonon-31m-en-ONNX`                                                                                                                                                           |
| Embeddings                      | `onnx-community/embeddinggemma-2-ONNX` (its default variant, and `fp32` for the WASM test when it is cached)                                                                               |
| GGUF chat                       | `bartowski/SmolLM2-135M-Instruct-GGUF` at revision `09816ac` (`Q4_0`, 92 MB; CI downloads it into a temporary cache)                                                                       |
| GGUF images, GGUF System One    | `mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M` with its `BF16-mmproj`, `ggml-org/Laya-GGUF:Q8_0`                                                                                     |

CI is [slicc-shared-web's pipeline](https://github.com/ai-ecoverse/slicc-shared-web#pipeline), called from `.github/workflows/ci.yml`: lint, the integration tests split over two shards, and a Claude and a Codex review on every pull request. CI has no Hugging Face cache, so only the tests that need no model run there: codecs, the server, streaming, the cache and the engines on a fake ORT or a fake wllama, plus the GGUF chat test, which downloads its 92 MB model into a temporary cache. On `main` it releases to npm with [semantic-release](https://semantic-release.gitbook.io) and provenance, through npm trusted publishing from `ci.yml`, so commit messages follow [Conventional Commits](https://www.conventionalcommits.org).

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
- [Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) by Conway Research, from NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and Moondream's [parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra), all CC-BY-4.0: the model the Parakeet TDT engine runs.
- [Ministral 3](https://huggingface.co/collections/mistralai/ministral-3), [Whisper](https://github.com/openai/whisper), [Silero VAD](https://github.com/snakers4/silero-vad), [Kokoro](https://github.com/hexgrad/kokoro) and [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2): the default models.
- [sentence-transformers](https://github.com/huggingface/sentence-transformers): the reference the embedding fixtures come from.
- [Phonon](https://github.com/gradium-ai/xn-ptts) (Gradium, built on Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)): the `phonon` speech engine's model, and `ptts`, its reference runtime.
- [jSquash](https://github.com/jamsinclair/jSquash): libwebp in WebAssembly for WebP images.

## License

[Apache-2.0](LICENSE)
