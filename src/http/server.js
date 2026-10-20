import { createServer as createHttpServer } from 'node:http';
import { errorBody, HttpError } from './errors.js';

const MAX_BODY = 16 * 1024 * 1024;
const DRAIN_MS = 1000;

const LOOPBACK = /^(localhost|127(\.\d{1,3}){3}|\[::1\]|::1|host\.slicc\.internal|.+\.localhost)$/;

export function allowedHost(host, extra = []) {
  if (!host || extra.includes('*')) return true;
  let hostname;
  try {
    hostname = new URL(`http://${host}`).hostname;
  } catch {
    return false;
  }
  return LOOPBACK.test(hostname) || extra.includes(hostname);
}

export function allowedOrigin(origin, extra = []) {
  if (!origin) return null;
  if (extra.includes(origin)) return origin;
  let url;
  try {
    url = new URL(origin);
  } catch {
    return null;
  }
  return LOOPBACK.test(url.hostname) && url.hostname !== 'host.slicc.internal' ? origin : null;
}

function stopReading(req, ms = DRAIN_MS) {
  if (req.complete) return;
  const timer = setTimeout(() => req.destroy(), ms);
  timer.unref?.();
  req.once('end', () => clearTimeout(timer));
}

function corsHeaders(req, options) {
  const origin = allowedOrigin(req.headers.origin, options.origins);
  if (!origin) return {};
  return {
    'access-control-allow-origin': origin,
    'access-control-allow-headers': 'authorization, content-type',
    'access-control-allow-methods': 'GET, POST, OPTIONS',
    'access-control-max-age': '600',
    vary: 'origin',
  };
}

export function readBody(req, limit = MAX_BODY) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > limit) {
        chunks.length = 0;
        reject(new HttpError(413, `request body is larger than ${limit} bytes`));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (size <= limit) resolve(Buffer.concat(chunks));
    });
    req.on('error', reject);
    req.on('close', () => {
      if (!req.complete) reject(new HttpError(400, 'the request body was cut off'));
    });
  });
}

export async function readJson(req, limit = MAX_BODY) {
  const text = (await readBody(req, limit)).toString('utf8');
  if (text.trim() === '') throw new HttpError(400, 'request body must be JSON');
  try {
    return JSON.parse(text);
  } catch {
    throw new HttpError(400, 'request body is not valid JSON');
  }
}

function abortion(signal) {
  const aborted = new Promise((_, reject) => {
    if (signal.aborted) reject(signal.reason);
    else signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
  aborted.catch(() => {});
  return aborted;
}

async function pipe(res, result, cors, signal) {
  const chunks = result.stream[Symbol.asyncIterator]();
  const aborted = abortion(signal);
  const next = () => Promise.race([chunks.next(), aborted]);
  const stop = () => Promise.resolve(chunks.return?.()).catch(() => {});
  let step;
  try {
    step = await next();
  } catch (error) {
    void stop();
    throw error;
  }
  res.writeHead(result.status ?? 200, { ...cors, ...result.headers });
  try {
    while (!step.done) {
      const written = new Promise((resolve) => res.write(step.value, resolve));
      await Promise.race([written, aborted]);
      step = await next();
    }
    res.end();
  } catch (error) {
    void stop();
    res.destroy(signal.aborted ? undefined : error);
  }
}

export function authorized(req, apiKey) {
  if (!apiKey) return true;
  const header = req.headers.authorization ?? '';
  return header === `Bearer ${apiKey}`;
}

async function sendEvents(res, events, headers) {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    ...headers,
  });
  try {
    for await (const event of events) {
      if (res.destroyed) return;
      res.write(`data: ${JSON.stringify(event)}\n\n`);
    }
    res.end('data: [DONE]\n\n');
  } catch (error) {
    res.end(`data: ${JSON.stringify(errorBody(error).body)}\n\n`);
  }
}

function respond(res, result, { cors, send, signal }) {
  if (result?.events) return sendEvents(res, result.events, cors);
  if (result?.stream) return pipe(res, result, cors, signal);
  send(result?.status ?? 200, result?.body ?? result);
}

export function createHandler(routes, options = {}) {
  return async (req, res) => {
    const cors = corsHeaders(req, options);
    const send = (status, body, headers = {}) => {
      if (res.headersSent || res.destroyed) return;
      const json = typeof body === 'string' ? body : JSON.stringify(body);
      res.writeHead(status, {
        'content-type': typeof body === 'string' ? 'text/plain; charset=utf-8' : 'application/json',
        ...cors,
        ...headers,
      });
      res.end(json);
    };
    if (req.method === 'OPTIONS') {
      res.writeHead(204, cors);
      res.end();
      return;
    }
    const controller = new AbortController();
    res.on('close', () => {
      if (!res.writableFinished) controller.abort();
    });
    try {
      if (!allowedHost(req.headers.host, options.hosts)) {
        throw new HttpError(
          403,
          `host ${req.headers.host} is not allowed; set ORTLLAMA_HOSTS to allow it`,
          { code: 'host_not_allowed' }
        );
      }
      const url = new URL(req.url, 'http://ortllama');
      const route = routes[`${req.method} ${url.pathname}`];
      if (!route)
        throw new HttpError(404, `no route for ${req.method} ${url.pathname}`, {
          code: 'unknown_route',
        });
      if (!route.public && !authorized(req, options.apiKey))
        throw new HttpError(401, 'missing or wrong API key', { code: 'invalid_api_key' });
      const read = route.raw ? readBody : readJson;
      const body = req.method === 'POST' ? await read(req, options.maxBody) : undefined;
      const result = await route.handle({ req, url, body, signal: controller.signal });
      await respond(res, result, { cors, send, signal: controller.signal });
    } catch (error) {
      if (controller.signal.aborted) return;
      const { status, body } = errorBody(error);
      if (status === 413) res.once('finish', () => stopReading(req, options.drainMs));
      send(status, body);
    }
  };
}

export function createServer(routes, options = {}) {
  return createHttpServer(createHandler(routes, options));
}
