import { createServer as createHttpServer } from 'node:http';
import { errorBody, HttpError } from './errors.js';

const MAX_BODY = 16 * 1024 * 1024;

export function allowedOrigin(origin, extra = []) {
  if (!origin) return null;
  if (extra.includes(origin)) return origin;
  let url;
  try {
    url = new URL(origin);
  } catch {
    return null;
  }
  const host = url.hostname;
  const local =
    host === 'localhost' || host === '127.0.0.1' || host === '[::1]' || host.endsWith('.localhost');
  return local ? origin : null;
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

export function readJson(req, limit = MAX_BODY) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new HttpError(413, `request body is larger than ${limit} bytes`));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8');
      if (text.trim() === '') return reject(new HttpError(400, 'request body must be JSON'));
      try {
        resolve(JSON.parse(text));
      } catch {
        reject(new HttpError(400, 'request body is not valid JSON'));
      }
    });
    req.on('error', reject);
  });
}

function authorized(req, apiKey) {
  if (!apiKey) return true;
  const header = req.headers.authorization ?? '';
  return header === `Bearer ${apiKey}`;
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
      const url = new URL(req.url, 'http://ortllama');
      const route = routes[`${req.method} ${url.pathname}`];
      if (!route)
        throw new HttpError(404, `no route for ${req.method} ${url.pathname}`, {
          code: 'unknown_route',
        });
      if (!route.public && !authorized(req, options.apiKey))
        throw new HttpError(401, 'missing or wrong API key', { code: 'invalid_api_key' });
      const body = req.method === 'POST' ? await readJson(req, options.maxBody) : undefined;
      const result = await route.handle({ req, url, body, signal: controller.signal });
      send(result?.status ?? 200, result?.body ?? result);
    } catch (error) {
      if (controller.signal.aborted) return;
      const { status, body } = errorBody(error);
      send(status, body);
    }
  };
}

export function createServer(routes, options = {}) {
  return createHttpServer(createHandler(routes, options));
}
