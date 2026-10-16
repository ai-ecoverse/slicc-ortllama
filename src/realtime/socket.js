import { errorBody, HttpError } from '../http/errors.js';
import { allowedHost, allowedOrigin, authorized } from '../http/server.js';
import { RealtimeSession } from './session.js';
import { upgrade } from './websocket.js';

const KEY_PROTOCOL = 'openai-insecure-api-key.';
const REASONS = { 400: 'Bad Request', 401: 'Unauthorized', 403: 'Forbidden', 404: 'Not Found' };

function protocolsOf(req) {
  return (req.headers['sec-websocket-protocol'] ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

export function checkUpgrade(req, options = {}) {
  if (!allowedHost(req.headers.host, options.hosts))
    return new HttpError(
      403,
      `host ${req.headers.host} is not allowed; set ORTLLAMA_HOSTS to allow it`,
      {
        code: 'host_not_allowed',
      }
    );
  if (req.headers.origin && !allowedOrigin(req.headers.origin, options.origins))
    return new HttpError(
      403,
      `origin ${req.headers.origin} is not allowed; set ORTLLAMA_ORIGINS to allow it`,
      {
        code: 'origin_not_allowed',
      }
    );
  const url = new URL(req.url, 'http://ortllama');
  if (url.pathname !== '/v1/realtime')
    return new HttpError(404, `no WebSocket route for ${url.pathname}`, { code: 'unknown_route' });
  if (url.searchParams.get('intent') !== 'transcription')
    return new HttpError(400, 'only intent=transcription is supported', { param: 'intent' });
  const keyed = protocolsOf(req).find((p) => p.startsWith(KEY_PROTOCOL));
  const viaProtocol = options.apiKey && keyed === `${KEY_PROTOCOL}${options.apiKey}`;
  if (!viaProtocol && !authorized(req, options.apiKey))
    return new HttpError(401, 'missing or wrong API key', { code: 'invalid_api_key' });
  return null;
}

function refuse(socket, error) {
  const body = JSON.stringify(errorBody(error).body);
  socket.end(
    `HTTP/1.1 ${error.status} ${REASONS[error.status]}\r\ncontent-type: application/json\r\ncontent-length: ${Buffer.byteLength(body)}\r\nconnection: close\r\n\r\n${body}`
  );
}

export function attachRealtime(server, { catalog, runtime, ...options }) {
  const sessions = new Map();
  server.on('upgrade', (req, socket, head) => {
    const problem = checkUpgrade(req, options);
    if (problem) {
      refuse(socket, problem);
      return;
    }
    const ws = upgrade(req, socket, head, { protocols: ['realtime'], idleMs: options.idleMs });
    if (!ws) {
      refuse(socket, new HttpError(400, 'the request is not a WebSocket upgrade'));
      return;
    }
    const session = new RealtimeSession({
      catalog,
      runtime,
      send: (event) => ws.send(JSON.stringify(event)),
    });
    sessions.set(ws, session);
    ws.on('message', (data, binary) => {
      if (binary)
        session.fail(new HttpError(400, 'events must be JSON text', { code: 'invalid_event' }));
      else session.receive(data);
    });
    ws.on('close', () => {
      session.close().finally(() => sessions.delete(ws));
    });
    session.start();
  });
  return {
    sessions,
    async close() {
      const open = [...sessions];
      await Promise.all(open.map(([, session]) => session.close()));
      for (const [ws] of open) ws.terminate();
    },
  };
}
