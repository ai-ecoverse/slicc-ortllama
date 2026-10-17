import { WebSocketServer } from 'ws';
import { errorBody, HttpError } from '../http/errors.js';
import { allowedHost, allowedOrigin, authorized } from '../http/server.js';
import { RealtimeSession } from './session.js';

const KEY_PROTOCOL = 'openai-insecure-api-key.';
const MAX_PAYLOAD = 16 * 1024 * 1024;
const MAX_FRAGMENTS = 1024;
const IDLE_MS = 60_000;
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

function closeWhenIdle(ws, ms) {
  let timer = null;
  const touch = () => {
    clearTimeout(timer);
    timer = setTimeout(() => ws.close(1000, 'idle'), ms);
    timer.unref?.();
  };
  for (const event of ['message', 'ping', 'pong']) ws.on(event, touch);
  ws.on('close', () => clearTimeout(timer));
  touch();
}

export function attachRealtime(server, { catalog, runtime, idleMs = IDLE_MS, ...options }) {
  const wss = new WebSocketServer({
    noServer: true,
    clientTracking: false,
    perMessageDeflate: false,
    maxPayload: MAX_PAYLOAD,
    maxFragments: MAX_FRAGMENTS,
    handleProtocols: (protocols) => (protocols.has('realtime') ? 'realtime' : false),
  });
  const sessions = new Map();
  const connect = (ws) => {
    const session = new RealtimeSession({
      catalog,
      runtime,
      send: (event) => {
        if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(event));
      },
    });
    sessions.set(ws, session);
    closeWhenIdle(ws, idleMs);
    ws.on('message', (data, binary) => {
      if (binary)
        session.fail(new HttpError(400, 'events must be JSON text', { code: 'invalid_event' }));
      else session.receive(data.toString('utf8'));
    });
    ws.on('error', () => {});
    ws.on('close', () => {
      session.close().finally(() => sessions.delete(ws));
    });
    session.start();
  };
  server.on('upgrade', (req, socket, head) => {
    const problem = checkUpgrade(req, options);
    if (problem) {
      refuse(socket, problem);
      return;
    }
    wss.handleUpgrade(req, socket, head, connect);
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
