import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';

const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const OPEN = 1;
const CLOSED = 3;
const MAX_FRAGMENTS = 1024;
const DATA = new Set([0x0, 0x1, 0x2]);
const CONTROL = new Set([0x8, 0x9, 0xa]);

export function acceptKey(key) {
  return createHash('sha1').update(`${key}${GUID}`).digest('base64');
}

export function frame(opcode, payload) {
  const length = payload.length;
  const size = length < 126 ? 2 : length < 65536 ? 4 : 10;
  const header = Buffer.alloc(size);
  header[0] = 0x80 | opcode;
  if (size === 2) header[1] = length;
  else if (size === 4) {
    header[1] = 126;
    header.writeUInt16BE(length, 2);
  } else {
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(length), 2);
  }
  return Buffer.concat([header, payload]);
}

export class WebSocketConnection extends EventEmitter {
  constructor(socket, { protocol = '', maxPayload = 16 * 1024 * 1024, idleMs = 60_000 } = {}) {
    super();
    this.socket = socket;
    this.protocol = protocol;
    this.maxPayload = maxPayload;
    this.idleMs = idleMs;
    this.timer = null;
    this.readyState = OPEN;
    this.buffer = Buffer.alloc(0);
    this.parts = [];
    this.partsLength = 0;
    this.partsOpcode = 0;
    socket.on('data', (chunk) => this.receive(chunk));
    socket.on('close', () => this.closed());
    socket.on('error', () => this.closed());
    this.touch();
  }

  touch() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.close(1000, 'idle'), this.idleMs);
    this.timer.unref?.();
  }

  receive(chunk) {
    this.touch();
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (this.readyState === OPEN) {
      const parsed = this.parse();
      if (!parsed) return;
      this.handle(parsed);
    }
  }

  parse() {
    const b = this.buffer;
    if (b.length < 2) return null;
    const fin = (b[0] & 0x80) !== 0;
    const opcode = b[0] & 0x0f;
    const masked = (b[1] & 0x80) !== 0;
    let length = b[1] & 0x7f;
    let at = 2;
    if (length === 126) {
      if (b.length < 4) return null;
      length = b.readUInt16BE(2);
      at = 4;
    } else if (length === 127) {
      if (b.length < 10) return null;
      length = Number(b.readBigUInt64BE(2));
      at = 10;
    }
    if (!masked) return { error: 1002, reason: 'client frames must be masked' };
    if (!DATA.has(opcode) && !CONTROL.has(opcode)) return { error: 1002, reason: 'unknown opcode' };
    if (CONTROL.has(opcode) && (!fin || length > 125))
      return { error: 1002, reason: 'control frames must be whole and at most 125 bytes' };
    if (opcode === 0 && this.parts.length === 0)
      return { error: 1002, reason: 'continuation without a message' };
    if ((opcode === 1 || opcode === 2) && this.parts.length > 0)
      return { error: 1002, reason: 'new message before the last one ended' };
    if (DATA.has(opcode) && this.parts.length >= MAX_FRAGMENTS)
      return { error: 1009, reason: 'too many fragments' };
    if (length + this.partsLength > this.maxPayload)
      return { error: 1009, reason: 'message too big' };
    if (b.length < at + 4 + length) return null;
    const mask = b.subarray(at, at + 4);
    const payload = Buffer.from(b.subarray(at + 4, at + 4 + length));
    for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
    this.buffer = b.subarray(at + 4 + length);
    return { fin, opcode, payload };
  }

  handle({ fin, opcode, payload, error, reason }) {
    if (error) {
      this.close(error, reason);
      return;
    }
    if (opcode === 0x8) {
      const code = payload.length >= 2 ? payload.readUInt16BE(0) : 1000;
      this.close(
        code >= 1000 && code < 5000 && ![1004, 1005, 1006, 1015].includes(code) ? code : 1002
      );
      return;
    }
    if (opcode === 0x9) {
      this.write(0xa, payload);
      return;
    }
    if (opcode === 0xa) return;
    if (opcode !== 0) this.partsOpcode = opcode;
    this.parts.push(payload);
    this.partsLength += payload.length;
    if (!fin) return;
    const data = Buffer.concat(this.parts);
    const binary = this.partsOpcode === 0x2;
    this.parts = [];
    this.partsLength = 0;
    this.emit('message', binary ? data : data.toString('utf8'), binary);
  }

  write(opcode, payload) {
    if (this.readyState === OPEN) this.socket.write(frame(opcode, payload));
  }

  send(text) {
    this.write(0x1, Buffer.from(text, 'utf8'));
  }

  close(code = 1000, reason = '') {
    if (this.readyState !== OPEN) return;
    clearTimeout(this.timer);
    const payload = Buffer.alloc(2 + Buffer.byteLength(reason));
    payload.writeUInt16BE(code, 0);
    payload.write(reason, 2);
    this.write(0x8, payload);
    this.readyState = 2;
    this.socket.end();
  }

  terminate() {
    this.socket.destroy();
    this.closed();
  }

  closed() {
    clearTimeout(this.timer);
    if (this.readyState === CLOSED) return;
    this.readyState = CLOSED;
    this.emit('close');
  }
}

export function upgrade(req, socket, head, { protocols = [], maxPayload, idleMs } = {}) {
  const key = req.headers['sec-websocket-key'];
  const valid =
    req.method === 'GET' &&
    /\bwebsocket\b/i.test(req.headers.upgrade ?? '') &&
    req.headers['sec-websocket-version'] === '13' &&
    typeof key === 'string' &&
    Buffer.from(key, 'base64').length === 16;
  if (!valid) return null;
  const offered = (req.headers['sec-websocket-protocol'] ?? '').split(',').map((s) => s.trim());
  const protocol = protocols.find((p) => offered.includes(p)) ?? '';
  socket.setNoDelay?.(true);
  socket.write(
    `HTTP/1.1 101 Switching Protocols\r\nupgrade: websocket\r\nconnection: Upgrade\r\nsec-websocket-accept: ${acceptKey(key)}\r\n${protocol ? `sec-websocket-protocol: ${protocol}\r\n` : ''}\r\n`
  );
  const ws = new WebSocketConnection(socket, { protocol, maxPayload, idleMs });
  if (head?.length) queueMicrotask(() => ws.receive(head));
  return ws;
}
