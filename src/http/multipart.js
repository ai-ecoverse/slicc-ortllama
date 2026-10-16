import { HttpError } from './errors.js';

const CRLF = Buffer.from('\r\n');
const HEADER_END = Buffer.from('\r\n\r\n');

function boundaryOf(contentType = '') {
  const match = /^multipart\/form-data\s*;(?:.*;)?\s*boundary=(?:"([^"]+)"|([^\s;]+))/i.exec(
    contentType
  );
  return match ? (match[1] ?? match[2]) : null;
}

function nextDelimiter(body, separator, from) {
  for (let at = body.indexOf(separator, from); at >= 0; at = body.indexOf(separator, at + 1)) {
    const after = body
      .subarray(at + separator.length, at + separator.length + 2)
      .toString('latin1');
    if (after === '--' || after === '\r\n') return at;
  }
  return -1;
}

function headersOf(text) {
  const headers = {};
  for (const line of text.split('\r\n')) {
    const colon = line.indexOf(':');
    if (colon > 0)
      headers[line.slice(0, colon).trim().toLowerCase()] = line.slice(colon + 1).trim();
  }
  return headers;
}

function dispositionOf(value = '') {
  const out = {};
  for (const match of value.matchAll(/;\s*([a-z*]+)=(?:"((?:[^"\\]|\\.)*)"|([^;\s]+))/gi))
    out[match[1].toLowerCase()] = (match[2] ?? match[3]).replace(/\\(.)/g, '$1');
  return out;
}

export function parseMultipart(body, contentType) {
  const boundary = boundaryOf(contentType);
  if (!boundary)
    throw new HttpError(400, 'the request must be multipart/form-data with a boundary');
  const delimiter = Buffer.from(`--${boundary}`);
  const separator = Buffer.concat([CRLF, delimiter]);
  const fields = {};
  const files = {};
  let at = body.indexOf(delimiter);
  if (at < 0) throw new HttpError(400, 'the multipart body has no parts');
  for (;;) {
    at += delimiter.length;
    if (body[at] === 0x2d && body[at + 1] === 0x2d) break;
    if (body.subarray(at, at + 2).equals(CRLF)) at += 2;
    const headerEnd = body.indexOf(HEADER_END, at);
    if (headerEnd < 0) throw new HttpError(400, 'a multipart part has no header end');
    const headers = headersOf(body.subarray(at, headerEnd).toString('utf8'));
    const start = headerEnd + HEADER_END.length;
    const next = nextDelimiter(body, separator, start);
    if (next < 0) throw new HttpError(400, 'the multipart body was cut off');
    const { name, filename } = dispositionOf(headers['content-disposition']);
    const data = body.subarray(start, next);
    if (name !== undefined) {
      if (filename !== undefined)
        files[name] = {
          data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
          filename,
          type: headers['content-type'] ?? '',
        };
      else fields[name] = data.toString('utf8');
    }
    at = next + CRLF.length;
  }
  return { fields, files };
}
