import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { externalFiles } from '../engines/external.js';
import { HttpError } from '../http/errors.js';

const CHUNK = 16 * 1024 * 1024;
const SUBTLE = { sha1: 'SHA-1', sha256: 'SHA-256' };
const LABEL = { sha1: 'git blob sha1', sha256: 'sha256' };
const verified = new Map();

const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');

async function streamed(path, algorithm, prefix) {
  const hash = createHash(algorithm).update(prefix);
  for await (const chunk of createReadStream(path, { highWaterMark: CHUNK })) hash.update(chunk);
  return hash.digest('hex');
}

async function whole(path, algorithm, prefix) {
  const data = new Uint8Array(await readFile(path));
  const head = new TextEncoder().encode(prefix);
  let bytes = data;
  if (head.length > 0) {
    bytes = new Uint8Array(head.length + data.length);
    bytes.set(head);
    bytes.set(data, head.length);
  }
  return hex(new Uint8Array(await crypto.subtle.digest(SUBTLE[algorithm], bytes)));
}

export async function blobDigest(blob, { read = 'stream' } = {}) {
  const algorithm = blob.etag.length === 40 ? 'sha1' : 'sha256';
  const prefix = algorithm === 'sha1' ? `blob ${blob.size}\0` : '';
  const digest = await (read === 'whole' ? whole : streamed)(blob.path, algorithm, prefix);
  return { algorithm, digest };
}

function corrupt(resolved, rel, { algorithm, digest }, etag) {
  const hint = resolved.redownload ? `; run: ${resolved.redownload(rel)}` : '';
  return new HttpError(
    500,
    `model ${resolved.id}:${resolved.variant} has a corrupt file ${rel}: its ${LABEL[algorithm]} is ${digest}, but the hub's etag is ${etag}${hint}`,
    { code: 'model_corrupt', param: 'model' }
  );
}

async function verifiable(files, patterns) {
  const out = await files.matches(patterns);
  for (const rel of [...out]) {
    if (!rel.endsWith('.onnx')) continue;
    for (const name of await externalFiles(files, rel))
      if (!out.includes(name) && (await files.exists(name))) out.push(name);
  }
  return out;
}

export async function verifyModel(resolved, { read = 'stream', memo = verified } = {}) {
  const { files, engine, info, variant } = resolved;
  if (!files?.blob || !engine.include) return 0;
  let checked = 0;
  for (const rel of await verifiable(files, engine.include(info, variant))) {
    const blob = await files.blob(rel);
    const stamp = blob && `${blob.size}:${blob.mtimeMs}`;
    if (!blob || memo.get(blob.path) === stamp) continue;
    const found = await blobDigest(blob, { read });
    if (found.digest !== blob.etag) throw corrupt(resolved, rel, found, blob.etag);
    memo.set(blob.path, stamp);
    checked++;
  }
  return checked;
}
