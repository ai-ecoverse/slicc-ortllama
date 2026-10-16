import { readFile } from 'node:fs/promises';
import { jpegExif, orient, orientation, webpExif } from './exif.js';
import { badRequest, HttpError } from './http/errors.js';
import { checkInflate, pngChunks, rawSize } from './png.js';

export const IMAGE_LIMITS = { maxBytes: 10 * 2 ** 20, maxImages: 8, maxPixels: 40_000_000 };

const DATA_URL = /^data:([\w.+-]+\/[\w.+-]+)?((?:;[\w.+-]+=[^;,]*)*);base64,/i;

const u16be = (b, i) => (b[i] << 8) | b[i + 1];
const u16le = (b, i) => b[i] | (b[i + 1] << 8);
const u24le = (b, i) => b[i] | (b[i + 1] << 8) | (b[i + 2] << 16);
const u32be = (b, i) => b[i] * 2 ** 24 + ((b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]);
const ascii = (b, i, n) => String.fromCharCode(...b.subarray(i, i + n));

export function imageBytes(url, { param = 'messages', limits = IMAGE_LIMITS } = {}) {
  if (typeof url !== 'string') throw badRequest('an image needs a url string', param);
  if (/^https?:/i.test(url))
    throw badRequest(
      'ortllama does not fetch image URLs; send the image inline as a data: URL (data:image/png;base64,…)',
      param
    );
  const match = DATA_URL.exec(url.slice(0, 256));
  if (!match) throw badRequest('an image url must be a base64 data: URL', param);
  const b64 = url.slice(match[0].length);
  if (Math.floor((b64.length * 3) / 4) > limits.maxBytes + 2)
    throw badRequest(`an image is larger than ${limits.maxBytes} bytes`, param);
  if (/[^A-Za-z0-9+/=\s]/.test(b64))
    throw badRequest('an image data: URL is not valid base64', param);
  return new Uint8Array(Buffer.from(b64, 'base64'));
}

export function sniff(bytes) {
  if (bytes.length >= 24 && u32be(bytes, 0) === 0x89504e47 && ascii(bytes, 12, 4) === 'IHDR')
    return 'png';
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff)
    return 'jpeg';
  if (bytes.length >= 30 && ascii(bytes, 0, 4) === 'RIFF' && ascii(bytes, 8, 4) === 'WEBP')
    return 'webp';
  return null;
}

const SOF = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);

function jpegSize(b) {
  let i = 2;
  while (i + 9 < b.length) {
    if (b[i] !== 0xff) return null;
    const marker = b[i + 1];
    if (marker === 0xff) {
      i++;
      continue;
    }
    if (SOF.has(marker)) return { height: u16be(b, i + 5), width: u16be(b, i + 7) };
    i += 2 + u16be(b, i + 2);
  }
  return null;
}

function webpSize(b) {
  const chunk = ascii(b, 12, 4);
  if (chunk === 'VP8 ') return { width: u16le(b, 26) & 0x3fff, height: u16le(b, 28) & 0x3fff };
  if (chunk === 'VP8L')
    return {
      width: 1 + (((b[22] & 0x3f) << 8) | b[21]),
      height: 1 + (((b[24] & 0xf) << 10) | (b[23] << 2) | ((b[22] & 0xc0) >> 6)),
    };
  if (chunk === 'VP8X') return { width: 1 + u24le(b, 24), height: 1 + u24le(b, 27) };
  return null;
}

export function dimensions(bytes, kind) {
  if (kind === 'png') return { width: u32be(bytes, 16), height: u32be(bytes, 20) };
  if (kind === 'jpeg') return jpegSize(bytes);
  return webpSize(bytes);
}

function unpack({ width, height, data, channels, depth }) {
  if (depth >= 8) return data;
  const per = width * channels;
  const stride = Math.ceil((per * depth) / 8);
  const mask = 2 ** depth - 1;
  const out = new Uint8Array(per * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < per; x++) {
      const bit = x * depth;
      out[y * per + x] = (data[y * stride + (bit >> 3)] >> (8 - depth - (bit & 7))) & mask;
    }
  return out;
}

function toRgba(png) {
  const { width, height, channels, depth } = png;
  const n = width * height;
  const data = unpack(png);
  const shift = depth === 16 ? 8 : 0;
  const scale = depth < 8 ? 255 / (2 ** depth - 1) : 1;
  const out = new Uint8Array(n * 4);
  const gray = channels < 3;
  for (let i = 0; i < n; i++) {
    const at = i * channels;
    const v = (k) => Math.round((data[at + k] >> shift) * scale);
    out[i * 4] = v(0);
    out[i * 4 + 1] = gray ? v(0) : v(1);
    out[i * 4 + 2] = gray ? v(0) : v(2);
    out[i * 4 + 3] = channels === 2 ? v(1) : channels === 4 ? v(3) : 255;
  }
  return out;
}

let webpReady = null;

async function webpDecoder() {
  webpReady ??= (async () => {
    const { default: decode, init } = await import('@jsquash/webp/decode.js');
    const wasm = new URL('codec/dec/webp_dec.wasm', import.meta.resolve('@jsquash/webp/decode.js'));
    await init(await WebAssembly.compile(await readFile(wasm)));
    return decode;
  })();
  return webpReady;
}

async function decodePng(bytes, param) {
  const { ihdr, idat, exif, file } = pngChunks(bytes, param);
  checkInflate(idat, rawSize(ihdr, param), param);
  const { decode, convertIndexedToRgb } = await import('fast-png');
  const png = decode(file, { checkCrc: true });
  const rgb = png.palette ? convertIndexedToRgb(png) : null;
  const flat = rgb
    ? { ...png, data: rgb, depth: 8, channels: rgb.length / (png.width * png.height) }
    : png;
  return { width: png.width, height: png.height, data: toRgba(flat), exif };
}

async function decodeAs(kind, bytes, limits, param) {
  if (kind === 'png') return decodePng(bytes, param);
  if (kind === 'jpeg') {
    const { default: jpeg } = await import('jpeg-js');
    const img = jpeg.decode(bytes, {
      useTArray: true,
      formatAsRGBA: true,
      maxResolutionInMP: limits.maxPixels / 1e6,
      maxMemoryUsageInMB: 1024,
    });
    return { width: img.width, height: img.height, data: img.data, exif: jpegExif(bytes) };
  }
  const decode = await webpDecoder();
  const img = await decode(bytes);
  return {
    width: img.width,
    height: img.height,
    data: new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.length),
    exif: webpExif(bytes),
  };
}

async function sha256(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function checkDecoded(image, kind, limits, param) {
  const pixels = image.width * image.height;
  if (image.data.length !== pixels * 4 || pixels > limits.maxPixels)
    throw badRequest(`the ${kind} image decoded to the wrong size`, param);
}

export async function decodeImage(bytes, { param = 'messages', limits = IMAGE_LIMITS } = {}) {
  const kind = sniff(bytes);
  if (!kind) throw badRequest('an image must be a PNG, JPEG or WebP file', param);
  const size = dimensions(bytes, kind);
  if (!size || !(size.width > 0 && size.height > 0))
    throw badRequest(`the ${kind} image has no readable size`, param);
  if (size.width * size.height > limits.maxPixels)
    throw badRequest(
      `an image of ${size.width}x${size.height} has more than ${limits.maxPixels} pixels`,
      param
    );
  let image;
  try {
    image = await decodeAs(kind, bytes, limits, param);
  } catch (error) {
    if (error instanceof HttpError) throw error;
    throw badRequest(`the ${kind} image could not be decoded: ${error.message}`, param);
  }
  checkDecoded(image, kind, limits, param);
  const { width, height, data } = orient(image, orientation(image.exif));
  return { width, height, data, hash: await sha256(bytes) };
}

export async function imageFromUrl(url, options) {
  return decodeImage(imageBytes(url, options), options);
}

export async function requestImage(value, resolved, limits = IMAGE_LIMITS) {
  if (!resolved.info.inputModalities.includes('image'))
    throw badRequest(`model ${resolved.id} takes no images; use a vision bundle`, 'image');
  const url = typeof value === 'string' ? value : value?.url;
  const { width, height, data } = await imageFromUrl(url, { param: 'image', limits });
  return { width, height, data };
}
