import { describe, readHeader } from './header.js';

const GGUF = /\.gguf$/i;
const SPLIT = /-(\d{5})-of-(\d{5})\.gguf$/i;
const QUANT =
  /(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[0-9A-Z]+)*|TQ\d_\d|MXFP4(?:_MOE)?|BF16|F16|F32))(?=[-_.]|$)/gi;

export const FILE_TYPES = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  7: 'Q8_0',
  8: 'Q5_0',
  9: 'Q5_1',
  10: 'Q2_K',
  11: 'Q3_K_S',
  12: 'Q3_K_M',
  13: 'Q3_K_L',
  14: 'Q4_K_S',
  15: 'Q4_K_M',
  16: 'Q5_K_S',
  17: 'Q5_K_M',
  18: 'Q6_K',
  19: 'IQ2_XXS',
  20: 'IQ2_XS',
  21: 'Q2_K_S',
  22: 'IQ3_XS',
  23: 'IQ3_XXS',
  24: 'IQ1_S',
  25: 'IQ4_NL',
  26: 'IQ3_S',
  27: 'IQ3_M',
  28: 'IQ2_S',
  29: 'IQ2_M',
  30: 'IQ4_XS',
  31: 'IQ1_M',
  32: 'BF16',
  36: 'TQ1_0',
  37: 'TQ2_0',
  38: 'MXFP4_MOE',
};

export const PREFERRED = [
  'Q4_K_M',
  'Q4_K_S',
  'Q4_0',
  'Q5_K_M',
  'Q5_K_S',
  'Q6_K',
  'Q8_0',
  'IQ4_XS',
  'IQ4_NL',
  'Q3_K_M',
  'Q3_K_L',
  'Q3_K_S',
  'Q2_K',
  'F16',
  'BF16',
  'F32',
];

const PROJECTORS = ['Q8_0', 'F16', 'BF16', 'F32'];

export function quantOf(name) {
  const stem = name.replace(SPLIT, '').replace(GGUF, '');
  const found = [...stem.matchAll(QUANT)].map((m) => m[1].toUpperCase());
  return found.at(-1) ?? null;
}

export function splitOf(name) {
  const m = SPLIT.exec(name);
  return m ? { part: Number(m[1]), count: Number(m[2]), base: name.slice(0, m.index) } : null;
}

export function partNames(base, count) {
  const n = String(count).padStart(5, '0');
  return Array.from(
    { length: count },
    (_, i) => `${base}-${String(i + 1).padStart(5, '0')}-of-${n}.gguf`
  );
}

function rank(name, order) {
  const plain = name.replace(/^UD-/, '');
  const at = order.indexOf(plain);
  return (at < 0 ? order.length : at) + (plain === name ? 0 : 0.5);
}

export function byPreference(names, order = PREFERRED) {
  return [...names].sort((a, b) => rank(a, order) - rank(b, order) || a.localeCompare(b));
}

function stemOf(name) {
  return name
    .replace(SPLIT, '')
    .replace(GGUF, '')
    .replace(/[^A-Za-z0-9._-]/g, '_');
}

async function headerOf(files, rel) {
  try {
    return describe(await readHeader(files, rel));
  } catch {
    return null;
  }
}

export async function scanGguf(files) {
  const names = (await files.list('')).filter((n) => GGUF.test(n)).sort();
  const models = [];
  const projectors = [];
  for (const name of names) {
    const split = splitOf(name);
    if (split && split.part !== 1) continue;
    const head = await headerOf(files, name);
    if (!head) continue;
    if (head.projector) {
      if (head.vision)
        projectors.push({ file: name, quant: quantOf(name) ?? FILE_TYPES[head.fileType] ?? null });
      continue;
    }
    const parts = split ? partNames(split.base, split.count) : [name];
    models.push({ name, parts, head, quant: quantOf(name) ?? FILE_TYPES[head.fileType] ?? null });
  }
  return { models, projectors };
}

export function variantsOf(models) {
  const out = {};
  const taken = new Map();
  for (const m of models) taken.set(m.quant, (taken.get(m.quant) ?? 0) + 1);
  for (const m of models) {
    const name = m.quant && taken.get(m.quant) === 1 ? m.quant : stemOf(m.name);
    out[name] = {
      files: m.parts,
      architecture: m.head.architecture,
      contextLength: m.head.contextLength,
    };
  }
  return out;
}

export function pickProjector(projectors) {
  if (projectors.length === 0) return null;
  const order = byPreference(
    projectors.map((p) => p.quant ?? ''),
    PROJECTORS
  );
  return projectors.find((p) => (p.quant ?? '') === order[0]).file;
}
