// Downloads Xenova/all-MiniLM-L6-v2 (quantized ONNX + tokenizer + config) into demo/models/
// so the demo never talks to Hugging Face at runtime.
import { mkdir, writeFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const MODEL = 'Xenova/all-MiniLM-L6-v2';
const REVISION = 'main';
const FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'onnx/model_quantized.onnx',
];
const DEST = new URL(`../demo/models/${MODEL}/`, import.meta.url);

for (const file of FILES) {
  const out = new URL(file, DEST);
  if (existsSync(out)) { console.log(`have ${file} (${(await stat(out)).size} bytes)`); continue; }
  await mkdir(new URL('./', out), { recursive: true });
  const url = `https://huggingface.co/${MODEL}/resolve/${REVISION}/${file}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(out, buf);
  console.log(`got  ${file} (${buf.length} bytes)`);
}
