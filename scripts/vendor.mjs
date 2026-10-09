// Copies the browser build of transformers.js and the onnxruntime-web wasm runtime into demo/vendor/,
// so nothing is fetched from a CDN at runtime.
//
//   transformers.min.js          self-contained ESM browser build (bundles onnxruntime-web's JS)
//   ort-wasm-simd-threaded.{mjs,wasm}  the plain CPU wasm runtime (no WebGPU/asyncify needed for the 'wasm' device)
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';

const NM = new URL('./node_modules/', import.meta.url);
const DEST = new URL('../demo/vendor/', import.meta.url);
const tjs = JSON.parse(await readFile(new URL('@huggingface/transformers/package.json', NM), 'utf8'));
const ort = JSON.parse(await readFile(new URL('onnxruntime-web/package.json', NM), 'utf8'));

const FILES = [
  ['@huggingface/transformers/dist/transformers.min.js', 'transformers.min.js'],
  ['@huggingface/transformers/LICENSE', 'LICENSE.transformers.txt'],
  ['onnxruntime-web/dist/ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.mjs'],
  ['onnxruntime-web/dist/ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.wasm'],
];

await mkdir(DEST, { recursive: true });
for (const [from, to] of FILES) {
  await copyFile(new URL(from, NM), new URL(to, DEST));
  console.log(`vendor/${to}`);
}
await writeFile(new URL('VERSIONS.json', DEST), JSON.stringify({
  '@huggingface/transformers': `${tjs.version} (Apache-2.0, see LICENSE.transformers.txt)`,
  'onnxruntime-web': `${ort.version} (MIT, Copyright (c) Microsoft Corporation; the npm package ships no LICENSE file)`,
}, null, 2) + '\n');
console.log(`transformers.js ${tjs.version}, onnxruntime-web ${ort.version}`);
