// img2threejs — Görselden renkli 3D modele, tarayıcıda (sunucusuz, kotasız)
// Kullanım:
//   import { imageToGLB, imageToMesh, setProgressLogger } from "./src/index.js";
//   const glb = await imageToGLB(fileOrBlob);   // → Blob ("model/gltf-binary")
//   const mesh = await imageToMesh(fileOrBlob); // → THREE.Mesh (kendi sahnenizde)
//
// WebGPU varsa AI modelleri WebGPU'da, yoksa WASM'da çalışır.
// Modeller (ilk kullanımda iner, sonra tarayıcı önbelleğinden):
//   - onnx-community/depth-anything-v2-small  (~19–27 MB)
//   - Xenova/modnet (background-removal)      (~7–13 MB)

import * as THREE from "three";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";

const TJS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

let tjs = null, depthPipe = null, cutPipe = null;
let progressLogger = null;

export function setProgressLogger(fn) { progressLogger = fn; }
function notify(msg) { if (progressLogger) progressLogger(msg); }

async function loadTJS() {
  if (tjs) return tjs;
  tjs = await import(TJS_URL);
  return tjs;
}

async function getDepthPipe() {
  if (depthPipe) return depthPipe;
  const { pipeline } = await loadTJS();
  const MODEL = "onnx-community/depth-anything-v2-small";
  const opts = {
    progress_callback: (p) => {
      if (p && p.status === "progress")
        notify(`depth modeli: ${Math.round(p.loaded / 1e6)}/${p.total ? Math.round(p.total / 1e6) + " MB" : "?"}`);
    }
  };
  if (navigator.gpu) {
    for (const dtype of ["q4f16", "fp16"]) {
      try {
        depthPipe = await pipeline("depth-estimation", MODEL, { ...opts, device: "webgpu", dtype });
        notify(`WebGPU hazır (depth, ${dtype})`);
        return depthPipe;
      } catch (e) { /* sıradaki dtype */ }
    }
    notify("WebGPU kullanılamadı — WASM'a geçiliyor");
  }
  for (const dtype of ["q8", "fp32"]) {
    try {
      depthPipe = await pipeline("depth-estimation", MODEL, { ...opts, device: "wasm", dtype });
      notify(`WASM hazır (depth, ${dtype})`);
      return depthPipe;
    } catch (e) { /* sıradaki dtype */ }
  }
  throw new Error("Yerel derinlik modeli yüklenemedi");
}

async function getCutPipe() {
  if (cutPipe) return cutPipe;
  const { pipeline } = await loadTJS();
  const opts = {
    progress_callback: (p) => {
      if (p && p.status === "progress")
        notify(`kesme modeli: ${Math.round(p.loaded / 1e6)}/${p.total ? Math.round(p.total / 1e6) + " MB" : "?"}`);
    }
  };
  if (navigator.gpu) {
    try {
      cutPipe = await pipeline("background-removal", "Xenova/modnet", { ...opts, device: "webgpu", dtype: "fp16" });
      return cutPipe;
    } catch (e) { /* wasm'a düş */ }
  }
  cutPipe = await pipeline("background-removal", "Xenova/modnet", { ...opts, device: "wasm", dtype: "q8" });
  return cutPipe;
}

/**
 * imageToMesh — File/Blob/URL → Three.js Mesh (texture'lı rölyef)
 * @param {File|Blob|string} input
 * @param {object} [opts] { gridN=150, relH=0.55 }
 * @returns {Promise<THREE.Mesh>}
 */
export async function imageToMesh(input, opts = {}) {
  const gridN = opts.gridN || 150;
  const relH = opts.relH ?? 0.55;

  const { RawImage } = await loadTJS();
  const img = typeof input === "string"
    ? await RawImage.fromURL(input)
    : await RawImage.fromBlob(input);
  notify("Görsel işleniyor…");

  const cutOut = await (await getCutPipe())(img);
  const cut = Array.isArray(cutOut) ? cutOut[0] : cutOut;
  if (!cut || !cut.width) throw new Error("Ön plan kesme başarısız — görselde ana nesne belirgin mi?");

  const depthRes = await (await getDepthPipe())(img);
  let depthImg = depthRes.depth;

  const W = cut.width, H = cut.height;
  if (depthImg.width !== W || depthImg.height !== H) depthImg = depthImg.resize(W, H);

  const maskData = cut.data;
  const dData = depthImg.data;

  // Yumuşatılmış alpha
  const softA = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4 + 3;
      let a = maskData[i];
      if (x > 0) a += maskData[i - 4];
      if (x < W - 1) a += maskData[i + 4];
      if (y > 0) a += maskData[i - 4 * W];
      if (y < H - 1) a += maskData[i + 4 * W];
      softA[y * W + x] = a / 5;
    }
  }

  // Konu bbox
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
      if (softA[y * W + x] > 90) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
  if (x1 <= x0 || y1 <= y0) throw new Error("Ön plan bulunamadı — görselde ana nesne belirgin mi?");
  const padX = Math.max(4, Math.round((x1 - x0) * 0.03));
  const padY = Math.max(4, Math.round((y1 - y0) * 0.03));
  x0 = Math.max(0, x0 - padX); y0 = Math.max(0, y0 - padY);
  x1 = Math.min(W - 1, x1 + padX); y1 = Math.min(H - 1, y1 + padY);

  // Konu derinlik aralığı
  let dMin = 1e9, dMax = -1e9;
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++)
      if (softA[y * W + x] > 150) {
        const d = dData[y * W + x];
        if (d < dMin) dMin = d;
        if (d > dMax) dMax = d;
      }
  if (!(dMax > dMin)) { dMin = 0; dMax = 255; }

  const bw = x1 - x0, bh = y1 - y0;
  const pos = [], uv = [], idx = [];
  const alphaCols = new Float32Array((gridN + 1) * (gridN + 1));

  let vi = 0;
  for (let j = 0; j <= gridN; j++) {
    for (let i2 = 0; i2 <= gridN; i2++) {
      const fx = x0 + (i2 / gridN) * bw;
      const fy = y0 + (j / gridN) * bh;
      const cx = Math.round(fx), cy = Math.round(fy);
      let a = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const ok = cy + dy >= 0 && cy + dy < H && cx + dx >= 0 && cx + dx < W;
          if (ok) { a += softA[(cy + dy) * W + (cx + dx)]; n++; }
        }
      a /= n; alphaCols[j * (gridN + 1) + i2] = a;

      const cxi = Math.max(0, Math.min(W - 1, cx)), cyi = Math.max(0, Math.min(H - 1, cy));
      const dRaw = dData[cyi * W + cxi];
      const t = Math.max(0, Math.min(1, (dRaw - dMin) / (dMax - dMin)));
      pos.push(
        fx / W - (x0 + x1) / (2 * W),
        -(fy / H - (y0 + y1) / (2 * H)) * (bh / bw),
        t * relH
      );
      uv.push(fx / W, 1 - fy / H);
      vi++;
    }
  }

  for (let j = 0; j < gridN; j++) {
    for (let i2 = 0; i2 < gridN; i2++) {
      const a1 = alphaCols[j * (gridN + 1) + i2], a2 = alphaCols[j * (gridN + 1) + i2 + 1],
        a3 = alphaCols[(j + 1) * (gridN + 1) + i2], a4 = alphaCols[(j + 1) * (gridN + 1) + i2 + 1];
      if (a1 > 120 && a2 > 120 && a3 > 120 && a4 > 120) {
        const tl = j * (gridN + 1) + i2, tr = tl + 1, bl = (j + 1) * (gridN + 1) + i2, br = bl + 1;
        idx.push(tl, bl, tr, tr, bl, br);
      }
    }
  }
  if (!idx.length) throw new Error("Mesh yüzü oluşamadı — daha sade arka planlı görsel deneyin");

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();

  const texCanvas = cut.toCanvas();
  const tex = new THREE.CanvasTexture(texCanvas);
  tex.colorSpace = THREE.SRGBColorSpace;

  const mat = new THREE.MeshStandardMaterial({
    map: tex,
    transparent: true,
    alphaTest: 0.4,
    side: THREE.DoubleSide,
    roughness: 0.85,
    metalness: 0.0,
    emissive: 0xffffff,
    emissiveIntensity: 0.18,
    emissiveMap: tex
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = "img2threejs_relief";
  notify("Mesh hazır (texture'lı)");
  return mesh;
}

/**
 * imageToGLB — File/Blob/URL → renkli .glb Blob
 * @param {File|Blob|string} input
 * @param {object} [opts] { gridN=150, relH=0.55 }
 * @returns {Promise<Blob>}
 */
export async function imageToGLB(input, opts = {}) {
  const mesh = await imageToMesh(input, opts);
  const group = new THREE.Group();
  group.add(mesh);
  notify("GLB paketleniyor…");
  return await new Promise((resolve, reject) => {
    new GLTFExporter().parse(
      group,
      (r) => resolve(new Blob([r], { type: "model/gltf-binary" })),
      (e) => reject(e),
      { binary: true }
    );
  });
}

export default { imageToGLB, imageToMesh, setProgressLogger };
