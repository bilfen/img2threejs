# img2threejs — Görselden Renkli 3D Modelo, Tarayıcıda

**Sunucusuz · Kotasız · Ücretsiz · Özgür (MIT)**

Bir görseli (çizim, fotoğraf, logo) **tarayıcıda**, kullanıcının kendi cihazında, **renkli 3D modele (.glb)** dönüştürür. Hiçbir sunucuya veri gitmez, GPU kotası/bekleme/anahtar yoktur. İlk üretimde ~26 MB'lık iki küçük AI modeli (derinlik + ön plan kesme) indirilir ve tarayıcı önbelleğine alınır; sonraki tüm üretimler ağsız çalışır.

> Bu proje, img2threejs adlı AI-ajan-is akışından ilhamla, tarayıcıya taşınmış **gerçek bir kütüphane**dir: `import { imageToGLB }` → Blob. Demo: https://bilfen.github.io/img2threejs/

## Kurulum

Kütüphane saf ESM'dir; npm'e gerek yok. Three.js ile birlikte importmap kullanın:

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
  import { imageToGLB, setProgressLogger } from "https://bilfen.github.io/img2threejs/src/index.js";
  // veya kendi sunucunuzdan / npm'den: "./src/index.js"
</script>
```

## API

### `imageToGLB(input, opts?) → Promise<Blob>`
| Param | Açıklama |
|---|---|
| `input` | `File \| Blob \| URL-string` |
| `opts.gridN` | Grid çözünürlüğü (default 150; yüksek = detaylı, yavaş) |
| `opts.relH` | Rölyef yüksekliği (default 0.55) |

### `imageToMesh(input, opts?) → Promise<THREE.Mesh>`
Texture'lı mesh (GLB'ye dönmeden kendi sahnenizde kullanmak için).

### `setProgressLogger(fn)`
Model indirme/işleme durum mesajları için (`fn(message: string)`).

## Nasıl çalışır?

1. **Ön plan kesme**: `Xenova/modnet` (background-removal pipeline) arka planı siler → RGBA öne-çıkan nesne.
2. **Derinlik tahmini**: `onnx-community/depth-anything-v2-small` gri derinlik haritası üretir.
3. **Mesh**: 150×150 grid geometri; öne plan maskesindeki pikseller derinliğe göre z'ye taşınır (rölyef).
4. **Texture**: kesilmiş ön plan görseli mesh'e renk haritası olarak sarılır → sonuç **renklidir**.
5. **GLB**: `GLTFExporter` ile ikili `.glb` blob üretir.

Hepsi transformers.js v3 ile tarayıcıda: **WebGPU** varsa (çoğu Chrome/Edge) saniyeler içinde, yoksa **WASM** ile birkaç kat yavaş ama yine çalışır.

## Sınırlar

- Sonuç *rölyef* tarzıdır (ön + arka yüz aynı renk); tam hacimsel AI mesh'i değil. Tam hacimsel HD model için Hunyuan3D-2 gibi servis kullanılmalı (kota/sunucu gerekir).
- İlk üretim ~26 MB model indirmesi: öğrenciler ilk tıklamada biraz bekler, sonrasında hızlıdır.
- Çok küçük/low kontrast nesnelerde masking zayıf olabilir; **sade/düz arka planlı görseller** en iyi sonucu verir.

## Demo

`demo/` klasöründeki sayfada drop-area ile görsel bırakın → 3D gösterim + **GLB indir** bağlantısı.

## Katkı / Lisans

MIT. Saf tarayıcı teknolojisi (WebGPU, WASM, three.js) kullanır; model dosyaları Hugging Face CDN'inden indirilir, veri işlenmez/saklanmaz.
