# Mini-Strata — Strata auf 4-GB-Grafikkarten

Ein Fork von [Strata](https://github.com/Niko1221/Strata) (MIT), geschnitten auf
**v0.2.18 (Engine 0.1.64)**, für Rechner mit kleiner VRAM-Karte und viel Arbeitsspeicher:
ein 35-Milliarden-MoE-Modell läuft lokal auf einer 4-GB-Grafikkarte.

## Was hier anders ist

Der q36-Pfad (`engine/src/qwen35`) der Engine lehnt 2-Bit-Expertenpakete ab
(„experts are IQ2_XXS/IQ2_S, which the native expert kernels do not take“), weil die
Down-Projektions-Kernel nur IQ4- und K-Quants kennen. Dieser Fork erweitert die
Typ-Liste `STRATA_D_FMTS` in `engine/src/kernels/cuda/iq_kernels.cu` um
**IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S** — damit passen die kleinen
Unsloth-UD-Pakete (Qwen3.6-35B-A3B, ~12,4 GB Datei) in die native Expertenschleife.
Mehr ändert der Fork nicht; alles andere ist Upstream.

## So fühlt es sich an (gemessen)

Maschine: RTX A2000 Laptop 4 GB (sm_86), 30 GB RAM, 8 Kerne, CUDA 12.8.
Modell: `Qwen3.6-35B-A3B-UD-IQ2_XXS` mit MTP-Block, Kontext 13312.

| Messgröße | Wert |
|---|---|
| Prefill | ~99 Token/s |
| Decode, kurzer Prompt | 48 Token/s |
| Decode, 8k-Prompt | 19 Token/s |
| GPU-Belegung | ~1,7 GiB Modell + Puffer (Karte bleibt unter 4 GiB) |
| Experten im RAM | 10,1 GiB pinned host memory |

Die Experten liegen komplett im Arbeitsspeicher; die Karte hält Dichte
(Embeddings, Attention, Output) und die Puffer. Entscheidend ist daher **freier RAM
(ab ~16 GB) und RAM-Bandbreite**, nicht VRAM.

## Voraussetzungen

- NVIDIA-Grafikkarte mit 4 GB VRAM, Compute-Fähigkeit 8.6 (gemessen) oder 7.5 (baubar)
- CUDA-12.x-Treiber; die Laufzeitbibliotheken (`libcudart.so.12`, `libcublas.so.12`,
  `libcublasLt.so.12`) müssen im `lib_dirs`-Pfad der Config stehen
- freier Arbeitsspeicher ab ~16 GB (Experten + Puffer), 8 physische Kerne empfohlen
- ~13 GB Platz auf der Zielmaschine für das Modell

## Modell holen

Bevor du lädst, die Datei prüfen: sie muss einen MTP-Block enthalten
(`--mtp` funktioniert sonst nicht). Der Scanner dafür liegt bei:
`mini/q36_scan.py DATEI` (läuft auch gegen die ersten ~16 MB einer Teildatei).

Bevorzugt die **MTP-Fassung** des Unsloth-Repos (gleicher Modellname, ~1 GB größer);
die Nicht-MTP-Fassung läuft auf diesem Pfad nicht. Download immer direkt auf der
Zielmaschine, z. B. `uvx hf download hf://<REPO>/<DATEI> --local-dir ~/models`.

## Bauen

```bash
cmake -B build -S engine \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86        # 75 fuer aeltere Karten
cmake --build build --target strata -j"$(nproc)"
```

Smoke-Test vor dem Modell: `LD_LIBRARY_PATH=<libdir> ./build/strata` muss die
`sm_86 …`-Zeile melden.

## Tokenizer erzeugen (einmal pro Familie)

```bash
uv run --with regex -- python3 terminal/tools/strata_tokenizer.py --gguf MODEL.gguf --out tokenizer_q36/tokenizer
```

## Starten

Direkt:

```bash
LD_LIBRARY_PATH=<libdir> ./build/strata --serve \
  --native ~/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  --max-context 13312 --prefill auto --mtp 2 \
  --cpu-threads 7 --expert-vram 0.3 --hybrid-rows 64
```

oder über den Serve mit Config (`mini/serve-config-template.json`):

```bash
nohup python3 terminal/serve/server.py --engine strata --config meine-config.json --port 11435 > run.log 2>&1 &
```

Fehlt etwas, zeigt der Wrapper einen `ModuleNotFoundError` — dann das genannte
Paket (`serve`, `strata_control`, `tools/strata_tokenizer.py`) an die vom Import
erwartete Stelle kopieren.

## Die wichtigsten Knöpfe

- `--max-context`: passt der Kontext nicht, **druckt die Engine selbst den größten
  passenden Wert** — diesen Wert benutzen, nicht rechnen. Achtung: auch ein
  KV-seitig passender Kontext kann am Expert-Stapel-Puffer `E.ecpu`
  (~268 MiB, wächst nur mit `--chunk`) scheitern. In 1k-Schritten antasten.
- `--cpu-threads`: nie mehr als physische Kerne (7 bei 8 Kernen). Der Experte-
  Durchsatz ist ram-bandbreitenbegrenzt, mehr Threads bringen fast nichts.
- `--expert-vram`: erst ab genug Budget (ganze Experten) wirksam; die Zeile
  „hybrid experts: X of 256 per layer in VRAM“ zeigt, ob wirklich welche platzierten.
  Auf 4 GB ist `0` normal.
- `--mtp 2`: wirkt nur bei greedy (Temperatur 0) und lohnt erst ab ~400 generierten
  Token (+16 % bei 400, kein Effekt bis 120; `--mtp 3` war schon schlechter).
- Ohne API-Key in der Config müssen alle Hostnamen, unter denen der Server
  angesprochen wird, in `allowed_hosts` stehen — sonst antwortet der
  DNS-Rebinding-Schutz mit 403.

## known pitfalls

- Nach Suspend kann der NVIDIA-Treiber die Karte aufhängen: `nvidia-smi` antwortet
  noch, aber CUDA-Programme fallen still auf CPU zurück (Diagnose:
  `ctypes.CDLL('libcuda.so.1').cuInit(0)`; Abhilfe: Neustart).
- Identische Dateinamen existieren in mehreren HF-Repos mit unterschiedlichem
  Inhalt — vor dem Akzeptieren Header-Scan (Größe + MTP-Zahl) laufen lassen.

## Stand halten

Der Fork steht auf `v0.2.18` (`6e38f588a`). Für ein Update: Upstream-Zweig aktualisieren,
dann die zwei Fork-Stellen nachtragen — `STRATA_D_FMTS` in
`engine/src/kernels/cuda/iq_kernels.cu` (Typ-Liste, beide `#ifdef`-Zweige) und, falls
Upstream die Handoff-Region in `engine/src/program/generate.cpp` erneut umschreibt,
diese mit der Fork-Handoff-Zeile (`hand_host`) zusammenführen.
