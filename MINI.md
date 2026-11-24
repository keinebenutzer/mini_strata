# Mini-Strata — Strata auf 4-GB-Grafikkarten

Ein Fork von [Strata](https://github.com/Niko1221/Strata) (MIT), geschnitten auf
**v0.2.21 (Engine 0.1.67)**, für Rechner mit kleiner VRAM-Karte und viel Arbeitsspeicher:
ein 35-Milliarden-MoE-Modell läuft lokal auf einer 4-GB-Grafikkarte.

## Was hier anders ist

Der q36-Pfad (`engine/src/qwen35`) der Engine lehnt 2-Bit-Expertenpakete ab
(„experts are IQ2_XXS/IQ2_S, which the native expert kernels do not take“), weil die
Down-Projektions-Kernel nur IQ4- und K-Quants kennen. Dieser Fork erweitert die
Typ-Liste `STRATA_D_FMTS` in `engine/src/kernels/cuda/iq_kernels.cu` um
**IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S** — damit passen die kleinen
Unsloth-UD-Pakete (Qwen3.6-35B-A3B, ~12,4 GB Datei) in die native Expertenschleife.
Dazu macht der Fork die 2-Bit-Größe in `setup.sh`/`setup.py` wählbar (mit
Automatik-Bau der mq8-Gewichte, `terminal/tools/build_mq8.py`) — Upstream bietet qwen36 nur
ab 4-Bit an. Sonst ändert der Fork nichts; alles andere ist Upstream.

## So fühlt es sich an (gemessen)

Maschine: RTX A2000 Laptop 4 GB (sm_86), 30 GB RAM, 8 Kerne, CUDA 12.8.
Produktivstandard: `Qwen3.6-35B-A3B-UD-IQ2_XXS` mit MTP-Block, q4_0-KV mit MTP2,
Kontext 32768.

| Messgröße | Wert |
|---|---|
| Modell | Qwen3.6-35B-A3B UD-IQ2_XXS (12,4 GB, mq8-Gewichte, MTP-Experten) |
| Kontext | 32768 |
| Prefill | 256 (Batch; `auto` = 512 OOMt auf 4 GB) |
| Decode (q4_0 + MTP2) | 35–38 Token/s |

Die Experten liegen komplett im Arbeitsspeicher; die Karte hält Dichte
(Embeddings, Attention, Output) und die Puffer. Entscheidend ist daher **freier RAM
(ab ~16 GB) und RAM-Bandbreite**, nicht VRAM.

## KV-Modus turbo3 (neu)

Der q36-Pfad kennt neben q4_0 jetzt ein 3-Bit-uniformes KV-Format, wählbar beim
Engine-Start mit `--kv turbo3` (8 Stufen, 14 Byte pro 32-Block, 448 Byte pro Zelle).
Es senkt den KV-Speicher auf **0,44 KiB/Token/Layer** statt 0,56 bei q4_0
(rund 21 % Ersparnis) — der Knopf für das maximale Kontextfenster.

Gemessen auf klapp (RTX A2000 4 GB, Qwen3.6-35B-A3B-UD-IQ2_XXS, Kontext 32768):
Decode ohne MTP 34,9 Token/s. Mit MTP2 ist turbo3 derzeit im Nachteil
(17–20 Token/s). **Produktivstandard bleibt daher q4_0 mit MTP2**; turbo3 ist die
Wahl, wenn das Kontextfenster wichtiger ist als der letzte Durchsatz.

## Voraussetzungen

- NVIDIA-Grafikkarte mit 4 GB VRAM, Compute-Fähigkeit 8.6 (gemessen) oder 7.5 (baubar)
- CUDA-12.x-Treiber; die Laufzeitbibliotheken (`libcudart.so.12`, `libcublas.so.12`,
  `libcublasLt.so.12`) müssen im `lib_dirs`-Pfad der Config stehen
- freier Arbeitsspeicher ab ~16 GB (Experten + Puffer), 8 physische Kerne empfohlen
- ~25 GB Platz auf der Zielmaschine (Modell 11,8 GB + die gebaute mq8-Fassung 12,4 GB)
- `g++` und `nvcc` (CUDA 12.x) für den Engine- und Gewichts-Bau

## Der automatische Weg (empfohlen)

```bash
./setup.sh --family qwen36 --model UD-IQ2_XXS --yes
```

Ein Befehl: Umgebung, Engine-Kompilierung, Modell-Download (SHA-256-geprüft),
mq8-Gewichts-Bau, Tokenizer, Config und Start — Einzelheiten in LESEN.md.
Die folgenden Abschnitte sind der Handbetrieb für Sonderfälle.

## Modell holen

Bevor du lädst, die Datei prüfen: sie muss einen MTP-Block enthalten
(`--mtp` funktioniert sonst nicht). Der Scanner dafür liegt bei:
`mini/q36_scan.py DATEI` (läuft auch gegen die ersten ~16 MB einer Teildatei).

Bevorzugt die **MTP-Fassung** des Unsloth-Repos (gleicher Modellname, ~1 GB größer);
die Nicht-MTP-Fassung läuft auf diesem Pfad nicht. Download immer direkt auf der
Zielmaschine, z. B. `uvx hf download hf://<REPO>/<DATEI> --local-dir ~/models`.
Danach die Gewichte bauen (einmal): `python3 terminal/tools/build_mq8.py DATEI DATEI.mq8.gguf`
— die Engine nimmt nur die mq8-Fassung (MTP-Experten auf Q8_0).

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
  --max-context 32768 --prefill 256 --mtp 2 \
  --cpu-threads 7 --expert-vram 0.3 --hybrid-rows 64
```

(`--prefill auto` = 512 OOMt auf 4-GB-Karten mit Engine 0.1.67:
„not enough VRAM for the hybrid experts' buffers and slots"; 256 ist das
gemessene Rezept — die setup-Routine setzt es auf Karten bis 6,5 GB selbst.)

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

Der Fork steht auf `v0.2.21`. Für ein Update: Upstream-Zweig aktualisieren,
dann die Fork-Stellen nachtragen — `STRATA_D_FMTS` in
`engine/src/kernels/cuda/iq_kernels.cu` (Typ-Liste, beide `#ifdef`-Zweige), die
Handoff-Zeile (`hand_host`) in `engine/src/program/generate.cpp` falls Upstream die
Region umschreibt, und die Setup-Automatik: `QWEN36_FILES` + `models` (UD-IQ2_XXS)
und die mq8-Bau-/`--expert-vram`-Blöcke in `terminal/setup.py`,
`BASE_GIB_MODEL`/`HYBRID_MIN_GIB` in `terminal/strata_control/own_path_fit.py`,
die Werkzeuge unter `tools/` (`build_mq8.py`, `mtp_requant.cpp`, `mtp_splice.py`,
`gguf_hdr.py`).
