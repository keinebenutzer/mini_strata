# LESEN.md — Mini-Strata: ein Befehl, alles andere automatisch

## Der automatische Weg (genau wie beim Original)

```bash
git clone <fork-url> mini_strata
cd mini_strata
./setup.sh --family qwen36 --model UD-IQ2_XXS --yes
```

Das ist alles. Die Routine führt direkt zu unserem Modell
**Qwen3.6-35B-A3B-UD-IQ2_XXS** (11,8 GB, Unsloth-Repo, SHA-256-geprüft):

1. Python-Umgebung und Pakete anlegen
2. **Engine kompilieren** — dieser Fork baut ihn selbst (seine Kernel nehmen
   die 2-Bit-Experten; der fertig gelieferte Engine kann das nicht)
3. Modell automatisch herunterladen
4. **Gewichte bauen**: der MTP-Block (blk.40) kommt als Q2_K/Q3_K daher,
   die Engine-Kernel wollen Q8_0 — `terminal/tools/build_mq8.py` hebt die drei
   Experten-Tensoren auf Q8_0 und schreibt die Datei neu (`.mq8.gguf`, +0,6 GB)
5. Tokenizer aus der GGUF erzeugen
6. Config schreiben (auf Karten bis 6,5 GB VRAM setzt sie `--expert-vram`
   auf das gemessene 4-GB-Rezept: dichte Seite auf der Karte, Experten aus dem RAM)
7. Server starten — API danach auf `http://127.0.0.1:<port>/v1`

Ohne `--yes` führt der interaktive Assistent durch dieselben Schritte
(Größe, Kontext, Bilder werden gefragt, Empfehlungen vorselektiert).
Auf Rechnern mit 16–31 GB RAM — den Zielrechnern dieses Forks — ist
**Qwen3.6-35B-A3B UD-IQ2_XXS** die vorausgewählte Empfehlung; die
Flash-Next-Größen, die dort nicht in den RAM passen, stehen weiterhin zur
Auswahl. Auf Karten bis 6,5 GB VRAM schreibt die Routine `--prefill 256`
in die Config (`auto` = 512 OOMt dort mit Engine 0.1.67).
Nächster Start: einfach `./setup.sh` — es startet sofort.

**Voraussetzungen:** NVIDIA-Karte (sm_75 oder neuer), CUDA-Compiler
(`nvcc`) und `g++` für den Engine-Bau, ~25 GB freier Festplatte
(11,8 + 12,4 GB Modell), 16 GB RAM.

## Der manuelle Weg (für Paketbauer und Sonderfälle)

### Bauen

```bash
cmake -B build -S engine \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86        # 75 fuer aeltere Karten
cmake --build build --target strata -j"$(nproc)"
LD_LIBRARY_PATH=<cuda-lib-dir> ./build/strata        # Smoke: muss "sm_86" melden
```

Ohne `-DSTRATA_ENABLE_CUDA=ON` gibt es gar kein `strata`-Ziel; auf manchen
Rechnern findet CMake `nvcc` nur mit gesetztem `CUDACXX=/pfad/zu/nvcc`.

### Gewichte

```bash
python3 terminal/tools/build_mq8.py Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf Qwen3.6-35B-A3B-UD-IQ2_XXS.mq8.gguf
```

### Starten

Config-Kopie von `mini/serve-config-template.json` machen, Pfade und
`allowed_hosts` anpassen, dann:

```bash
nohup python3 terminal/serve/server.py --engine strata --config meine-config.json --port 11435 > run.log 2>&1 &
```

Läuft es nicht: die Engine-Logdatei (Feld `log` in der Config) lesen — sie nennt
die entscheidende Zeile, nicht der Wrapper-Traceback.

## KV-Modus turbo3

Der Engine-Start kennt neben `q4_0` das 3-Bit-uniforme KV-Format `--kv turbo3`
(8 Stufen, 14 Byte pro 32-Block, 448 Byte pro Zelle). Es senkt den KV-Speicher auf
0,44 KiB/Token/Layer statt 0,56 bei `q4_0` (rund 21 % Ersparnis) und ist damit der
Knopf für das maximale Kontextfenster. Auf klapp (RTX A2000 4 GB,
Qwen3.6-35B-A3B-UD-IQ2_XXS, Kontext 32768) misst es ohne MTP 34,9 Token/s; mit MTP2
ist es derzeit im Nachteil (17–20 Token/s). Produktivstandard bleibt daher
`q4_0` mit MTP2.

## Upstream-Update

`git fetch <upstream> && git merge <upstream-tag>` — danach die Fork-Stellen
nachtragen (siehe MINI.md, Abschnitt „Stand halten“), neu bauen, fertig.
