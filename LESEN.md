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
in die Config (`auto` = 512 OOMt dort mit Engine 0.1.66).
Nächster Start: einfach `./setup.sh` — es startet sofort.

**Voraussetzungen:** NVIDIA-Karte (sm_75 oder neuer), CUDA-Compiler
(`nvcc`) und `g++` für den Engine-Bau, ~25 GB freier Festplatte
(11,8 + 12,4 GB Modell), 16 GB RAM.

**nvcc auffindbar machen:** die Routine sucht den Compiler nur in PATH,
`$CUDA_HOME`/`$CUDA_PATH`, `/usr/local/cuda*` und `/opt/cuda*`. Ein aus
einem Tarball ausgepacktes Toolkit (z. B. `~/cuda-12.8/cuda-12.8`) steht
daneben und muss genannt werden:

```bash
CUDA_HOME=$HOME/cuda-12.8/cuda-12.8 ./setup.sh --family qwen36 --model UD-IQ2_XXS --yes
# oder: STRATA_NVCC=$HOME/cuda-12.8/cuda-12.8/bin/nvcc ./setup.sh ...
```

Für sm_86 genügt CUDA ≥ 12.0; CUDA 13.0 verlangt der Bau nur für RTX 50
(sm_120 — nvcc 13.2.0/13.2.1 kompiliert dort die Kernel falsch). Ist
NVIDIAs apt-Quelle deaktiviert (`cuda-*.list.distUpgrade` nach einem
Dist-Upgrade), findet die Automatik kein `cuda-toolkit-13-0`; ein
vorhandenes Toolkit per `CUDA_HOME` ist dann der kürzere Weg.
Liegen Modell und `.mq8.gguf` schon im Ordner (Namen wie im HF-Repo), zeigt
`--gguf-dir <ordner>` der Routine die Dateien — sie lädt und baut sie dann
nicht neu.

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

## Upstream-Update

`git fetch <upstream> && git merge <upstream-tag>` — danach die Fork-Stellen
nachtragen (siehe MINI.md, Abschnitt „Stand halten“), neu bauen, fertig.
