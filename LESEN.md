# LESEN.md — Mini-Strata in vier Schritten

## 1. Holen

```bash
git clone <fork-url> mini_strata
cd mini_strata
```

## 2. Bauen

```bash
cmake -B build -S engine \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86        # 75 fuer aeltere Karten
cmake --build build --target strata -j"$(nproc)"
LD_LIBRARY_PATH=<cuda-lib-dir> ./build/strata        # Smoke: muss "sm_86" melden
```

Ohne `-DSTRATA_ENABLE_CUDA=ON` gibt es gar kein `strata`-Ziel; auf manchen
Rechnern findet CMake `nvcc` nur mit gesetztem `CUDACXX=/pfad/zu/nvcc`.

## 3. Modell und Tokenizer

- Modell direkt auf diese Maschine laden (siehe MINI.md; erst `mini/q36_scan.py`
  gegen die Datei laufen lassen — MTP-Block muss da sein).
- Tokenizer einmal erzeugen:
  `uv run --with regex -- python3 terminal/tools/strata_tokenizer.py --gguf MODEL.gguf --out tokenizer_q36/tokenizer`

## 4. Starten

Config-Kopie von `mini/serve-config-template.json` machen, Pfade und
`allowed_hosts` anpassen, dann:

```bash
nohup python3 terminal/serve/server.py --engine strata --config meine-config.json --port 11435 > run.log 2>&1 &
```

Läuft es nicht: die Engine-Logdatei (Feld `log` in der Config) lesen — sie nennt
die entscheidende Zeile, nicht der Wrapper-Traceback.

## Upstream-Update

`git fetch <upstream> && git merge <upstream-tag>` — danach die zwei Fork-Stellen
nachtragen (siehe MINI.md, Abschnitt „Stand halten“), neu bauen, fertig.
