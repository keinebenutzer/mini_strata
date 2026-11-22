#!/usr/bin/env python3
# build_mq8.py — der Gewichte-Bau dieses Forks fuer Qwen3.6-35B-A3B-UD-IQ2_XXS.
# Der MTP-Block (blk.40) der Unsloth-Datei fuehrt seine Experten als Q2_K/Q3_K;
# die nativen Experten-Kernel des Engines nehmen diese Typen nicht. Dieser Bau
# hebt die drei blk.40-Experten-Tensoren auf Q8_0 (dequantize + requantize,
# exakt nach ggml-quants.c) und schreibt die Datei neu: .mq8.gguf.
# Aufruf: python3 build_mq8.py <orig.gguf> <out.mq8.gguf>
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gguf_hdr import parse_header

# Blockgroessen nach Engine-Blockgeometrie (Bytes je 256 Werte), wie in mtp_splice.py
BS256 = {0:1024, 1:512, 30:512, 2:144, 3:160, 6:176, 7:192, 8:272, 9:288, 15:48,
         10:84, 11:110, 12:144, 13:176, 14:210, 16:66, 17:74, 18:98, 19:70,
         20:144, 21:110, 22:82, 23:136, 29:56}
# name -> (mtp_requant-Modus, GGUF-Typ in der Originaldatei)
REPL = {
    "blk.40.ffn_down_exps.weight": ("q3k", 11),
    "blk.40.ffn_gate_exps.weight": ("q2k", 10),
    "blk.40.ffn_up_exps.weight":   ("q2k", 10),
}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_mq8.py <orig.gguf> <out.mq8.gguf>")
    orig, outp = sys.argv[1], sys.argv[2]
    if not os.path.exists(orig):
        raise SystemExit(f"missing: {orig}")
    if os.path.exists(outp):
        print(f"mq8 already there: {outp}")
        return 0

    # 1) den Umtensorisierer bauen (nur g++, keine Fremdpakete)
    exe = os.path.join(HERE, "mtp_requant")
    if not os.path.exists(exe):
        src = os.path.join(HERE, "mtp_requant.cpp")
        print("compiling mtp_requant ...")
        subprocess.run(["g++", "-O2", "-o", exe, src], check=True)

    # 2) die drei blk.40-Tensoren ausschneiden (Offsets sind RELATIV zum Datenanfang)
    ver, meta, tensors, data_start, info_start = parse_header(orig)
    tmp = tempfile.mkdtemp(prefix="mq8_", dir=os.path.dirname(os.path.abspath(outp)) or ".")
    try:
        q8paths = {}
        with open(orig, "rb") as f:
            for name, (mode, want_type) in REPL.items():
                hit = [t for t in tensors if t[0] == name]
                if not hit:
                    raise SystemExit(f"{name} not in {orig} - is this the published IQ2_XXS MTP file?")
                _, typ, dims, off = hit[0]
                if typ != want_type:
                    raise SystemExit(f"{name}: type {typ} in the file, expected {want_type} "
                                     f"(already converted? build the mq8 from the published file)")
                n = 1
                for d in dims:
                    n *= d
                size = n * BS256[typ] // 256
                raw = os.path.join(tmp, name + ".raw")
                q8 = os.path.join(tmp, name + ".q8")
                f.seek(data_start + off)
                with open(raw, "wb") as w:
                    left = size
                    while left:
                        chunk = f.read(min(left, 8 * 1024 * 1024))
                        if not chunk:
                            raise SystemExit(f"{name}: file ends early")
                        w.write(chunk)
                        left -= len(chunk)
                subprocess.run([exe, mode, raw, q8], check=True)
                q8paths[name] = q8

        # 3) die Datei neu bauen: drei Tensoren ersetzt (Typ 8 = Q8_0), Offsets neu
        subprocess.run([sys.executable, os.path.join(HERE, "mtp_splice.py"),
                        orig, outp,
                        q8paths["blk.40.ffn_down_exps.weight"],
                        q8paths["blk.40.ffn_gate_exps.weight"],
                        q8paths["blk.40.ffn_up_exps.weight"]], check=True)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"mq8 built: {outp} ({os.path.getsize(outp)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
