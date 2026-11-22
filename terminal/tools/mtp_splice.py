# mtp_splice.py — baut GGUF mit umquantisierten blk.40-Experten (Q2_K/Q3_K -> Q8_0).
# GGUF-Konvention (llama.cpp und Strata, gguf_reader.hpp:430): Tensor-Offsets sind RELATIV
# zum Datenanfang: tensor_data = base + data_start + offset.
# Aufruf: python3 mtp_splice.py <orig.gguf> <out.gguf> <down.q8> <gate.q8> <up.q8>
import struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_hdr import parse_header

# Blockgroessen exakt nach Engine-Blockgeometrie (gguf_reader.hpp:140), Bytes je 256 Werte.
BS256 = {0:1024, 1:512, 30:512, 2:144, 3:160, 6:176, 7:192, 8:272, 9:288, 15:48,
         10:84, 11:110, 12:144, 13:176, 14:210, 16:66, 17:74, 18:98, 19:70,
         20:144, 21:110, 22:82, 23:136, 29:56}
REPL = {
    'blk.40.ffn_down_exps.weight': 11,   # Q3_K
    'blk.40.ffn_gate_exps.weight': 10,   # Q2_K
    'blk.40.ffn_up_exps.weight': 10,     # Q2_K
}

def enc_str(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b

def main():
    orig, outp, q8_down, q8_gate, q8_up = sys.argv[1:6]
    ver, meta, tensors, data_start, info_start = parse_header(orig)
    orig_size = os.path.getsize(orig)
    new_bytes = {
        'blk.40.ffn_down_exps.weight': open(q8_down, 'rb').read(),
        'blk.40.ffn_gate_exps.weight': open(q8_gate, 'rb').read(),
        'blk.40.ffn_up_exps.weight': open(q8_up, 'rb').read(),
    }
    sized = []
    for name, t, dims, off in tensors:
        n = 1
        for d in dims: n *= d
        bs = BS256.get(t)
        if bs is None:
            raise SystemExit(f'kein Blocktyp fuer {name} typ {t}')
        if (n * bs) % 256:
            raise SystemExit(f'{name}: n*bs nicht teilbar (n={n} t={t})')
        size = n * bs // 256
        if name in REPL:
            assert t == REPL[name], f'{name}: Typ {t} != erwartet {REPL[name]}'
        sized.append((name, t, dims, off, size))
    sized.sort(key=lambda x: x[3])
    payload = orig_size - data_start
    last = sized[-1]
    assert last[3] + last[4] <= payload, f'Layout unplausibel (relativ): {last[3]}+{last[4]} > {payload}'
    total = sum(s[4] for s in sized)
    assert total <= payload + 32*len(sized), f'Typ-Tabelle unplausibel: {total} vs {payload}'
    align = meta.get('general.alignment', 32)
    if not isinstance(align, int): align = 32
    # Neuer Plan: relative Offsets ab 0, in Offset-Reihenfolge des Originals
    cur = 0
    plan = []
    for name, t, dims, off, size in sized:
        if name in REPL:
            plan.append((name, 8, dims, cur, new_bytes[name], None))
            cur += len(new_bytes[name])
        else:
            plan.append((name, t, dims, cur, None, (off, size)))
            cur += size
        cur = (cur + align - 1) // align * align
    with open(orig, 'rb') as fi, open(outp, 'wb') as fo:
        fi.seek(0)
        raw_head = fi.read(info_start)   # magic, ver, n_tensors, n_kv, KV byte-treu
        fo.write(raw_head)
        for name, t, dims, off, nb, src in plan:
            fo.write(enc_str(name))
            fo.write(struct.pack('<I', len(dims)))
            for d in dims: fo.write(struct.pack('<Q', d))
            fo.write(struct.pack('<I', t))
            fo.write(struct.pack('<Q', off))
        pad = (align - (fo.tell() % align)) % align
        fo.write(b'\0' * pad)
        new_data_start = fo.tell()
        for name, t, dims, off, nb, src in plan:
            if nb is not None:
                fo.write(nb)
            else:
                soff, ssize = src
                fi.seek(data_start + soff)   # relativ -> absolut
                remaining = ssize
                while remaining:
                    chunk = fi.read(min(remaining, 8*1024*1024))
                    if not chunk: break
                    fo.write(chunk)
                    remaining -= len(chunk)
            pad = (align - (fo.tell() % align)) % align
            fo.write(b'\0' * pad)
    print('neu:', outp, 'groesse', os.path.getsize(outp), 'data_start', new_data_start)

if __name__ == '__main__':
    main()
