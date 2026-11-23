# /// script
# requires-python = ">=3.12"
# ///
"""GGUF-Scan fuer die qwen35moe/q36-Familie: Geometrie, Typ-Mix und Bytes je Kategorie,
MTP-Tensoren, native-Expert-Tauglichkeit. Laeuft auch auf einer Kopf-Datei (dd-Pipe der
ersten ~16 MB reicht, der Header endet bei ~11 MB). Aufruf: uv run q36_scan.py DATEI"""
import struct, sys
from collections import Counter, defaultdict

DT = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",
      12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",
      20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",29:"IQ1_M",30:"BF16"}
# Bytes pro Element (Superblock-Tabellen)
BB = {8:34/32,10:104/256,11:128/256,12:144/256,13:176/256,14:210/256,16:66/256,17:74/256,
      18:98/256,20:130/256,21:162/256,22:82/256,23:122/256,0:4,1:2,30:2}

def scan(path):
    with open(path,"rb") as f:
        buf = f.read(300*1024*1024)
    P=[24]
    def u8(): v=buf[P[0]]; P[0]+=1; return v
    def u32(): v=struct.unpack_from("<I",buf,P[0])[0]; P[0]+=4; return v
    def u64(): v=struct.unpack_from("<Q",buf,P[0])[0]; P[0]+=8; return v
    def s(): n=u64(); v=buf[P[0]:P[0]+n].decode("utf-8",errors="replace"); P[0]+=n; return v
    kv={}
    for _ in range(struct.unpack_from("<Q",buf,16)[0]):
        k=s(); t=u32()
        if t==8: v=s()
        elif t in (0,1,7): v=u8()
        elif t==2: v=struct.unpack_from("<H",buf,P[0])[0]; P[0]+=2
        elif t==3: v=struct.unpack_from("<h",buf,P[0])[0]; P[0]+=2
        elif t==4: v=u32()
        elif t==5: v=struct.unpack_from("<i",buf,P[0])[0]; P[0]+=4
        elif t==6: v=round(struct.unpack_from("<f",buf,P[0])[0],4); P[0]+=4
        elif t==9:
            et=u32(); c=u64()
            if et==8:
                vals=[s() for _ in range(c)]   # consume ALWAYS, store only small ones
                v=vals if c<10 else f"[{c} strings]"
            else:
                step={4:4,5:4,6:4,7:1,2:2,3:2,0:1,1:1}[et]
                v=[struct.unpack_from("<i",buf,P[0]+j*step)[0] for j in range(min(c,8))] if et==5 else f"[{c} x typ{et}]"
                P[0]+=step*c
        else: raise SystemExit(f"KV typ {t} bei {k}")
        kv[k]=v
    n_tensors=struct.unpack_from("<Q",buf,8)[0]
    tensors=[]
    for _ in range(n_tensors):
        name=s(); nd=u32(); dims=[u64() for _ in range(nd)]; dt=u32(); off=u64()
        tensors.append((name,dims,dt,off))
    return kv,tensors

def cat(n):
    l=n.lower()
    if "exps" in l: return "routed_experts"
    if "shared" in l: return "shared_expert"
    if "mtp" in l or "eh_proj" in l: return "MTP"
    if "attn" in l: return "attention"
    if any(x in l for x in ("ssm","conv","gdn","delta","beta","gate_a","recurrent")): return "ssm_gdn"
    if "ffn" in l or "mlp" in l: return "dense_ffn"
    if "embd" in l: return "embeddings"
    if "output" in l: return "output"
    return "sonstiges"

path=sys.argv[1]
kv,tensors=scan(path)
print(f"Datei: {path}  Tensoren: {len(tensors)}")
for k in sorted(kv):
    if "tokenizer" not in k: print(f"  {k} = {kv[k]}")
byc=defaultdict(lambda:[0,Counter(),0.0])
for name,dims,dt,off in tensors:
    c=cat(name); n=1
    for d in dims: n*=d
    byc[c][0]+=1; byc[c][1][DT.get(dt,"t"+str(dt))]+=1
    bb=BB.get(dt)
    if bb: byc[c][2]+=n*bb
print("\nKategorien:")
for c,(cnt,types,est) in sorted(byc.items()):
    print(f"  {c:16s} {cnt:4d} Tensoren  ~{est/2**30:6.2f} GiB  {dict(types.most_common(8))}")
mtp=[t[0] for t in tensors if "mtp" in t[0].lower() or "eh_proj" in t[0].lower()]
print(f"\nMTP-Tensoren: {len(mtp)}  Beispiele: {mtp[:6]}")
