# GGUF-Header-Parser (GGUF-Typenummern aus gguf.h: 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,8 string,9 array,10 u64,11 i64)
import struct, sys

def parse_header(path):
    with open(path, 'rb') as f:
        buf = f.read(32*1024*1024)   # Header + Tensor-Infos passen in 32 MB
    assert buf[:4] == b'GGUF'
    ver = struct.unpack_from('<I', buf, 4)[0]
    n_tensors_hdr = struct.unpack_from('<Q', buf, 8)[0]
    nkv = struct.unpack_from('<Q', buf, 16)[0]
    P = [24]
    def u8():
        v = buf[P[0]]; P[0] += 1; return v
    def u32():
        v = struct.unpack_from('<I', buf, P[0])[0]; P[0] += 4; return v
    def u64():
        v = struct.unpack_from('<Q', buf, P[0])[0]; P[0] += 8; return v
    def s():
        n = u64(); v = buf[P[0]:P[0]+n].decode('utf-8', 'replace'); P[0] += n; return v
    meta = {}
    for i in range(nkv):
        k = s(); t = u32()
        if t == 8: v = s()
        elif t in (0, 1, 7): v = u8()
        elif t == 2: v = struct.unpack_from('<H', buf, P[0])[0]; P[0] += 2
        elif t == 3: v = struct.unpack_from('<h', buf, P[0])[0]; P[0] += 2
        elif t == 4: v = u32()
        elif t == 5: v = struct.unpack_from('<i', buf, P[0])[0]; P[0] += 4
        elif t == 6: v = struct.unpack_from('<f', buf, P[0])[0]; P[0] += 4
        elif t in (10, 11): v = u64()
        elif t == 9:
            et = u32(); c = u64()
            if et == 8:
                v = [s() for _ in range(c)]
            else:
                sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8}.get(et, 1)
                v = f'[{c} x typ{et}]'; P[0] += c*sz
        else: v = '?'
        meta[k] = v
    tensors = []
    info_start = P[0]
    for i in range(n_tensors_hdr):
        name = s(); nd = u32()
        dims = [u64() for _ in range(nd)]
        t = u32(); o = u64()
        tensors.append((name, t, dims, o))
    align = meta.get('general.alignment', 32)
    if not isinstance(align, int) or align <= 0: align = 32
    data_start = (P[0] + align - 1) // align * align   # Engine: data_start = align(info_end)
    return ver, meta, tensors, data_start, info_start

if __name__ == '__main__':
    TYPES = {0:'F32',8:'Q8_0',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',16:'IQ2_XXS',18:'IQ3_XXS',20:'IQ4_NL',22:'IQ2_S',23:'IQ4_XS'}
    BS = {8:34,10:84,11:76,12:144,13:176,16:66,18:98,20:130,22:82,23:122}
    ver, meta, tensors, ds = parse_header(sys.argv[1])
    filt = sys.argv[2] if len(sys.argv) > 2 else 'blk.40'
    print('data_start', ds)
    for name, t, dims, o in tensors:
        if filt in name:
            n = 1
            for d in dims: n *= d
            print(f'{name:55s} {TYPES.get(t,"t"+str(t)):8s} dims={dims} n={n} offset={o} bytes={BS.get(t,0)*(n//256)}')
