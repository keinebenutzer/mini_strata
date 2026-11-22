// mtp_requant.cpp — Q2_K/Q3_K -> Q8_0 Umtensorisierung (Referenzlogik aus strata-llama ggml-quants.c)
// Bau: g++ -O2 -o mtp_requant mtp_requant.cpp
// Aufruf: ./mtp_requant q2k|q3k <raw-in> <raw-out>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdlib>

#define QK_K 256
#define QK8_0 32

struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };
struct block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(block_q2_K) == 84, "q2k");
static_assert(sizeof(block_q3_K) == 110, "q3k");
static_assert(sizeof(block_q8_0) == 34, "q80");

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign << 31;
        else {
            int e = 0;
            while (!(mant & 0x400)) { mant <<= 1; e++; }
            mant &= 0x3ff;
            bits = (sign << 31) | ((127u - 15u - (uint32_t)e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}
static uint16_t float_to_half(float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4);
    uint16_t sign = (uint16_t)((bits >> 31) << 15);
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp >= 31) return sign | 0x7c00;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x800000;
        uint32_t m = mant >> (14 - exp);
        if ((mant >> (13 - exp)) & 1) m++;
        return sign | (uint16_t)m;
    }
    uint16_t h = sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return h;
}

static void dequant_q2_K(const block_q2_K * x, float * y, int64_t k) {
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        const uint8_t * q = x[i].qs;
        int is = 0; float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}
static void dequant_q3_K(const block_q3_K * x, float * y, int64_t k) {
    const int nb = (int)(k / QK_K);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;
    for (int i = 0; i < nb; i++) {
        const float d_all = half_to_float(x[i].d);
        const uint8_t * q = x[i].qs;
        const uint8_t * hm = x[i].hmask;
        uint8_t m = 1;
        std::memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0; float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                shift += 2; m <<= 1;
            }
            q += 32;
        }
    }
}
static void quant_q8_0(const float * x, block_q8_0 * y, int64_t k) {
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = amax > std::fabs(v) ? amax : std::fabs(v);
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = float_to_half(d);
        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;
            y[i].qs[j] = (int8_t)roundf(x0);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 4) { fprintf(stderr, "usage: mtp_requant q2k|q3k in out\n"); return 2; }
    const char mode = argv[1][1];
    FILE * in = fopen(argv[2], "rb");
    if (!in) { perror("in"); return 1; }
    fseek(in, 0, SEEK_END); long sz = ftell(in); fseek(in, 0, SEEK_SET);
    std::vector<uint8_t> raw(sz);
    if (fread(raw.data(), 1, sz, in) != (size_t)sz) { fprintf(stderr, "read short\n"); return 1; }
    fclose(in);
    int64_t bs = (mode == '2') ? 84 : 110;
    if (sz % bs) { fprintf(stderr, "size %ld not multiple of block %ld\n", sz, bs); return 2; }
    int64_t n = (sz / bs) * QK_K;
    std::vector<float> f(n);
    if (mode == '2') dequant_q2_K((const block_q2_K*)raw.data(), f.data(), n);
    else             dequant_q3_K((const block_q3_K*)raw.data(), f.data(), n);
    std::vector<block_q8_0> q8(n / QK8_0);
    quant_q8_0(f.data(), q8.data(), n);
    FILE * out = fopen(argv[3], "wb");
    if (!out) { perror("out"); return 1; }
    fwrite(q8.data(), 1, q8.size() * sizeof(block_q8_0), out);
    fclose(out);
    printf("OK n=%lld out_bytes=%zu\n", (long long)n, q8.size() * sizeof(block_q8_0));
    return 0;
}
