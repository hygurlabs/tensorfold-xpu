// Glue du Qwen3.8-27B sur Intel Xe2 (OpenCL C, ocloc), d'après la référence torch du portage
// (tensorfold/families/qwen3_5/cuda/reference.py). Conventions :
//   x   : flux résiduel fp32 [T, 5120] ; h : entrée normée d'un GEMM, fp16 [T, K]
//   AHI, ALO : sommes de ligne de h par groupe de 64 (famille de GEMM « biais par DPAS ») : s = Σ h (fp32, k croissant)
//              / 64, AHI = fp16(s), ALO = fp16(s − AHI), [T, K / 64] ; mêmes bits que rowsum_hl du banc.
// Sous-groupes de 16 ; réductions dans un ordre fixe (sous-groupe puis mémoire locale, rang croissant).
#pragma OPENCL FP_CONTRACT OFF

#define GS 64
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {                   // au plus proche pair
  const uint u = as_uint(f);
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static inline float rbf(float f) { return bf2f(f2bf(f)); }
static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + exp(-x)); }

// Somme d'un groupe de travail de 256 dans l'ordre fixe : sous-groupes, puis les 16 sommes partielles dans l'ordre.
static float wg_sum256(float v, local float *red) {
  const float s = sub_group_reduce_add(v);
  const int sg = get_sub_group_id();
  if (get_sub_group_local_id() == 0) red[sg] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < 16; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

// Sommes de ligne d'une ligne fp16 rangée en mémoire locale (K valeurs), écrites en [row, K / 64].
static void rowsums(local const half *hl, int K, global half *AHI, global half *ALO, long row) {
  for (int g = get_local_id(0); g < K / GS; g += get_local_size(0)) {
    float s = 0.0f;
    for (int k = 0; k < GS; k++) s += (float)hl[g * GS + k];
    s *= 1.0f / GS;
    const half hi = convert_half_rte(s);
    AHI[row * (K / GS) + g] = hi;
    ALO[row * (K / GS) + g] = convert_half_rte(s - (float)hi);
  }
}

// Plongement : x = bf16(q·s + b) (MLX affine 4 bits, groupe 64 ; mots [V, D/8], quartet j = élément 8w + j).
KERNEL(256) void embed(global const int *IDS, global const uint *W, global const ushort *S, global const ushort *B,
                       global float *X, int D) {
  const int t = get_group_id(0);
  const long id = IDS[t];
  for (int e = get_local_id(0); e < D; e += 256) {
    const uint w = W[id * (D / 8) + e / 8];
    const float q = (float)((w >> (4 * (e % 8))) & 0xFu);
    const float s = bf2f(S[id * (D / GS) + e / GS]), b = bf2f(B[id * (D / GS) + e / GS]);
    X[(long)t * D + e] = rbf(q * s + b);
  }
}

// x ← bf16(x + r) (si R), h = fp16(bf16(rms(x)·w)), et ses sommes de ligne. Un groupe de 256 par ligne, D ≤ 5 120.
KERNEL(256) void add_rmsnorm(global float *X, global const half *R, global const ushort *WN, global half *H,
                             global half *AHI, global half *ALO, int D, float eps, int has_r) {
  local float red[16];
  local half hl[5120];
  const int t = get_group_id(0), lid = get_local_id(0);
  float ss = 0.0f;
  for (int e = lid; e < D; e += 256) {
    float x = X[(long)t * D + e];
    if (has_r) {
      x = rbf(x + (float)R[(long)t * D + e]);
      X[(long)t * D + e] = x;
    }
    ss += x * x;
  }
  const float inv = rsqrt(wg_sum256(ss, red) / D + eps);
  for (int e = lid; e < D; e += 256) {
    const half v = convert_half_rte(rbf(X[(long)t * D + e] * inv * bf2f(WN[e])));
    H[(long)t * D + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, D, AHI, ALO, t);
}

// GDN, avant la récurrence. P : sortie du GEMM gdn.in, fp16 [T, 16 640] = qkv (10 240) | z (6 144) | a (48) | b (48),
// colonnes 16 480 à 16 639 de bourrage (tuile du GEMM).
// Convolution causale de noyau 4 sur qkv avec l'état (3 lignes, fp16, CIN → COUT), SiLU, arrondi bf16 ; q, k normés
// par tête de 128 (q·1/128, k·128^-½ après RMS sans poids), v tel quel ; β = σ(b), g = exp(−e^A_log·softplus(a + dt)).
// Un groupe de 256 par jeton.
#define NQK 2048                                        // 16 têtes × 128
#define NCONV 10240
KERNEL(256) void gdn_pre(global const half *P, global const half *CIN, global half *COUT, global const ushort *CW,
                         global const ushort *ALOG, global const ushort *DT, global ushort *Q, global ushort *K,
                         global ushort *V, global float *G, global float *BETA, int T) {
  local float cv[NCONV];
  const int t = get_group_id(0), lid = get_local_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  for (int c = lid; c < NCONV; c += 256) {
    float s = 0.0f;
    for (int j = 0; j < 4; j++) {
      const int i = t + j;                              // ligne de concat(état, qkv)
      const float x = i < 3 ? (float)CIN[i * NCONV + c] : (float)P[(long)(i - 3) * 16640 + c];
      s += x * bf2f(CW[c * 4 + j]);
    }
    cv[c] = rbf(silu(s));
    if (t >= T - 3) COUT[(t - (T - 3)) * NCONV + c] = P[(long)t * 16640 + c];
  }
  if (T < 3)                                            // l'état garde une partie de l'ancien
    for (int c = lid; c < NCONV && t == 0; c += 256)
      for (int r = 0; r < 3 - T; r++) COUT[r * NCONV + c] = CIN[(r + T) * NCONV + c];
  barrier(CLK_LOCAL_MEM_FENCE);
  // q (têtes 0-15) et k (16-31) : RMS sur 128, un sous-groupe par tête, deux têtes par sous-groupe.
  for (int hh = sg; hh < 32; hh += 16) {
    float v[8], ss = 0.0f;
    for (int i = 0; i < 8; i++) {
      v[i] = cv[hh * 128 + i * 16 + lane];
      ss += v[i] * v[i];
    }
    const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + 1e-6f);
    const float scale = hh < 16 ? 1.0f / 128.0f : rsqrt(128.0f);
    global ushort *dst = hh < 16 ? Q + ((long)t * 16 + hh) * 128 : K + ((long)t * 16 + hh - 16) * 128;
    for (int i = 0; i < 8; i++) dst[i * 16 + lane] = f2bf(v[i] * inv * scale);
  }
  for (int c = lid; c < NCONV - NQK * 2; c += 256) V[(long)t * 6144 + c] = f2bf(cv[NQK * 2 + c]);
  if (lid < 48) {
    const float a = (float)P[(long)t * 16640 + 16384 + lid], b = (float)P[(long)t * 16640 + 16432 + lid];
    const float sp = a + bf2f(DT[lid]);
    const float soft = sp > 20.0f ? sp : log1p(exp(sp));
    BETA[(long)t * 48 + lid] = sigmoid_f(b);
    G[(long)t * 48 + lid] = exp(-exp(bf2f(ALOG[lid])) * soft);
  }
}

// Après la récurrence : out = fp16(bf16(SiLU(z)·rms(y)·w)) par tête de 128, et ses sommes de ligne (K = 6 144).
// Y : sortie du GDN bf16 [T, 48, 128] ; z dans P (colonnes 10 240 + …). Un groupe de 256 par jeton.
KERNEL(256) void gated_norm(global const ushort *Y, global const half *P, global const ushort *WN, global half *H,
                            global half *AHI, global half *ALO, float eps) {
  local half hl[6144];
  const int t = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  for (int hh = sg; hh < 48; hh += 16) {
    float v[8], ss = 0.0f;
    for (int i = 0; i < 8; i++) {
      v[i] = bf2f(Y[((long)t * 48 + hh) * 128 + i * 16 + lane]);
      ss += v[i] * v[i];
    }
    const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + eps);
    for (int i = 0; i < 8; i++) {
      const int e = hh * 128 + i * 16 + lane;
      const float yn = v[i] * inv * bf2f(WN[i * 16 + lane]);
      const float z = (float)P[(long)t * 16640 + 10240 + e];
      const half o = convert_half_rte(rbf(silu(z) * yn));
      H[(long)t * 6144 + e] = o;
      hl[e] = o;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, 6144, AHI, ALO, t);
}

// Attention, avant le noyau : P = sortie du GEMM attn.qkv fp16 [T, 14 336] = (q 256 | porte 256) × 24 | k 4 × 256 |
// v 4 × 256. q et k : RMS par tête avec poids, arrondi bf16, RoPE sur les 64 premières dimensions (moitiés de 32,
// θ^(−i/32)), arrondi bf16. Q [T, 24, 256] ; K, V écrits au cache [L, 4, 256] à la position p0 + t.
KERNEL(256) void attn_prep(global const half *P, global const ushort *QN, global const ushort *KN,
                           global const float *INVF, global ushort *Q, global ushort *KC, global ushort *VC, int p0,
                           float eps) {
  const int t = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const float pos = (float)(p0 + t);
  for (int hh = sg; hh < 32; hh += 16) {               // 0-23 : q, 24-27 : k, 28-31 : v
    if (hh >= 28) {
      for (int i = 0; i < 16; i++)
        VC[((long)(p0 + t) * 4 + hh - 28) * 256 + i * 16 + lane] = f2bf((float)P[(long)t * 14336 + 13312 + (hh - 28) * 256 + i * 16 + lane]);
      continue;
    }
    const bool isq = hh < 24;
    const long src = (long)t * 14336 + (isq ? hh * 512 : 12288 + (hh - 24) * 256);
    float v[16], ss = 0.0f;
    for (int i = 0; i < 16; i++) {
      v[i] = (float)P[src + i * 16 + lane];
      ss += v[i] * v[i];
    }
    const float inv = rsqrt(sub_group_reduce_add(ss) / 256.0f + eps);
    for (int i = 0; i < 16; i++) v[i] = rbf(v[i] * inv * bf2f((isq ? QN : KN)[i * 16 + lane]));
    // RoPE : dimensions 0-31 (x1) et 32-63 (x2), soit v[0], v[1] (x1) et v[2], v[3] (x2) pour ce couloir.
    for (int i = 0; i < 2; i++) {
      const float ang = pos * INVF[i * 16 + lane], c = cos(ang), s = sin(ang);
      const float x1 = v[i], x2 = v[i + 2];
      v[i] = x1 * c - x2 * s;
      v[i + 2] = x2 * c + x1 * s;
    }
    global ushort *dst = isq ? Q + ((long)t * 24 + hh) * 256 : KC + ((long)(p0 + t) * 4 + hh - 24) * 256;
    for (int i = 0; i < 16; i++) dst[i * 16 + lane] = f2bf(v[i]);
  }
}

// Sortie de l'attention : fp16(bf16(o·σ(porte))) [T, 6 144] et ses sommes de ligne. O bf16 [T, 24, 256].
KERNEL(256) void attn_gate(global const ushort *O, global const half *P, global half *H, global half *AHI,
                           global half *ALO) {
  local half hl[6144];
  const int t = get_group_id(0);
  for (int e = get_local_id(0); e < 6144; e += 256) {
    const int hh = e / 256, d = e % 256;
    const float g = (float)P[(long)t * 14336 + hh * 512 + 256 + d];
    const half o = convert_half_rte(rbf(bf2f(O[(long)t * 6144 + e]) * sigmoid_f(g)));
    H[(long)t * 6144 + e] = o;
    hl[e] = o;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, 6144, AHI, ALO, t);
}

// SwiGLU : GU fp16 [T, 34 816] = gate (17 408) | up (17 408) ; act = fp16(bf16(SiLU(g)·u)) et ses sommes de ligne.
KERNEL(256) void swiglu(global const half *GU, global half *H, global half *AHI, global half *ALO) {
  local half hl[17408];
  const int t = get_group_id(0);
  for (int e = get_local_id(0); e < 17408; e += 256) {
    const half o = convert_half_rte(rbf(silu((float)GU[(long)t * 34816 + e]) * (float)GU[(long)t * 34816 + 17408 + e]));
    H[(long)t * 17408 + e] = o;
    hl[e] = o;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, 17408, AHI, ALO, t);
}

// Arg max d'une ligne de logits fp16 [V] (le premier indice en cas d'égalité). Un groupe de 256.
KERNEL(256) void argmax(global const half *L, int V, global int *OUT) {
  local float bv[256];
  local int bi[256];
  const int lid = get_local_id(0);
  float best = -INFINITY;
  int idx = 0;
  for (int i = lid; i < V; i += 256) {
    const float v = (float)L[i];
    if (v > best) { best = v; idx = i; }
  }
  bv[lid] = best;
  bi[lid] = idx;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    for (int i = 1; i < 256; i++)
      if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
    OUT[0] = idx;
  }
}

// Réempaquetage au chargement d'une projection MLX (mots [N, K/8], échelles et biais bf16 [N, K/64]) vers la
// disposition du GEMM : mots [K/8, NT] aux colonnes n0 + n, quartets entrelacés (paire k = 2p, 2p + 1 : bits 4p et
// 16 + 4p) ; échelles fp16 [K/64, NT] ; biais × 64 en fp16 [K/64, NT] (famille « biais par DPAS »).
kernel void repack(global const uint *MW, global const ushort *MS, global const ushort *MB, global uint *W,
                   global half *S, global half *BG, int N, int K, int NT, int n0) {
  const int n = get_global_id(0), w = get_global_id(1);
  if (n >= N || w >= K / 8) return;
  const uint m = MW[(long)n * (K / 8) + w];
  uint o = 0;
  for (int p = 0; p < 4; p++) {
    o |= ((m >> (8 * p)) & 0xFu) << (4 * p);
    o |= ((m >> (8 * p + 4)) & 0xFu) << (16 + 4 * p);
  }
  W[(long)w * NT + n0 + n] = o;
  if (w % (GS / 8) == 0) {
    const int g = w / (GS / 8);
    S[(long)g * NT + n0 + n] = convert_half_rte(bf2f(MS[(long)n * (K / GS) + g]));
    BG[(long)g * NT + n0 + n] = convert_half_rte(bf2f(MB[(long)n * (K / GS) + g]) * GS);
  }
}
