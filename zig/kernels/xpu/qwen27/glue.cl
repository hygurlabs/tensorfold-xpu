// Glue du Qwen3.8-27B sur Intel Xe2 (OpenCL C, ocloc), d'après la référence torch du portage
// (tensorfold/families/qwen3_5/cuda/reference.py). Conventions :
//   x   : flux résiduel fp32 [T, 5120] ; h : entrée normée d'un GEMM, fp16 [T, K]
//   AHI, ALO : sommes de ligne de h par groupe de GSZ (famille de GEMM « biais par DPAS ») : s = Σ h (fp32, k croissant)
//              / GSZ, AHI = fp16(s), ALO = fp16(s − AHI), [T, K / GSZ] ; mêmes bits que rowsum_hl du banc. GSZ (64 ou
//              128) est un argument : celui de la projection qui lit h.
// Sous-groupes de 16 ; réductions dans un ordre fixe (sous-groupe puis mémoire locale, rang croissant).
#pragma OPENCL FP_CONTRACT OFF

#define GS 64
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {                   // au plus proche pair
  return intel_convert_bfloat16_as_ushort(f); // conversion matérielle (au plus proche pair)
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

static void quant_row(local const half *hl, int K, global char *A8, global float *SA, int MS, global half *AHI,
                      global half *ALO, int t);

// Sommes de ligne d'une ligne fp16 rangée en mémoire locale (K valeurs), écrites en [row, K / gs] : un sous-groupe par
// groupe de gs, gs / 16 valeurs par couloir (k croissant), puis la somme du sous-groupe ; ordre fixe.
static void rowsums(local const half *hl, int K, global half *AHI, global half *ALO, long row, int gs) {
  const int lane = get_sub_group_local_id(), nsg = get_num_sub_groups(), per = gs / 16;
  for (int g = get_sub_group_id(); g < K / gs; g += nsg) {
    float s = 0.0f;
    for (int k = 0; k < per; k++) s += (float)hl[g * gs + lane * per + k];
    s = sub_group_reduce_add(s) * (1.0f / gs);
    if (lane == 0) {
      const half hi = convert_half_rte(s);
      AHI[row * (K / gs) + g] = hi;
      ALO[row * (K / gs) + g] = convert_half_rte(s - (float)hi);
    }
  }
}

// Poids tournés (checkpoint marqué hadamard128, tools/xpu/rotate_w4.py) : chaque bloc de 128 d'une entrée de GEMM est
// tourné par Hadamard (Sylvester, normé 1/√128 ; élément 16 i + couloir du bloc au couloir), en mémoire locale, par les
// sous-groupes du groupe ; l'appelant met les barrières autour.
static void rot128_local(local half *hl, int K) {
  const int lane = get_sub_group_local_id();
  for (int b = get_sub_group_id(); b < K / 128; b += get_num_sub_groups()) {
    float v[8];
    for (int i = 0; i < 8; i++) v[i] = (float)hl[b * 128 + i * 16 + lane];
    for (int m = 1; m < 16; m <<= 1)
      for (int i = 0; i < 8; i++) {
        const float o = sub_group_shuffle_xor(v[i], m);
        v[i] = (lane & m) ? o - v[i] : v[i] + o;
      }
    for (int m = 1; m < 8; m <<= 1)
      for (int i = 0; i < 8; i++)
        if (!(i & m)) {
          const float x = v[i], y = v[i | m];
          v[i] = x + y;
          v[i | m] = x - y;
        }
    for (int i = 0; i < 8; i++) hl[b * 128 + i * 16 + lane] = convert_half_rte(v[i] * 0.08838834764831845f);
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

// Plongement dense : x = bf16(E[id]) (table fp16 [V, D], checkpoint GPTQ).
KERNEL(256) void embed_dense(global const int *IDS, global const half *E, global float *X, int D) {
  const int t = get_group_id(0);
  const long id = IDS[t];
  for (int e = get_local_id(0); e < D; e += 256) X[(long)t * D + e] = rbf((float)E[id * D + e]);
}

// Petite projection dense fp16 (portes a, b du GDN au format GPTQ, gardées en fp16 comme le checkpoint) :
// P[t, c0 + n] = fp16(Σ_k H[t, k]·W[n, k]), N = 96, K = 5 120, par DPAS. Un groupe de 256 par (16 lignes, 16 colonnes) :
// le sous-groupe s somme les K de la tranche s (320, dans l'ordre), puis les 16 tranches s'ajoutent dans l'ordre ;
// le sous-groupe r écrit la ligne r. Les bits d'une ligne ne dépendent ni de T ni de sa place.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
KERNEL(256) void dense_rows(global const half *H, global const half *W, global half *P, int T, int ldp, int c0) {
  local float red[16 * 16 * 16];
  const int K = 5120, KS = K / 16;
  const int m0 = get_group_id(0) * 16, n0 = get_group_id(1) * 16, sg = get_sub_group_id(), lane = get_sub_group_local_id();
  float8 acc0 = 0.0f, acc1 = 0.0f;
  global const uint *wr = (global const uint *)(W + (long)(n0 + lane) * K);
  for (int k = sg * KS; k < (sg + 1) * KS; k += 16) {
    short8 a0, a1;
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)H, K * 2, T, K * 2, (int2)(k, m0), (private ushort *)&a0);
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)H, K * 2, T, K * 2, (int2)(k, m0 + 8), (private ushort *)&a1);
    const int8 bb = as_int8(vload8(0, wr + k / 2));
    acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bb, acc0);
    acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bb, acc1);
  }
  for (int r = 0; r < 8; r++) {
    red[(sg * 16 + r) * 16 + lane] = acc0[r];
    red[(sg * 16 + 8 + r) * 16 + lane] = acc1[r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float v = 0.0f;
  for (int s = 0; s < 16; s++) v += red[(s * 16 + sg) * 16 + lane];
  if (m0 + sg < T) P[(long)(m0 + sg) * ldp + c0 + n0 + lane] = convert_half_rte(v);
}

// x ← bf16(x + r) (si R), h = fp16(bf16(rms(x)·w)), et ses sommes de ligne. Un groupe de 512 par ligne (10 valeurs
// par fil pour D = 5 120 ; 512 est le plafond en mode 256 registres) ; somme des carrés : sous-groupes, puis les 32
// sommes partielles dans l'ordre.
KERNEL(128) void add_rmsnorm(global float *X, global const half *R, global const ushort *WN, global half *H,
                              global half *AHI, global half *ALO, int D, float eps, int has_r, int gs,
                              global char *A8, global float *SA, int MS, int rot) {
  // 128 fils par ligne (plusieurs groupes par cœur Xe en mode 256 registres), 4 éléments consécutifs par lecture :
  // le fil prend les éléments 4 (128 i + fil) … + 3 ; D multiple de 512, au plus 5 120.
  local float red[8];
  local half hl[5120];
  const int t = get_group_id(0), lid = get_local_id(0);
  float4 xv[10];
  float ss = 0.0f;
  for (int i = 0; i < D / 512; i++) {
    const int e = (i * 128 + lid) * 4;
    float4 x = vload4(0, X + (long)t * D + e);
    if (has_r) {
      const float4 r = convert_float4(vload4(0, R + (long)t * D + e));
      x = (float4)(rbf(x.s0 + r.s0), rbf(x.s1 + r.s1), rbf(x.s2 + r.s2), rbf(x.s3 + r.s3));
      vstore4(x, 0, X + (long)t * D + e);
    }
    xv[i] = x;
    ss += x.s0 * x.s0 + x.s1 * x.s1 + x.s2 * x.s2 + x.s3 * x.s3;
  }
  const float s = sub_group_reduce_add(ss);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float tot = 0.0f;
  for (int i = 0; i < 8; i++) tot += red[i];
  const float inv = rsqrt(tot / D + eps);
  for (int i = 0; i < D / 512; i++) {
    const int e = (i * 128 + lid) * 4;
    const ushort4 w = vload4(0, WN + e);
    const half4 v = (half4)(convert_half_rte(rbf(xv[i].s0 * inv * bf2f(w.s0))), convert_half_rte(rbf(xv[i].s1 * inv * bf2f(w.s1))),
                            convert_half_rte(rbf(xv[i].s2 * inv * bf2f(w.s2))), convert_half_rte(rbf(xv[i].s3 * inv * bf2f(w.s3))));
    if (!rot) vstore4(v, 0, H + (long)t * D + e);
    vstore4(v, 0, hl + e);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (rot) {
    rot128_local(hl, D);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int e = lid * 4; e < D; e += 512) vstore4(vload4(0, hl + e), 0, H + (long)t * D + e);
  }
  if (A8)                                             // W4A8 : h quantifié pour la projection suivante
    quant_row(hl, D, A8, SA, MS, AHI, ALO, t);
  else
    rowsums(hl, D, AHI, ALO, t, gs);
}

// Même calcul, 512 fils par ligne : pour le décodage et la vérification (16 lignes au plus), où la ligne doit occuper
// plus de fils.
KERNEL(512) void add_rmsnorm_dec(global float *X, global const half *R, global const ushort *WN, global half *H,
                                  global half *AHI, global half *ALO, int D, float eps, int has_r, int gs,
                                  global char *A8, global float *SA, int MS, int rot) {
  local float red[32];
  local half hl[5120];
  const int t = get_group_id(0), lid = get_local_id(0);
  float xv[10], ss = 0.0f;
  for (int i = 0; i < 10; i++) {
    const int e = lid + i * 512;
    float x = e < D ? X[(long)t * D + e] : 0.0f;
    if (has_r && e < D) {
      x = rbf(x + (float)R[(long)t * D + e]);
      X[(long)t * D + e] = x;
    }
    xv[i] = x;
    ss += x * x;
  }
  const float s = sub_group_reduce_add(ss);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float tot = 0.0f;
  for (int i = 0; i < 32; i++) tot += red[i];
  const float inv = rsqrt(tot / D + eps);
  for (int i = 0; i < 10; i++) {
    const int e = lid + i * 512;
    if (e >= D) continue;
    const half v = convert_half_rte(rbf(xv[i] * inv * bf2f(WN[e])));
    if (!rot) H[(long)t * D + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (rot) {
    rot128_local(hl, D);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int e = lid; e < D; e += 512) H[(long)t * D + e] = hl[e];
  }
  if (A8)                                             // W4A8 : h quantifié pour la projection suivante
    quant_row(hl, D, A8, SA, MS, AHI, ALO, t);
  else
    rowsums(hl, D, AHI, ALO, t, gs);
}

// GDN, avant la récurrence. P : sortie du GEMM gdn.in, fp16 [T, 16 640] = qkv (10 240) | z (6 144) | a (48) | b (48),
// colonnes 16 480 à 16 639 de bourrage (tuile du GEMM).
// Convolution causale de noyau 4 sur qkv avec l'état (3 lignes, fp16, CIN → COUT), SiLU, arrondi bf16 ; q, k normés
// par tête de 128 (q·1/128, k·128^-½ après RMS sans poids), v tel quel ; β = σ(b), g = exp(−e^A_log·softplus(a + dt)).
// Un groupe de 256 par jeton.
#define NQK 2048                                        // 16 têtes × 128
#define NCONV 10240
// Grille (T, 40) : un groupe de 256 par bloc de 256 canaux (deux têtes de 128) ; le bloc 0 calcule aussi g et β.
KERNEL(256) void gdn_pre(global const half *P, global const half *CIN, global half *COUT, global const ushort *CW,
                         global const ushort *ALOG, global const ushort *DT, global ushort *Q, global ushort *K,
                         global ushort *V, global float *G, global float *BETA, int T, global const int *TREE,
                         global const ulong *MT) {
  // MT (si non nul) : plusieurs flux, chacun une chaîne de lignes ; la ligne t, j-ième de sa fenêtre (première ligne
  // MT[8t + 6], largeur MT[8t + 7]), lit l'état d'entrée de son flux en MT[8t + 2] et écrit son état après la fenêtre
  // en MT[8t + 3]
  local float cv[256];
  const int t = get_group_id(0), blk = get_group_id(1), lid = get_local_id(0), sg = get_sub_group_id();
  const int lane = get_sub_group_local_id(), c = blk * 256 + lid;
  float s = 0.0f;
  if (MT) {
    CIN = (global const half *)MT[t * 8 + 2];
    COUT = (global half *)MT[t * 8 + 3];
  }
  const int wf = MT ? (int)MT[t * 8 + 6] : 0, ww = MT ? (int)MT[t * 8 + 7] : T;
  const int dep = MT ? t - wf : TREE ? TREE[t * 17] & 255 : t;
  for (int j = 0; j < 4; j++) {
    const int dd = dep - 3 + j;                       // profondeur de l'entrée (négative : l'état d'entrée)
    const int row = MT ? wf + dd : dd < 0 ? 0 : (TREE ? TREE[t * 17 + 1 + dd] : dd);
    const float x = dd < 0 ? (float)CIN[(3 + dd) * NCONV + c] : (float)P[(long)row * 16640 + c];
    s += x * bf2f(CW[c * 4 + j]);
  }
  cv[lid] = rbf(silu(s));
  if (MT) {                                            // la fenêtre comme un bloc de ww lignes
    if (dep >= ww - 3) COUT[(dep - (ww - 3)) * NCONV + c] = P[(long)t * 16640 + c];
    if (ww < 3 && dep == 0)
      for (int r = 0; r < 3 - ww; r++) COUT[r * NCONV + c] = CIN[(r + ww) * NCONV + c];
  } else {
    if (t >= T - 3) COUT[(t - (T - 3)) * NCONV + c] = P[(long)t * 16640 + c];
    if (T < 3 && t == 0)                                // l'état garde une partie de l'ancien
      for (int r = 0; r < 3 - T; r++) COUT[r * NCONV + c] = CIN[(r + T) * NCONV + c];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (c < NQK * 2) {                                    // têtes q (0-15) et k (16-31) : deux par bloc
    if (sg < 2) {
      const int hh = blk * 2 + sg;
      float v[8], ss = 0.0f;
      for (int i = 0; i < 8; i++) {
        v[i] = cv[sg * 128 + i * 16 + lane];
        ss += v[i] * v[i];
      }
      const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + 1e-6f);
      const float scale = hh < 16 ? 1.0f / 128.0f : rsqrt(128.0f);
      global ushort *dst = hh < 16 ? Q + ((long)t * 16 + hh) * 128 : K + ((long)t * 16 + hh - 16) * 128;
      for (int i = 0; i < 8; i++) dst[i * 16 + lane] = f2bf(v[i] * inv * scale);
    }
  } else
    V[(long)t * 6144 + c - NQK * 2] = f2bf(cv[lid]);
  if (blk == 0 && lid < 48) {
    const float a = (float)P[(long)t * 16640 + 16384 + lid], b = (float)P[(long)t * 16640 + 16432 + lid];
    const float sp = a + bf2f(DT[lid]);
    const float soft = sp > 20.0f ? sp : log1p(exp(sp));
    BETA[(long)t * 48 + lid] = sigmoid_f(b);
    G[(long)t * 48 + lid] = exp(-exp(bf2f(ALOG[lid])) * soft);
  }
}

// Même calcul, une ligne par groupe de 256 (tous les canaux du jeton) : meilleur pour les longs morceaux de prompt.
KERNEL(256) void gdn_pre_rows(global const half *P, global const half *CIN, global half *COUT, global const ushort *CW,
                              global const ushort *ALOG, global const ushort *DT, global ushort *Q, global ushort *K,
                              global ushort *V, global float *G, global float *BETA, int T) {
  // Grille (⌈T/8⌉, 5) : 8 lignes et 16 têtes de 128 canaux (0-15 q, 16-31 k, 32-79 v) par groupe, un sous-groupe par
  // tête, le canal 128 tête + 16 i + couloir. Les poids de la convolution sont lus une fois, la fenêtre glisse dans les
  // registres (entrée X[i] = CIN[i] pour i < 3, P[i − 3] ensuite ; la ligne t lit X[t … t + 3]). Mêmes additions dans
  // le même ordre que ligne par ligne : mêmes bits.
  const int t0 = get_group_id(0) * 8, sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);
  const int head = get_group_id(1) * 16 + sg;
  float w[8][4], x[8][3];
  for (int i = 0; i < 8; i++) {
    const int c = head * 128 + i * 16 + lane;
    for (int j = 0; j < 4; j++) w[i][j] = bf2f(CW[c * 4 + j]);
    for (int j = 0; j < 3; j++) {
      const int r = t0 + j;
      x[i][j] = r < 3 ? (float)CIN[r * NCONV + c] : (float)P[(long)(r - 3) * 16640 + c];
    }
  }
  for (int t = t0; t < min(t0 + 8, T); t++) {
    float v[8], ss = 0.0f;
    for (int i = 0; i < 8; i++) {
      const int c = head * 128 + i * 16 + lane, r = t + 3;
      const float x3 = r < 3 ? (float)CIN[r * NCONV + c] : (float)P[(long)(r - 3) * 16640 + c];
      float s = 0.0f;
      s += x[i][0] * w[i][0];
      s += x[i][1] * w[i][1];
      s += x[i][2] * w[i][2];
      s += x3 * w[i][3];
      x[i][0] = x[i][1];
      x[i][1] = x[i][2];
      x[i][2] = x3;
      v[i] = rbf(silu(s));
      ss += v[i] * v[i];
      if (t >= T - 3) COUT[(t - (T - 3)) * NCONV + c] = P[(long)t * 16640 + c];
      if (T < 3 && t == 0)
        for (int q = 0; q < 3 - T; q++) COUT[q * NCONV + c] = CIN[(q + T) * NCONV + c];
    }
    if (head < 32) {
      const float inv = rsqrt(sub_group_reduce_add(ss) / 128.0f + 1e-6f);
      const float scale = head < 16 ? 1.0f / 128.0f : rsqrt(128.0f);
      global ushort *dst = head < 16 ? Q + ((long)t * 16 + head) * 128 : K + ((long)t * 16 + head - 16) * 128;
      for (int i = 0; i < 8; i++) dst[i * 16 + lane] = f2bf(v[i] * inv * scale);
    } else
      for (int i = 0; i < 8; i++) V[(long)t * 6144 + (head - 32) * 128 + i * 16 + lane] = f2bf(v[i]);
    if (get_group_id(1) == 0 && lid < 48) {
      const float a = (float)P[(long)t * 16640 + 16384 + lid], b = (float)P[(long)t * 16640 + 16432 + lid];
      const float sp = a + bf2f(DT[lid]);
      const float soft = sp > 20.0f ? sp : log1p(exp(sp));
      BETA[(long)t * 48 + lid] = sigmoid_f(b);
      G[(long)t * 48 + lid] = exp(-exp(bf2f(ALOG[lid])) * soft);
    }
  }
}

// Après la récurrence : out = fp16(bf16(SiLU(z)·rms(y)·w)) par tête de 128, et ses sommes de ligne (K = 6 144).
// Y : sortie du GDN bf16 [T, 48, 128] ; z dans P (colonnes 10 240 + …). Un groupe de 256 par jeton.
KERNEL(256) void gated_norm(global const ushort *Y, global const half *P, global const ushort *WN, global half *H,
                            global half *AHI, global half *ALO, float eps, int gs, int rot) {
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
      if (!rot) H[(long)t * 6144 + e] = o;
      hl[e] = o;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (rot) {
    rot128_local(hl, 6144);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int e = get_local_id(0) * 4; e < 6144; e += 1024) vstore4(vload4(0, hl + e), 0, H + (long)t * 6144 + e);
  }
  rowsums(hl, 6144, AHI, ALO, t, gs);
}

// Attention, avant le noyau : P = sortie du GEMM attn.qkv fp16 [T, 14 336] = (q 256 | porte 256) × 24 | k 4 × 256 |
// v 4 × 256. q et k : RMS par tête avec poids, arrondi bf16, RoPE sur les 64 premières dimensions (moitiés de 32,
// θ^(−i/32)), arrondi bf16. Q [T, 24, 256] ; K, V écrits au cache [L, 4, 256] à la position p0 + t.
// Rotation de Hadamard d'une tête de 256 (Sylvester, normée par 1/16), répartie sur le sous-groupe : le couloir tient
// les dimensions 16 i + couloir. Quatre étages entre couloirs (shuffle), quatre dans le couloir. Orthogonale et
// symétrique : elle est sa propre inverse, et (qH)·(kH) = q·k.
static void fwht256(float v[16]) {
  const int lane = get_sub_group_local_id();
  for (int m = 1; m < 16; m <<= 1)
    for (int i = 0; i < 16; i++) {
      const float o = sub_group_shuffle_xor(v[i], m);
      v[i] = (lane & m) ? o - v[i] : v[i] + o;
    }
  for (int m = 1; m < 16; m <<= 1)
    for (int i = 0; i < 16; i++)
      if (!(i & m)) {
        const float a = v[i], b = v[i | m];
        v[i] = a + b;
        v[i | m] = a - b;
      }
  for (int i = 0; i < 16; i++) v[i] *= 1.0f / 16.0f;
}

// Étude du cache K/V : une tête quantifiée en entiers symétriques de `levels` pas de chaque côté (127 : int8, 7 : int4,
// échelle max |x| / levels) puis remise en float.
static void quant8_head(float v[16], float levels) {
  float m = 0.0f;
  for (int i = 0; i < 16; i++) m = fmax(m, fabs(v[i]));
  m = sub_group_reduce_max(m);
  const float s = m > 0.0f ? m / levels : 1.0f;
  for (int i = 0; i < 16; i++) v[i] = rint(v[i] / s) * s;
}

// Sortie de l'attention ramenée de la base tournée (KV8 & 1) : O bf16 [T, 24, 256], grille (T, 24).
KERNEL(16) void rot_heads(global ushort *O) {
  const int t = get_group_id(0), hh = get_group_id(1), lane = get_sub_group_local_id();
  global ushort *o = O + ((long)t * 24 + hh) * 256;
  float v[16];
  for (int i = 0; i < 16; i++) v[i] = bf2f(o[i * 16 + lane]);
  fwht256(v);
  for (int i = 0; i < 16; i++) o[i * 16 + lane] = f2bf(v[i]);
}

// Une tête de K ou de V dans le cache int8 (cap positions) : 256 int8 à la rangée row, l'échelle max |x| / 127 après les
// cap · 4 rangées.
static void store_kv8(global ushort *C, long row, int h, const float v[16], int cap) {
  const int lane = get_sub_group_local_id();
  float m = 0.0f;
  for (int i = 0; i < 16; i++) m = fmax(m, fabs(v[i]));
  m = sub_group_reduce_max(m);
  const float s = m > 0.0f ? m / 127.0f : 1.0f;
  global char *c = (global char *)C + (row * 4 + h) * 256;
  for (int i = 0; i < 16; i++) c[i * 16 + lane] = (char)rint(v[i] / s);
  if (lane == 0) ((global float *)((global char *)C + (long)cap * 1024))[row * 4 + h] = s;
}

KERNEL(256) void attn_prep(global const half *P, global const ushort *QN, global const ushort *KN,
                           global const float *INVF, global ushort *Q, global ushort *KC, global ushort *VC, int p0,
                           float eps, global const int *TREE, global const ulong *MT, global const int *POS, int kv8,
                           int cap) {
  // cap > 0 : cache K/V en int8 (store_kv8)
  // kv8 (étude du cache K/V) : & 1 q, k, v tournés par Hadamard ; & 2 k, v quantifiés par (jeton, tête), en int8, ou
  // en int4 avec & 4
  // TREE (si non nul) : la position de la ligne t est p0 + sa profondeur ; ses K, V restent rangés à p0 + t.
  // MT (si non nul) : plusieurs flux, une ligne chacun : caches MT[8t], MT[8t + 1], position POS[t], où K, V vont.
  const int t = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const float pos = (float)(MT ? POS[t] : p0 + (TREE ? TREE[t * 17] & 255 : t));
  const long krow = MT ? POS[t] : p0 + t;               // rangée du cache où vont K et V
  if (MT) {
    KC = (global ushort *)MT[t * 8];
    VC = (global ushort *)MT[t * 8 + 1];
  }
  for (int hh = sg; hh < 32; hh += 16) {               // 0-23 : q, 24-27 : k, 28-31 : v
    if (hh >= 28) {
      float vv[16];
      for (int i = 0; i < 16; i++) vv[i] = (float)P[(long)t * 14336 + 13312 + (hh - 28) * 256 + i * 16 + lane];
      if (kv8 & 1) fwht256(vv);
      if (kv8 & 2) quant8_head(vv, (kv8 & 4) ? 7.0f : 127.0f);
      if (cap)
        store_kv8(VC, krow, hh - 28, vv, cap);
      else
        for (int i = 0; i < 16; i++) VC[(krow * 4 + hh - 28) * 256 + i * 16 + lane] = f2bf(vv[i]);
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
    if (kv8 & 1) fwht256(v);
    if ((kv8 & 2) && !isq) quant8_head(v, (kv8 & 4) ? 7.0f : 127.0f);
    if (!isq && cap) {
      store_kv8(KC, krow, hh - 24, v, cap);
      continue;
    }
    global ushort *dst = isq ? Q + ((long)t * 24 + hh) * 256 : KC + (krow * 4 + hh - 24) * 256;
    for (int i = 0; i < 16; i++) dst[i * 16 + lane] = f2bf(v[i]);
  }
}

// Sortie de l'attention : fp16(bf16(o·σ(porte))) [T, 6 144] et ses sommes de ligne. O bf16 [T, 24, 256].
KERNEL(256) void attn_gate(global const ushort *O, global const half *P, global half *H, global half *AHI,
                           global half *ALO, int gs, int rot) {
  local half hl[6144];
  const int t = get_group_id(0);
  for (int e = get_local_id(0); e < 6144; e += 256) {
    const int hh = e / 256, d = e % 256;
    const float g = (float)P[(long)t * 14336 + hh * 512 + 256 + d];
    const half o = convert_half_rte(rbf(bf2f(O[(long)t * 6144 + e]) * sigmoid_f(g)));
    if (!rot) H[(long)t * 6144 + e] = o;
    hl[e] = o;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (rot) {
    rot128_local(hl, 6144);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int e = get_local_id(0) * 4; e < 6144; e += 1024) vstore4(vload4(0, hl + e), 0, H + (long)t * 6144 + e);
  }
  rowsums(hl, 6144, AHI, ALO, t, gs);
}

// SwiGLU : GU fp16 [T, 34 816] = gate (17 408) | up (17 408) ; act = fp16(bf16(SiLU(g)·u)) et ses sommes de ligne.
// Grille (T, 17) : un groupe de 256 par bloc de 1 024 colonnes (1 024 / gs groupes de sommes).
// ilv : gate|up entrelacés par blocs de 16 (gate de la colonne c en (c / 16)·32 + c % 16, up 16 plus loin).
KERNEL(256) void swiglu(global const half *GU, global half *H, global half *AHI, global half *ALO, int gs, int ilv,
                        int rot) {
  local half hl[1024];
  const int t = get_group_id(0), blk = get_group_id(1), lid = get_local_id(0);
  {                                                   // 4 colonnes consécutives par fil, lectures et écriture vectorielles
    const int e = lid * 4, c = blk * 1024 + e;
    const long gc = ilv ? (c / 16) * 32 + c % 16 : c, uc = ilv ? gc + 16 : 17408 + c;
    const half4 gv = vload4(0, GU + (long)t * 34816 + gc), uv = vload4(0, GU + (long)t * 34816 + uc);
    half4 o;
    for (int i = 0; i < 4; i++) o[i] = convert_half_rte(rbf(silu((float)gv[i]) * (float)uv[i]));
    if (!rot) vstore4(o, 0, H + (long)t * 17408 + c);
    vstore4(o, 0, hl + e);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (rot) {                                          // les 8 blocs de 128 de ces 1 024 colonnes
    rot128_local(hl, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    vstore4(vload4(0, hl + lid * 4), 0, H + (long)t * 17408 + blk * 1024 + lid * 4);
  }
  const int lane = get_sub_group_local_id(), per = gs / 16, ng = 1024 / gs;
  for (int g = get_sub_group_id(); g < ng; g += 16) {
    float s = 0.0f;
    for (int k = 0; k < per; k++) s += (float)hl[g * gs + lane * per + k];
    s = sub_group_reduce_add(s) * (1.0f / gs);
    if (lane == 0) {
      const half hi = convert_half_rte(s);
      AHI[(long)t * (17408 / gs) + blk * ng + g] = hi;
      ALO[(long)t * (17408 / gs) + blk * ng + g] = convert_half_rte(s - (float)hi);
    }
  }
}

// Sommes de ligne de l'activation écrite par l'épilogue SwiGLU (H fp16 [T, 17 408]) : grille T, groupes de 256.
KERNEL(128) void rowsums_ff(global half *H, global half *AHI, global half *ALO, int gs, int rot) {
  // les sommes de rowsums lues directement en mémoire globale (mêmes additions, même ordre : mêmes bits)
  const int t = get_group_id(0), lane = get_sub_group_local_id(), per = gs / 16;
  global half *h = H + (long)t * 17408;
  if (rot) {                                          // poids tournés (gs = 128) : chaque bloc tourné sur place, puis sa somme
    for (int b = get_sub_group_id(); b < 17408 / 128; b += get_num_sub_groups()) {
      float v[8];
      for (int i = 0; i < 8; i++) v[i] = (float)h[b * 128 + i * 16 + lane];
      for (int m = 1; m < 16; m <<= 1)
        for (int i = 0; i < 8; i++) {
          const float o = sub_group_shuffle_xor(v[i], m);
          v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
      for (int m = 1; m < 8; m <<= 1)
        for (int i = 0; i < 8; i++)
          if (!(i & m)) {
            const float x = v[i], y = v[i | m];
            v[i] = x + y;
            v[i | m] = x - y;
          }
      float s = 0.0f;
      for (int i = 0; i < 8; i++) {
        const half o = convert_half_rte(v[i] * 0.08838834764831845f);
        h[b * 128 + i * 16 + lane] = o;
        s += (float)o;
      }
      s = sub_group_reduce_add(s) * (1.0f / 128);
      if (lane == 0) {
        const half hi = convert_half_rte(s);
        AHI[(long)t * 136 + b] = hi;
        ALO[(long)t * 136 + b] = convert_half_rte(s - (float)hi);
      }
    }
    return;
  }
  for (int g = get_sub_group_id(); g < 17408 / gs; g += get_num_sub_groups()) {
    float s = 0.0f;
    if (per == 8) {
      const half8 x = vload8(0, h + g * gs + lane * 8);
      for (int k = 0; k < 8; k++) s += (float)x[k];
    } else {
      const half4 x = vload4(0, h + g * gs + lane * 4);
      for (int k = 0; k < 4; k++) s += (float)x[k];
    }
    s = sub_group_reduce_add(s) * (1.0f / gs);
    if (lane == 0) {
      const half hi = convert_half_rte(s);
      AHI[(long)t * (17408 / gs) + g] = hi;
      ALO[(long)t * (17408 / gs) + g] = convert_half_rte(s - (float)hi);
    }
  }
}

// Étude W4A8 (quantification simulée) : chaque ligne de X fp16 [T, K] quantifiée en int8 symétrique par groupes de qg
// (échelle = max |x| / 127), remise en fp16 sur place, puis ses sommes de ligne par groupe de gs recalculées.
// rot : chaque bloc de 128 de la ligne d'abord tourné par Hadamard (Sylvester, normé par 1/√128 ; les poids le sont
// aussi, W·H, par le convertisseur tools/xpu/rotate_w4.py) ; levels : 127 (int8) ou 7 (int4, valeurs −8 … 7).
KERNEL(256) void fakequant_rows(global half *X, global half *AHI, global half *ALO, int K, int qg, int gs, int rot,
                                int levels) {
  local half hl[17408];
  const int t = get_group_id(0), lane = get_sub_group_local_id();
  for (int e = get_local_id(0) * 4; e < K; e += 1024) vstore4(vload4(0, X + (long)t * K + e), 0, hl + e);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int b = get_sub_group_id(); rot && b < K / 128; b += get_num_sub_groups()) {
    float v[8];                                      // élément 16 i + couloir du bloc
    for (int i = 0; i < 8; i++) v[i] = (float)hl[b * 128 + i * 16 + lane];
    for (int m = 1; m < 16; m <<= 1)
      for (int i = 0; i < 8; i++) {
        const float o = sub_group_shuffle_xor(v[i], m);
        v[i] = (lane & m) ? o - v[i] : v[i] + o;
      }
    for (int m = 1; m < 8; m <<= 1)
      for (int i = 0; i < 8; i++)
        if (!(i & m)) {
          const float x = v[i], y = v[i | m];
          v[i] = x + y;
          v[i | m] = x - y;
        }
    for (int i = 0; i < 8; i++) hl[b * 128 + i * 16 + lane] = convert_half_rte(v[i] * 0.08838834764831845f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int g = get_sub_group_id(); g < K / qg; g += get_num_sub_groups()) {
    float m = 0.0f;
    for (int k = lane; k < qg; k += 16) m = fmax(m, fabs((float)hl[g * qg + k]));
    m = sub_group_reduce_max(m);
    const float s = m > 0.0f ? m / (float)levels : 1.0f;
    for (int k = lane; k < qg; k += 16) {
      const float q = clamp(rint((float)hl[g * qg + k] / s), -(float)levels - (levels == 7 ? 1.0f : 0.0f), (float)levels);
      hl[g * qg + k] = convert_half_rte(q * s);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int e = get_local_id(0) * 4; e < K; e += 1024) vstore4(vload4(0, hl + e), 0, X + (long)t * K + e);
  rowsums(hl, K, AHI, ALO, t, gs);
}

// W4A8 du prompt : quantification des activations. Ligne t (en mémoire locale, K valeurs fp16) en int8 symétrique par
// groupe de 128 (échelle s = max |x| / 127 dans SA [K/128, MS] fp32), rangée dans A8 [T, K] par blocs de 16 dans l'ordre
// [k0 k4 k1 k5 | k8 k12 k9 k13 | k2 k6 k3 k7 | k10 k14 k11 k15] (celui du dépliage des mots dans w4a8) ; AHI, ALO :
// Σ (s·q) / 128 en hi/lo fp16, la somme des activations déquantifiées (terme de biais). Un sous-groupe par groupe de
// 128 ; le couloir tient 8 K (un mot : w = couloir % 2 dans son bloc de 16).
static void quant_row(local const half *hl, int K, global char *A8, global float *SA, int MS, global half *AHI,
                      global half *ALO, int t) {
  const int lane = get_sub_group_local_id();
  for (int g = get_sub_group_id(); g < K / 128; g += get_num_sub_groups()) {
    const half8 xv = vload8(0, hl + g * 128 + lane * 8);
    float m = 0.0f;
    for (int j = 0; j < 8; j++) m = fmax(m, fabs((float)xv[j]));
    m = sub_group_reduce_max(m);
    const float s = m > 0.0f ? m / 127.0f : 1.0f;
    char4 q0, q1;                                     // mot double (w & m), puis ((w >> 4) & m)
    int qs = 0;
    for (int j = 0; j < 8; j++) {
      const int v = (int)clamp(rint((float)xv[j] / s), -127.0f, 127.0f);
      const int b = (j % 2) * 2 + j / 4;              // octet dans son mot double
      if ((j % 4) < 2) q0[b] = (char)v; else q1[b] = (char)v;
      qs += v;
    }
    global char *blk = A8 + (long)t * K + g * 128 + (lane / 2) * 16;
    vstore4(q0, 0, blk + (lane % 2) * 4);
    vstore4(q1, 0, blk + 8 + (lane % 2) * 4);
    qs = sub_group_reduce_add(qs);
    if (lane == 0) {
      SA[(long)g * MS + t] = s;
      const float a = s * (float)qs * (1.0f / 128.0f);
      const half hi = convert_half_rte(a);
      AHI[(long)t * (K / 128) + g] = hi;
      ALO[(long)t * (K / 128) + g] = convert_half_rte(a - (float)hi);
    }
  }
}

// La quantification seule, ligne X fp16 [T, K] : un groupe de 256 par ligne.
KERNEL(256) void quant_a8(global const half *X, global char *A8, global float *SA, global half *AHI, global half *ALO,
                          int K, int MS, int rot) {
  local half hl[17408];
  const int t = get_group_id(0);
  for (int e = get_local_id(0) * 4; e < K; e += 1024) vstore4(vload4(0, X + (long)t * K + e), 0, hl + e);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (rot) {                                          // l'entrée de down (sortie SwiGLU de l'épilogue du GEMM)
    rot128_local(hl, K);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  quant_row(hl, K, A8, SA, MS, AHI, ALO, t);
}

// Arg max de chaque ligne de logits fp16 [T, V] (le premier indice en cas d'égalité), en deux noyaux : maxima
// partiels de 256 groupes par ligne (grille (256, T) ; PV, PI [T, 256]), puis le maximum des partiels (grille T).
KERNEL(256) void argmax_part(global const half *L, int V, global float *PV, global int *PI) {
  local float bv[256];
  local int bi[256];
  const int lid = get_local_id(0), gid = get_group_id(0), row = get_group_id(1), per = (V + 255) / 256;
  float best = -INFINITY;
  int idx = 0x7fffffff;
  for (int i = gid * per + lid; i < min(V, (gid + 1) * per); i += 256) {
    const float v = (float)L[(long)row * V + i];
    if (v > best) { best = v; idx = i; }
  }
  bv[lid] = best;
  bi[lid] = idx;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    for (int i = 1; i < 256; i++)
      if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
    PV[row * 256 + gid] = best;
    PI[row * 256 + gid] = idx;
  }
}
KERNEL(256) void argmax(global const float *PV, global const int *PI, global int *OUT) {
  local float bv[256];
  local int bi[256];
  const int lid = get_local_id(0), row = get_group_id(0);
  bv[lid] = PV[row * 256 + lid];
  bi[lid] = PI[row * 256 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    float best = bv[0];
    int idx = bi[0];
    for (int i = 1; i < 256; i++)
      if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
    OUT[row] = idx;
  }
}

// Prises pour le brouillonneur : TAPS[t][slot·D + e] = bf16(x + r) (l'état caché complet après la couche), [T, 5 D].
KERNEL(256) void tap(global const float *X, global const half *R, global ushort *TAPS, int D, int slot) {
  const int t = get_group_id(1), e = get_global_id(0);
  if (e < D) TAPS[(long)t * 5 * D + slot * D + e] = f2bf(rbf(X[(long)t * D + e] + (float)R[(long)t * D + e]));
}

// État de convolution après un chemin gardé de KEEP lignes (PATH : leurs lignes, ou les KEEP premières si nul) :
// COUT = 3 dernières lignes de concat(CIN, qkv du chemin), qkv lu dans P (pas 16 640). Grille 40 groupes de 256 canaux.
KERNEL(256) void conv_commit(global const half *CIN, global const half *P, global half *COUT, int keep,
                             global const int *PATH) {
  const int c = get_global_id(0);
  for (int r = 0; r < 3; r++) {
    const int i = keep + r;                           // ligne de concat(état, qkv), 3 + keep lignes
    COUT[r * NCONV + c] = i < 3 ? CIN[i * NCONV + c] : P[(long)(PATH ? PATH[i - 3] : i - 3) * 16640 + c];
  }
}

// Réempaquetage au chargement d'une projection MLX (mots [N, K/8], échelles et biais bf16 [N, K/64]) vers la
// disposition du GEMM : mots [K/8, NT] aux colonnes n0 + n, quartets entrelacés (paire k = 2p, 2p + 1 : bits 4p et
// 16 + 4p) ; échelles fp16 [K/64, NT] ; biais × 64 en fp16 [K/64, NT] (famille « biais par DPAS »).
// ILV (gate|up entrelacés pour l'épilogue SwiGLU) : 1 gate, 2 up ; la colonne n va en (n / 16)·32 + (ILV − 1)·16 + n % 16.
kernel void repack(global const uint *MW, global const ushort *MS, global const ushort *MB, global uint *W,
                   global half *S, global half *BG, int N, int K, int NT, int n0, int ILV) {
  const int n = get_global_id(0), w = get_global_id(1);
  if (n >= N || w >= K / 8) return;
  const long col = ILV ? (n / 16) * 32 + (ILV - 1) * 16 + n % 16 : n0 + n;
  const uint m = MW[(long)n * (K / 8) + w];
  uint o = 0;
  for (int p = 0; p < 4; p++) {
    o |= ((m >> (8 * p)) & 0xFu) << (4 * p);
    o |= ((m >> (8 * p + 4)) & 0xFu) << (16 + 4 * p);
  }
  W[(long)w * NT + col] = o;
  if (w % (GS / 8) == 0) {
    const int g = w / (GS / 8);
    S[(long)g * NT + col] = convert_half_rte(bf2f(MS[(long)n * (K / GS) + g]));
    BG[(long)g * NT + col] = convert_half_rte(bf2f(MB[(long)n * (K / GS) + g]) * GS);
  }
}

// Réempaquetage au chargement d'une projection GPTQ symétrique 4 bits (mots [K/8, N] int32, quartet j = k 8w + j ;
// échelles fp16 [K/G, N] ; zéro 8) vers la disposition du GEMM : mêmes mots entrelacés, S = s, biais × G = −8·s·G.
kernel void repack_gptq(global const uint *QW, global const half *SC, global uint *W, global half *S, global half *BG,
                        int N, int K, int NT, int n0, int G, int ILV) {
  const int n = get_global_id(0), w = get_global_id(1);
  if (n >= N || w >= K / 8) return;
  const long col = ILV ? (n / 16) * 32 + (ILV - 1) * 16 + n % 16 : n0 + n;
  const uint m = QW[(long)w * N + n];
  uint o = 0;
  for (int p = 0; p < 4; p++) {
    o |= ((m >> (8 * p)) & 0xFu) << (4 * p);
    o |= ((m >> (8 * p + 4)) & 0xFu) << (16 + 4 * p);
  }
  W[(long)w * NT + col] = o;
  if (w % (G / 8) == 0) {
    const int g = w / (G / 8);
    const half s = SC[(long)g * N + n];
    S[(long)g * NT + col] = s;
    BG[(long)g * NT + col] = convert_half_rte((float)s * (-8.0f * G));
  }
}

// Quantification affine 4 bits, groupes de 64, d'un poids fp16 [N, K] (tête GPTQ) par l'algorithme de MLX
// (affine_quantize : bord de plus grande valeur absolue sur une valeur entière, échelles et biais arrondis bf16),
// vers la disposition MLX (mots [N, K/8], échelles et biais bf16 [N, K/64]) que `repack` lit. Un fil par (n, g).
kernel void quantize_mlx(global const half *Wt, int N, int K, global uint *MW, global ushort *MS, global ushort *MB) {
  const int n = get_global_id(0), g = get_global_id(1);
  if (n >= N || g >= K / 64) return;
  global const half *row = Wt + (long)n * K + g * 64;
  float lo = INFINITY, hi = -INFINITY;
  for (int i = 0; i < 64; i++) {
    const float v = (float)row[i];
    lo = fmin(lo, v);
    hi = fmax(hi, v);
  }
  const bool neg = fabs(lo) > fabs(hi);
  float scale = fmax((hi - lo) / 15.0f, 1e-7f);
  scale = neg ? scale : -scale;
  const float edge = neg ? lo : hi;
  const float q0 = rint(edge / scale);
  if (q0 != 0.0f) scale = edge / q0;
  const float bias = q0 == 0.0f ? 0.0f : edge;
  const float s = rbf(scale), b = rbf(bias);
  MS[(long)n * (K / 64) + g] = f2bf(s);
  MB[(long)n * (K / 64) + g] = f2bf(b);
  for (int w = 0; w < 8; w++) {
    uint word = 0;
    for (int j = 0; j < 8; j++) {
      const int q = clamp((int)rint(((float)row[w * 8 + j] - b) / s), 0, 15);
      word |= (uint)q << (4 * j);
    }
    MW[(long)n * (K / 8) + g * 8 + w] = word;
  }
}
