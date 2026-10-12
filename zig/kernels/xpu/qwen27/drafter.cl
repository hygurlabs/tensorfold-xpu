// DFlash2 drafter for Qwen3.8-27B on Intel Xe2 (after tensorfold/families/qwen3_5/cuda/dflash2.py): 5 layers, hidden
// 5 120, 32 query heads and 8 key/value heads of 128, a block of L rows (the pending token and L − 1 masks) attending
// its context window (the last 2 047 committed target rows) and itself (not causal). Projections run on the int4 GEMM
// (weights quantized to affine 4/64 at load), whose input is fp16 with group-of-64 row sums (AHI, ALO).
#pragma OPENCL FP_CONTRACT OFF

#define GS 64
#define D 5120
#define HD 128
#define NH 32
#define NKV 8
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {
  const uint u = as_uint(f);
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static inline float rbf(float f) { return bf2f(f2bf(f)); }

// Row sums of a fp16 row held in local memory (K values): one subgroup per group of 64.
static void rowsums(local const half *hl, int K, global half *AHI, global half *ALO, long row) {
  const int lane = get_sub_group_local_id(), nsg = get_num_sub_groups();
  for (int g = get_sub_group_id(); g < K / GS; g += nsg) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[g * GS + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / GS);
    if (lane == 0) {
      const half hi = convert_half_rte(s);
      AHI[row * (K / GS) + g] = hi;
      ALO[row * (K / GS) + g] = convert_half_rte(s - (float)hi);
    }
  }
}

// Two-tap grouped dynamic causal convolution inside the block: y[r] = bf16(x[r]·k0 + x[r − 1]·k1), k = bf16(base +
// dyn[group]), x[−1] = 0. X fp16 [L, D] (a projection's output or a norm's); DYN fp16 [L, 2 branches, 2 taps, 320]
// (the kernel projection, padded row stride DS); BASE bf16 [2, 2, D]. With RES: res = bf16(res + y) (fp32 residual
// stream), else y as the next GEMM's input (fp16 with row sums). Grid (L, D / 512), groups of 512.
KERNEL(512) void dconv(global const half *X, global const half *DYN, int DS, global const ushort *BASE, int branch,
                       global float *RES, int has_res, global half *H, global half *AHI, global half *ALO) {
  local half hl[512];
  const int r = get_group_id(0), blk = get_group_id(1), c = blk * 512 + get_local_id(0), g = c / 16;
  const float x = (float)X[(long)r * D + c], prev = r > 0 ? (float)X[(long)(r - 1) * D + c] : 0.0f;
  const float d0 = (float)DYN[(long)r * DS + (branch * 2) * 320 + g], d1 = (float)DYN[(long)r * DS + (branch * 2 + 1) * 320 + g];
  const float k0 = rbf(bf2f(BASE[(branch * 2) * D + c]) + d0), k1 = rbf(bf2f(BASE[(branch * 2 + 1) * D + c]) + d1);
  const float y = rbf(x * k0 + prev * k1);
  if (has_res) {
    RES[(long)r * D + c] = rbf(RES[(long)r * D + c] + y);
    return;
  }
  const half o = convert_half_rte(y);
  H[(long)r * D + c] = o;
  hl[get_local_id(0)] = o;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
  if (sg < 8) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[sg * GS + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / GS);
    if (lane == 0) {
      const half hi = convert_half_rte(s);
      AHI[(long)r * (D / GS) + blk * 8 + sg] = hi;
      ALO[(long)r * (D / GS) + blk * 8 + sg] = convert_half_rte(s - (float)hi);
    }
  }
}

// q, k: RMS per head of 128 with weight, bf16, then rotate-half RoPE over the whole head (θ^(−2i/128)), bf16; v copied.
// P fp16 [L, PS] = q (4 096) | k (1 024) | v (1 024) (row stride PS; with QOFF < 0 only k | v, from column 0).
// Q bf16 [L, 32, 128]; K, V bf16 written at rows POS0 + r of the caches [NKV][CAP][128]. One subgroup per (row, head):
// heads 0-31 q, 32-39 k, 40-47 v. Grid (L, 48 or 16).
KERNEL(16) void dprep(global const half *P, int PS, int qoff, global const ushort *QN, global const ushort *KN,
                      global const float *INVF, global ushort *Q, global ushort *KC, global ushort *VC, int CAP, int pos0,
                      float eps) {
  const int r = get_group_id(0), lane = get_sub_group_local_id();
  const int hh = get_group_id(1) + (qoff < 0 ? NH : 0);
  const int kv0 = qoff < 0 ? 0 : NH * HD;            // column of k in P
  const float pos = (float)(pos0 + r);
  if (hh >= NH + NKV) {                              // v
    const int h = hh - NH - NKV;
    for (int i = 0; i < 8; i++)
      VC[((long)h * CAP + pos0 + r) * HD + i * 16 + lane] = f2bf((float)P[(long)r * PS + kv0 + NKV * HD + h * HD + i * 16 + lane]);
    return;
  }
  const bool isq = hh < NH;
  const long src = (long)r * PS + (isq ? hh * HD : kv0 + (hh - NH) * HD);
  float v[8], ss = 0.0f;
  for (int i = 0; i < 8; i++) {
    v[i] = (float)P[src + i * 16 + lane];
    ss += v[i] * v[i];
  }
  const float inv = rsqrt(sub_group_reduce_add(ss) / HD + eps);
  for (int i = 0; i < 8; i++) v[i] = rbf(v[i] * inv * bf2f((isq ? QN : KN)[i * 16 + lane]));
  for (int i = 0; i < 4; i++) {                      // halves: dims 0-63 (v[0..3]) and 64-127 (v[4..7])
    const float ang = pos * INVF[i * 16 + lane], c = cos(ang), s = sin(ang);
    const float a = v[i], b = v[i + 4];
    v[i] = a * c - b * s;
    v[i + 4] = b * c + a * s;
  }
  global ushort *dst = isq ? Q + ((long)r * NH + hh) * HD : KC + ((long)(hh - NH) * CAP + pos0 + r) * HD;
  for (int i = 0; i < 8; i++) dst[i * 16 + lane] = f2bf(v[i]);
}

// Block attention: query row r (position POS0 + r) sees context rows [C0, POS0) within the window
// (POS0 + r − j ≤ WIN) and every block row (rows POS0 .. POS0 + L − 1 of the cache: not causal). One subgroup per
// (row, key/value head) handles its 4 query heads. O fp16 [L, 4 096] with row sums (o_proj's input).
KERNEL(16) void dattn(global const ushort *Q, global const ushort *KC, global const ushort *VC, int CAP, int c0,
                      int pos0, int L, int WIN, float scale, global half *O) {
  const int r = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id();
  float q[4][8], o[4][8], m[4], l[4];
  for (int a = 0; a < 4; a++) {
    for (int i = 0; i < 8; i++) {
      q[a][i] = bf2f(Q[((long)r * NH + h * 4 + a) * HD + i * 16 + lane]);
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  const int first = max(c0, pos0 + r - WIN);
  for (int j = first; j < pos0 + L; j++) {
    float kv[8];
    for (int i = 0; i < 8; i++) kv[i] = bf2f(KC[((long)h * CAP + j) * HD + i * 16 + lane]);
    float sc[4];
    for (int a = 0; a < 4; a++) {
      float d = 0.0f;
      for (int i = 0; i < 8; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    for (int i = 0; i < 8; i++) kv[i] = bf2f(VC[((long)h * CAP + j) * HD + i * 16 + lane]);
    for (int a = 0; a < 4; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 8; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < 4; a++)
    for (int i = 0; i < 8; i++) O[(long)r * NH * HD + (h * 4 + a) * HD + i * 16 + lane] = convert_half_rte(rbf(o[a][i] / l[a]));
}

// Row sums for a fp16 matrix already written (o_proj's input after dattn): grid L, groups of 256.
KERNEL(256) void rowsums_rows(global const half *H, int K, global half *AHI, global half *ALO) {
  local half hl[25600];
  const int r = get_group_id(0);
  for (int e = get_local_id(0); e < K; e += 256) hl[e] = H[(long)r * K + e];
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, K, AHI, ALO, r);
}

// Target taps bf16 [n, 25 600] to fp16 with row sums (fc's input). Grid n, groups of 256.
KERNEL(256) void taps_in(global const ushort *TAPS, global half *H, global half *AHI, global half *ALO) {
  local half hl[25600];
  const int r = get_group_id(0);
  for (int e = get_local_id(0); e < 25600; e += 256) {
    const half v = convert_half_rte(bf2f(TAPS[(long)r * 25600 + e]));
    H[(long)r * 25600 + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, 25600, AHI, ALO, r);
}

// fp16 [n, D] to the fp32 residual stream (rounded to bf16): the drafter's x after fc's norm or the embedding.
KERNEL(256) void to_res(global const half *H, global float *X) {
  const long i = get_global_id(0);
  X[i] = rbf((float)H[i]);
}

// Top 16 of each row of logits fp16 [R, N] (N real columns of stride NS), in two passes: 256 groups per row keep
// their top 16 (PV, PI [R, 256, 16]), the host merges them. Ties keep the lower index.
KERNEL(16) void topk_part(global const half *L, int N, int NS, global float *PV, global int *PI) {
  const int gid = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
  const int per = (N + 255) / 256, a = gid * per, b = min(N, a + per);
  float bv[16];
  int bi[16];
  for (int i = 0; i < 16; i++) { bv[i] = -INFINITY; bi[i] = 0x7fffffff; }
  if (lane == 0)
    for (int j = a; j < b; j++) {
      const float v = (float)L[(long)row * NS + j];
      if (v <= bv[15]) continue;
      int k = 15;
      while (k > 0 && v > bv[k - 1]) { bv[k] = bv[k - 1]; bi[k] = bi[k - 1]; k--; }
      bv[k] = v;
      bi[k] = j;
    }
  if (lane == 0)
    for (int i = 0; i < 16; i++) {
      PV[((long)row * 256 + gid) * 16 + i] = bv[i];
      PI[((long)row * 256 + gid) * 16 + i] = bi[i];
    }
}

// Affine 4-bit quantization of a bf16 weight [N, K] in groups of 64 along K, as dflash2.quantize4 (MLX layout:
// words [N, K/8], scales and biases bf16 [N, K/64]): scale = bf16(max(hi − lo, ·) / 15), bias = bf16(lo),
// q = clamp(round((w − bias) / scale), 0, 15). One work-item per (row, group).
kernel void quantize4(global const ushort *W, int N, int K, global uint *MW, global ushort *MS, global ushort *MB) {
  const int n = get_global_id(0), g = get_global_id(1);
  if (n >= N || g >= K / GS) return;
  float lo = INFINITY, hi = -INFINITY;
  for (int i = 0; i < GS; i++) {
    const float v = bf2f(W[(long)n * K + g * GS + i]);
    lo = fmin(lo, v);
    hi = fmax(hi, v);
  }
  const float scale = rbf(fmax((hi - lo) / 15.0f, 1e-8f)), bias = rbf(lo);
  MS[(long)n * (K / GS) + g] = f2bf(scale);
  MB[(long)n * (K / GS) + g] = f2bf(bias);
  for (int w = 0; w < GS / 8; w++) {
    uint word = 0;
    for (int j = 0; j < 8; j++) {
      const float v = bf2f(W[(long)n * K + g * GS + w * 8 + j]);
      const int q = clamp((int)rint((v - bias) / scale), 0, 15);
      word |= (uint)q << (4 * j);
    }
    MW[(long)n * (K / 8) + g * (GS / 8) + w] = word;
  }
}
