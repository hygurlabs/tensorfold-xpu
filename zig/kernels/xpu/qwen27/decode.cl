// Noyaux du décodage du Qwen3.8-27B sur Intel Xe2 : GDN en série (quelques jetons) et attention sur le cache.
#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {
  const uint u = as_uint(f);
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Table d'arbre d'un bloc vérifié (≤ 16 lignes) : TREE[t·17] = profondeur de la ligne t (bits 0-7), plus SAVE (son
// état GDN sert à un enfant non consécutif) et LOAD (son parent n'est pas la ligne t − 1) ; TREE[t·17 + 1 + d] = ligne
// de son ancêtre à la profondeur d (lui-même à sa profondeur). Une chaîne : profondeur t, ancêtres 0 … t.
#define DEPTH(t) (TREE[(t) * 17] & 255)
#define SAVE(t) (TREE[(t) * 17] & 256)
#define LOAD(t) (TREE[(t) * 17] & 512)
#define ANC(t, d) TREE[(t) * 17 + 1 + (d)]

// GDN en série, T jetons (décodage, vérification) : S ← g·S ; u = β·(v − S·k) ; S ← S + u·kᵀ ; y = S·q.
// Un sous-groupe par (bloc de 8 rangées v, tête V) ; le couloir tient 8 colonnes k (k = 16 i + couloir).
// Q, K bf16 [·, 16, 128] ; V, Y bf16 [·, 48, 128] ; G, BETA fp32 [·, 48] ; état fp32 [48, 128 (v), 128 (k)] lu dans S,
// écrit dans SO (en place si S = SO). ROWS (si non nul) : la ligne d'entrée du pas t (rejeu d'un chemin), Y suit t.
// TREE (si non nul) : chaque ligne part de l'état de son parent (SL [16, 48, 128, 128], états par ligne) ou de S à la
// racine ; SO n'est pas écrit (le commit rejoue le chemin gardé).
KERNEL(16) void gdn_step(global const ushort *Q, global const ushort *K, global const ushort *V, global const float *G,
                         global const float *BETA, global const float *S, global float *SO, global ushort *Y, int T,
                         global const int *ROWS, global const int *TREE, global float *SL, global const ulong *MT) {
  // MT (si non nul) : plusieurs flux, grille (16, 48, lignes) : la première ligne t0 de chaque fenêtre (MT[8t + 6] = t,
  // largeur MT[8t + 7]) part de l'état de son flux, MT[8t + 4], et écrit l'état après ses lignes en MT[8t + 5]
  const int vb = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id(), kh = h / 3;
  int t0 = 0;
  if (MT) {
    t0 = get_group_id(2);
    if ((int)MT[t0 * 8 + 6] != t0) return;            // not a window's first row
    S = (global const float *)MT[t0 * 8 + 4];
    SO = (global float *)MT[t0 * 8 + 5];
    T = t0 + (int)MT[t0 * 8 + 7];
  }
  float s[8][8];
  for (int r = 0; r < 8; r++)
    for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
  for (int t = t0; t < T; t++) {
    const int row = ROWS ? ROWS[t] : t;
    if (TREE && DEPTH(t) > 0 && LOAD(t)) {              // l'état du parent, depuis son emplacement
      const int par = ANC(t, DEPTH(t) - 1);
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) s[r][i] = SL[(((long)par * 48 + h) * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    } else if (TREE && DEPTH(t) == 0)
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    float q[8], k[8];
    for (int i = 0; i < 8; i++) {
      q[i] = bf2f(Q[((long)row * 16 + kh) * 128 + i * 16 + lane]);
      k[i] = bf2f(K[((long)row * 16 + kh) * 128 + i * 16 + lane]);
    }
    const float g = G[(long)row * 48 + h], beta = BETA[(long)row * 48 + h];
    for (int r = 0; r < 8; r++) {
      float mem = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] *= g;
        mem += s[r][i] * k[i];
      }
      mem = sub_group_reduce_add(mem);
      const float u = (bf2f(V[((long)row * 48 + h) * 128 + vb * 8 + r]) - mem) * beta;
      float y = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] += k[i] * u;
        y += s[r][i] * q[i];
      }
      y = sub_group_reduce_add(y);
      if (lane == 0) Y[((long)t * 48 + h) * 128 + vb * 8 + r] = f2bf(y);
    }
    if (TREE && SAVE(t))
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) SL[(((long)t * 48 + h) * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
  }
  if (!TREE)
    for (int r = 0; r < 8; r++)
      for (int i = 0; i < 8; i++) SO[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
}

// Attention de décodage, par tranches de clés. Q bf16 [T, 24, 256] ; caches KC, VC bf16 [L, 4, 256] ; la ligne t voit
// les positions 0 … p0 + profondeur(t) : avant p0 le cache, au-delà ses ancêtres, rangés à p0 + leur ligne (une
// chaîne : la position elle-même, mêmes bits que le décodage série). Un sous-groupe par (tranche, tête K/V, ligne t)
// traite les 6 têtes de requête de la tête K/V (le couloir tient 16 dimensions). Partiels : PM, PL [T, 24, NS],
// PO [T, 24, NS, 256] fp32.
#define SPLIT 32
KERNEL(16) void attn_decode(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM,
                            global float *PL, global float *PO, int p0, int NS, float scale, global const int *TREE,
                            global const ulong *MT, global const int *POS) {
  // MT (si non nul) : plusieurs flux, une ligne chacun : caches MT[8t], MT[8t + 1], clés 0 … POS[t] toutes en cache
  const int sp = get_group_id(0), hk = get_group_id(1), t = get_group_id(2), lane = get_sub_group_local_id();
  if (MT) {
    KC = (global const ushort *)MT[t * 8];
    VC = (global const ushort *)MT[t * 8 + 1];
  }
  const int last = MT ? POS[t] : p0 + DEPTH(t), j0 = sp * SPLIT, j1 = min(j0 + SPLIT, last + 1);
  float q[6][16], o[6][16], m[6], l[6];
  for (int a = 0; a < 6; a++) {
    for (int i = 0; i < 16; i++) {
      q[a][i] = bf2f(Q[((long)t * 24 + hk * 6 + a) * 256 + i * 16 + lane]);
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  for (int j = j0; j < j1; j++) {
    const long kr = (MT || j < p0) ? j : p0 + ANC(t, j - p0); // rangée du cache de la clé j
    float kv[16];
    for (int i = 0; i < 16; i++) kv[i] = bf2f(KC[(kr * 4 + hk) * 256 + i * 16 + lane]);
    float sc[6];
    for (int a = 0; a < 6; a++) {
      float d = 0.0f;
      for (int i = 0; i < 16; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    for (int i = 0; i < 16; i++) kv[i] = bf2f(VC[(kr * 4 + hk) * 256 + i * 16 + lane]);
    for (int a = 0; a < 6; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 16; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < 6; a++) {
    const long base = ((long)t * 24 + hk * 6 + a) * NS + sp;
    if (lane == 0) {
      PM[base] = m[a];
      PL[base] = l[a];
    }
    for (int i = 0; i < 16; i++) PO[base * 256 + i * 16 + lane] = o[a][i];
  }
}

// Fusion des tranches, dans l'ordre : O bf16 [T, 24, 256]. Un sous-groupe par (tête, ligne).
KERNEL(16) void attn_merge(global const float *PM, global const float *PL, global const float *PO, global ushort *O,
                           int p0, int NS, global const int *TREE, global const int *POS) {
  const int hh = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id();
  const int used = (POS ? POS[t] : p0 + DEPTH(t)) / SPLIT + 1; // tranches non vides de la ligne t
  const long base = ((long)t * 24 + hh) * NS;
  float m = -INFINITY;
  for (int s = 0; s < used; s++) m = fmax(m, PM[base + s]);
  float l = 0.0f, o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int s = 0; s < used; s++) {
    const float w = exp(PM[base + s] - m);
    l += PL[base + s] * w;
    for (int i = 0; i < 16; i++) o[i] += PO[(base + s) * 256 + i * 16 + lane] * w;
  }
  for (int i = 0; i < 16; i++) O[((long)t * 24 + hh) * 256 + i * 16 + lane] = f2bf(o[i] / l);
}

// Commit d'un chemin gardé : les K, V de ses lignes (rangées p0 + ligne) vont aux positions p0 + profondeur, dans
// l'ordre des profondeurs (une ligne du chemin n'est jamais avant sa profondeur : rien n'est écrasé avant d'être lu).
// PATH [n] : lignes du chemin. Un groupe de 16 par tête K/V.
KERNEL(16) void kv_move(global ushort *KC, global ushort *VC, int p0, global const int *PATH, int n) {
  const int hk = get_group_id(0), lane = get_sub_group_local_id();
  for (int d = 0; d < n; d++) {
    const long src = ((long)(p0 + PATH[d]) * 4 + hk) * 256, dst = ((long)(p0 + d) * 4 + hk) * 256;
    if (src == dst) continue;
    for (int i = 0; i < 16; i++) {
      KC[dst + i * 16 + lane] = KC[src + i * 16 + lane];
      VC[dst + i * 16 + lane] = VC[src + i * 16 + lane];
    }
  }
}

// Sonde EXL3 (codebook mul1, docs/recipes/exl3.md) : Y[ks, T, N] fp32 = X[T, K] · W_q sur la tranche ks des tuiles K,
// W_q décodé à la volée de TR [K/16, N/16, 16·bits] int16 (flux circulaire lu du bit de poids fort de chaque mot de
// 32 bits). Un sous-groupe par (colonne de tuiles, tranche de K) ; le couloir décode les valeurs 16·couloir … + 15 de
// chaque tuile, soit les colonnes couloir/2 et couloir/2 + 8 (disposition des fragments B). Mesure de vitesse seulement.
KERNEL(16) void exl3_gemv(global const ushort *TR, global const half *X, global float *Y, int K, int N, int T, int bits,
                          int kslices) {
  // la tuile (8·bits mots de 32 bits, 4 bits : 32 mots) est lue une fois par le sous-groupe (deux mots par couloir) et
  // distribuée par shuffle ; les activations de ses 16 lignes de K aussi (une par couloir et par ligne t)
  const int nt = get_group_id(0), ks = get_group_id(1), lane = get_sub_group_local_id();
  const int per = K / 16 / kslices, kt0 = ks * per, R = 256 * bits, nw = 8 * bits;
  const float c1 = (float)as_half((ushort)0x1EEE), c0 = (float)as_half((ushort)0xC931);
  float acc0[8], acc1[8];
  for (int t = 0; t < 8; t++) acc0[t] = acc1[t] = 0.0f;
  for (int kt = kt0; kt < kt0 + per; kt++) {
    global const uint *tw = (global const uint *)(TR + ((long)kt * (N / 16) + nt) * 16 * bits);
    const uint r0 = tw[lane], r1 = lane + 16 < nw ? tw[lane + 16] : 0u;
    float xr[8];
    for (int t = 0; t < 8; t++) xr[t] = t < T ? (float)X[(long)t * K + kt * 16 + lane] : 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int v = 0; v < 16; v++) {
      const int p = lane * 16 + v;
      int a = (p + 1) * bits - 16;
      a += a < 0 ? R : 0;
      const int wi = a >> 5, wj = wi + 1 == nw ? 0 : wi + 1, sh = a & 31;
      const uint w0a = sub_group_shuffle(r0, wi & 15), w0b = sub_group_shuffle(r1, wi & 15);
      const uint w1a = sub_group_shuffle(r0, wj & 15), w1b = sub_group_shuffle(r1, wj & 15);
      const uint w0 = wi < 16 ? w0a : w0b, w1 = wj < 16 ? w1a : w1b;
      const uint hi = sh ? (w0 << sh) | (w1 >> (32 - sh)) : w0; // les 32 bits qui commencent au bit a
      const uint s = hi >> 16;
      const uint xx = s * 0x83DCD12Du;
      const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
      const float w = (float)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * c1 + c0;
      const int l = p >> 3, j = p & 7, row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
      for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        const float xv = sub_group_shuffle(xr[t], row);
        if (j >> 2) acc1[t] += xv * w; else acc0[t] += xv * w;
      }
    }
  }
  for (int t = 0; t < T && t < 8; t++) {
    const float a0 = acc0[t] + sub_group_shuffle_xor(acc0[t], 1), a1 = acc1[t] + sub_group_shuffle_xor(acc1[t], 1);
    if ((lane & 1) == 0) {
      const long base = ((long)ks * T + t) * N + nt * 16 + lane / 2;
      Y[base] = a0;
      Y[base + 8] = a1;
    }
  }
}

// Attention du prompt (FlashAttention-2 sur DPAS bf16), causale, têtes de 256, 6 têtes de requête par tête K/V.
// Q bf16 [T, 24, 256] (ligne t à la position p0 + t) ; caches KC, VC bf16 [L, 4, 256] (clés 0 … p0 + T − 1 écrites) ;
// O bf16 [T, 24, 256]. Un groupe de travail : une tête K/V et 8 jetons, soit 48 lignes (6 tuiles de 8, une par tête de
// requête), 8 sous-groupes ; grille (⌈T/8⌉, 4). Q est copié une fois en mémoire locale. Par bloc de 64 clés :
// - S = Q·Kᵀ : le sous-groupe sg prend les clés 16 (sg % 4) … + 15 de 3 tuiles de lignes (sg / 4) : chaque tuile K sert
//   3 fois ; S en mémoire locale ;
// - softmax en ligne : 6 lignes par sous-groupe (max, somme, facteur de rattrapage), P en bf16 en mémoire locale, dans la
//   disposition des DPAS ;
// - O += P·V : le sous-groupe sg tient les dimensions 32 sg … + 31 des 48 lignes : chaque tuile V sert 6 fois.
// L'ordre des clés est fixe : une ligne a les mêmes bits quel que soit T ou son morceau.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1)))
kernel void attn_prompt(global const ushort *Q, global const ushort *KC, global const ushort *VC, global ushort *O,
                        int p0, int T, float scale) {
  local float S[48 * 64];
  local ushort P[6 * 4 * 128];                         // tuiles A des DPAS : (tuile de lignes, 16 clés), r × 16 + couloir
  local ushort QL[6 * 16 * 128];                       // Q des 48 lignes, même disposition par (tuile, 16 dimensions)
  local float A[48];
  const int t0 = get_group_id(0) * 8, hk = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nkeys = p0 + min(t0 + 7, T - 1) + 1;
  const int kt = sg % 4, rh = sg / 4;
  float8 o[6][2];
  for (int i = 0; i < 6; i++) o[i][0] = o[i][1] = 0.0f;
  float m[6], l[6];
  for (int j = 0; j < 6; j++) {
    m[j] = -INFINITY;
    l[j] = 0.0f;
  }
  for (int e = sg; e < 96; e += 8) {                   // Q lu une fois : 96 tuiles de 8 lignes × 16 dimensions
    const int i = e / 16, ds = e % 16;
    short8 qa;
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)(Q + (hk * 6 + i) * 256), 256 * 2, T, 24 * 256 * 2,
                                              (int2)(ds * 16, t0), (private ushort *)&qa);
    intel_sub_group_block_write_us8(QL + e * 128, as_ushort8(qa));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k0 = 0; k0 < nkeys; k0 += 64) {
    float8 s[3] = {0.0f, 0.0f, 0.0f};
    for (int ds = 0; ds < 16; ds++) {
      // 16 clés × 16 dimensions, transposées par paires de bf16 : le couloir reçoit sa clé (hors des clés écrites : 0)
      uint8 kw;
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)(KC + hk * 256), 256 * 2, nkeys, 4 * 256 * 2,
                                                          (int2)(ds * 8, k0 + kt * 16), (private uint *)&kw);
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < 3; i++) {
        const short8 qa = as_short8(intel_sub_group_block_read_us8(QL + ((rh * 3 + i) * 16 + ds) * 128));
        s[i] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa, as_int8(kw), s[i]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);                      // le bloc précédent a fini de lire P
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 3; i++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) S[((rh * 3 + i) * 8 + r) * 64 + kt * 16 + lane] = s[i][r] * scale;
    barrier(CLK_LOCAL_MEM_FENCE);
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 6; j++) {
      const int R = sg * 6 + j, qpos = p0 + t0 + R % 8;
      float x[4], mx = -INFINITY;
      for (int c = 0; c < 4; c++) {
        const int key = k0 + c * 16 + lane;
        x[c] = key > qpos ? -INFINITY : S[R * 64 + c * 16 + lane];
        mx = fmax(mx, x[c]);
      }
      mx = sub_group_reduce_max(mx);
      const float mn = fmax(m[j], mx);
      const float alpha = mn == -INFINITY ? 1.0f : exp(m[j] - mn);
      float sum = 0.0f;
      for (int c = 0; c < 4; c++) {
        const float p = mn == -INFINITY ? 0.0f : exp(x[c] - mn);
        P[((R / 8) * 4 + c) * 128 + (R % 8) * 16 + lane] = f2bf(p);
        sum += p;
      }
      l[j] = l[j] * alpha + sub_group_reduce_add(sum);
      m[j] = mn;
      if (lane == 0) A[R] = alpha;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 6; i++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) {
        const float al = A[i * 8 + r];
        o[i][0][r] *= al;
        o[i][1][r] *= al;
      }
    for (int c = 0; c < 4; c++) {
      short8 pa[6];
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < 6; i++) pa[i] = as_short8(intel_sub_group_block_read_us8(P + (i * 4 + c) * 128));
      __attribute__((opencl_unroll_hint))
      for (int dt = 0; dt < 2; dt++) {
        uint8 vw;
        intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)(VC + hk * 256), 256 * 2, nkeys,
                                                             4 * 256 * 2, (int2)((sg * 2 + dt) * 16, k0 + c * 16),
                                                             (private uint *)&vw);
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < 6; i++) o[i][dt] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[i], as_int8(vw), o[i][dt]);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0)
    for (int j = 0; j < 6; j++) A[sg * 6 + j] = 1.0f / l[j];
  barrier(CLK_LOCAL_MEM_FENCE);
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 6; i++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      if (t0 + r >= T) break;
      const float inv = A[i * 8 + r];
      global ushort *dst = O + ((long)(t0 + r) * 24 + hk * 6 + i) * 256 + sg * 32 + lane;
      dst[0] = f2bf(o[i][0][r] * inv);
      dst[16] = f2bf(o[i][1][r] * inv);
    }
}
