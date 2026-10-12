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
  // MT (si non nul) : plusieurs flux, une ligne chacun, grille (16, 48, T) : la ligne get_group_id(2) met à jour sur
  // place l'état de son flux, MT[8t + 4]
  const int vb = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id(), kh = h / 3;
  int t0 = 0;
  if (MT) {
    t0 = get_group_id(2);
    S = (global const float *)MT[t0 * 8 + 4];
    SO = (global float *)MT[t0 * 8 + 4];
    T = t0 + 1;
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
