// GDN du prompt sous forme par morceaux (C = 64 jetons) sur les XMX de Xe2, mode rapide : produits en DPAS bf16,
// état et accumulations en fp32. Maths : chunk_ref.py (validées en FP64). Trois noyaux :
//   gdn_prep  (morceau, tête V ; 4 sous-groupes)  la (cumul de log g), T = (I + A)⁻¹, W = T·diag(β e^la)·K,
//                                                U0 = T·diag(β)·V
//   gdn_state (tête V ; 16 sous-groupes, 8 rangées v de S chacun), morceaux en série :
//                                                Uᵀ = U0ᵀ − S·Wᵀ, S ← e^la_C·S + (Uᵀ·diag(e^(la_C − la)))·K
//   gdn_out   (morceau, tête V ; 8 sous-groupes) Y = e^la·Q·Sᵀ + (M ⊙ Q·Kᵀ)·U
// Dispositions : Q, K [T, HK, 128] bf16 ; V [T, HV, 128] bf16 ; G, BETA [T, HV] fp32 (g multiplicatif) ;
// S0, SF [HV, 128 (v), 128 (k)] fp32 ; Y [T, HV, 128] bf16. Intermédiaires : LA [HV, NC, 64] fp32,
// WB [NC, HV, 64 (t), 128 (k)] bf16, U0 [NC, HV, 64 (t), 128 (v)] fp32, HS [NC, HV, 128 (v), 128 (k)] bf16 (état en
// entrée de morceau), VN [NC, HV, 128 (v), 64 (t)] bf16 (U transposé). T multiple de 64. Opérandes des DPAS par
// lectures 2D : normales (A : lignes × 16), transposées (B dont les paires suivent une ligne), en VNNI (B dont les
// paires suivent une colonne).
#pragma OPENCL FP_CONTRACT OFF

#define CK 64
#define D 128
#define DPAS(a, b, c) intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(a), as_int8(b), c)

static inline ushort bf(float f) {                     // fp32 → bf16 au plus proche pair (pas de NaN ici)
  uint u = as_uint(f);
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

#define KERNEL_WG(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel
#define RD16(base, wbytes, h, pitch, x, y, dst) \
  intel_sub_group_2d_block_read_16b_8r16x1c((global void *)(base), wbytes, h, pitch, (int2)(x, y), (private ushort *)&(dst))
#define RDT(base, wbytes, h, pitch, x, y, dst) \
  intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)(base), wbytes, h, pitch, (int2)(x, y), (private uint *)&(dst))
#define RDV(base, wbytes, h, pitch, x, y, dst) \
  intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)(base), wbytes, h, pitch, (int2)(x, y), (private uint *)&(dst))

KERNEL_WG(64) void gdn_prep(global const ushort *K, global const ushort *V, global const float *G,
                            global const float *BETA, global float *LA, global ushort *WB, global float *U0, int HK,
                            int HV, int NC, int skip) {
  local float as_[CK * CK], ts[CK * CK], la[CK], be[CK], co[CK];
  const int c = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id(), kh = h / (HV / HK);
  const int sg = get_sub_group_id(), T = NC * CK;
  const global ushort *Kh = K + kh * D, *Vh = V + h * D;
  // Décroissance cumulée dans le morceau (balayage par paquets de 16, ordre fixe ; sous-groupe 0).
  float carry = 0.0f;
  for (int b = 0; b < CK / 16 && sg == 0; b++) {
    const int t = c * CK + b * 16 + lane;
    // g sous-normal ou nul (porte très forte) : log g borné, sinon −∞ − (−∞) = NaN plus bas ; e^−87 < 1e−37
    const float s = sub_group_scan_inclusive_add(fmax(log(G[(long)t * HV + h]), -87.0f)) + carry;
    const float bt = BETA[(long)t * HV + h];
    la[b * 16 + lane] = s;
    be[b * 16 + lane] = bt;
    co[b * 16 + lane] = bt * exp(s);
    LA[((long)h * NC + c) * CK + b * 16 + lane] = s;
    carry = sub_group_broadcast(s, 15);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // A[t, i] = β_t e^(la_t − la_i) (k_t·k_i), i < t : M = t, N = i, K = k. Tuiles de lignes {sg, 7 − sg} (même
  // charge par sous-groupe, A étant triangulaire), colonnes tj ≤ ti / 2.
  for (int w = 0; w < 2 && !(skip & 1); w++) {
    const int ti = w == 0 ? sg : CK / 8 - 1 - sg;
    float8 acc[CK / 16];
    for (int tj = 0; tj < CK / 16; tj++) acc[tj] = 0.0f;
    for (int kk = 0; kk < D / 16; kk++) {
      ushort8 a;
      RD16(Kh, D * 2, T, HK * D * 2, kk * 16, c * CK + ti * 8, a);
      for (int tj = 0; tj <= ti / 2; tj++) {
        uint8 b;
        RDT(Kh, D * 2, T, HK * D * 2, kk * 8, c * CK + tj * 16, b);
        acc[tj] = DPAS(a, b, acc[tj]);
      }
    }
    for (int tj = 0; tj < CK / 16; tj++) {
      const int i = tj * 16 + lane;
      for (int r = 0; r < 8; r++) {
        const int t = ti * 8 + r;
        as_[t * CK + i] = i < t ? be[t] * ((skip & 8) ? 1.0f : exp(la[t] - la[i])) * acc[tj][r] : 0.0f;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // T = (I + A)⁻¹ par blocs 16 × 16 (L = I + A, unitriangulaire inférieure) : D_i = L_ii⁻¹ par substitution, puis
  // T_ij = −D_i·Σ_{j ≤ k < i} L_ik·T_kj. Le couloir tient une colonne du bloc ; les éléments de L et D sont lus en
  // diffusion depuis la mémoire locale ; ordre des sommes fixe.
  local float dsm[CK / 16][16 * 16];
  for (int bi = sg; bi == sg && !(skip & 2); bi++) {        // bloc diagonal bi = sous-groupe
    float d[16];
#pragma unroll
    for (int r = 0; r < 16; r++) {
      float x = r == lane ? 1.0f : 0.0f;
#pragma unroll
      for (int k = 0; k < r; k++) x -= as_[(bi * 16 + r) * CK + bi * 16 + k] * d[k];
      d[r] = lane <= r ? x : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < 16; r++) dsm[bi][r * 16 + lane] = d[r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int bj = sg; bj == sg && !(skip & 2); bj++) {        // bloc de colonnes bj = sous-groupe

    float tcol[CK / 16][16];                         // T_kj, colonne du couloir, k = bj..3
#pragma unroll
    for (int r = 0; r < 16; r++) tcol[bj][r] = dsm[bj][r * 16 + lane];
    for (int bi = bj + 1; bi < CK / 16; bi++) {
      float x[16];
#pragma unroll
      for (int r = 0; r < 16; r++) x[r] = 0.0f;
      for (int bk = bj; bk < bi; bk++)
#pragma unroll
        for (int k = 0; k < 16; k++) {
          const float tk = tcol[bk][k];
#pragma unroll
          for (int r = 0; r < 16; r++) x[r] += as_[(bi * 16 + r) * CK + bk * 16 + k] * tk;
        }
#pragma unroll
      for (int r = 0; r < 16; r++) {
        float y = 0.0f;
#pragma unroll
        for (int k = 0; k <= r; k++) y -= dsm[bi][r * 16 + k] * x[k];
        tcol[bi][r] = y;
      }
    }
    for (int bi = 0; bi < CK / 16; bi++)
#pragma unroll
      for (int r = 0; r < 16; r++) ts[(bi * 16 + r) * CK + bj * 16 + lane] = bi < bj ? 0.0f : tcol[bi][r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // W = (T·diag(β e^la))·K puis U0 = (T·diag(β))·V : M = t, N = colonne, K = i ; l'échelle des colonnes de T se fait
  // avant l'arrondi bf16 de l'opérande A (K et V restent exacts).
  for (int pass = 0; pass < 2 && !(skip & 4); pass++) {
    const local float *sc = pass == 0 ? co : be;
    for (int w = 0; w < 2; w++) {                    // tuiles de lignes {sg, 7 − sg}, toutes les colonnes
      const int ti = w == 0 ? sg : CK / 8 - 1 - sg;
      ushort8 a[CK / 16];
      for (int q = 0; q <= ti / 2; q++) {
        const float f = sc[q * 16 + lane];
        for (int r = 0; r < 8; r++) a[q][r] = bf(ts[(ti * 8 + r) * CK + q * 16 + lane] * f);
      }
      float8 acc[D / 16];
      for (int tj = 0; tj < D / 16; tj++) acc[tj] = 0.0f;
      for (int q = 0; q <= ti / 2; q++)              // T[t][i] = 0 si i > t
        for (int tj = 0; tj < D / 16; tj++) {
          uint8 b;
          if (pass == 0)
            RDV(Kh, D * 2, T, HK * D * 2, tj * 16, c * CK + q * 16, b);
          else
            RDV(Vh, D * 2, T, HV * D * 2, tj * 16, c * CK + q * 16, b);
          acc[tj] = DPAS(a[q], b, acc[tj]);
        }
      const int row = (c * HV + h) * CK + ti * 8;
      for (int tj = 0; tj < D / 16; tj++) {
        if (pass == 0) {
          ushort8 o;
          for (int r = 0; r < 8; r++) o[r] = bf(acc[tj][r]);
          intel_sub_group_2d_block_write_16b_8r16x1c((global void *)WB, D * 2, NC * HV * CK, D * 2,
                                                     (int2)(tj * 16, row), (private ushort *)&o);
        } else
          intel_sub_group_2d_block_write_32b_8r16x1c((global void *)U0, D * 4, NC * HV * CK, D * 4,
                                                     (int2)(tj * 16, row), (private uint *)&acc[tj]);
      }
    }
  }
}

// Récurrence d'état : un sous-groupe par bloc de 8 rangées v de S (16 par groupe de travail, une tête), S en
// registres (8 tuiles 8 × 16 fp32) et opérande A direct ; tuiles du morceau suivant préchargées (2D). Essais sans
// gain : opérandes B partagés par la mémoire locale (même temps : la limite est la latence du chemin en série d'un
// morceau, pas le trafic L2), en double tampon (64 Ko, résultats faux, pas plus rapide).
#define RV 8
KERNEL_WG(16 * (D / RV)) void gdn_state(global const ushort *K, global const ushort *WB, global const float *U0,
                                        global const float *LA, global const float *S0, global float *SF,
                                        global ushort *HS, global ushort *VN, int HK, int HV, int NC) {
  const int h = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id(), kh = h / (HV / HK);
  const int v0 = sg * RV, T = NC * CK;
  const global ushort *Kh = K + kh * D;
  float8 s[D / 16];
  for (int tj = 0; tj < D / 16; tj++)
    intel_sub_group_2d_block_read_32b_8r16x1c((global void *)S0, D * 4, HV * D, D * 4, (int2)(tj * 16, h * D + v0),
                                              (private uint *)&s[tj]);
  for (int c = 0; c < NC; c++) {
    const int row = (c * HV + h) * D + v0;           // rangée v de HS et VN
    const int trow = (c * HV + h) * CK;              // rangée t de WB et U0
    if (c + 1 < NC) {                                // morceau suivant : W, K
      const int nrow = ((c + 1) * HV + h) * CK;
      for (int tt = 0; tt < CK / 16; tt++) {
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)WB, D * 2, NC * HV * CK, D * 2, (int2)(0, nrow + tt * 16));
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)WB, D * 2, NC * HV * CK, D * 2, (int2)(64, nrow + tt * 16));
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)Kh, D * 2, T, HK * D * 2, (int2)(0, (c + 1) * CK + tt * 16));
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)Kh, D * 2, T, HK * D * 2, (int2)(64, (c + 1) * CK + tt * 16));
      }
    }
    for (int tj = 0; tj < D / 16; tj++) {
      ushort8 o;
      for (int r = 0; r < 8; r++) o[r] = bf(s[tj][r]);
      intel_sub_group_2d_block_write_16b_8r16x1c((global void *)HS, D * 2, NC * HV * D, D * 2, (int2)(tj * 16, row),
                                                 (private ushort *)&o);
    }
    // Uᵀ[v][t] = U0[t][v] − Σ_k S[v][k] W[t][k].
    float8 u[CK / 16];
    for (int tt = 0; tt < CK / 16; tt++) u[tt] = 0.0f;
    for (int kk = 0; kk < D / 16; kk++) {
      ushort8 a;
      for (int r = 0; r < 8; r++) a[r] = bf(s[kk][r]);
      for (int tt = 0; tt < CK / 16; tt++) {
        uint8 b;
        RDT(WB, D * 2, NC * HV * CK, D * 2, kk * 8, trow + tt * 16, b);
        u[tt] = DPAS(a, b, u[tt]);
      }
    }
    const float lc = LA[((long)h * NC + c) * CK + CK - 1];
    ushort8 a[CK / 16];                              // Uᵀ·diag(e^(la_C − la)), opérande A de la mise à jour
    for (int tt = 0; tt < CK / 16; tt++) {
      const float d = exp(lc - LA[((long)h * NC + c) * CK + tt * 16 + lane]);
      float8 u0;
      RDT(U0, D * 4, NC * HV * CK, D * 4, v0, trow + tt * 16, u0);
      const float8 ut = u0 - u[tt];
      ushort8 o;
      for (int r = 0; r < 8; r++) {
        o[r] = bf(ut[r]);
        a[tt][r] = bf(ut[r] * d);
      }
      intel_sub_group_2d_block_write_16b_8r16x1c((global void *)VN, CK * 2, NC * HV * D, CK * 2, (int2)(tt * 16, row),
                                                 (private ushort *)&o);
    }
    // S ← e^la_C·S + Σ_t a[v][t] K[t][k].
    const float dc = exp(lc);
    for (int tj = 0; tj < D / 16; tj++) {
      s[tj] *= dc;
      for (int tt = 0; tt < CK / 16; tt++) {
        uint8 b;
        RDV(Kh, D * 2, T, HK * D * 2, tj * 16, c * CK + tt * 16, b);
        s[tj] = DPAS(a[tt], b, s[tj]);
      }
    }
  }
  for (int tj = 0; tj < D / 16; tj++)
    intel_sub_group_2d_block_write_32b_8r16x1c((global void *)SF, D * 4, HV * D, D * 4, (int2)(tj * 16, h * D + v0),
                                               (private uint *)&s[tj]);
}

// Sortie : P = M ⊙ Q·Kᵀ et les tuiles A de Q, rangées en mémoire locale en bf16 dans l'ordre des couloirs (une fois
// par groupe de travail), puis chaque sous-groupe une tuile de 16 colonnes v : Y = e^la·Q·Sᵀ + P·U.
KERNEL_WG(128) void gdn_out(global const ushort *Q, global const ushort *K, global const float *LA,
                            global const ushort *HS, global const ushort *VN, global ushort *Y, int HK, int HV,
                            int NC) {
  local ushort pa[(CK / 8) * (CK / 16) * 128], qa[(CK / 8) * (D / 16) * 128];
  local float la[CK];
  const int c = get_group_id(0), h = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int kh = h / (HV / HK), T = NC * CK;
  const global ushort *Qh = Q + kh * D, *Kh = K + kh * D;
  if (sg < CK / 16) la[sg * 16 + lane] = LA[((long)h * NC + c) * CK + sg * 16 + lane];
  {
    const int ti = sg;                               // lignes t = 8 sg .. 8 sg + 7
    ushort8 aq[D / 16];
    for (int kk = 0; kk < D / 16; kk++) {
      RD16(Qh, D * 2, T, HK * D * 2, kk * 16, c * CK + ti * 8, aq[kk]);
      intel_sub_group_block_write_us8(qa + (ti * (D / 16) + kk) * 128, aq[kk]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int tj = 0; tj < CK / 16; tj++) {
      float8 acc = 0.0f;
      for (int kk = 0; kk < D / 16 && tj <= ti / 2; kk++) {
        uint8 b;
        RDT(Kh, D * 2, T, HK * D * 2, kk * 8, c * CK + tj * 16, b);
        acc = DPAS(aq[kk], b, acc);
      }
      const int i = tj * 16 + lane;
      ushort8 o;
      for (int r = 0; r < 8; r++) {
        const int t = ti * 8 + r;
        o[r] = bf(i <= t ? exp(la[t] - la[i]) * acc[r] : 0.0f);
      }
      intel_sub_group_block_write_us8(pa + (ti * (CK / 16) + tj) * 128, o);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n = sg;                                  // colonnes v = 16 n .. 16 n + 15
  const int vrow = (c * HV + h) * D + n * 16;
  float8 acc[CK / 8];
  for (int ti = 0; ti < CK / 8; ti++) acc[ti] = 0.0f;
  for (int kk = 0; kk < D / 16; kk++) {              // Q·Sᵀ : B = (S[v][2p], S[v][2p+1]), lecture transposée de HS
    uint8 b;
    RDT(HS, D * 2, NC * HV * D, D * 2, kk * 8, vrow, b);
    for (int ti = 0; ti < CK / 8; ti++) acc[ti] = DPAS(intel_sub_group_block_read_us8(qa + (ti * (D / 16) + kk) * 128), b, acc[ti]);
  }
  for (int ti = 0; ti < CK / 8; ti++)
    for (int r = 0; r < 8; r++) acc[ti][r] *= exp(la[ti * 8 + r]);
  for (int kk = 0; kk < CK / 16; kk++) {             // P·U : B = (U[2p][v], U[2p+1][v]), lecture transposée de VN
    uint8 b;
    RDT(VN, CK * 2, NC * HV * D, CK * 2, kk * 8, vrow, b);
    for (int ti = kk * 2; ti < CK / 8; ti++)
      acc[ti] = DPAS(intel_sub_group_block_read_us8(pa + (ti * (CK / 16) + kk) * 128), b, acc[ti]);
  }
  for (int ti = 0; ti < CK / 8; ti++) {
    ushort8 o;
    for (int r = 0; r < 8; r++) o[r] = bf(acc[ti][r]);
    intel_sub_group_2d_block_write_16b_8r16x1c((global void *)Y, HV * D * 2, T, HV * D * 2,
                                               (int2)(h * D + n * 16, c * CK + ti * 8), (private ushort *)&o);
  }
}
