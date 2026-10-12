// GEMM int4 de prompt en ISA, généré par nGEN : même arithmétique et même tuilage que w4n_prompt (../gemm/w4.cl),
// donc mêmes bits. C[M, N] = A[M, K] · Wᵀ ; poids [K/8, N] aux nibbles entrelacés, échelles et biais [K/GSZ, N] fp16,
// sommes de ligne Asum [K/GSZ, Mp] (noyau rowsum de w4.cl).
//
// Sous-groupe (un thread SIMD16) : 32 lignes × 16 colonnes ; groupe de travail : 2 × 8 sous-groupes (64 × 128).
// Grille 3D (GM, N/128, paquets) : x = tuile de lignes dans le paquet de GM, y = tuile de colonnes, z = paquet ;
// le matériel parcourt x d'abord, puis y : toutes les colonnes d'un paquet de lignes avant le suivant (A reste dans
// le L2), sans division entière.
// Par groupe g de GSZ : somme partielle DPAS à partir de 0 (blocs de 64 K, quatre pas k16, deux lectures 2D de A de
// 32 lignes × 32 K), puis acc = acc + part·(s·2²⁴) et acc = acc + Σa·b (mad fp32 à un arrondi).
//
// Deux corps : « simple » (chargements attendus juste avant usage) et « pipeline » (par défaut : le demi-bloc suivant
// se charge pendant le calcul du courant, registres en double tampon), aux mêmes bits.
//
//   g++ -std=c++17 -O1 -fno-operator-names -I<ngen> w4_ngen.cpp -o w4_ngen && ./w4_ngen xe2|xe3 64|128 SORTIE.zebin [simple]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <vector>

#include "ngen_elf.hpp"

using namespace ngen;

static constexpr int GM = 8;

template <HW hw>
class W4Prompt : public ELFCodeGenerator<hw> {
    NGEN_FORWARD_ELF(hw)

    // dst = a·b en 32 bits : le multiplicateur entier est en 32 × 16 bits, un mul d × d tronque b à 16 bits (la tête,
    // N = 248 320, lisait ses échelles à une adresse fausse). a < 2¹⁶.
    void mul32(const Subregister &dst, const Subregister &a, const Subregister &b, const Subregister &tmp) {
        mul(1, tmp, a, b.uw(1));
        shl(1, tmp, tmp, 16);
        mul(1, dst, a, b.uw(0));
        add(1, dst, dst, tmp);
    }

    // En-tête 2D : adresse, largeur − 1, hauteur − 1, pas − 1 (octets), x, y, forme du bloc.
    void header(const GRF &h, const Subregister &base, const Subregister &wbytes_m1, const Subregister &height_m1,
                int bw, int bh, int count) {
        mov(1, h.uq(0), base);
        mov(1, h.ud(2), wbytes_m1);
        mov(1, h.ud(3), height_m1);
        mov(1, h.ud(4), wbytes_m1);
        mov(1, h.ud(7), (bw - 1) | ((bh - 1) << 8) | ((count - 1) << 16));
    }

public:
    // Ablations (mesure seulement, bits faux) : sans dépliage int4, sans mise à l'échelle par groupe, sans chargement
    // de A ou des mots (les registres gardent leur contenu).
    struct Ablate { bool unpack = false, fold = false, aload = false, wload = false, sload = false, pf = false,
                    bias = false; };   // bias : sans sommes de ligne ni terme de biais (plafond du GPTQ signé)

    // tn : tuiles de 16 colonnes par sous-groupe (1 : 32 × 16, 4 accumulateurs ; 2 : 32 × 32, 8 accumulateurs
    // indépendants, groupe de travail 64 × 256). Le pipeline n'existe qu'en tn = 1.
    // lag : la mise à l'échelle du groupe g − 1 est émise tuile par tuile juste avant le premier DPAS du groupe g sur
    // la même tuile (vectoriel et XMX en parallèle) ; une dernière passe traite le dernier groupe. Mêmes bits.
    // dpb : biais par DPAS en fin de noyau (Ahi, Alo, b × GSZ), plus de sommes de ligne ni de mad de biais par groupe.
    // slmA : A partagé en mémoire locale. Chaque demi-bloc (64 lignes × 32 K) est lu une fois par le groupe de travail
    // (un seizième par sous-groupe, bloc 2D 8 × 16), rangé en tuiles DPAS de 256 octets en double tampon, puis relu
    // par chaque sous-groupe ; une barrière par demi-bloc. Mêmes bits (seule la provenance de A change).
    // s4 (avec lag, dpb, tn = 2) : famille s4dpb, K en 4 tranches de groupes sommées chacune à partir de 0, total =
    // ((t0 + t1) + t2) + t3 gardé dans la zone de 4 Kio du fil matériel (P, 16 Mio, reste dans le L2 ; en mémoire
    // locale, les 64 Ko réservés par groupe de travail coûtent 16-33 % d'occupation), puis le biais par DPAS. Mêmes
    // bits que les noyaux w4k_* de w4.cl.
    W4Prompt(Product product, int gsz, bool pipe, Ablate ab = {}, int tn = 1, int simdFold = 16, bool lag = false,
             bool dpb = false, bool slmA = false, int layout = 0, bool ilv = false, bool s4 = false, bool asplit = false,
             bool b3 = false, bool swiglu = false)
        : ELFCodeGenerator<hw>(product) {
        const int subs = gsz / 64;
        newArgument("A", ExternalArgumentType::GlobalPtr);
        newArgument("W", ExternalArgumentType::GlobalPtr);
        newArgument("S", ExternalArgumentType::GlobalPtr);
        newArgument("B", ExternalArgumentType::GlobalPtr);
        newArgument("C", ExternalArgumentType::GlobalPtr);
        newArgument("M", DataType::d);
        newArgument("N", DataType::d);
        newArgument("K", DataType::d);
        if (dpb) {
            newArgument("Ahi", ExternalArgumentType::GlobalPtr);
            newArgument("Alo", ExternalArgumentType::GlobalPtr);
            newArgument("Mp", DataType::d);
            newArgument("G", DataType::d);
            if (s4) newArgument("P", ExternalArgumentType::GlobalPtr);
        } else {
            newArgument("Asum", ExternalArgumentType::GlobalPtr);
            newArgument("Mp", DataType::d);
        }
        // swiglu (dpb, tn = 2, sans s4) : poids gate|up entrelacés par blocs de 16 colonnes (tuile j = 0 gate, j = 1 up
        // des mêmes 16 colonnes d'activation) ; la sortie est l'activation fp16(bf16(SiLU(g)·u)), g et u arrondis fp16
        // d'abord, C [M, N/2]. SiLU(x) = x / (1 + 2^(−x·log2 e)), exp et inverse par l'unité mathématique.
        externalName(swiglu ? "w4g_dpb_swiglu" : s4 ? "w4g_s4dpb" : dpb ? "w4g_dpb" : tn == 1 ? "w4g_prompt" : "w4g_prompt_tn2");
        requireSIMD(16);
        requireGRF(256);
        requireLocalID(1);
        requireDPAS();
        if (slmA) {
            requireBarrier();
            requireSLM(8192);
        }
        finalizeInterface();

        setDefaultNoMask();
        setDefaultAutoSWSB(true);
        prologue();
        or_(1, cr0, cr0, uint16_t(0x14C0));            // arrondi IEEE, dénormaux f16/f32/f64 gardés (comme oneDNN)

        const auto A = getArgument("A"), W = getArgument("W"), S = getArgument("S"), B = getArgument("B"),
                   C = getArgument("C"), M = getArgument("M"), N = getArgument("N"), K = getArgument("K"),
                   Mp = getArgument("Mp");
        const auto Asum = dpb ? Subregister() : getArgument("Asum");

        // Registres.
        const GRF hA = r32, hW = r33, hC = r34, s = r35, t = r36, hPA = r37, hPW = r38, addr = r39;
        const GRF sh = r40, bh = r41, sf = r42, bf = r43, asum0 = r46;   // sf/bf : r42-r45 ; asum : r46-r47
        // tn = 1 : W 48-55, B 56-63 / 168-175, A 64-95 / 176-207, part 96-127, acc 128-159, H 160-163, W' 208-215.
        // tn = 2 : W 48-63, B 64-71 / 72-79, A 80-111, part 112-175, acc 176-239, H 240-243.
        // layout (tn = 2 seulement) : 0 actuel ; 1 A, part, acc décalés d'un registre ; 2 B aussi ; 3 part et acc
        // seuls (étude des conflits de bancs de registres entre opérandes du DPAS). H passe en r250 si décalé.
        const int L = (tn == 2) ? layout : 0;
        const int sA = (L == 1 || L == 2) ? 1 : 0, sB = (L == 2) ? 1 : 0, sP = L ? 1 : 0;
        const int W0 = 48, B0 = tn == 1 ? 56 : 64 + sB, B1 = tn == 1 ? 168 : 72 + sB, A0 = tn == 1 ? 64 : 80 + sA,
                  PART = tn == 1 ? 96 : 112 + sP, ACC = tn == 1 ? 128 : 176 + sP,
                  H = b3 ? 80 + sA : tn == 1 ? 160 : (sP ? 250 : 240), A0b = 176, W0b = 208;
        const int TILES = 4 * tn;                         // accumulateurs 8 × 16 par sous-groupe
        const int SF2 = 244;                               // échelle dupliquée pour le SIMD32 : r244-r247
        // b3 (dpb, tn = 2, sans slmA) : B tourne sur trois tampons (r64, r72, r248) au lieu de deux, pour que le dépliage
        // d'une unité n'attende pas les DPAS de l'avant-dernière. r248-r255 libérés : biais décalé en r242, sortie fp16
        // dans A, en-têtes de la passe de biais en r39-r41, compteurs de tranches en r46, en-tête de P en r47.
        const int PBF = b3 ? 242 : 248;                    // décalage : biais du groupe précédent (2 registres)
        const int B2 = 248;
        Label lEnd, lGroup;

        // Tuile : mt = z·GM + x, nt = y ; sous-groupe = id local du canal 0 / 16.
        const Subregister m0 = s.d(0), n0 = s.d(1), g = s.d(2), groups = s.d(3), mt = s.d(4), sg = s.d(5);
        const Subregister kx = s.d(6), tmp = s.d(7);
        mul(1, mt, r0.ud(7), GM);
        add(1, mt, mt, r0.ud(1));
        shl(1, tmp, mt, 6);
        cmp(1 | ge | f0[0], tmp, M);
        jmpi(1 | f0[0], lEnd);                           // paquet incomplet : tuile hors de M
        shr(1, sg, getLocalID(0).uw(0), 4);
        and_(1, t.d(0), sg, 1);
        shl(1, t.d(0), t.d(0), 5);
        add(1, m0, tmp, t.d(0));                          // m0 = mt·64 + (sg % 2)·32
        shr(1, t.d(1), sg, 1);
        shl(1, t.d(1), t.d(1), tn == 1 ? 4 : 5);
        shl(1, n0, r0.ud(6), tn == 1 ? 7 : 8);
        add(1, n0, n0, t.d(1));                           // n0 = nt·128·tn + (sg / 2)·16·tn

        // En-têtes 2D (largeurs et hauteurs − 1).
        shl(1, t.d(2), K, 1);
        add(1, t.d(2), t.d(2), -1);                       // K·2 − 1
        add(1, t.d(3), M, -1);
        header(hA, A, t.d(2), t.d(3), 16, 32, 2);
        mov(1, hA.ud(6), m0);
        header(hPA, A, t.d(2), t.d(3), 16, 32, 2);
        mov(1, hPA.ud(6), m0);
        shl(1, t.d(4), N, 2);
        add(1, t.d(4), t.d(4), -1);                       // N·4 − 1
        shr(1, t.d(5), K, 3);
        add(1, t.d(5), t.d(5), -1);                       // K/8 − 1
        header(hW, W, t.d(4), t.d(5), 16, 8, 1);
        mov(1, hW.ud(5), n0);
        mov(1, hW.ud(6), 0);                               // ligne de mots du bloc courant
        header(hPW, W, t.d(4), t.d(5), 16, 8, 1);
        mov(1, hPW.ud(5), n0);
        if (swiglu)
            add(1, t.d(6), N, -1);                         // (N/2)·2 − 1
        else {
            shl(1, t.d(6), N, 1);
            add(1, t.d(6), t.d(6), -1);                   // N·2 − 1
        }
        header(hC, C, t.d(6), t.d(3), 16, 8, 1);
        mov(1, hC.ud(5), n0);
        // A en mémoire locale. la.ud(0) : adresse locale des envois (seul le premier mot compte) ; la.d(1) : base des
        // tuiles lues par ce sous-groupe ((sg % 2)·2048) ; la.d(2) : place de sa part (sg·256) ; la.d(3) : x de sa
        // part dans le demi-bloc (16·(sg % 2)). Part : 8 lignes × 16 K, rangée sg / 2 du groupe de travail.
        const GRF hP = r254, la = r255;
        const int PIECE = 240;                             // part en transit : r240-r243 (H ne sert qu'à la fin)
        if (slmA) {
            and_(1, t.d(0), sg, 1);
            shl(1, la.d(1), t.d(0), 11);
            shl(1, la.d(3), t.d(0), 4);
            shl(1, la.d(2), sg, 8);
            shl(1, t.d(0), t.d(0), 5);
            add(1, t.d(1), m0, -32);
            sel(1 | ge, t.d(1), t.d(1), 0);
            // base des lignes du groupe de travail : m0 − 32·(sg % 2)
            add(1, t.d(4), m0, 0);
            mul(1, t.d(5), t.d(0), -1);
            add(1, t.d(4), t.d(4), t.d(5));
            shr(1, t.d(5), sg, 1);
            shl(1, t.d(5), t.d(5), 3);
            add(1, t.d(4), t.d(4), t.d(5));               // y de la part : base + 8·(sg / 2)
            shl(1, t.d(6), K, 1);
            add(1, t.d(6), t.d(6), -1);
            add(1, t.d(7), M, -1);
            header(hP, A, t.d(6), t.d(7), 16, 8, 1);
            mov(1, hP.ud(6), t.d(4));
        }

        for (int r = 0; r < 8 * TILES; r++) mov(16, GRF(ACC + r).f(), 0.0f);
        mov(1, g, 0);

        // Échelle et biais du groupe gi (registre) : 16 fp16 en (gi·N + n0)·2, convertis en fp32 (échelle × 2²⁴).
        auto scales = [&](const Subregister &gi) {
            if (ab.sload) return;
            mul32(t.d(0), gi, N, t.d(5));
            add(1, t.d(0), t.d(0), n0);
            shl(1, t.d(0), t.d(0), 1);
            mov(1, addr.uq(0), t.ud(0));
            add(1, addr.uq(0), addr.uq(0), S);
            load.ugm(1, sh, block(DataSizeLSC::D32, 8 * tn), A64, addr);
            if (!dpb) {
                mov(1, addr.uq(0), t.ud(0));
                add(1, addr.uq(0), addr.uq(0), B);
                load.ugm(1, bh, block(DataSizeLSC::D32, 8 * tn), A64, addr);
            }
            for (int j = 0; j < tn; j++) {                // tuile j : sf/bf en r42+2j / r43+2j (r42-r45)
                mov(16, GRF(42 + 2 * j).f(), sh.hf(16 * j)(1));
                mul(16, GRF(42 + 2 * j).f(), GRF(42 + 2 * j).f(), 16777216.0f);
                if (!dpb) mov(16, GRF(43 + 2 * j).f(), bh.hf(16 * j)(1));
                if (simdFold == 32 && !lag) {
                    mov(16, GRF(SF2 + 2 * j).f(), GRF(42 + 2 * j).f());
                    mov(16, GRF(SF2 + 2 * j + 1).f(), GRF(42 + 2 * j).f());
                }
            }
        };
        // Fin de groupe : acc += part·sf, puis acc += Σa·bf, Σa = Asum[gi·Mp + m0 .. + 31].
        auto fold = [&](const Subregister &gi) {
            if (ab.fold) return;
            mul32(t.d(0), gi, Mp, t.d(5));
            add(1, t.d(0), t.d(0), m0);
            shl(1, t.d(0), t.d(0), 2);
            mov(1, addr.uq(0), t.ud(0));
            add(1, addr.uq(0), addr.uq(0), Asum);
            load.ugm(1, asum0, block(DataSizeLSC::D32, 32), A64, addr);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < tn; j++)
                    for (int r = 0; r < 8; r += simdFold / 16) {
                        const int tile = i * tn + j, row = 8 * i + r;
                        const GRF acc = GRF(ACC + 8 * tile + r), part = GRF(PART + 8 * tile + r);
                        if (simdFold == 32) {
                            // Échelle : deux lignes par instruction, l'échelle de la colonne dupliquée dans deux
                            // registres contigus (SF2). Biais : une ligne par instruction (somme de ligne scalaire).
                            mad(32, acc.f(), acc.f(), part.f(), GRF(SF2 + 2 * j).f());
                            for (int q = 0; q < 2; q++)
                                mad(16, GRF(acc.getBase() + q).f(), GRF(acc.getBase() + q).f(),
                                    GRF(46 + (row + q) / 16).f((row + q) % 16), GRF(43 + 2 * j).f());
                        } else {
                            mad(16, acc.f(), acc.f(), part.f(), GRF(42 + 2 * j).f());
                            mad(16, acc.f(), acc.f(), GRF(46 + row / 16).f(row % 16), GRF(43 + 2 * j).f());
                        }
                    }
        };
        // Mise à l'échelle d'une tuile avec l'échelle dupliquée SF2 (SIMD32) et le biais bias_j, sommes de ligne en
        // r46-r47 (déjà chargées).
        auto foldTile = [&](int i, int j, int biasReg) {
            if (ab.fold) return;
            for (int r = 0; r < 8; r += 2) {
                const int tile = i * tn + j, row = 8 * i + r;
                const GRF acc = GRF(ACC + 8 * tile + r), part = GRF(PART + 8 * tile + r);
                mad(32, acc.f(), acc.f(), part.f(), GRF(SF2 + 2 * j).f());
                for (int q = 0; q < 2 && !ab.bias && !dpb; q++)
                    mad(16, GRF(acc.getBase() + q).f(), GRF(acc.getBase() + q).f(),
                        GRF(46 + (row + q) / 16).f((row + q) % 16), GRF(biasReg + j).f());
            }
        };
        auto loadAsum = [&](const Subregister &gi) {
            if (ab.sload || ab.bias || dpb) return;
            mul32(t.d(0), gi, Mp, t.d(5));
            add(1, t.d(0), t.d(0), m0);
            shl(1, t.d(0), t.d(0), 2);
            mov(1, addr.uq(0), t.ud(0));
            add(1, addr.uq(0), addr.uq(0), Asum);
            load.ugm(1, asum0, block(DataSizeLSC::D32, 32), A64, addr);
        };
        // Demi-bloc (32 K) : deux pas k16, mots 2·step et 2·step + 1, A en abuf (deux blocs de 16 GRF).
        // Dépliage d'une unité (pas step, tuile de colonnes j) dans le tampon B de sa parité, découpé en morceaux pour
        // pouvoir l'intercaler entre des DPAS (ilv).
        auto unpackOps = [&](int wbuf, int step, int j, int bb) {
            std::vector<std::function<void()>> ops;
            const GRF w0 = GRF(wbuf + 8 * j + 2 * step), w1 = GRF(wbuf + 8 * j + 2 * step + 1);
            if (ab.unpack) return ops;
            for (int q = 0; q < 4; q++) {
                const GRF d0 = GRF(bb + q), d1 = GRF(bb + 4 + q);
                if (q == 0) {
                    ops.push_back([=] { and_(16, d0.ud(), w0.ud(), 0x000F000Fu); });
                    ops.push_back([=] { and_(16, d1.ud(), w1.ud(), 0x000F000Fu); });
                } else {
                    ops.push_back([=] { shr(16, d0.ud(), w0.ud(), 4 * q); });
                    ops.push_back([=] { and_(16, d0.ud(), d0.ud(), 0x000F000Fu); });
                    ops.push_back([=] { shr(16, d1.ud(), w1.ud(), 4 * q); });
                    ops.push_back([=] { and_(16, d1.ud(), d1.ud(), 0x000F000Fu); });
                }
            }
            return ops;
        };
        // Tampon B de l'unité n du demi-bloc hb (rang dans l'itération) : parité de la tuile, ou rotation sur trois (b3 ;
        // à la frontière d'itération, deux unités consécutives peuvent partager un tampon).
        auto bufB = [&](int hb, int n, int j) {
            if (!b3) return (j % 2) ? B1 : B0;
            const int idx = (hb * 2 * tn + n) % 3;
            return idx == 0 ? B0 : idx == 1 ? B1 : B2;
        };
        auto half = [&](int abuf, int wbuf, int t2, bool first, std::function<void(int)> afterH = {}, int hb = 0) {
            // Unités du demi-bloc : (pas, tuile de colonnes), dans l'ordre des DPAS.
            std::vector<std::pair<int, int>> units;
            for (int h = 0; h < 2; h++)
                for (int j = 0; j < tn; j++) units.push_back({2 * t2 + h, j});
            if (ilv) for (auto &op : unpackOps(wbuf, units[0].first, units[0].second, bufB(hb, 0, units[0].second))) op();
            for (size_t n = 0; n < units.size(); n++) {
                const int step = units[n].first, j = units[n].second, h = step % 2;
                const int bb = bufB(hb, (int)n, j);
                std::vector<std::function<void()>> nextOps;
                if (!ilv) {
                    for (auto &op : unpackOps(wbuf, step, j, bb)) op();
                } else if (n + 1 < units.size())
                    nextOps = unpackOps(wbuf, units[n + 1].first, units[n + 1].second, bufB(hb, (int)n + 1, units[n + 1].second));
                size_t k = 0;
                for (int i = 0; i < 4; i++) {
                    const GRF part = GRF(PART + 8 * (i * tn + j));
                    const GRF a = GRF(abuf + h * 16 + i * 4);
                    if (first && h == 0 && lag) foldTile(i, j, PBF);
                    if (first && h == 0)
                        dpas(16, 8, 8, part.f(), null.f(), GRF(bb).hf(), a.hf());
                    else
                        dpas(16, 8, 8, part.f(), part.f(), GRF(bb).hf(), a.hf());
                    // Après chaque DPAS, une part du dépliage de l'unité suivante.
                    const size_t share = (nextOps.size() + 3 - i) / (4 - i);
                    for (size_t c = 0; c < share && k < nextOps.size(); c++) nextOps[k++]();
                }
                while (k < nextOps.size()) nextOps[k++]();
                // Les DPAS de la moitié h émis : sa moitié de A peut être rechargée (asplit).
                if (afterH && (n + 1 == units.size() || units[n + 1].first % 2 != h)) afterH(h);
            }
        };
        // asplit : A lu en deux moitiés de 16 K (en-tête r241, bloc 16 × 32), la moitié h du demi-bloc suivant lancée
        // dès que les DPAS de la moitié h courante sont émis : la lecture se recouvre avec les 8 DPAS de l'autre moitié
        // au lieu de vider la file XMX à chaque demi-bloc.
        const GRF hA1 = r241;
        auto loadAHalf = [&](int h, int dx) {
            add(1, hA1.ud(5), kx, dx + 16 * h);
            load(1, GRF(A0 + 16 * h), block_2d(DataSizeLSC::D16, 16, 32, 1), A64, hA1);
        };
        if (asplit) {
            shl(1, t.d(2), K, 1);
            add(1, t.d(2), t.d(2), -1);
            add(1, t.d(3), M, -1);
            header(hA1, A, t.d(2), t.d(3), 16, 32, 1);
            mov(1, hA1.ud(6), m0);
        }

        // s4 : tranche courante, fin de la tranche, groupes par tranche ; adresse locale des envois.
        const Subregister sl = b3 ? r46.d(0) : r254.d(0), slEnd = b3 ? r46.d(1) : r254.d(1), per = b3 ? r46.d(2) : r254.d(2);
        Label lSlice, lNoLoad, lNoStore;
        if (!pipe) {
            if (s4) {
                shr(1, per, K, gsz == 64 ? 6 : 7);
                shr(1, per, per, 2);
                mov(1, sl, 0);
                mov(1, slEnd, 0);
                mov(1, g, 0);
                mov(1, kx, 0);
                if (asplit) for (int h = 0; h < 2; h++) loadAHalf(h, 0);
                mark(lSlice);
                add(1, slEnd, slEnd, per);
            }
            if (lag) {
                for (int j = 0; j < tn; j++) {
                    mov(16, GRF(42 + 2 * j).f(), 0.0f);
                    mov(16, GRF(43 + 2 * j).f(), 0.0f);
                }
                for (int r = 0; r < 8 * TILES; r++) mov(16, GRF(PART + r).f(), 0.0f);
            }
            if (slmA) {                                    // demi-bloc 0 rangé dans le tampon 0
                mov(1, hP.ud(5), la.d(3));
                load(1, GRF(PIECE), block_2d(DataSizeLSC::D16, 16, 8), A64, hP);
                mov(1, la.ud(0), la.d(2));
                store.slm(1, block(DataSizeLSC::D32, 64), SLM, la, GRF(PIECE));
                slmfence(r253);
                fencewait();
                barrier(r252, r0);
            }
            if (s4)
                mov(1, groups, slEnd);
            else {
                shr(1, groups, K, gsz == 64 ? 6 : 7);     // groupes
                mov(1, kx, 0);
                if (asplit) for (int h = 0; h < 2; h++) loadAHalf(h, 0);
            }
            mark(lGroup);
            for (int u = 0; u < subs; u++) {
                if (!ab.pf) {
                    add(1, hPW.ud(6), hW.ud(6), 8);
                    load(1, null, block_2d(DataSizeLSC::D32, 16, 8), A64, hPW);
                    add(1, hPA.ud(5), kx, 64);
                    load(1, null, block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hPA);
                    add(1, hPA.ud(5), kx, 96);
                    load(1, null, block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hPA);
                }
                for (int j = 0; j < tn && !ab.wload; j++) {
                    if (j) add(1, hW.ud(5), hW.ud(5), 16);
                    load(1, GRF(W0 + 8 * j), block_2d(DataSizeLSC::D32, 16, 8), A64, hW);
                }
                if (tn > 1 && !ab.wload) add(1, hW.ud(5), hW.ud(5), -16 * (tn - 1));
                if (u == 0 && lag) {
                    // Le groupe précédent passe en « précédent » (échelle dupliquée, biais), ses sommes de ligne
                    // sont chargées ; au groupe 0, échelle et biais précédents valent 0 (acc + 0 exact).
                    for (int j = 0; j < tn; j++) {
                        mov(16, GRF(SF2 + 2 * j).f(), GRF(42 + 2 * j).f());
                        mov(16, GRF(SF2 + 2 * j + 1).f(), GRF(42 + 2 * j).f());
                        mov(16, GRF(PBF + j).f(), GRF(43 + 2 * j).f());
                    }
                    add(1, t.d(7), g, -1);
                    sel(1 | ge, t.d(7), t.d(7), 0);
                    loadAsum(t.d(7));
                }
                if (u == 0) scales(g);
                for (int t2 = 0; t2 < 2; t2++) {
                    if (slmA) {
                        const int buf = t2 % 2;            // demi-blocs par itération pairs : parité connue
                        // Part du demi-bloc suivant (x = kx + 32·(t2 + 1) + 16·(sg % 2)).
                        add(1, hP.ud(5), kx, 32 * (t2 + 1));
                        add(1, hP.ud(5), hP.ud(5), la.d(3));
                        load(1, GRF(PIECE), block_2d(DataSizeLSC::D16, 16, 8), A64, hP);
                        // Tuiles du demi-bloc courant : (rangée sm·4 + i, moitié h) → la.d(1) + i·512 + h·256.
                        for (int h = 0; h < 2; h++)
                            for (int i = 0; i < 4; i++) {
                                add(1, la.ud(0), la.d(1), buf * 4096 + i * 512 + h * 256);
                                load.slm(1, GRF(A0 + h * 16 + i * 4), block(DataSizeLSC::D32, 64), SLM, la);
                            }
                        half(A0, W0, t2, u == 0 && t2 == 0);
                        add(1, la.ud(0), la.d(2), (1 - buf) * 4096);
                        store.slm(1, block(DataSizeLSC::D32, 64), SLM, la, GRF(PIECE));
                        slmfence(r253);
                        fencewait();
                        barrier(r252, r0);
                    } else if (asplit) {
                        half(A0, W0, t2, u == 0 && t2 == 0, [&](int h) { loadAHalf(h, 32 * (t2 + 1)); }, u * 2 + t2);
                    } else {
                        add(1, hA.ud(5), kx, 32 * t2);
                        if (!ab.aload) load(1, GRF(A0), block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hA);
                        half(A0, W0, t2, u == 0 && t2 == 0, {}, u * 2 + t2);
                    }
                }
                add(1, kx, kx, 64);
                add(1, hW.ud(6), hW.ud(6), 8);
            }
            if (!lag) fold(g);
            add(1, g, g, 1);
            cmp(1 | lt | f1[0], g, groups);
            jmpi(1 | f1[0], lGroup);
            if (lag) {                                     // dernier groupe
                for (int j = 0; j < tn; j++) {
                    mov(16, GRF(SF2 + 2 * j).f(), GRF(42 + 2 * j).f());
                    mov(16, GRF(SF2 + 2 * j + 1).f(), GRF(42 + 2 * j).f());
                    mov(16, GRF(PBF + j).f(), GRF(43 + 2 * j).f());
                }
                add(1, t.d(7), groups, -1);
                loadAsum(t.d(7));
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < tn; j++) foldTile(i, j, PBF);
            }
            if (s4) {
                // total = total + tranche, tuile 8 × 16 par tuile (lectures dans A0.., libre ici) ; la tranche 0 est
                // rangée telle quelle, la dernière reste dans acc.
                const GRF hP4 = b3 ? r47 : r255;
                const auto P = getArgument("P");
                shl(1, t.d(0), N, 2);
                add(1, t.d(0), t.d(0), -1);               // N·4 − 1
                add(1, t.d(1), M, -1);
                header(hP4, P, t.d(0), t.d(1), 16, 8, 1);
                cmp(1 | eq | f1[0], sl, 0);
                jmpi(1 | f1[0], lNoLoad);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < tn; j++) {
                        const int tile = i * tn + j, tmp = asplit ? W0 + 8 * (tile % 4) : A0 + 8 * (tile % 4);   // asplit : 48-79
                        add(1, hP4.ud(5), n0, 16 * j);
                        add(1, hP4.ud(6), m0, 8 * i);
                        load(1, GRF(tmp), block_2d(DataSizeLSC::D32, 16, 8), A64, hP4);
                        for (int r = 0; r < 8; r++)
                            add(16, GRF(ACC + 8 * tile + r).f(), GRF(tmp + r).f(), GRF(ACC + 8 * tile + r).f());
                    }
                mark(lNoLoad);
                cmp(1 | eq | f1[0], sl, 3);
                jmpi(1 | f1[0], lNoStore);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < tn; j++) {
                        add(1, hP4.ud(5), n0, 16 * j);
                        add(1, hP4.ud(6), m0, 8 * i);
                        store(1, block_2d(DataSizeLSC::D32, 16, 8), A64, hP4, GRF(ACC + 8 * (i * tn + j)));
                    }
                memfence(b3 ? r45 : r253);
                fencewait();
                for (int r = 0; r < 8 * TILES; r++) mov(16, GRF(ACC + r).f(), 0.0f);
                mark(lNoStore);
                add(1, sl, sl, 1);
                cmp(1 | lt | f1[0], sl, 4);
                jmpi(1 | f1[0], lSlice);
            }
            if (dpb) {
                // acc += Ahi·B + Alo·B par pas de 16 groupes : A[32 lignes × 16 groupes] (hi puis lo), B[16 groupes ×
                // 16 colonnes] en VNNI (lecture 2D transformée) ; hors surface, les lectures 2D rendent 0.
                const auto Ahi = getArgument("Ahi"), Alo = getArgument("Alo"), Gs = getArgument("G");
                const GRF hAH = b3 ? r40 : r250, hAL = b3 ? r41 : r251, hBB = b3 ? r39 : r252;
                const Subregister st = b3 ? r46.d(4) : r253.d(0), nst = b3 ? r46.d(5) : r253.d(1);
                Label lBias;
                shl(1, t.d(0), Gs, 1);
                add(1, t.d(0), t.d(0), -1);               // G·2 − 1
                add(1, t.d(1), M, -1);
                header(hAH, Ahi, t.d(0), t.d(1), 16, 32, 1);
                header(hAL, Alo, t.d(0), t.d(1), 16, 32, 1);
                mov(1, hAH.ud(6), m0);
                mov(1, hAL.ud(6), m0);
                shl(1, t.d(2), N, 1);
                add(1, t.d(2), t.d(2), -1);               // N·2 − 1
                add(1, t.d(3), Gs, -1);
                header(hBB, B, t.d(2), t.d(3), 16, 16, 1);
                add(1, nst, Gs, 15);
                shr(1, nst, nst, 4);
                mov(1, st, 0);
                mark(lBias);
                shl(1, hAH.ud(5), st, 4);
                shl(1, hAL.ud(5), st, 4);
                shl(1, hBB.ud(6), st, 4);
                load(1, GRF(A0), block_2d(DataSizeLSC::D16, 16, 32), A64, hAH);
                load(1, GRF(A0 + 16), block_2d(DataSizeLSC::D16, 16, 32), A64, hAL);
                for (int j = 0; j < tn; j++) {
                    add(1, hBB.ud(5), n0, 16 * j);
                    load(1, GRF(j ? B1 : B0), block_2d(DataSizeLSC::D16, 16, 16) | vnni, A64, hBB);
                }
                for (int j = 0; j < tn; j++)
                    for (int i = 0; i < 4; i++) {
                        const GRF acc = GRF(ACC + 8 * (i * tn + j));
                        dpas(16, 8, 8, acc.f(), acc.f(), GRF(j ? B1 : B0).hf(), GRF(A0 + 4 * i).hf());
                        dpas(16, 8, 8, acc.f(), acc.f(), GRF(j ? B1 : B0).hf(), GRF(A0 + 16 + 4 * i).hf());
                    }
                add(1, st, st, 1);
                cmp(1 | lt | f1[0], st, nst);
                jmpi(1 | f1[0], lBias);
            }
        } else {
            // Pipeline : deux blocs de 64 K par itération (quatre demi-blocs), A en double tampon par demi-bloc, mots
            // en double tampon par bloc ; le chargement du demi-bloc suivant part avant le calcul du courant.
            const int AB[2] = {A0, A0b}, WB[2] = {W0, W0b};
            const int gpi = (subs == 1) ? 2 : 1;            // groupes par itération
            shr(1, groups, K, 7);                          // itérations : K / 128
            mov(1, hW.ud(6), 0);
            load(1, GRF(WB[0]), block_2d(DataSizeLSC::D32, 16, 8), A64, hW);
            mov(1, hA.ud(5), 0);
            load(1, GRF(AB[0]), block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hA);
            mark(lGroup);
            for (int h = 0; h < 4; h++) {
                const int blk = h / 2, t2 = h % 2, nh = h + 1;
                if (nh % 2 == 0) {                         // le demi-bloc suivant ouvre un bloc : ses mots
                    add(1, hW.ud(6), hW.ud(6), 8);
                    if (!ab.wload) load(1, GRF(WB[(nh / 2) % 2]), block_2d(DataSizeLSC::D32, 16, 8), A64, hW);
                    add(1, hPW.ud(6), hW.ud(6), 16);       // préchargement deux blocs plus loin
                    load(1, null, block_2d(DataSizeLSC::D32, 16, 8), A64, hPW);
                    add(1, hPA.ud(5), hA.ud(5), 32 + 128);
                    load(1, null, block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hPA);
                    add(1, hPA.ud(5), hA.ud(5), 64 + 128);
                    load(1, null, block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hPA);
                }
                add(1, hA.ud(5), hA.ud(5), 32);
                if (!ab.aload) load(1, GRF(AB[nh % 2]), block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hA);
                const int gl = blk / subs;                 // groupe dans l'itération
                const bool start = (h % (2 * subs)) == 0, end = ((h + 1) % (2 * subs)) == 0;
                if (start) {
                    add(1, t.d(7), g, gl);
                    scales(t.d(7));
                }
                half(AB[h % 2], WB[blk % 2], t2, start);
                if (end) {
                    add(1, t.d(7), g, gl);
                    fold(t.d(7));
                }
            }
            add(1, g, g, gpi);
            add(1 | gt | f1[0], groups, groups, -1);
            jmpi(1 | f1[0], lGroup);
        }

        if (swiglu) {
            // Activation de 8 lignes × 16 colonnes par tuile de lignes i : colonnes d'activation n0/2 … + 15.
            const GRF X = GRF(PART), E = GRF(PART + 1), Q = GRF(PART + 2), U = GRF(PART + 3), O = GRF(PART + 4);
            shr(1, hC.ud(5), n0, 1);
            for (int i = 0; i < 4; i++) {
                for (int r = 0; r < 8; r++) {
                    const GRF g = GRF(ACC + 8 * (2 * i) + r), u = GRF(ACC + 8 * (2 * i + 1) + r);
                    mov(16, Q.hf(0)(1), g.f());              // g arrondi fp16
                    mov(16, X.f(), Q.hf(0)(1));
                    mov(16, Q.hf(16)(1), u.f());             // u arrondi fp16
                    mov(16, U.f(), Q.hf(16)(1));
                    mul(16, E.f(), X.f(), -1.4426950408889634f);
                    math(16, MathFunction::exp, E.f(), E.f());
                    add(16, E.f(), E.f(), 1.0f);
                    math(16, MathFunction::inv, E.f(), E.f());
                    mul(16, X.f(), X.f(), E.f());            // SiLU(g)
                    mul(16, O.f(), X.f(), U.f());
                    shr(16, E.ud(), O.ud(), 16);             // arrondi bf16 au plus proche pair
                    and_(16, E.ud(), E.ud(), 1);
                    add(16, E.ud(), E.ud(), 0x7FFF);
                    add(16, O.ud(), O.ud(), E.ud());
                    and_(16, O.ud(), O.ud(), 0xFFFF0000u);
                    mov(16, GRF(H + r / 2).hf((r % 2) * 16)(1), O.f());
                }
                add(1, hC.ud(6), m0, 8 * i);
                store(1, block_2d(DataSizeLSC::D16, 16, 8), A64, hC, GRF(H));
            }
        } else
        // Sortie : fp16 arrondi au plus proche pair, écrite par blocs 2D de 8 lignes × 16 colonnes.
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < tn; j++) {
                for (int r = 0; r < 8; r++)
                    mov(16, GRF(H + r / 2).hf((r % 2) * 16)(1), GRF(ACC + 8 * (i * tn + j) + r).f());
                add(1, hC.ud(6), m0, 8 * i);
                add(1, hC.ud(5), n0, 16 * j);
                store(1, block_2d(DataSizeLSC::D16, 16, 8), A64, hC, GRF(H));
            }
        mark(lEnd);
        epilogue();
    }
};

// W4A8 (option locale, prompt) : C[M, N] = A8[M, K] · Wᵀ sur DPAS int8. A8 int8 [M, K], quantifié par (ligne, groupe de
// 128) avec l'échelle SA [K/128, Mp] fp32, ses K rangés par blocs de 16 (deux mots w0, w1) dans l'ordre des quatre
// mots doubles w0 & m, w1 & m, (w0 >> 4) & m, (w1 >> 4) & m (m = 0x0F0F0F0F, nos mots aux quartets entrelacés) :
// [k0 k4 k1 k5 | k8 k12 k9 k13 | k2 k6 k3 k7 | k10 k14 k11 k15], trois instructions SIMD32 pour deux mots. Par
// groupe : quatre DPAS k32 en int32 à partir de 0, puis acc += float(part)·s_w·s_a (deux lignes par instruction).
// Mots par groupe et A par moitié de groupe (64 K) : des chargements plus fins par pas k32, en double tampon, sont
// plus lents (158 contre 193 TOPS sur gate|up).
// Le biais (b × 128, Ahi/Alo des activations déquantifiées) par DPAS fp16 en fin de noyau, comme w4g_dpb.
// Sous-groupe 32 lignes × 32 colonnes, groupe de travail 64 × 256, même grille (GM, N/256, paquets). GPTQ g128.
// swiglu : épilogue SwiGLU de w4g_dpb_swiglu (gate|up entrelacés par 16).
template <HW hw>
class W4A8Prompt : public ELFCodeGenerator<hw> {
    NGEN_FORWARD_ELF(hw)

    void mul32(const Subregister &dst, const Subregister &a, const Subregister &b, const Subregister &tmp) {
        mul(1, tmp, a, b.uw(1));
        shl(1, tmp, tmp, 16);
        mul(1, dst, a, b.uw(0));
        add(1, dst, dst, tmp);
    }
    void header(const GRF &h, const Subregister &base, const Subregister &wbytes_m1, const Subregister &height_m1,
                int bw, int bh, int count) {
        mov(1, h.uq(0), base);
        mov(1, h.ud(2), wbytes_m1);
        mov(1, h.ud(3), height_m1);
        mov(1, h.ud(4), wbytes_m1);
        mov(1, h.ud(7), (bw - 1) | ((bh - 1) << 8) | ((count - 1) << 16));
    }

public:
    W4A8Prompt(Product product, bool swiglu) : ELFCodeGenerator<hw>(product) {
        newArgument("A", ExternalArgumentType::GlobalPtr);
        newArgument("W", ExternalArgumentType::GlobalPtr);
        newArgument("S", ExternalArgumentType::GlobalPtr);
        newArgument("B", ExternalArgumentType::GlobalPtr);
        newArgument("C", ExternalArgumentType::GlobalPtr);
        newArgument("M", DataType::d);
        newArgument("N", DataType::d);
        newArgument("K", DataType::d);
        newArgument("Ahi", ExternalArgumentType::GlobalPtr);
        newArgument("Alo", ExternalArgumentType::GlobalPtr);
        newArgument("Mp", DataType::d);
        newArgument("G", DataType::d);
        newArgument("SA", ExternalArgumentType::GlobalPtr);
        externalName(swiglu ? "w4a8_swiglu" : "w4a8");
        requireSIMD(16);
        requireGRF(256);
        requireLocalID(1);
        requireDPAS();
        finalizeInterface();

        setDefaultNoMask();
        setDefaultAutoSWSB(true);
        prologue();
        or_(1, cr0, cr0, uint16_t(0x14C0));

        const auto A = getArgument("A"), W = getArgument("W"), S = getArgument("S"), B = getArgument("B"),
                   C = getArgument("C"), M = getArgument("M"), N = getArgument("N"), K = getArgument("K"),
                   Mp = getArgument("Mp"), SA = getArgument("SA");

        // Registres : en-têtes et scalaires r32-r47 ; W 48-79 (deux tuiles × 16 lignes de mots = 128 K) ; A 80-111
        // (64 K × 32 lignes int8) ; B 112-119 / 120-127 ; part (int32) 128-191 ; acc (fp32) 192-255.
        const GRF hA = r32, hW = r33, hC = r34, s = r35, t = r36, addr = r37, swraw = r38, hPW = r39;
        const GRF sw0 = r40, sw1 = r41, sa0 = r42, ftmp = r44, hPA = r46;   // ftmp : r44-r45
        const int W0 = 48, A0 = 80, BB[2] = {112, 120}, PART = 128, ACC = 192;
        Label lEnd, lGroup;

        const Subregister m0 = s.d(0), n0 = s.d(1), g = s.d(2), groups = s.d(3), mt = s.d(4), sg = s.d(5);
        const Subregister kx = s.d(6), tmp = s.d(7);
        mul(1, mt, r0.ud(7), GM);
        add(1, mt, mt, r0.ud(1));
        shl(1, tmp, mt, 6);
        cmp(1 | ge | f0[0], tmp, M);
        jmpi(1 | f0[0], lEnd);
        shr(1, sg, getLocalID(0).uw(0), 4);
        and_(1, t.d(0), sg, 1);
        shl(1, t.d(0), t.d(0), 5);
        add(1, m0, tmp, t.d(0));                          // m0 = mt·64 + (sg % 2)·32
        shr(1, t.d(1), sg, 1);
        shl(1, t.d(1), t.d(1), 5);
        shl(1, n0, r0.ud(6), 8);
        add(1, n0, n0, t.d(1));                           // n0 = nt·256 + (sg / 2)·32

        add(1, t.d(2), K, -1);                            // K − 1 (octets d'une ligne de A8)
        add(1, t.d(3), M, -1);
        header(hA, A, t.d(2), t.d(3), 32, 32, 2);
        mov(1, hA.ud(6), m0);
        header(hPA, A, t.d(2), t.d(3), 32, 32, 2);
        mov(1, hPA.ud(6), m0);
        shl(1, t.d(4), N, 2);
        add(1, t.d(4), t.d(4), -1);                       // N·4 − 1
        shr(1, t.d(5), K, 3);
        add(1, t.d(5), t.d(5), -1);                       // K/8 − 1
        header(hW, W, t.d(4), t.d(5), 16, 16, 1);
        header(hPW, W, t.d(4), t.d(5), 16, 16, 1);
        if (swiglu)
            add(1, t.d(6), N, -1);
        else {
            shl(1, t.d(6), N, 1);
            add(1, t.d(6), t.d(6), -1);
        }
        header(hC, C, t.d(6), t.d(3), 16, 8, 1);
        mov(1, hC.ud(5), n0);

        for (int r = 0; r < 64; r++) mov(16, GRF(ACC + r).f(), 0.0f);
        mov(1, g, 0);
        shr(1, groups, K, 7);

        mark(lGroup);
        // Mots du groupe (16 lignes de mots par tuile), préchargement du groupe suivant (mots et A).
        shl(1, hW.ud(6), g, 4);
        mov(1, hW.ud(5), n0);
        load(1, GRF(W0), block_2d(DataSizeLSC::D32, 16, 16), A64, hW);
        add(1, hW.ud(5), n0, 16);
        load(1, GRF(W0 + 16), block_2d(DataSizeLSC::D32, 16, 16), A64, hW);
        add(1, hPW.ud(6), hW.ud(6), 16);
        mov(1, hPW.ud(5), n0);
        load(1, null, block_2d(DataSizeLSC::D32, 16, 16), A64, hPW);
        add(1, hPW.ud(5), n0, 16);
        load(1, null, block_2d(DataSizeLSC::D32, 16, 16), A64, hPW);
        shl(1, kx, g, 7);
        add(1, hPA.ud(5), kx, 128);
        load(1, null, block_2d(DataSizeLSC::D8, 32, 32, 2), A64, hPA);
        add(1, hPA.ud(5), kx, 192);
        load(1, null, block_2d(DataSizeLSC::D8, 32, 32, 2), A64, hPA);
        // Échelles des poids (32 colonnes, fp16 → fp32) et des activations (32 lignes, fp32).
        mul32(t.d(0), g, N, t.d(5));
        add(1, t.d(0), t.d(0), n0);
        shl(1, t.d(0), t.d(0), 1);
        mov(1, addr.uq(0), t.ud(0));
        add(1, addr.uq(0), addr.uq(0), S);
        load.ugm(1, swraw, block(DataSizeLSC::D32, 16), A64, addr);
        mul32(t.d(1), g, Mp, t.d(5));
        add(1, t.d(1), t.d(1), m0);
        shl(1, t.d(1), t.d(1), 2);
        mov(1, addr.uq(0), t.ud(1));
        add(1, addr.uq(0), addr.uq(0), SA);
        load.ugm(1, sa0, block(DataSizeLSC::D32, 32), A64, addr);
        for (int h = 0; h < 2; h++) {
            add(1, hA.ud(5), kx, 64 * h);
            load(1, GRF(A0), block_2d(DataSizeLSC::D8, 32, 32, 2), A64, hA);
            for (int st = 0; st < 2; st++) {
                const int step = 2 * h + st;              // pas k32 du groupe : lignes de mots 4·step … + 3
                for (int j = 0; j < 2; j++) {
                    const int bb = BB[j];
                    for (int p = 0; p < 2; p++) {          // B : [w0 & m, w1 & m, (w0 >> 4) & m, (w1 >> 4) & m, puis w2, w3]
                        const GRF wd = GRF(W0 + 16 * j + 4 * step + 2 * p);
                        and_(32, GRF(bb + 4 * p).ud(), wd.ud(), 0x0F0F0F0Fu);
                        shr(32, GRF(bb + 4 * p + 2).ud(), wd.ud(), 4);
                        and_(32, GRF(bb + 4 * p + 2).ud(), GRF(bb + 4 * p + 2).ud(), 0x0F0F0F0Fu);
                    }
                    for (int i = 0; i < 4; i++) {
                        const GRF part = GRF(PART + 8 * (i * 2 + j)), a = GRF(A0 + 16 * st + 4 * i);
                        if (step == 0)
                            dpas(16, 8, 8, part.d(), null.d(), GRF(bb).ub(), a.b());
                        else
                            dpas(16, 8, 8, part.d(), part.d(), GRF(bb).ub(), a.b());
                    }
                }
            }
        }
        // acc += float(part)·s_w·s_a
        mov(16, sw0.f(), swraw.hf(0)(1));
        mov(16, sw1.f(), swraw.hf(16)(1));
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                for (int r = 0; r < 8; r += 2) {
                    const int row = 8 * i + r;
                    const GRF part = GRF(PART + 8 * (i * 2 + j) + r), acc = GRF(ACC + 8 * (i * 2 + j) + r);
                    mov(32, ftmp.f(), part.d());
                    mul(32, ftmp.f(), ftmp.f(), (j ? sw1 : sw0).f(0)(0, 16, 1));
                    for (int q = 0; q < 2; q++)
                        mad(16, GRF(acc.getBase() + q).f(), GRF(acc.getBase() + q).f(),
                            GRF(sa0.getBase() + (row + q) / 16).f((row + q) % 16), GRF(ftmp.getBase() + q).f());
                }
        add(1, g, g, 1);
        cmp(1 | lt | f1[0], g, groups);
        jmpi(1 | f1[0], lGroup);

        // Biais par DPAS fp16 : acc += Ahi·(b·128) + Alo·(b·128), par pas de 16 groupes (comme w4g_dpb).
        {
            const auto Ahi = getArgument("Ahi"), Alo = getArgument("Alo"), Gs = getArgument("G");
            const GRF hAH = r46, hAL = r47, hBB = r43;
            const Subregister st = r45.d(0), nst = r45.d(1);
            Label lBias;
            shl(1, t.d(0), Gs, 1);
            add(1, t.d(0), t.d(0), -1);
            add(1, t.d(1), M, -1);
            header(hAH, Ahi, t.d(0), t.d(1), 16, 32, 1);
            header(hAL, Alo, t.d(0), t.d(1), 16, 32, 1);
            mov(1, hAH.ud(6), m0);
            mov(1, hAL.ud(6), m0);
            shl(1, t.d(2), N, 1);
            add(1, t.d(2), t.d(2), -1);
            add(1, t.d(3), Gs, -1);
            header(hBB, B, t.d(2), t.d(3), 16, 16, 1);
            add(1, nst, Gs, 15);
            shr(1, nst, nst, 4);
            mov(1, st, 0);
            mark(lBias);
            shl(1, hAH.ud(5), st, 4);
            shl(1, hAL.ud(5), st, 4);
            shl(1, hBB.ud(6), st, 4);
            load(1, GRF(A0), block_2d(DataSizeLSC::D16, 16, 32), A64, hAH);
            load(1, GRF(A0 + 16), block_2d(DataSizeLSC::D16, 16, 32), A64, hAL);
            for (int j = 0; j < 2; j++) {
                add(1, hBB.ud(5), n0, 16 * j);
                load(1, GRF(BB[j]), block_2d(DataSizeLSC::D16, 16, 16) | vnni, A64, hBB);
            }
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 4; i++) {
                    const GRF acc = GRF(ACC + 8 * (i * 2 + j));
                    dpas(16, 8, 8, acc.f(), acc.f(), GRF(BB[j]).hf(), GRF(A0 + 4 * i).hf());
                    dpas(16, 8, 8, acc.f(), acc.f(), GRF(BB[j]).hf(), GRF(A0 + 16 + 4 * i).hf());
                }
            add(1, st, st, 1);
            cmp(1 | lt | f1[0], st, nst);
            jmpi(1 | f1[0], lBias);
        }

        const int H = A0;                                  // sortie fp16 : 4 registres
        if (swiglu) {
            const GRF X = GRF(PART), E = GRF(PART + 1), Q = GRF(PART + 2), U = GRF(PART + 3), O = GRF(PART + 4);
            shr(1, hC.ud(5), n0, 1);
            for (int i = 0; i < 4; i++) {
                for (int r = 0; r < 8; r++) {
                    const GRF gg = GRF(ACC + 8 * (2 * i) + r), u = GRF(ACC + 8 * (2 * i + 1) + r);
                    mov(16, Q.hf(0)(1), gg.f());
                    mov(16, X.f(), Q.hf(0)(1));
                    mov(16, Q.hf(16)(1), u.f());
                    mov(16, U.f(), Q.hf(16)(1));
                    mul(16, E.f(), X.f(), -1.4426950408889634f);
                    math(16, MathFunction::exp, E.f(), E.f());
                    add(16, E.f(), E.f(), 1.0f);
                    math(16, MathFunction::inv, E.f(), E.f());
                    mul(16, X.f(), X.f(), E.f());
                    mul(16, O.f(), X.f(), U.f());
                    shr(16, E.ud(), O.ud(), 16);
                    and_(16, E.ud(), E.ud(), 1);
                    add(16, E.ud(), E.ud(), 0x7FFF);
                    add(16, O.ud(), O.ud(), E.ud());
                    and_(16, O.ud(), O.ud(), 0xFFFF0000u);
                    mov(16, GRF(H + r / 2).hf((r % 2) * 16)(1), O.f());
                }
                add(1, hC.ud(6), m0, 8 * i);
                store(1, block_2d(DataSizeLSC::D16, 16, 8), A64, hC, GRF(H));
            }
        } else
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++) {
                    for (int r = 0; r < 8; r++)
                        mov(16, GRF(H + r / 2).hf((r % 2) * 16)(1), GRF(ACC + 8 * (i * 2 + j) + r).f());
                    add(1, hC.ud(6), m0, 8 * i);
                    add(1, hC.ud(5), n0, 16 * j);
                    store(1, block_2d(DataSizeLSC::D16, 16, 8), A64, hC, GRF(H));
                }
        mark(lEnd);
        epilogue();
    }
};

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage : %s xe2|xe3 64|128 SORTIE.zebin [simple]\n", argv[0]);
        return 2;
    }
    const int gsz = std::atoi(argv[2]);
    bool pipe = true;
    int tn = 1, simdFold = 16;
    bool lag = false, dpb = false, slmA = false;
    int layout = 0;
    bool ilv = false, s4 = false, asplit = false, b3 = false, swiglu = false;
    typename W4Prompt<HW::Xe2>::Ablate ab;
    for (int i = 4; i < argc; i++) {
        if (!std::strcmp(argv[i], "simple")) pipe = false;
        if (!std::strcmp(argv[i], "nounpack")) ab.unpack = true;
        if (!std::strcmp(argv[i], "nofold")) ab.fold = true;
        if (!std::strcmp(argv[i], "noaload")) ab.aload = true;
        if (!std::strcmp(argv[i], "nowload")) ab.wload = true;
        if (!std::strcmp(argv[i], "nosload")) ab.sload = true;
        if (!std::strcmp(argv[i], "nopf")) ab.pf = true;
        if (!std::strcmp(argv[i], "nobias")) ab.bias = true;
        if (!std::strcmp(argv[i], "tn2")) tn = 2;
        if (!std::strcmp(argv[i], "fold32")) simdFold = 32;
        if (!std::strcmp(argv[i], "lag")) lag = true;
        if (!std::strcmp(argv[i], "dpb")) dpb = true;
        if (!std::strcmp(argv[i], "slmA")) slmA = true;
        if (!std::strncmp(argv[i], "layout", 6)) layout = std::atoi(argv[i] + 6);
        if (!std::strcmp(argv[i], "ilv")) ilv = true;
        if (!std::strcmp(argv[i], "s4")) s4 = true;
        if (!std::strcmp(argv[i], "asplit")) asplit = true;
        if (!std::strcmp(argv[i], "b3")) b3 = true;
        if (!std::strcmp(argv[i], "swiglu")) swiglu = true;
    }
    bool a8 = false;
    for (int i = 4; i < argc; i++)
        if (!std::strcmp(argv[i], "a8")) a8 = true;
    if (a8) {                                          // W4A8 (GPTQ g128) : seules options, swiglu
        auto bin8 = W4A8Prompt<HW::Xe2>(Product(ProductFamily::GenericXe2, 0, PlatformType::Discrete), swiglu).getBinary();
        std::ofstream(argv[3], std::ios::binary).write(reinterpret_cast<const char *>(bin8.data()), bin8.size());
        std::printf("%s : %zu octets\n", argv[3], bin8.size());
        return 0;
    }
    if (tn == 2) pipe = false;
    std::vector<uint8_t> bin;
    if (std::strcmp(argv[1], "xe2") == 0)
        bin = W4Prompt<HW::Xe2>(Product(ProductFamily::GenericXe2, 0, PlatformType::Discrete), gsz, pipe, ab, tn, simdFold, lag, dpb, slmA, layout, ilv, s4, asplit, b3, swiglu).getBinary();
    else
        bin = W4Prompt<HW::Xe3>(Product(ProductFamily::GenericXe3, 0, PlatformType::Integrated), gsz, pipe,
                                {ab.unpack, ab.fold, ab.aload, ab.wload, ab.sload, ab.pf, ab.bias}, tn, simdFold, lag, dpb, slmA, layout, ilv, s4, asplit, b3, swiglu)
                  .getBinary();
    std::ofstream(argv[3], std::ios::binary).write(reinterpret_cast<const char *>(bin.data()), bin.size());
    std::printf("%s : %zu octets\n", argv[3], bin.size());
    return 0;
}
