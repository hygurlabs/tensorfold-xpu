# Backend XPU : plan pour le service multi-flux et les MoE

État de départ (2026-10-12) : le backend XPU sert **un flux à la fois** (`serial_host`) et **un modèle dense**,
Qwen3.8-27B (`qwen3_5`, 48 couches GDN + 16 d'attention), en MLX affine 4/64 ou GPTQ symétrique 4/128. Ce document
fixe l'ordre des travaux avant de les lancer.

## 1. Multi-flux

### Ce qui existe déjà dans le moteur

Le cœur sert plusieurs flux par l'hôte « lanes » (`src/core/lane_host.zig`, `src/core/lanes/`) : un fil possède le
cœur et enchaîne des tours partagés tant qu'un flux vit. Un backend l'alimente par l'interface `lanes.Backend`
(`src/core/lanes/backend.zig`) :

| Entrée | Rôle |
|---|---|
| `prefill` / `prefill_step` | le prompt d'un flux dans son emplacement |
| `first`, `queue`, `read` | le premier jeton, puis les jetons suivants, gardés sur le GPU derrière des poignées |
| `verify(windows)` | un forward commun : une fenêtre (bloc ou arbre drafté) par flux |
| `keep(windows, paths)` | le chemin gardé de chaque fenêtre |
| `draft(requests)` | les brouillons de plusieurs flux |
| `release` | libère l'emplacement |

Le backend CUDA du 27B l'implémente déjà (`src/native/qwen27_lanes.zig`, `qwen27_slots.zig`) : c'est le modèle à
suivre.

### Ce qu'il faut côté XPU

1. **Emplacements par flux** : état GDN (151 Mo), état de convolution, cache KV (64 Ko par jeton, soit 1 Gio à
   16 k jetons). Mesuré : le serveur chargé (GPTQ, DFlash2, un flux à 16 k) occupe 17,8 Gio sur 31,9 ; il reste
   ~14 Gio, soit une douzaine de flux à 16 k ou une vingtaine à 8 k. Le nombre d'emplacements découle de
   `--context` et de `--parallel`.
2. **Forward multi-flux** : les lignes de plusieurs flux dans un même forward. Les GEMM n'y voient que des lignes :
   rien à changer jusqu'à 16 lignes (noyaux de décodage), puis le chemin prompt au-delà. Le reste reçoit une
   **table par ligne** : flux, position, ancêtres. C'est la généralisation directe de la table d'arbre déjà en
   place (`TREE` dans `decode.cl` et `glue.cl`) :
   - attention de décodage : cache K/V et position de la ligne lus via son flux ;
   - GDN en série (`gdn_step`) : état d'entrée et de sortie du flux de la ligne ;
   - convolution (`gdn_pre`) : état d'entrée du flux.
3. **Vérification et commit par fenêtre** : `verifyTree` et `commitPath` acceptent une liste de fenêtres. Le rejeu
   GDN et le déplacement K/V se font par flux.
4. **Brouillons** : DFlash2 par flux (contexte de taps par emplacement), lancés ensemble.
5. **Contrats** : réponse d'un flux seul == la même requête parmi d'autres (`bench_concurrent --alone`), et
   drafté == série (`--serial`), déjà outillés.

**Gain attendu (mesuré).** Le décodage est borné par la lecture des poids. Un forward de 1 ligne coûte 29,1 ms,
de 16 lignes 33,0 ms (`XPU_ROWS_PROBE`, contexte court) : seize flux à une ligne par tour donneraient ~485 tok/s
agrégés, contre 34 pour un flux seul. L'attention grandit avec le contexte de chaque flux, ce qui réduira ce gain
sur de longs contextes. Avec des brouillons, les lignes se partagent entre flux : les noyaux de décodage s'arrêtent
à 16 lignes par forward. Au-delà, il faut des noyaux de décodage à 32-64 lignes, ou passer au chemin prompt.

**Ordre** : (1) emplacements et forward multi-flux, contrat « seul == parmi d'autres » ; (2) fenêtres draftées par
flux ; (3) bancs `bench_concurrent --levels 1,2,4,8`.

**État (2026-10-12)** : (1) fait, avec les fenêtres en chaîne (copies du contexte) et la conservation partielle par
flux (`verifyMulti`, `keepMulti`, `native/xpu_lanes.zig`). Un flux seul garde l'arbre DFlash2. Reste (2) : le
brouillonneur pour chaque flux dans les tours partagés (taps par emplacement, arbres par fenêtre).

## 2. MoE

### Ce qui existe déjà dans le moteur

Des familles MoE sur d'autres backends : `flashnext` (`qwen4_exp`), `glm` (`glm5_next`), `kimi_k3`, ainsi qu'un
routage sur Metal (`src/core/moe_route.zig`).

### La contrainte de la carte

32 Go de VRAM. Les MoE déjà servis par le moteur sont grands (Flash-Next fait plus de 120 Go en NVFP4) : ils ne
tiennent pas sur une B70. Deux voies :

- **A. Un MoE qui tient** (30-35 B paramètres totaux, ~3 B actifs, 15-20 Go en 4 bits) : tout en VRAM, le plus
  simple et le plus rapide. Le décodage ne lit que les experts actifs (~1,5-2 Go par jeton), soit plus de 100 tok/s
  en série en théorie.
- **B. Experts hors carte** : les experts froids en RAM hôte épinglée, lus par PCIe 5.0 (56 Go/s mesurés), les experts
  chauds en VRAM. Le décodage est alors borné par le PCIe pour les experts manqués. Avec 32 Go de RAM sur la
  machine, cette voie reste limitée à ~60 Go de poids.

### Ce qu'il faut côté XPU (voie A d'abord)

1. **Routeur** : GEMM du routeur, softmax ou sigmoïde, top-k, normalisation. Petit noyau, sur le modèle du routage
   Metal existant.
2. **GEMM groupé par expert** :
   - au décodage (≤ 16 lignes), une boucle sur les experts actifs, en réutilisant nos noyaux de décodage int4 sur des
     pointeurs d'expert ;
   - au prompt, les lignes triées par expert (permutation, puis décalages par expert), un GEMM nGEN par segment ou un
     noyau « grouped » qui lit la table des segments.
3. **Combinaison** : la somme pondérée des sorties d'experts et de l'expert partagé, dans l'ordre fixe des experts
   (déterminisme).
4. **Chargement** : les experts empilés par couche dans la disposition GEMM existante (mots entrelacés, échelles
   et biais × G), GPTQ ou MLX.
5. **W4A8** : l'option int8 du prompt s'applique telle quelle aux GEMM d'experts.

**Ordre** : choisir le modèle cible (une famille déjà décrite par le moteur si possible, sinon une nouvelle
famille `config.json`) ; routeur et GEMM groupé au décodage, comparés à une référence torch fp32 ; puis le prompt ;
puis les brouillons, s'il existe un drafter pour ce modèle.

## 3. Prérequis communs

- **Dimensions lues dans `config.json`** au lieu des constantes de `src/xpu/qwen27.zig`, et passées aux noyaux en
  arguments. C'est le prérequis de tout autre modèle, dense ou MoE.
- **Généraliser l'attention** : dimension de tête (64/128/256), rapport de têtes, RoPE partielle ou complète,
  normes q/k optionnelles.
- **Commentaires des noyaux en anglais** avant toute discussion amont.

## 4. EXL3 (mesuré le 2026-10-12, sans poids)

Le décodage EXL3 (codebook `mul1`, `docs/recipes/exl3.md`) est procédural : sa vitesse se mesure sur des flux de bits
aléatoires (`XPU_EXL3_PROBE`, noyau `exl3_gemv` de `decode.cl`). Sur gate|up (K 5 120, N 34 816), une ligne :

| | ms | débit de poids |
|---|---|---|
| notre int4 (GEMV de décodage) | 0,172 | 517 Go/s |
| EXL3 4 / 3 / 2 bits (sonde, tuile en registres) | 5,2 / 5,1 / 5,1 | 17 / 13 / 9 Go/s |

Le temps ne baisse pas avec les bits : le décodage EXL3 est borné par le calcul vectoriel de la B70, pas par la
mémoire. Même parfait (~15 opérations par poids, `dp4a` pour la somme des octets), il prendrait ~0,27 ms : environ
1,6 fois notre int4, quelle que soit la largeur. Au prompt, son décodage par tuile est 10 à 15 fois plus coûteux que
notre dépliage int4, sans voie int8 possible (valeurs fp16 non uniformes). **Pas de gain de vitesse sur cette
carte** ; seul intérêt restant : la capacité (un 70B dense en 3 bits ≈ 26 Go) et la justesse aux bas débits.

## 5. Cache K/V en 8 et 4 bits, rotation de Hadamard (étudié le 2026-10-12)

Mode d'étude `XPU_KV8` (`attn_prep`, `rot_heads`) : q, k, v tournés par Hadamard (256 par tête, la sortie ramenée avant
o_proj : (qH)·(kH) = q·k), k et v quantifiés par (jeton, tête) puis remis en bf16. Perplexité sur les trois textes bruts
de l'étude W4A8 (GPTQ), écart au cache bf16 :

| | doc | code | français |
|---|---|---|---|
| rotation seule (bruit d'arrondi) | +0,00 %, KL 2,9e-4 | +0,03 %, 3,7e-4 | +0,01 %, 3,7e-4 |
| int8 | −0,05 %, 4,4e-4 | +0,02 %, 5,1e-4 | +0,06 %, 4,9e-4 |
| rotation + int8 | −0,02 %, 3,4e-4 | +0,04 %, 4,6e-4 | +0,00 %, 4,2e-4 |
| int4 | +1,10 %, 2,0e-2 | +0,58 %, 1,9e-2 | +1,90 %, 2,9e-2 |
| rotation + int4 | +0,30 %, 8,1e-3 | +0,00 %, 8,4e-3 | +0,90 %, 1,2e-2 |

Le cache int8 est sans perte mesurable, rotation ou non ; la rotation divise par 2,4 l'erreur de l'int4, qui reste
au-dessus du W4A8. Pour l'écrire vraiment en int8 il faut que l'attention du prompt lise l'int8 : le noyau Triton
capturé ne le sait pas, d'où une attention de prompt maison.

**Cache int8 réel (2026-10-12, `XPU_KV_INT8=1`)** : K et V en int8 par (position, tête K/V), échelle float après les
positions ; Q quantifié par requête dans l'attention du prompt (DPAS int8 × int8), P·V en fp16 (int8 → fp16 exact en une
instruction). Perplexité sur les trois textes : −0,02 / −0,02 / +0,04 % (KL 4,6-5,9e-4), contrats drafté == série et
multi-flux == série tenus. Mémoire : 12 flux au lieu de 7 à 16 k jetons, 7 au lieu de 3 à 32 k. Vitesse : attention du
prompt 163 ms contre 155 en bf16 (6,7 k jetons), décodage à 15 k inchangé (l'attention de décodage n'est pas bornée par la
lecture du cache). La rotation de Hadamard n'est utile qu'en int4.

Attention du prompt maison (`attn_prompt`, par défaut) : 155 ms contre 185 pour le noyau Triton sur 6,7 k jetons, mêmes
jetons ; le sens de oneDNN (S = K·Qᵀ) supprime les échanges entre couloirs du softmax, qui coûtaient autant qu'un produit.

## 6. W4A4 et poids tournés (étudié le 2026-10-12, poids bf16 d'origine)

DPAS mesurés (`tools/xpu/dpas_types.cpp`) : fp16 183, int8 367, **int4 × int4 733 TOPS** (résultat vérifié) ; l'int2
accepté par l'assembleur ne calcule que K = 64. Poids : `tools/xpu/rotate_w4.py` refait chaque projection depuis le bf16,
RTN int4 g128 (format GPTQ du moteur), avec ou sans rotation de Hadamard par blocs de 128 le long de K ; le moteur tourne
les activations de même (`XPU_A8_ROT=1`, `XPU_A8_BITS=4`, `XPU_A8_SITES4`). Perplexité face au GPTQ actuel (W4A16), doc /
code / français :

| | doc | code | français |
|---|---|---|---|
| GPTQ + int8 (W4A8 actuel) | +0,04 % | +0,10 % | +0,02 % |
| GPTQ + int4, sans rotation | +12,8 % | +9,1 % | +20,1 % |
| RTN sans rotation, W4A16 | +0,25 % | +0,57 % | +2,13 % |
| **RTN tourné + int8** | **−0,23 %** | **−0,65 %** | **−0,52 %** |
| RTN tourné + int4 partout | +2,77 % | +1,47 % | +4,01 % |
| RTN tourné, int4 sur l'entrée de mixer seulement | +0,34 % | −0,40 % | +0,59 % |
| RTN tourné, int4 sur la sortie de mixer seulement | +0,35 % | −0,13 % | +0,08 % |
| RTN tourné, int4 sur gate\|up seulement | +1,01 % | −0,00 % | +1,74 % |
| RTN tourné, int4 sur down seulement | +0,49 % | −0,61 % | +0,06 % |
| RTN tourné, int4 sauf down | +2,25 % | +0,87 % | +2,92 % |

Sans rotation, l'int4 des activations est inutilisable ; avec, la perte tombe à 1,5-4 %, portée surtout par gate|up.
Les poids RTN tournés battent le GPTQ calibré, dès le W4A8. L'int4 sur down seul (ou sur les sorties de mixer) reste au
niveau du GPTQ actuel.

**Intégration (2026-10-13)** : checkpoint tourné détecté par le fichier `hadamard128` ; chaque entrée de GEMM est tournée là
où elle est produite (normes, portes d'attention et de GDN, SwiGLU ; portes a/b tournées dans le checkpoint), le flux
résiduel et la tête non. Chemin réel, perplexité face au GPTQ W4A16 (doc / code / français) et préremplissage à 6,7 k :

| | perplexité | préremplissage |
|---|---|---|
| GPTQ + W4A8 | +0,17 / +0,11 / +0,15 % | 2 552 tok/s |
| tourné + W4A8 | −0,25 / −0,61 / −0,46 % | 2 517 tok/s |
| tourné + W4A8, int4 sur down (`XPU_PREFILL_INT4=1`) | 0,00 / −0,32 / +0,57 % | 2 753 tok/s |
| tourné + W4A8, int4 sur down et sorties de mixer | +1,09 / +0,02 / +1,06 % | 2 813 tok/s |

Le GEMM W4A4 (`W4A8Prompt` avec `a4`, `ngen_a4`) prend les mots de poids tels quels comme opérande u4 (8 valeurs de K
par mot, ordre des quartets de `repack_gptq`, reproduit par `quant_a4`) : down et sorties de mixer passent de 635 à
385 ms. Le brouillonneur DFlash2 accepte un peu moins sur les poids tournés (4,1 jetons par tour contre 4,7).
