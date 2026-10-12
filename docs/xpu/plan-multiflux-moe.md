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

1. **Emplacements par flux** : état GDN (151 Mo), état de convolution, cache KV (64 Ko par jeton). Avec 32 Go et
   ~15 Go de poids GPTQ, la place restante couvre par exemple 4 flux à 16 k jetons (4 × (0,15 + 1) Go), ou 8 flux à
   8 k. Le nombre d'emplacements découle de `--context` et de `--parallel`.
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

**Gain attendu.** Le décodage est borné par la lecture des poids : 16 lignes coûtent à peine plus qu'une. Le
débit agrégé devrait monter presque linéairement jusqu'à 4-8 flux. Le portage Python mesurait 146 tok/s à 4 flux ;
en Zig, avec l'arbre DFlash2, viser 250-400 tok/s agrégés.

**Ordre** : (1) emplacements et forward multi-flux sans brouillons, contrat « seul == parmi d'autres » ;
(2) fenêtres draftées par flux ; (3) bancs `bench_concurrent --levels 1,2,4,8`.

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
