# Backend XPU (Intel Arc Pro B70) du moteur Zig de TensorFold

Portage **non officiel** : un backend `xpu` pour le moteur Zig de TensorFold, qui pilote une carte Intel Xe2
(Arc Pro B70, Battlemage G31, 32 Go) directement par **Level Zero**, sans PyTorch, sans SYCL, sans Python à
l'exécution. Il sert **Qwen3.8-27B** derrière l'API OpenAI du serveur `tensorfold-native`, avec le brouillonneur
DFlash2 en arbre.

## Ce qu'il sert

| | |
|---|---|
| Modèle | Qwen3.8-27B (`model_type` `qwen3_5` : 48 couches GDN + 16 d'attention) |
| Formats | MLX affine 4 bits groupes de 64 (`mlx-q4g64`), GPTQ symétrique 4 bits groupes de 128 (`gptq-int4-g128`) |
| Brouillons | DFlash2 (`--drafter`), arbre best-first de 15 nœuds, glouton et échantillonné |
| Service | plusieurs flux (`--parallel N`, 8 par défaut, hôte « lanes » du moteur) : tours partagés, un flux seul en arbre DFlash2 ; `--parallel 1` : hôte série. MoE : voir [plan-multiflux-moe.md](plan-multiflux-moe.md) |

Au format GPTQ, les couches sont en int4 g128 telles que stockées. Les portes `in_proj_a/b` du GDN restent en fp16
dense, comme le checkpoint, et l'embedding est fp16. La tête de sortie est requantifiée au chargement en affine
4/64 par l'algorithme de MLX : sa justesse est celle de la tête 4 bits du checkpoint MLX (même jeton que la tête
fp16 dans 97,4 % des cas, KL 3,9e-3 ; `tools/xpu/head_study.py`).

## Construire

Prérequis : Zig 0.17, le pilote `xe` et le runtime de calcul Intel (Level Zero), `ocloc` (compilateur hors ligne
d'Intel), un compilateur C++17 et les en-têtes nGEN de oneDNN.

```sh
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=<chemin vers ngen>   # zig-out/xpu/bin/tensorfold-native
zig build xpu-qwen27 -Doptimize=ReleaseFast -Dngen-include=<chemin vers ngen>   # exécutable de test xpu-qwen27
```

Les noyaux OpenCL C sont compilés par `ocloc` pour `-Dxpu-device` (défaut `bmg`). Les GEMM du prompt sont générés en
ISA par nGEN au moment du build (`zig/tools/xpu/w4_ngen.cpp`) ; tout est embarqué dans l'exécutable.

## Lancer

```sh
tensorfold-native serve MODELE --drafter DFLASH2 --parallel 8 --context 16384 --host 127.0.0.1 --port 8081
```

Avec `--parallel N` (N > 1), chaque flux a son emplacement sur la carte : état GDN (2 × 151 Mo, l'état et sa
sauvegarde pour la vérification) et cache K/V (64 Ko par jeton de `--context`). Le nombre d'emplacements est borné
par la mémoire restante après le chargement (`slotsThatFit`). Un flux seul qui demande des brouillons tourne en
arbre DFlash2. Dès qu'un autre arrive, le cœur le reprend dans des tours partagés : un forward pour les fenêtres de
tous les flux (chacune une chaîne : le jeton en attente, puis les copies du contexte ou une chaîne DFlash2), 16 lignes
au plus. Chaque emplacement a son contexte DFlash2 (K/V du brouillonneur en anneau de 2 080 positions, ~42 Mo) qui
absorbe toutes ses lignes gardées ; les chaînes de tous les flux sont proposées en une passe du brouillonneur. Quand
les autres flux finissent, celui qui reste repasse en arbre DFlash2 (`Lone.resume`).

Options par variables d'environnement :

| Variable | Effet |
|---|---|
| `XPU_PREFILL_INT4=1` | **Option locale**, avec `XPU_PREFILL_INT8=1` et un checkpoint tourné (`zig/tools/xpu/rotate_w4.py`, repère `hadamard128`) : entrée de `down` en int4 (DPAS u4 × s4), +8 % de préremplissage à qualité égale au GPTQ W4A8 ; `XPU_PREFILL_INT4_SITES=10` y ajoute les sorties de mixer (+12 %, perplexité +1 %). |
| `XPU_KV_INT8=1` | Cache K/V en int8 (échelle par position et tête K/V) : moitié de la mémoire, 12 flux au lieu de 7 à 16 k jetons, 7 au lieu de 3 à 32 k ; perplexité inchangée (±0,04 %). Attention du prompt `attn_prompt8` : Q·Kᵀ en DPAS int8, P·V en fp16. |
| `XPU_ATTN_TRITON=1` | Attention du prompt par le noyau Triton du portage Python au lieu de la nôtre (`attn_prompt`) |
| `XPU_PREFILL_INT8=1` | **Option locale** : GEMM du prompt en W4A8, activations int8 par (ligne, groupe de 128), GPTQ seulement. Le décodage reste en fp16. |
| `XPU_TREE=N` | Nœuds de l'arbre DFlash2 par tour (15 par défaut ; 0 : une chaîne) |
| `XPU_BLOCK=N` | Lignes du bloc du brouillonneur (8) |

## Chiffres (B70 à 230 W, serveur `--parallel 8 --context 16384`, validation du 2026-10-13)

| | MLX g64 | GPTQ g128 | GPTQ g128 + W4A8 | RTN tourné + W4A4 |
|---|---|---|---|---|
| Préremplissage 1 k / 4 k / 8 k jetons (tok/s) | 1 764 / 1 754 / 1 723 | 1 956 / 1 932 / 1 918 | 2 576 / 2 554 / 2 505 | 2 791 / 2 769 / 2 714 |
| Un flux seul, arbre DFlash2, code / chat, glouton | 128 / 98 | 132 / 89 | 132 / 96 | 125 / 98 |
| 2 / 4 / 8 flux simultanés, code, glouton (agrégé) | 197 / 293 / 324 | 222 / 309 / 269 | 222 / 309 / 269 | 202 / 287 / 260 |
| 2 / 4 / 8 flux simultanés, chat, glouton (agrégé) | 143 / 238 / 299 | 129 / 216 / 207 | 153 / 230 / 223 | 155 / 249 / 226 |
| Décodage `bench_openai`, 1 / 2 / 4 / 8 requêtes (agrégé) | 172 / 245 / 340 / 345 | 165 / 304 / 336 / 469 | 165 / 321 / 364 / 457 | 178 / 308 / 366 / 474 |

Avant les chaînes DFlash2 par flux (copies seules dans les tours partagés), 2 / 4 / 8 flux de code faisaient
64 / 120 / 182 tok/s en GPTQ. Validation : `zig build test`, oracle, drafté == série, multi-flux == série, et
`bench_concurrent --alone --serial` sur 1, 2, 4 et 8 flux, en glouton et en échantillonné, pour les quatre
configurations : 64 cellules toutes égales, aucune erreur dans les journaux des serveurs.

Bancs : `bench_openai.py` (préremplissage, décodage) et `bench_concurrent.py --alone --serial` (brouillons), dans
`tools/` de la branche `main` de ce dépôt. Le débit drafté dépend du texte produit ; les deux formats n'écrivent pas le
même.

Le préremplissage est **borné par la puissance** : à 230 W, la carte tourne à ~2 180 MHz pendant un prompt (PL2),
pas aux 2 600 du DPAS pur. Le GEMM gate|up seul tient 146 TFLOPS fp16 et ~209 TOPS en int8.

## Justesse : ce qui est vérifié

- **Oracle** : les jetons gloutons égalent ceux de la référence torch fp32 du portage Python (même prompt, 32
  jetons), en MLX et en GPTQ, avec et sans W4A8.
- **Drafté == série** : bit à bit, avec des brouillons parfaits, faux et réels, en chaîne et en arbre, en glouton
  et en échantillonné (`xpu-qwen27` avec `XPU_VERIFY_TEST`, et `bench_concurrent --alone --serial` sur le serveur).
- **Plusieurs flux == un seul** : bit à bit, jusqu'à 16 emplacements, avec des fenêtres de 1 et 4 lignes et des
  conservations partielles (`XPU_MULTI_TEST`), et chaque réponse concurrente égale à la même requête seule
  (`bench_concurrent --alone`).
- **W4A8** : perplexité sur trois textes bruts (anglais, code, français) +0,04 à +0,13 %, KL ≈ 3e-3 face au fp16 ;
  pour comparaison, MLX et GPTQ s'écartent l'un de l'autre de 0,06 à 0,11 en KL
  (`tools/xpu/a8_study.py`).
- **Mesurer sur du texte brut** : dans un tour utilisateur de chat, le modèle (référence fp32 comprise) prédit mal
  le jeton suivant, car il n'est entraîné que sur les réponses. Les perplexités y sont sans signification.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `zig/src/xpu/ze.zig` | Accès à Level Zero : mémoire, modules, noyaux, listes de commandes |
| `zig/src/xpu/qwen27.zig` | Le modèle : chargement (MLX, GPTQ), forward couche par couche, `Runner` (`prompt`, `step`, `verify`, `verifyTree`, `commit`, `commitPath` ; emplacements `select`, `verifyMulti`, `keepMulti`) |
| `zig/src/xpu/drafter.zig` | DFlash2 : contexte de taps, bloc proposé, sélecteur, arbre best-first, `treeRound` |
| `zig/src/native/xpu.zig` | La prise sur le serveur : hôte série (`--parallel 1`) ou hôte « lanes » |
| `zig/src/native/xpu_lanes.zig` | Le backend `lanes` : emplacements, tours partagés (`verifyMulti`, `keepMulti`), flux seul en arbre (`lone`) |
| `zig/src/xpu/qwen27_main.zig` | Exécutable de test : oracle, contrats, profils, sondes |
| `zig/kernels/xpu/qwen27/*.cl` | Noyaux OpenCL C : GEMM int4 du décodage, GDN par morceaux et en série, attention de décodage, normes, SwiGLU, quantification int8, brouillonneur |
| `zig/kernels/xpu/qwen27/decode.cl` (`attn_prompt`, `attn_prompt8`) | Attention du prompt : FlashAttention-2 sur DPAS dans le sens de oneDNN (S = K·Qᵀ, une requête par couloir, softmax dans le couloir) ; cache bf16 ou int8 |
| `zig/kernels/xpu/qwen27/triton/prompt_attention.spv` | Ancienne attention du prompt (noyau Triton du portage Python, en SPIR-V), avec `XPU_ATTN_TRITON=1` |
| `zig/tools/xpu/w4_ngen.cpp` | Générateur nGEN des GEMM du prompt : int4 fp16 (`w4g_dpb`, `w4g_s4dpb`, épilogue SwiGLU) et W4A8 (`w4a8`) |
| `zig/build/xpu.zig` | Étapes de build |

Fichiers du cœur modifiés (`zig build test` passe) : `zig/src/core/serial_host.zig` (9 lignes, `draft_sampled` :
brouillons pour les requêtes échantillonnées ; `null` pour les autres backends) ; `zig/src/core/engine_api.zig`
(`Lone.resume`, optionnel), `zig/src/core/lane_host.zig` (un flux resté seul repasse au pilote seul quand le backend le
propose), `zig/src/core/lanes/engine.zig` (`Engine.detach`, l'inverse d'`adopt`).

## Exécutable de test

```sh
xpu-qwen27 MODELE ids.txt 32          # prompt (ids séparés par des espaces), 32 jetons gloutons, temps
xpu-qwen27 MODELE --bench=4096 2      # prompt synthétique de 4 096 jetons (attention : ne remplace pas de vrais jetons)
```

| Variable | Effet |
|---|---|
| `XPU_DRAFTER=DIR`, `XPU_TREE`, `XPU_BLOCK` | tours draftés, comparés à la série |
| `XPU_VERIFY_TEST=1` | contrat drafté == série avec des brouillons parfaits et faux |
| `XPU_MULTI_TEST=N` | contrat multi-flux : N emplacements, tours partagés (une ligne, puis fenêtres de 4 lignes) == série, et débit agrégé |
| `XPU_ROWS_PROBE=1` | coût d'un forward de 1 à 16 lignes |
| `XPU_PROFILE=1` | temps GPU par noyau |
| `XPU_DUMP_H=FICHIER` | états cachés finaux de chaque ligne (études de justesse) |
| `XPU_A8=128` ou `token`, `XPU_A8_SITES` | étude W4A8 : quantification simulée des entrées des GEMM du prompt |
| `XPU_CHUNK`, `XPU_CTX` | lignes par morceau de prompt (1 024), contexte (4 096) |
| `XPU_OVERLAP_PROBE`, `XPU_GEMM_ZEBIN`, `XPU_A8_ZEBIN` (+ `XPU_A8_PROJ`) | sondes : recouvrement GEMM/GDN, GEMM fp16 ou W4A8 en boucle |

## Limites

- Un seul modèle : les dimensions du 27B sont des constantes (`qwen27.zig`) et certaines formes sont écrites dans
  les noyaux (GDN de 128, attention de 256, rapport de têtes 6).
- En multi-flux, les tours partagés vérifient des chaînes (pas d'arbre DFlash2 par flux). Sur un texte très répétitif,
  où les copies suffisent, la passe du brouillonneur coûte ~6-9 % à 4-8 flux. Pas encore de cache de préfixes ni de
  reprise de réponse (`decode_chunk`).
- Formats 4 bits seulement (MLX g64, GPTQ g128 symétrique). Le W4A8 demande le GPTQ g128.
- Les commentaires des noyaux sont en français : à passer en anglais avant toute proposition amont.
