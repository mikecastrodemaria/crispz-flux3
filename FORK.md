# crispz-flux3 : fork FLUX.3 Image Klein de crispz-klein

Fork **texte vers image + édition multi-référence** basé sur
[`black-forest-labs/flux-3-image-klein`](https://huggingface.co/black-forest-labs/flux-3-image-klein)
(5,6B, distillé 16 steps, sans CFG), sorti le 15 octobre 2026.

- Remotes : `origin` = crispz-flux3, `klein` = crispz-klein.
- Base du fork : `3316ef8` (crispz-klein/main, 1.36.6).
- Historique du fork précédent (crispz-qwen-edit vers crispz-klein) : [FORK-klein.md](FORK-klein.md).

> **État : étape 1 sur 5 (squelette).** Le code est encore celui de crispz-klein,
> renommé. Rien ne charge FLUX.3 pour l'instant : voir la feuille de route.

## Pourquoi un fork et pas une 3e variante dans crispz-klein

FLUX.3 n'est pas un FLUX.2 plus gros, c'est une autre mécanique :

| | FLUX.2 klein (crispz-klein) | FLUX.3 Image Klein (ce fork) |
|---|---|---|
| Pipeline diffusers | `Flux2KleinPipeline` + `Flux2KleinInpaintPipeline` | `ModularPipeline` (`Flux3AutoBlocks`), pas d'inpaint |
| Transformer | `Flux2Transformer2DModel` (3072 / 4096) | `Flux3Transformer2DModel`, 5,6B |
| Encodeur texte | Qwen3-4B / Qwen3-8B (texte seul) | Qwen3-VL-4B-Instruct (vision-langage) |
| Layout du repo HF | sous-dossiers diffusers | `model.safetensors`, `ae.safetensors`, `text_encoder/`, `variants/{fp8r,nvfp4}` |
| Steps / CFG | 4 / aucun | 16 / aucun (Base : 50 / CFG) |
| Références | liste `image=[...]` | liste `image=[...]` + `<ref_image_N>` dans le prompt, jusqu'à 10 |
| Extras | | bounding boxes (grille 0-1000, y d'abord), sortie 4K native |
| diffusers | commit figé `de6b049` | diffusers récent (ModularPipeline) |
| Licence | Apache 2.0 (4B) | **FLUX Kommunity License v1.0** |

Mélanger les deux dans un même venv imposait de monter diffusers pour tout le monde,
au risque de casser crispz-klein, validé de bout en bout sur son commit. Ici, le fork
a son propre `.venv` et sa propre pile.

## Licence : à lire avant tout usage commercial

`flux-3-image-klein` est sous **FLUX Kommunity License v1.0** (repo HF à accès
conditionnel : accepter les conditions avec son compte HF, puis un token READ dans
la config `hf_token`).

- Usage non commercial : libre.
- Usage commercial des sorties : gratuit pour un "Qualifying User" dont le chiffre
  d'affaires annualisé (sociétés affiliées comprises) est inférieur à 5 M$.
- Conditions : filtrer ou relire les sorties avant publication, signaler le contenu
  généré par IA quand la loi l'exige, ne pas utiliser les sorties pour entraîner un
  modèle d'image concurrent.
- Au-delà de 5 M$ : licence payante ou API BFL.

L'encodeur Qwen3-VL-4B-Instruct est sous Apache 2.0. Le prompt upsampler
`flux-3-image-klein-pup` (9B) a sa propre licence, à vérifier avant de l'activer.

## Mémoire attendue (chiffres BFL / Hugging Face, à remesurer sur RTX 5090)

| Variante | Pic après chargement | Pic après inférence |
|---|---|---|
| bf16 (H100) | 18,8 Go | 20,7 Go |
| fp8r (H100) | 14,4 Go | 16,3 Go |
| nvfp4 (RTX 6000, Blackwell) | 13,4 Go | 15,3 Go |

La model card annonce des chiffres plus hauts (31 Go en bf16 à 1024x1024 sur B200) :
seule une mesure sur la 5090 tranchera.

Le prompt upsampler ajoute environ 28 Go : sur une 5090 (32 Go), il ne cohabite pas
avec le générateur. Schéma prévu : upsampler, puis déchargement, puis génération
(ou upsampler confié à Ollama, déjà intégré via `cz_ollama.py`).

## Feuille de route

1. **Squelette** (fait) : fork, renommage `crispz-klein` vers `crispz-flux3`,
   version `0.1.0-dev`, ce document.
2. **Cœur** : `cz_pipeline._ensure_base()` charge `ModularPipeline` + `load_components`.
   Défauts 16 steps / sans CFG. Édition multi-référence avec `<ref_image_N>`.
   Nouveau `requirements.txt` (diffusers récent, transformers compatible Qwen3-VL).
3. **Nettoyage de ce qui n'a pas d'équivalent**, avec un message clair plutôt qu'une
   erreur diffusers : inpaint / img2img (pas de `Flux3...InpaintPipeline`), LoRAs et
   checkpoints single-file FLUX.2, gardes 3072/4096, remplacement d'encodeur, cache
   d'embeddings (à réécrire sur le modèle "encode once, generate many"), sélecteur
   4B/9B (remplacé par distillé / base, puis bf16 / fp8r / nvfp4).
4. **Spécificités FLUX.3** : variantes quantifiées (`BFLQuantizationConfig`, nvfp4 sur
   Blackwell avec flashinfer), bounding boxes dans l'UI, prompt upsampler optionnel.
5. **Launcher** `crispz-flux3.pinokio`, puis validation de bout en bout sur RTX 5090
   (mesures du tableau ci-dessus remplacées par les nôtres).

## Sources

- Model card : https://huggingface.co/black-forest-labs/flux-3-image-klein
- Diffusers : https://huggingface.co/blog/flux-3
- Docs BFL : https://docs.bfl.ml/flux_3/flux3_image_klein
- Code de référence : https://github.com/black-forest-labs/flux-3-image-klein
