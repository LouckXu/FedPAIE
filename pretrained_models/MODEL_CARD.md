# FedPAIE pretrained models

This directory contains two global, non-personalized components for the FedPAIE
pipeline described in the [FedPAIE paper](https://arxiv.org/abs/2607.27659).
They are initialization checkpoints, not finished models for a particular user.

## Included models

| File | Purpose | Parameters | Provenance |
|---|---|---:|---|
| `fedpaie_global_scorer_best.safetensors` | Global dual-cue aesthetic scorer used to initialize a user's locally calibrated scorer | 787,202 | Federated round 13, selected by macro validation SRCC = 0.562287 |
| `fedpaie_generic_enhancer_prior.safetensors` | Generic CLUT enhancement prior used to initialize local enhancer personalization | 292,541 | `20+05+20` CLUT-Net, LUT dimension 33 |

Neither file contains a personalized scorer or personalized enhancer. Both use
the non-executable `safetensors` format and contain tensors plus small public
architecture metadata.

## Loading prerequisites

Use the matching FedPAIE release code and install its dependencies, including
PyTorch, torchvision, and `safetensors`. The pretrained MobileNetV3-Large and
LPIPS dependency weights are not embedded in this bundle and may be downloaded
by their respective libraries when first used.

```bash
pip install safetensors
```

Load the global scorer with the release model definition:

```python
from models.scorer.fusion_scorer import FusionScorer
from safetensors.torch import load_file

scorer = FusionScorer()
state_dict = load_file(
    "pretrained_models/fedpaie_global_scorer_best.safetensors",
    device="cpu",
)
scorer.load_state_dict(state_dict, strict=True)
scorer.eval()
```

The scorer expects the release pipeline's 24-dimensional HSV/Lab descriptor
and 960-dimensional frozen MobileNetV3-Large descriptor.

Load the generic enhancer prior as follows:

```python
from models.clut_net import CLUTNet
from safetensors.torch import load_file

enhancer = CLUTNet("20+05+20")
state_dict = load_file(
    "pretrained_models/fedpaie_generic_enhancer_prior.safetensors",
    device="cpu",
)
enhancer.load_state_dict(state_dict, strict=True)
enhancer.eval()
```

The enhancer expects RGB tensors in `[0, 1]` and preserves input resolution.
Always use strict loading so an incompatible code revision fails explicitly.

## What a deployer still needs to do

FedPAIE personalization has two local stages after downloading these global
checkpoints:

1. Collect a local pool of 1–5 user ratings for ordinary photographs. Follow
   one of the paper protocols by sampling 10 or 100 support ratings for scorer
   calibration. The reference code also uses disjoint rated validation data
   for model selection and a test split for research evaluation; 10/100 denotes
   the support size rather than the user's total rated pool. Normalize ratings
   with the release preprocessing and calibrate a local scorer from
   `fedpaie_global_scorer_best.safetensors`.
2. Starting from `fedpaie_generic_enhancer_prior.safetensors`, adapt the
   enhancer on the user's unpaired local photos. The calibrated scorer supplies
   the preference signal, so expert-retouched target images are not required.
3. Save the resulting personalized enhancer checkpoint and validate its output
   on representative user images before use.
4. Deploy only the personalized enhancer for normal image enhancement. The
   scorer is needed during personalization and evaluation, but it is not needed
   for final enhancer inference.

The supplied CLI reproduces a Flickr-AES-style four-split research simulation.
Its shared dataset class currently requires an `image_name` and 1–5 `score`
column even for the `personalization` split, although enhancer adaptation does
not consume those ratings. A real application should implement a local data
adapter that accepts genuinely unrated adaptation photos and may omit the
research-only test split.

The 10- and 100-rating settings are experimental protocols, not a guarantee of
quality for every person or image domain. A deployer should define consent,
storage, deletion, access control, and fallback behavior for local ratings and
photos. Production use also requires device-specific latency, memory, color,
resolution, and output-quality testing.

## Validation evidence and limitations

- The global scorer was selected at federated round 13 by macro validation SRCC
  (`0.5622870247903683`). The corresponding macro validation PLCC is
  `0.577001842838821`, and MSE is `0.06091655246520846`. These are validation
  selection values, not independent test-set claims.
- Under the paper-aligned 224 x 224 evaluation on 500 MIT-Adobe FiveK input to
  Expert C pairs, the generic enhancer prior recorded PSNR
  `22.5984 +/- 3.7648` dB, SSIM `0.903875 +/- 0.055264`, and LPIPS-Alex
  `0.086722 +/- 0.034781`. These measure reference similarity, not subjective
  aesthetics or personalized preference.
- The generic prior is a FedPAIE checkpoint using the CLUT-Net architecture,
  rather than a repackaging of the upstream CLUT checkpoint.
- These research checkpoints provide no production safety, robustness, or
  quality guarantee. The aesthetic score is a learned proxy rather than a
  direct measurement of human preference.

## Data and privacy boundary

No Flickr-AES or MIT-Adobe FiveK image is included. The bundle also excludes
client identities, client lists, personalized checkpoints, per-client metrics,
ratings, photos, optimizer states, training histories, source paths, and full
Python model objects.

FedPAIE is designed so that personal ratings and photos used after release can
remain in the deployer's controlled local environment. This bundle by itself
does not enforce storage or network isolation, differential privacy, secure
aggregation, or any formal privacy guarantee. Deployers remain responsible for
the surrounding system and applicable data-protection requirements.

## License

The two `.safetensors` weight files are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
See `MODEL_LICENSE`. The repository source code remains under its Apache-2.0
license. Datasets and third-party dependency weights are not distributed here
and remain subject to their own terms.

## Integrity verification

From this directory, verify every distributed file listed in the manifest:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

The detailed machine-readable architecture, provenance, validation values, and
per-model hashes are retained in `model_index.json`.

## References

- [FedPAIE paper](https://arxiv.org/abs/2607.27659)
- [CLUT-Net implementation](https://github.com/Xian-Bei/CLUT)
- [Flickr-AES source repository](https://github.com/alanspike/personalizedImageAesthetics)
- [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/)
