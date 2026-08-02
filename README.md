<div align="center">
  <h1>FedPAIE</h1>
  <h3>Learning Color Grading, No Photo Sharing</h3>
  <p><strong>Federated Aesthetic Preference Learning for Personalized Image Enhancement</strong></p>
  <p>
    <a href="https://arxiv.org/abs/2607.27659"><img src="https://img.shields.io/badge/arXiv-2607.27659-b31b1b.svg" alt="arXiv"></a>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-2C5AA0.svg" alt="Apache License 2.0"></a>
    <img src="https://img.shields.io/badge/on--device-0.293M%20parameters-F97316.svg" alt="0.293M on-device parameters">
  </p>
  <p>
    <a href="#method-at-a-glance">Overview</a> ·
    <a href="#installation">Installation</a> ·
    <a href="#pretrained-models">Models</a> ·
    <a href="#data">Data</a> ·
    <a href="#training-and-personalization">Training</a> ·
    <a href="#inference">Inference</a> ·
    <a href="#citation">Citation</a>
  </p>
</div>

<p align="center">
  <a href="assets/teaser.webp"><img src="assets/teaser.webp" width="720" alt="FedPAIE learns personalized color grading while raw photos and ratings remain on users' devices."></a>
</p>

<p align="center"><sub>Global preference knowledge is aggregated without centralizing raw user data, while scorer calibration and color-grading adaptation remain user-specific.</sub></p>

FedPAIE learns personalized image color grading from decentralized, sparse user
ratings. It first simulates federated training of a lightweight dual-cue aesthetic
scorer, calibrates that scorer for an unseen user, and then freezes it to guide local
adaptation of a compact CLUT enhancer from unpaired photographs. Deployment retains
only the personalized enhancer.

> **Release scope.** The source checkout contains code, documentation figures, and the
> two global initialization checkpoints used for personalization. Source datasets,
> per-image ratings, cached features, raw evaluation outputs, and user-specific
> checkpoints are not included.

## Highlights

- **Data-local personalization.** Raw user photos and ratings remain on the simulated
  client while only scorer parameters are aggregated.
- **Sparse preference adaptation.** Personalized Scorer Calibration supports both
  10-shot and 100-shot user feedback.
- **Lightweight throughout.** The scorer has 0.787M trainable parameters, enhancer
  adaptation updates only 0.265M parameters, and on-device inference retains a 0.293M
  personalized enhancer.

## Method at a glance

1. **Federated Aesthetic Preference Learning** trains a global aesthetic scorer with
   weighted regression and square-root FedAvg.
2. **Generic Enhancement Prior Learning** trains a compact CLUT enhancer on paired
   [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) images.
3. **Personalized Scorer Calibration** adapts the global scorer from 10 or 100 local
   support ratings using regression, pairwise ordering, and variance preservation.
4. **Frozen-Scorer-Guided Enhancer Adaptation** updates only the CLUT coefficient
   predictor from unpaired local images. The LUT bases and scorer remain frozen.
5. **Personalized On-Device Inference** uses only the 292,541-parameter enhancer.

| Stage | Trainable parameters | Role |
|---|---:|---|
| Federated scorer training | 0.787M | Shared preference initialization |
| 10-shot scorer calibration | 0.527M | Masked local calibration |
| 100-shot scorer calibration | 0.787M | Full local calibration |
| Enhancer adaptation | 0.265M | CNN backbone and coefficient head |
| On-device inference | 0.293M retained | Personalized color grading |

The scorer combines a 24-D HSV/Lab statistical descriptor with a frozen 960-D
MobileNetV3-Large semantic descriptor. The default enhancer configuration is
`20+05+20`: 20 LUT bases, spatial rank 5, and width rank 20.

<p align="center">
  <a href="assets/pipeline.webp"><img src="assets/pipeline.webp" width="100%" alt="FedPAIE training, personalization, and on-device inference pipeline."></a>
</p>

<p align="center"><sub>FedPAIE pipeline from Federated Aesthetic Preference Learning and Generic Enhancement Prior Learning to local scorer calibration and frozen-scorer-guided enhancer adaptation.</sub></p>

## Personalized color-grading examples

The global enhancer supplies a common initialization, while local adaptation produces
distinct client-specific color transformations without changing spatial content.

<p align="center">
  <a href="assets/qualitative_results.webp"><img src="assets/qualitative_results.webp" width="100%" alt="The same input images receive different personalized color grading for five clients."></a>
</p>

<p align="center"><sub>Representative personalized variation across five unseen clients. Each column uses the same input images and a different calibrated user preference.</sub></p>

## Repository layout

```text
datasets/                 portable Flickr-AES and FiveK data pipelines
evaluation/               FiveK reference evaluation
losses/                   paper-aligned scorer and enhancer objectives
models/scorer/            lightweight dual-cue aesthetic scorer
models/clut_net/          compact residual CLUT enhancer
pretrained_models/        included global scorer and enhancer initializations
training/                 federated training and local personalization
tests/                    lightweight unit tests
inference.py              personalized enhancement entry point
```

## Installation

Python 3.10 or newer is required. Clone the repository and create an isolated
environment from its root:

```bash
git clone https://github.com/LouckXu/FedPAIE.git
cd FedPAIE
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS or Linux, or with
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the runtime
dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For tests and code-quality checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
ruff check .
```

The first semantic-feature extraction may download the standard ImageNet
MobileNetV3-Large weights through Torchvision. LPIPS may likewise download its AlexNet
weights. Pre-download or cache them for offline execution.

## Pretrained models

The two global initialization checkpoints are included directly in the source
repository under the following layout:

```text
pretrained_models/
├── fedpaie_global_scorer_best.safetensors
├── fedpaie_generic_enhancer_prior.safetensors
├── MODEL_CARD.md
├── model_index.json
├── SHA256SUMS.txt
└── MODEL_LICENSE
```

These tensor-only checkpoints are the two shared initializations described in the
paper. The scorer initializes local preference calibration, and the generic enhancer
initializes local CLUT adaptation. They do not contain source images, ratings, client
identifiers, or ready-to-deploy personalized models. Verify the files before use:

```bash
cd pretrained_models
shasum -a 256 -c SHA256SUMS.txt
cd ..
```

## Data

> **Reproduction scope.** The commands below build a deterministic release dataset
> from legally obtained source data. The released global initializations avoid
> retraining the two shared priors, but exact paper numbers additionally depend on
> the paper-matched manifests, preprocessing, cohort, and resolved software environment.

Obtain each dataset from its owner and follow its usage terms:

- [Flickr-AES author repository](https://github.com/alanspike/personalizedImageAesthetics)
- [MIT-Adobe FiveK project page](https://data.csail.mit.edu/graphics/fivek/)

The Flickr-AES pipeline expects:

```text
data/
├── raw/
│   ├── FLICKR-AES_image_labeled_by_each_worker.csv
│   └── flickr_images/
├── split/
│   └── flickr_dp4_split/          generated client CSVs
└── precomputed/
    └── flickr_features/           generated tensor features and index
```

Create deterministic 70/10/10/10 client partitions and cache the fixed descriptors:

```bash
python -m datasets.preprocess_flickr_aes --data-root data
python -m datasets.precompute_features --data-root data --device auto
```

The generated split files contain numeric client IDs only. Original Flickr worker IDs
are discarded before processed outputs are written. Do not publish derived data unless
the dataset terms explicitly permit it.

The official FiveK release provides raw inputs in DNG format and expert renditions as
16-bit ProPhoto RGB TIFF files. Before running the commands below, consistently develop
or export the DNG inputs to ProPhoto RGB TIFF files and place them in
`data/fivek/raw_input`. Place the Expert C TIFF files in
`data/fivek/raw_expert_c`. The current preprocessing command accepts TIFF input and
converts each collection to resized sRGB JPEGs:

```bash
python -m datasets.preprocess_fivek \
  --input-dir data/fivek/raw_input \
  --output-dir data/fivek/input \
  --collection-label input

python -m datasets.preprocess_fivek \
  --input-dir data/fivek/raw_expert_c \
  --output-dir data/fivek/expert_c \
  --collection-label expert_c
```

Create filename-only train/validation/test manifests without copying the images:

```bash
python -m datasets.split_fivek \
  --input-dir data/fivek/input \
  --target-dir data/fivek/expert_c \
  --output-dir data/fivek/splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

The DNG development step is not implemented here and its settings affect the resulting
images. The 80/10/10 split is a deterministic release default rather than the exact
paper evaluation subset. Use the paper-matched manifest when comparing directly with
reported numbers.

### Data required for a new user

The pretrained files are not the final personalized model. A deployer must obtain the
local data lawfully and, with the user's permission, prepare:

- a local support pool from which exactly 10 or 100 images are sampled, each rated on
  the 1–5 scale;
- disjoint rated validation images for checkpoint selection, plus held-out test images
  when reproducing the paper-style evaluation;
- a separate collection of ordinary local photographs for enhancer adaptation, with
  no paired user-specific retouches required.

The current data loader expects the same per-client layout used by the research
simulation:

```text
data/split/flickr_dp4_split/
├── CLIENT_ID_train_fl.csv          rated support pool
├── CLIENT_ID_personalization.csv  unpaired adaptation photographs
├── CLIENT_ID_val.csv               rated validation set
└── CLIENT_ID_test.csv              rated held-out test set
```

Each CSV must contain `image_name` and `score`, and the referenced images must be in
`data/raw/flickr_images`. The scorer draws its 10- or 100-shot support set from
`train_fl`. Enhancer adaptation uses only the photographs in `personalization` and no
retouched targets. Because the shared loader validates a 1–5 `score` column in every
split, that column must currently also be present in the personalization CSV even
though the adaptation objective does not use it. A production data adapter should
relax that schema check for truly unrated photographs instead of inventing ratings.
The 10/100-shot label denotes the sampled calibration support size, not the total
number of rated images expected by this research CLI. The supplied implementation also
uses disjoint validation data for model selection and a test split for reporting. A
production adapter may omit the research-only test split, but must replace the current
four-split loader accordingly.

After creating the manifests, cache the fixed scorer descriptors:

```bash
python -m datasets.precompute_features --data-root data --device auto
```

## Training and personalization

### 1. Federated scorer

```bash
python -m training.train_federated_scorer \
  --data-root data \
  --output-dir outputs/federated_scorer \
  --rounds 20 \
  --device auto
```

By default, the script holds out 37 unseen identities, uses 10 fixed validation
identities, includes every eligible client in each round, and aggregates updates in
proportion to the square root of the actual processed example count. Eligibility for
the unseen cohort requires at least 100 `personalization` samples and at least 10
validation and test samples. Scorer support ratings are sampled from `train_fl`; the
`personalization` split supplies unpaired images for enhancer adaptation. Global scorer
updates use AdamW (`weight_decay=1e-4`) and the paper's global class-reweighting rule.
Validation correlations are computed per validation client and macro-averaged.
This stage is optional when using the released global scorer checkpoint.

### 2. Personalized scorer calibration

```bash
python -m training.personalize_scorer \
  --data-root data \
  --global-checkpoint pretrained_models/fedpaie_global_scorer_best.safetensors \
  --client-ids CLIENT_ID \
  --output-dir outputs/personalized_scorers \
  --support-sizes 10,100 \
  --mode hpo \
  --device auto
```

Use `--mode fixed_hp` for the fixed-hyperparameter protocol. The HPO mode uses a
deterministically seeded Optuna TPE sampler. Checkpoint selection uses validation data;
the test split is evaluated only after selection.

For a lower-cost deployment trial, start with `--mode fixed_hp`. In the following
enhancer command, pair it with `--scorer-mode fixed_hp --preset fixed_hp`. Per-user
scorer HPO and the shared-HPO enhancer preset are primarily research-analysis options.

The command above runs up to 20 trials for the selected user at both support sizes.
For the full research simulation, replace `--client-ids CLIENT_ID` with
`--split-file outputs/federated_scorer/open_world_user_split.json`. For a pipeline
smoke test, append `--max-clients 1 --trials 1 --epochs 1 --max-batches 1`.

### 3. Generic enhancement prior

```bash
python -m training.train_generic_enhancer \
  --input-dir data/fivek/input \
  --target-dir data/fivek/expert_c \
  --train-manifest data/fivek/splits/train.txt \
  --val-manifest data/fivek/splits/val.txt \
  --output-dir outputs/generic_enhancer \
  --device auto
```

This release example trains the generic prior against Expert C. Exact comparison with
the paper requires its common pretrained prior and paper-matched manifest.
This stage is optional when using the released generic enhancer checkpoint.

### 4. Personalized enhancer adaptation

For a new user, replace `CLIENT_ID` below with the ID used in that user's manifest
names. In the full research simulation, choose an ID from the `unseen_users` field of
`outputs/federated_scorer/open_world_user_split.json`. Use the same value in subsequent
inference and evaluation commands.

```bash
python -m training.adapt_enhancer \
  --data-root data \
  --global-enhancer-checkpoint pretrained_models/fedpaie_generic_enhancer_prior.safetensors \
  --scorer-dir outputs/personalized_scorers \
  --output-dir outputs/personalized_enhancers \
  --client-ids CLIENT_ID \
  --support-size 100 \
  --scorer-mode hpo \
  --preset shared_hpo \
  --min-validation-srcc 0.10 \
  --device auto
```

Both enhancer presets use `--min-validation-srcc` as their shared scorer-eligibility
threshold. Its default of 0.10 matches the paper protocol and keeps their cohorts
aligned. A skipped client is recorded in `adaptation_summary.json` without producing an
enhancer checkpoint. Use `--preset fixed_hp` for the prespecified open-world reference.
The `shared_hpo` preset applies one validation-selected configuration across the
eligible cohort for matched trade-off and ablation analyses. It does not rerun enhancer
HPO for each user. Both presets use seed 60 by default.

Portable enhancer checkpoints record the CLUT architecture automatically.
`--architecture` is needed only for an older state-dict checkpoint without metadata.

Run scorer calibration before enhancer adaptation and use the same client ID, support
size, and scorer mode in both commands. The personalized scorer is frozen guidance for
adaptation rather than a deployment dependency. After adaptation, copy only
`outputs/personalized_enhancers/client_CLIENT_ID/client_CLIENT_ID_personalized_enhancer.pt`
to the inference endpoint.

### Production deployment requirements

The commands above provide a single-machine reference implementation. Production
integration additionally needs to:

1. implement a consent-aware local data adapter for rated support/validation images and
   unrated adaptation photos, rather than relying on the Flickr-AES CSV layout;
2. cache the MobileNetV3-Large and LPIPS-AlexNet dependency weights when personalization
   must run offline;
3. run scorer calibration and enhancer adaptation locally, enforce the validation-SRCC
   eligibility rule, inspect representative outputs, and define a safe fallback when
   personalization is rejected;
4. package or export the resulting personalized enhancer for the target runtime and
   validate latency, memory, resolution, color handling, and numerical consistency on
   the actual device;
5. integrate image decoding, orientation, color management, output encoding, model
   versioning, rollback, and secure local storage into the application; and
6. if the global scorer will be retrained across deployed users, implement the actual
   federated client/server protocol and its authentication, encrypted transport,
   aggregation, monitoring, and chosen privacy protections.

If the released global initializations are used as-is, a new user's calibration,
enhancer adaptation, and inference require no further federated communication. Normal
inference uses only the personalized enhancer and does not require ratings, the scorer,
MobileNetV3, or LPIPS.

## Inference

Personalized enhancer checkpoints written by the release code contain a tensor-only
`model_state_dict` and are loaded without arbitrary Python object deserialization:

```bash
python inference.py \
  --checkpoint outputs/personalized_enhancers/client_CLIENT_ID/client_CLIENT_ID_personalized_enhancer.pt \
  --input /path/to/input.jpg \
  --output-dir outputs/inference \
  --device auto
```

The output preserves the source resolution. For old, trusted full-model PyTorch files,
`--allow-legacy-checkpoint` enables pickle loading. Never use that option with an
untrusted checkpoint.

## Evaluation

The following command evaluates one personalized checkpoint using the paper's
224 x 224 FiveK metric protocol:

```bash
python -m evaluation.evaluate_fivek \
  --checkpoint outputs/personalized_enhancers/client_CLIENT_ID/client_CLIENT_ID_personalized_enhancer.pt \
  --input-dir data/fivek/input \
  --target-dir data/fivek/expert_c \
  --manifest data/fivek/splits/test.txt \
  --image-size 224 \
  --output outputs/evaluation/client_CLIENT_ID.json \
  --device auto
```

This reports per-client PSNR, SSIM, and LPIPS against Expert C. Aggregating paper-style
means and standard deviations across clients requires evaluating every retained
checkpoint. The personalized-scorer predicted gain reported during local adaptation is
an optimization-aligned proxy, not direct evidence of improved human preference.

## Paper notation and code parameters

| Paper term | Public code name |
|---|---|
| `L_reg`, `L_pair`, `L_var` | `lambda_reg`, `lambda_pair`, `lambda_var` |
| `L_pref`, `L_aes` | `lambda_pref`, `lambda_aes` |
| `L_1`, `L_perc`, `L_gap` | `lambda_l1`, `lambda_perc`, `lambda_gap` |
| excess-gain threshold | `mu` |
| LUT bases `beta` | `enhancer.CLUTs` |
| coefficient predictor `psi` | `enhancer.backbone` + `enhancer.classifier` |
| color projection `theta_c` | `scorer.color_proj` |
| semantic projection `theta_s` | `scorer.semantic_proj` |
| fusion head `theta_f` | `scorer.mlp` |
| score temperature `tau` | `scorer.temperature` |
| minimum validation SRCC | `--min-validation-srcc` |

## Privacy scope and limitations

**Research simulation.** Raw photos and ratings are treated as client-local under the
experimental protocol, but this code executes client computation and aggregation in a
single process on one machine. It is not a networked federated-learning service and
does not establish a formal privacy guarantee. Model updates may still reveal
information in adversarial settings.

**Real deployment.** A production operator must separately implement client enrollment
and orchestration, model/version distribution, authenticated clients and servers,
encrypted transport and storage, secure aggregation where required, access control,
key management, failure recovery, monitoring, and an explicit privacy threat model.
Differential privacy or other protections must be designed and validated for the
intended risk and utility requirements. After the two initializations have been
delivered, new-user scorer calibration and enhancer adaptation can remain local, and
the inference application needs only the personalized enhancer.

## Reproducibility notes

- All public entry points accept paths through command-line arguments; no machine-local
  paths or usernames are embedded in the code.
- Seeds are exposed and used by Python, NumPy, PyTorch, client sampling, support-set
  construction, and Optuna sampling.
- `--deterministic` is available for federated scorer training, scorer calibration,
  and generic-prior training. Enhancer adaptation seeds its stochastic components with
  seed 60. Exact numerical equality across hardware and library versions is not
  guaranteed.
- Data, cached features, outputs, and personalized checkpoints are intentionally
  excluded from Git. Only the two documented global initialization weights are tracked.
- Exact paper results additionally require the same raw dataset release and resulting
  client cohort. This public release does not ship data-derived cohort manifests.

The reported experiments used Windows 11, an Intel Core Ultra 9 285K CPU, an NVIDIA
RTX 5090 GPU, approximately 64 GB of system memory, CUDA 12.8, PyTorch 2.2.2, and
Torchvision 0.17.2. The dependency files use compatible lower bounds rather than an
environment lock, so record the resolved versions for new experiments.

## Checkpoints

The repository contains only the global scorer initialization and generic enhancer
prior. It does not contain personalized user checkpoints. Model terms, metadata, and
file hashes are provided in `pretrained_models/`. This code uses native PyTorch
`grid_sample` and does not include the upstream CLUT custom extension.

## License

The FedPAIE source code is licensed under the [Apache License 2.0](LICENSE). The two
released weight files are distributed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); see
[MODEL_LICENSE](pretrained_models/MODEL_LICENSE). The paper, datasets, and other
external artifacts retain their respective terms and are not relicensed by either
license.

## Citation

```bibtex
@misc{xu2026fedpaie,
  title         = {Learning Color Grading, No Photo Sharing: Federated Aesthetic Preference Learning for Personalized Image Enhancement},
  author        = {Xu, Chuanzhi and Tao, Ziyuan and KNell, Jean Julien and Chen, Yanrong and Guo, Haolan and Yin, Xuanhua and Mahmood, Adnan and Cai, Weidong},
  year          = {2026},
  eprint        = {2607.27659},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.27659}
}
```
