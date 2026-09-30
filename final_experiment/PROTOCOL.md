# Bounded final experiment: matched hard versus soft crowd supervision

Frozen before any outcome of these six new training runs. Earlier FER2013
PrivateTest outcomes and exploratory questions are known. This is a new,
prespecified within-dataset experiment, NOT preregistered external confirmation.

## Question and fixed scope

Does retaining crowd-distribution information during training, rather than
collapsing it to a hard category, change expression recognition and selective
vote disagreement under identical data, architecture and compute?

Exactly two conditions, three paired seeds (17, 42, 89), 12 full-backbone epochs.
No outcome-dependent extra runs, hyperparameter changes or new optional studies.
Historical original-label/hybrid runs remain separate exploratory context.

## Data and independent operational roles

Keep FER2013 Training as the training partition. Obtain ten-category vote counts
from the already archived official FER+ annotations, joined by original row.
Remove PublicTest images whose exact pixel hash occurs in Training. Assign each
remaining PublicTest pixel cluster to selection or calibration by SHA-256 of
`final-20260921:` followed by its pixel SHA-256: even final hexadecimal digit is
selection, odd is calibration. Assignment is annotation- and outcome-independent.
No identical pixels may occur across these roles. Selection alone chooses
checkpoints. Calibration is not loaded for training or checkpoint selection.

The primary test population is PrivateTest without pixels present anywhere in
Training or PublicTest. Full PrivateTest is secondary historical comparability.
Test/calibration inference begins only after all six selected checkpoints exist.
This is exact-pixel separation, not verified subject/near-duplicate separation.
The previously viewed test outcomes still make generalization claims exploratory.

## Matched supervision and model

For each row, q is the ten-category empirical vote distribution. Let
w = sum(q[seven expression categories]) and r = q[:7]/w when w > 0.
Both conditions use all the same images and the same w. A zero-w image has zero
supervised loss in BOTH conditions; it is not assigned an invented category.

- Hard: weighted cross-entropy against argmax(r), ties resolved by the shared
  canonical class order (angry, disgust, fear, happy, neutral, sad, surprise).
- Soft: weighted cross-entropy against r, i.e. -sum(q[:7]*log p).

Loss is the mean of weighted losses over the batch, not divided by sum of weights.
This isolates collapsing versus retaining the same supported vote distribution.
It is not a replication of the prior strict-majority/original-label fallback rule
or of FER+'s original eight-class published benchmark. Training out-of-scope vote
mass is handled through identical weights; evaluation retains all ten categories.

Architecture: existing channel-gated torchvision Swin-T, ImageNet-1K V1, one pooled
768->48->768 gate and seven-class classifier. All parameters fine-tuned with
AdamW lr 1e-4, weight decay 1e-4, cosine minimum 1e-6, batch 32, BF16 autocast on
CUDA. Existing grayscale3/224/bilinear/mean=.5/std=.5 transform. Only horizontal
flip p=.5, deterministically keyed to seed, epoch and original row ID. Both
conditions share initialization, epoch order, flips and budget within each seed.

Choose minimum ten-category squared distribution distance on the SELECTION
partition (seven probabilities padded with zeros); ties retain earlier epoch.
Same objective for both conditions. Do not choose using original-label F1,
calibration or test. Checkpoint and execution state saved every epoch for restart.

## Outcomes fixed in advance

Primary contrast: soft minus hard expected vote disagreement at 80% coverage,
using maximum probability ranking, on the primary duplicate-excluded test set.
Exact common-coverage analysis is descriptive ranking evaluation, not a deployed
threshold; ties use original row ID. Report all three paired seed differences.

Secondary: full-coverage vote disagreement, excess above per-image available-class
oracle floor, ten-category squared distribution distance, original-label weighted
and macro F1, crowd-majority weighted F1 on its explicitly counted eligible subset,
and risk at 20%, 50%, 80% and 100% common coverage. Area under the complete
risk-coverage curve is descriptive; no claimed novel selector or risk guarantee.

For deployment-style evaluation, select maximum-probability thresholds on the
CALIBRATION subset at empirical vote-risk targets 10%, 20%, 30%, requiring >=100
accepted rows and retaining score ties. Maximize calibration coverage subject to
the empirical target. Empty feasible sets cause full abstention. Freeze before
test application. Report achieved test coverage and disagreement, annotation
oracle floor and excess; all three targets and seeds are released. Empirical
constraints are NOT confidence-bound risk guarantees.

Use 5,000 paired exact-pixel-cluster test resamples, RNG seed 20260921, for the
primary common-coverage contrast, reranking each resample at the fixed coverage.
Apply identical resampled clusters to both conditions and all fixed seeds. Report
mean paired difference and percentile 95% interval, explicitly conditional on
the fitted models and annotations. No seed-population or external-domain claim.
No multiplicity-adjusted significance claims from secondary endpoints.

## Decision and release

Report null/adverse results as prominently as favorable ones. An interval covering
zero does not prove equivalence. If soft labels help, describe the measured
effect under this protocol; do not claim a new learning algorithm. If they do not,
state the bounded negative result and whether the evidence warrants submission.

Keep the sender-controlled prototype and its existing behavior. No participant
study, new personal data collection, external dataset experiment or communication
benefit is claimed. No automated journal submission or public upload is authorized.
