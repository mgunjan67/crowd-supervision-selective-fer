# Vocabulary and minority-allocation follow-up

Specified on 20 October 2026 after all previous manuscript, test, external, and
review outcomes were known. This is an adaptive explanatory follow-up, not
preregistered or independent confirmation. Its own sources and run list are
hashed before full training. No outcome-dependent conditions or seeds may be
added. Training uses the already acquired FER2013/FER+ source only.

## Questions and complete run list

1. Does the observed minority-category allocation differ from a fixed permutation
   preserving each image's complete supported-target probability multiset,
   entropy, and every maximum? Train the seven-output permutation condition
   using Swin-T at seeds 17, 42, 89. Compare it with all retained seven-output
   Soft/Uniform/Tie-Uniform states, without changing those states.
2. Do the matched target comparisons persist when contempt is available as an
   eighth expression output? Train eight-output Soft, Uniform, and Tie-Uniform
   at seeds 17, 42, 89: nine runs. Unknown and not-a-face remain outside the
   expression vocabulary and remain in complete-reference scoring.

Total: twelve full Swin-T runs, twelve epochs each. No restricted/new images or
new human ratings are collected. No favorable result is required. All seeds,
conditions, selected states and fixed epoch-12 states are reported.

## Fixed target construction

Supported mass is the sum of the first K votes in the canonical annotation order
angry, disgust, fear, happy, neutral, sad, surprise, contempt, unknown, NF.
For positive mass, normalize supported votes to r. Loss is supported mass times
cross-entropy, averaged by batch size. Zero supported mass gives zero loss.

Soft keeps r. Uniform preserves the canonical first maximum m and puts
(1-m)/(K-1) elsewhere. Tie-Uniform preserves every tied maximum and divides
remaining mass equally over other categories, with the all-tied case uniform.

Permutation retains EVERY position attaining the maximum. Other category
positions are deterministically cycled in a row-specific shuffled order, using
PCG64 seeded from SHA256('minority-permutation-20261020:' + original row ID).
This permutation is fixed across epochs and training seeds. Fewer than two
nonmaximum positions are unchanged. Duplicate probabilities can also leave a
target unchanged. Report the changed-row fraction and weighted class-marginal
changes; this control does NOT claim to preserve aggregate class priors or to
isolate every regularization mechanism. No epoch-wise randomization, which
would average back toward uniform minority allocation, is used.

## Training and selection

Use the original exact-pixel-isolated roles, same Swin-T ImageNet-1K V1 weights,
post-pooling gate, preprocessing, stateless horizontal flips, batch size 32,
AdamW 1e-4, weight decay 1e-4, cosine schedule ending at 1e-6, BF16 forward,
and twelve epochs. Match initialization, order and augmentation within K/seed.
Seven-output permutation initialization must match retained seven-output runs.
All new epoch weights and selection predictions are retained.

The primary checkpoint rule remains minimum ten-category squared distance on
selection, padding missing outputs with zeros; earliest ties. Retain epoch12.
A predeclared stopping sensitivity selects the earliest epoch minimizing mean
complete-vote disagreement at 80% own-MSP coverage on SELECTION only. Report
this rule for every eight-output condition; it does not retrospectively
reconstruct unavailable historical seven-output epochs.

Finish and hash all twelve runs before new calibration or test inference.
Smoke tests may exercise training and shape contracts, without calibration/test
access or hyperparameter tuning. Repairs are recorded before resumed execution.

## Analysis

Primary follow-up reporting is descriptive: eight-output Soft-minus-Uniform at
80% own-MSP coverage, selected and epoch12, all three seed differences and mean.
Tie-Uniform and permutation contrasts are mechanistic sensitivities. Also report
full-coverage ten-category squared distance, floor/excess components, fixed-set
cross-evaluation, unsupported-mass strata, and risk-coverage curves. For each K
retain complete ten-category votes; do not compare conventional FER+ published
accuracy with these different references or image populations.

Use the original 5000 shared pixel-cluster resampling scheme, resampling clusters
with replacement, averaging the three paired-seed differences per replicate and
reranking at ceil(.8 * resampled image count). These intervals are conditional
on the fitted models, observed pixels and votes and do not undo adaptive design
or quantify unrestricted seed/rater/person variation. Report seed ranges and
sample SD without claiming three seeds establish population robustness.

New findings may support an explanation, reject it, or remain inconclusive.
Vocabulary changes also alter supported weights and output dimension, so this
is a specified vocabulary-policy intervention, not a uniquely isolated contempt
effect. No new application reliability or communication claim is introduced.
