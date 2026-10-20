# Final scientific diagnostics, 20 October 2026

This protocol was written before computing the new diagnostic outcomes. All
historical within-source and RAVDESS outcomes were already known. The analyses
are exploratory, are not independent confirmation, and use no new fitted models.
Historical evidence files are read only. CPU execution only.

## Centered logit normalization

Primary source: Xia, Laurent, Franchi and Bouganis, ICLR 2025, equations (16)
and (28), https://arxiv.org/html/2403.14715 (version 3, 20 February 2025), and
https://proceedings.iclr.cc/paper_files/paper/2025/hash/9dc5accb1e4f4a9798eae145f2e4869b-Abstract-Conference.html.
The method ranks by max(z-mean(z))/norm_p(z-mean(z)); it does not apply softmax
to the normalized logits. Since log softmax(z)=z-logsumexp(z), centered logits
equal log(p)-mean(log(p)) in exact arithmetic. Original uncentered logits cannot
be recovered, and the common offset may not be used for an uncentered variant.
Stored float32 probabilities are rounded outputs of BF16-forward, float32-softmax
inference: recovered centered logits are approximate, even if the formula is
algebraically exact for unrounded positive probabilities. Count zeros/minima for
all inputs; if zeros exist, clip them at float32 tiny (not an optimized floor),
and report counts. Uniform rows receive score zero.

For each of all 48 historical fitted inference states independently, choose
the p-norm exponent from [1, 1.5, 2, 3, 4, 6, 8, 16] by minimum all-prefix AURC
on the existing 1,642-image calibration role, using complete-vocabulary expected
vote disagreement. Ties choose the first grid value. This adapts the source's
validation-AURC selection to the present crowd-disagreement endpoint and uses a
declared small grid; it is not a reproduction of the source training experiments.
Compare MSP, normalized score, and calibration-AURC choice between the two scores
(MSP wins exact ties), evaluated on the fixed 3,275-image primary test role.
Preserve categorical predictions. Report all seeds, both checkpoint states and
backbones at 20%, 50%, 80%, 100% common coverage, with curves at 5% increments.
Image-row identifiers resolve rank ties. No test outcome selects an exponent.

## External temporal aggregation and constant predictions

For all 36 retained external inference states, compare mean of all eight frame
probability vectors against the fixed middle four (sample positions 2,3,4,5,
zero based, after sorting each clip's retained frames chronologically). All eight
retained frames per clip were valid in the historical experiment. Use all 1,440
clips and the same raw-nine-category and calm-to-neutral references. Recompute
MSP ranking separately for each aggregation at 80% and 100% coverage. Report
per-seed and averaged Soft-minus-control contrasts, aggregation changes, neutral
prediction fraction, and central/full prediction agreement. Constant-neutral and
constant-happy predictions are descriptive references on all clips and each
model's exact accepted subset; constants have no invented confidence ranking.
No frame ground truth, naturalistic evaluation, or causal explanation is claimed.

## Target statistics and bootstrap documentation

Reconstruct the five historical seven-class targets for all 28,709 training
images from retained vote annotations. Compute target entropy (natural logs),
unweighted means restricted to positive supervised mass, supervised-mass-weighted
means, and corresponding seven-class marginal distributions. Zero-mass images
have zero supervised contribution and are reported separately. Report scalar
number of distinct exact-pixel primary-test clusters and their size distribution.
The historical bootstrap shares each resampled cluster multiset between models,
sets accepted count to ceil(0.8 * resampled image count), computes each paired
seed difference, and then averages the three differences within each replicate.
Intervals condition on fitted models, observed annotations, and the reused source.

## Integrity and delivery

Record hashes of protocol, code and input files; retain CSV ledgers and a JSON
audit. Calculation checks must verify shift/scale invariance and exact centered
logit recovery for positive synthetic softmax inputs, unchanged argmax, 100%
coverage score invariance, all-eight reconstruction against stored probabilities,
component decomposition, constant-reference identity, and target normalization.
New figures use the CSV outputs without hand-entered result values.
