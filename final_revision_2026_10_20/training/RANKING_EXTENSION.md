# Fixed ranking checks for the new vocabulary/target family

Specified 20 October 2026 while training is incomplete and before any new
calibration or test inference. Earlier manuscript, historical test and ranking
diagnostic results are known. This is an adaptive explanatory follow-up on a
previously exposed image source, not independent or preregistered confirmation.
No additional training, images, seeds, learned selector or new inference is added.

## Complete family and contrasts

Evaluate every one of the 33 declared new inference states: seven-output
Permutation at seeds 17/42/89, distance-selected and epoch12 (six states), and
eight-output Soft/Uniform/Tie-Uniform at the same seeds, distance-selected,
epoch12 and selection-risk-selected (27 states). Include the six historical
seven-output Swin-T Soft distance-selected/epoch12 states solely as the paired
Permutation comparator. These 39 nominal states are not 39 independent fits.

The eight contrast groups are seven-output Soft minus Permutation at selected
and epoch12, plus eight-output Soft minus Uniform and Tie-Uniform at each of
selected, epoch12 and risk_selected. Report all three seed differences and
their descriptive mean, sample SD and range at coverage 20/50/80/100 percent.
No contrast, seed or endpoint may be omitted because its direction is adverse.

## Fixed scores and parameter rule

Use MSP, top-two probability margin and negative entropy. Additionally use
Xia et al.'s mean-centered logit normalization: z=log(p)-mean(log(p)), score
max(z)/norm_a(z). The declared ordered exponent grid is exactly
[1,1.5,2,3,4,6,8,16], matching the earlier sensitivity. Select each state's
exponent by minimum all-prefix expected-complete-vote-disagreement AURC on
the existing 1,642-image calibration role; exact ties retain the first grid
entry. Compare that score with MSP on the same calibration AURC and retain
MSP on exact ties. This fallback compares those two scores only, not all five.

For positive exact probabilities, mean-centering removes the unrecoverable
softmax offset. Stored float32 probabilities are rounded, so reconstruction
only approximates original centered logits. Count zeros and entries below
float32 tiny; clip only those entries at that fixed floor. Uniform vectors
receive score zero. Do not claim uncentered logits were recovered or that
these experiments reproduce Xia et al.'s original training study.

Evaluate all scores at ceil(coverage * N) for the frozen 3,275-image primary
test role, resolving score ties by original image-row ID. Predictions remain
the original argmax. The ten annotation categories remain in the denominator
for both vocabularies. Full-coverage disagreement must be score-invariant.
Retain calibration AURC for every exponent and test AURC/coverage outcomes for
every exponent, including candidates that were not selected. Candidate test
results are descriptive and never select parameters. No new bootstrap intervals
are required here: paired seed descriptors must not be interpreted as uncertainty
over future training runs or as multiple confirmatory hypothesis tests.

## Verification and delivery

Hash this protocol, the extension script and its CPU tests before new inference.
Record hashes of old comparison predictions and the old diagnostic ledgers at
the same freeze. Reconcile all six old Soft normalization candidate AURCs,
chosen exponents, selected-score results and original margin/entropy outputs.
Reconcile all 33 new MSP 80-percent rows against the primary frozen analysis.

Validate the existing 33-state evaluation and its 99 role archives using the
protected contracts; numerical replay must work without photographs, GPU use,
unreleased epoch weights or optional released weights. Keep verification of
available bytes distinct from retained-only provenance hashes.

Write separate `results/new_ranking_*` CSV/JSON/TeX files. Never modify the
primary frozen result summary. A compact descriptive S11 table reports all
eight mean contrasts at 80 percent; full per-seed and candidate outcomes remain
in CSV. Synthetic tests and any layout fixtures stay under the excluded checks
tree and must be explicitly marked as non-empirical.
