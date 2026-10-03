# Final bounded robustness response

Preserve the earlier review-response matrix in evidence_revision/. This follow-up addresses the latest requested improvement, not every possible research limitation.

| Concern | Action and boundary |
| --- | --- |
| Only one backbone | Nine ResNet-18 runs: Soft, Uniform, Tie-Uniform, seeds 17/42/89. Same roles/transform/training budget and matched within-seed state. Two families, still one image source. No architecture leaderboard. |
| Selective improvement conflates accepted images | Both control comparisons cross-evaluated on fixed image sets; selected and epoch-12 matrices retained. |
| Selective reporting / stopping | Frozen pre-training protocol; all nine 12-epoch runs, both states, all paired seeds/contrasts, logs and technical failure record included. |
| Small effect / weak practical meaning | Absolute disagreement changes retained; no claim of corrected-image count or messaging utility. |
| Unpublished bibliography entries | Vats–Chadha and Wang–Li–Wang kept as explicitly unpublished in-text links, removed from numbered list; published ResNet CVPR source added. |
| Contribution unclear | Introduction identifies matched target controls, fixed-image comparisons and annotation-reference/vocabulary diagnostics. |
| Independent dataset / adaptive reuse | Not solved. FER+ reannotates FER2013. Four adaptive phases do not establish external generalization. |
| Learned supported-mass selector / entropy mechanism | Not implemented and not claimed. |
| Ethics / photograph rights | Truthful disclosures retained; no original images redistributed. |

ResNet reported effects (conditional descriptive intervals): selected Soft-minus-uniform: +0.173 pp [-0.167, +0.455]; selected Soft-minus-tie_uniform: +0.042 pp [-0.258, +0.343]; epoch12 Soft-minus-uniform: +0.170 pp [-0.157, +0.467]; epoch12 Soft-minus-tie_uniform: +0.190 pp [-0.126, +0.515].
Internal checks are not independent external replication. Author approval of v1.1.0 is confirmed; later results need final review before actual upload.
