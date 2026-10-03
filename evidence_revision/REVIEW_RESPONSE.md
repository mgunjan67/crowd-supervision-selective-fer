# Bounded response to the two supplied reviews

The reviews are advisory, not a journal decision. This revision adds evidence rather than asserting that acceptance is assured. Version 1.0.0 is preserved.

| Criticism | Action and remaining boundary |
| --- | --- |
| Uniform does not preserve tied maxima | Three matched Tie-Uniform runs, seeds 17/42/89, preserve every tied maximum and otherwise equal Uniform. Both selected and epoch-12 models are reported. |
| Own-set risk mixes ranking with prediction | Cross-evaluate Soft/Uniform predictions on both 80% accepted sets, report overlap and descriptive difference curves. Predictor differences depend on the accepted set. No unique causal decomposition is claimed. |
| Original versus majority F1 conflates references and populations | Report original/all, original/majority-subset and majority/identical-subset F1 plus the reference confusion matrix. Raw row/class mappings independently checked. |
| Unsupported categories hidden by aggregate metrics | Main-text per-seed full/partial/zero-mass retention and separate contempt/unknown/not-a-face vote-mass tables. Soft's adverse zero-mass retention is reported. |
| Selection criterion and early stopping affect conclusions | All 15 selected epochs, full retained learning curves and selected/fixed-epoch contrasts; no intermediate inference results invented. |
| “Ambiguity-preserving” is misleading | Replace with dominant-confidence-preserving. Explicitly note changes in entropy and class marginals. |
| Missing closest conceptual prior | Add Khurana et al., Crowd-Calibrator, COLM 2024, with primary-source URL; distinguish language-task conceptual precedent from this static FER comparison. |
| Prototype distracts from scientific argument | Preserve motivation and runnable prototype; move detailed runtime/UI/processing boundary to supplementary material. No communication benefit is asserted. |
| Repetition and disproportionate caveats | Consolidate interpretation and conclusion; keep factual scope limitations where needed. |
| Need stronger generalization | Not resolved by this bounded revision. One image source, one backbone and three paired seeds remain. FER+ is not a second image dataset. |
| Need a supported-mass-aware learned selector | Not implemented; no annotation-derived oracle is presented as deployable. Vocabulary diagnostics do not substitute for that experiment. |
| Need entropy/class-marginal matched targets | Not implemented; Tie-Uniform fixes ties but does not establish minority-identity causality. |
| External reproduction | Second arithmetic implementation is internal verification, not external human replication. All inference artifacts are released for such a check. |
| Ethics and photograph rights | Truthful no-approval/no-exemption disclosure retained. Public data is not proof of consent or rights; images are not redistributed. |

This is a limited empirical study, not a SOTA claim or universally superior method. Additional backbones, a mass-aware selector and an independent source may strengthen a subsequent study, but were not silently added or claimed here.
