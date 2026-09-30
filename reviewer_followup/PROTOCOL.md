# Reviewer-motivated follow-up: frozen before new training

This follow-up is specified after examining the previous hard/soft results and the supplied review. It is not preregistered or independent confirmation. The previous experiment and release remain unchanged. No outcome-dependent additional runs are permitted.

## Matched controls

Train two additional conditions, each at seeds 17, 42, and 89, using the existing training implementation, identical ImageNet initialization, image roles, per-image supported-mass weights, stateless augmentation, minibatch order, optimizer, learning-rate schedule, BF16 precision, and 12-epoch budget. All new runs must finish before any new calibration/test predictions are examined. Select checkpoints by the existing ten-category squared distance. Retain both selected and epoch-12 weights.

For q, the ten-category vote distribution, w=sum(q[:7]) and r=q[:7]/w:

1. `uniform`: let h be the canonical first maximum of r and m=max(r). Set t[h]=m and t[c]=(1-m)/6 otherwise. Minimize -w sum(t log p), averaged over batch size. This preserves dominant confidence while removing the observed minority allocation. It is image-dependent smoothing, not constant label smoothing. Ties retain the canonical first maximum.
2. `tie_hard`: distribute target mass uniformly among all supported categories tied for the largest vote. Minimize the same weighted cross-entropy. Unique maxima equal the old hard target. This is the expected objective of uniform random tie resolution, without adding RNG calls.

Zero-supported-mass rows contribute exactly zero loss. No relabeling or exclusion changes are made. Confirm initialization hashes and matched configuration against the old runs. The original hard and soft losses and artifacts are untouched.

## Endpoints and interpretation

Primary follow-up contrast: soft minus uniform mean complete-vote disagreement at exactly ceil(0.8*N) retained primary test rows, ranked by maximum probability with original row order breaking ties. Report all three paired differences and a 5,000-replicate paired pixel-cluster percentile interval, conditional on these fitted models and annotations. Compare tie_hard with hard and soft descriptively. Do not describe failure to detect a difference as equivalence. If uniform matches or improves on soft, narrow the claim: the full minority allocation has not demonstrated added benefit here.

Repeat all common-coverage endpoints using fixed epoch 12 for all four conditions. This is sensitivity to selection versus a fixed budget endpoint, not a replay of alternative selection criteria, because every historical epoch's weights were not retained.

## Vocabulary and calibration diagnostics

For predicted class h, decompose disagreement exactly into unsupported mass (1-w), supported ambiguity (w-max(q[:7])), and excess (max(q[:7])-q[h]). Report these on identical accepted sets, including 80% global coverage. Describe separately w=0, 0<w<1, and w=1 populations. At positive w, also report conditional supported-vote disagreement 1-r[h]; it has a different estimand. Within-stratum 80% coverage is distinct from restricting the global accepted set. No annotation-derived score is presented as deployable.

Document the idealized mismatch: weighted soft CE has optimum p=r for w>0, whereas seven-output probabilities padded with zeros minimizing ten-category squared distance have optimum p[c]=q[c]+(1-w)/7. Under ideal p=r, complete-vote disagreement is 1-w*max(p), so MSP cannot by itself represent it. Empirical diagnostics must not be called proof of a unique failure cause.

For each old empirical calibration target (.10,.20,.30), report the calibration/test accepted risk, coverage, and all three decomposition components. Describe distribution differences without identifying them causally. Repeat empirical threshold fitting on 100 fixed-seed pixel-cluster half-splits of calibration; evaluate on the disjoint other half, preserving score ties and minimum accepted count 100. This describes internal threshold-search optimism/variability, not independent validation.

A conservative comparator uses a fixed 101-threshold grid from 0 to 1 inclusive, minimum n=100, and upper bound min(1, empirical risk + sqrt(log(101/.05)/(2*n))). Choose maximum coverage meeting each target or abstain completely. This is a per-model simultaneous fixed-grid Hoeffding bound under i.i.d. bounded observations and unchanged population; dependence and population shift are not verified absent here, so no operational guarantee is claimed. Report calibration/test trade-offs including empty policies and do not compare different achieved risks as equal-risk superiority. No tuning of grid, delta, or target after viewing outcomes.

## Auditing and stopping boundary

Preserve row mapping, original labels, and annotation mappings. Recheck label disagreement and denominators independently. Hash all new sources before training. Independently recompute central metrics, controls, threshold policies, and derived table values. Package source, predictions, selected/final inference weights, protocol and limitations; render and inspect every final PDF page. All secondary analyses are descriptive. One reused image source, three seeds, no human study, no ethics determination, and no architecture novelty remain limitations rather than wording problems to hide.
