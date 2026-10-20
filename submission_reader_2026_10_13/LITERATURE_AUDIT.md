# Focused primary-source and attribution audit

Checked 12 October 2026. This is not a systematic review, implementation
reproduction or proof of priority. The retained 21 September primary-source
audit supplies the earlier 19 entries; the current checks below cover the
closest literature, added sources and current attributions. No published
benchmark percentage is substituted for this study's vote-disagreement endpoint.

| Work | Primary source / identity | Attribution retained in this edition |
| --- | --- | --- |
| Barsoum, Zhang, Canton Ferrer, Zhang (2016) | [Author manuscript](https://arxiv.org/abs/1608.01041), [official FER+ repository](https://github.com/microsoft/FERPlus); ICMI, DOI 10.1145/2993148.2993165 | Majority, multi-label, probabilistic drawing and distribution cross-entropy already precede this study. FER+ reannotates FER2013, not a new image source. |
| Washington et al. (2021) | [Accepted author manuscript](https://par.nsf.gov/servlets/purl/10466641), DOI 10.1007/s12559-021-09936-4; Cognitive Computation 13:1363–1373 | Hard/soft ResNet-152 on CAFE; categorical versus distributional agreement is prior art. Use the final 17-author record, not the earlier arXiv author order. |
| Khurana, Nalisnick, Fokkens, Swayamdipta (2024) | [Primary full text](https://arxiv.org/html/2408.14141v1), Sections 4.1.1–4.1.2; [COLM primary PDF](https://openreview.net/pdf?id=VWWzO3ewMS) | Learned crowd estimates from representations; distance-based rejection, optionally adding predictive entropy. NLP results are not FER baselines. The learned comparator is not implemented here. |
| Le et al. (WACV 2023) | [CVF accepted paper](https://openaccess.thecvf.com/content/WACV2023/html/Le_Uncertainty-Aware_Label_Distribution_Learning_for_Facial_Expression_Recognition_WACV_2023_paper.html), [author university record](https://livrepository.liverpool.ac.uk/3169130/), DOI 10.1109/WACV56688.2023.00603 | Valence–arousal neighborhood target construction. The attribution is limited to the verified abstract/method description, not a claimed implementation audit. |
| Kawamura, Hayashi, Takemura, Nagahara (WACV 2024) | [CVF primary PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Kawamura_MIDAS_Mixing_Ambiguous_Data_With_Soft_Labels_for_Dynamic_Facial_WACV_2024_paper.pdf), DOI 10.1109/WACV57701.2024.00642 | Soft-label mixing in dynamic FER, not the direct static-image vote-target comparison used here. |
| Yiming Wang, Frederick W. B. Li, Jingyun Wang (2026) | [arXiv v1](https://arxiv.org/abs/2609.17130), submitted 15 September 2026; comments say submitted to ICASSP 2027 | Dirichlet–multinomial vote-count modeling, predicted disagreement, rejection combining ambiguity/vacuity/temporal instability/input quality. Unpublished, not verified as accepted; prose-only citation under the journal's rule. No numerical superiority is borrowed. |
| Vats and Chadha (2023) | [arXiv v7](https://arxiv.org/abs/2301.10906) | Swin and SE for FER predate the present fixed head. Unpublished status explicit; no benchmark value is used. |
| Livingstone and Russo (2018) | [Primary publication](https://doi.org/10.1371/journal.pone.0196391), original S5 workbook; [original Zenodo record](https://zenodo.org/records/1188976) | Source collection, video-only speech eligibility, response semantics and source licensing; original ethics approval is not approval of this secondary analysis. |

Metadata details: Le's IEEE DOI-backed pages are 6077–6086, whereas the CVF
accepted version uses 6088–6097. MIDAS similarly uses IEEE 6538–6548 and CVF
6552–6562. The retained DOI-backed pagination follows the prior IEEE-deposited
Crossref audit; the different CVF pages are not a numerical-results correction.
An attempted DOI/Crossref fetch in this turn was unavailable. The accessible
author-institution record confirms Le's DOI, authors and title. Washington's
NSF-hosted accepted manuscript supplies the final publication header and author
record; PMC access was CAPTCHA-limited. No unavailable full text is represented
as newly read.

All 24 numbered entries are cited exactly by key, with no duplicate keys or
DOIs. The two unpublished arXiv works are attributed in prose, not represented
as accepted numbered references. EmotionPush authors are Shih-Ming Wang,
Chun-Hui Scott Lee, Yu-Chun Lo, Ting-Hao Huang and Lun-Wei Ku; the primary
[COLING demonstration](https://aclanthology.org/C16-2030/) concerns text-based
notifications, not facial inference. Existing component/method references retain
the primary-source audit in `final_experiment/REFERENCE_AUDIT.md`.

Final bounded-revision additions verified against primary arXiv records:

| Work | Primary record / status | Bounded attribution |
| --- | --- | --- |
| Linwei Tao, Haoyang Luo, Minjing Dong, Chang Xu (2026) | [Confidence Calibration under Ambiguous Ground Truth](https://arxiv.org/abs/2603.22879), v1, 24 March 2026; no accepted venue verified | Majority-label versus annotator-distribution calibration; post-hoc calibrators not implemented here. Lower squared distance is not asserted to establish calibration. |
| Keito Inoshita, Takato Ueno (2026) | [Interpretable Uncertainty Routing Separating Emotion Ambiguity from Distribution Shift in Facial Expression Recognition](https://arxiv.org/abs/2606.22725), v1, 21 June 2026; no accepted venue verified | DINOv2-ensemble uncertainty routing, distinct from annotation accounting and our multi-factor frozen-pipeline stress test. |

Both additions are unpublished prose-only attributions under the journal's
reference-list rule. The24 numbered bibliography entries remain unchanged;
there are now four prose-only unpublished works. No external benchmark score
is substituted for this study's endpoints. No unresolved attribution supports
a central empirical claim.
