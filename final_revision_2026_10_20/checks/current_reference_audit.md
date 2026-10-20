# Current bibliography and citation-support audit

Checked 20 October 2026 (Asia/Katmandu) for the final revision. This audit covers all 26 bibliography entries and the citation contexts in the five current `*_revised.tex` main sections. It is not a systematic review, proof of priority or reproduction of the cited experiments.

## Result

All 26 keys are cited; no missing, duplicate or unused keys were found. All 17 DOI identities were checked against live publisher-deposited Crossref records. The compact fetched metadata is in `current_crossref_metadata.json`; the complete per-entry support map and access limitations are in `current_reference_audit.json`.

No unresolved author/title/year/venue identity or sentence-support error was found after the corrections below. No experimental code, data, predictions, checkpoints or numerical results were changed.

## Changes in this audit

- Replaced the arXiv DOI on accepted Geifman (2017) with its official NeurIPS proceedings URL.
- Replaced the arXiv DOI on accepted PyTorch (2019) with its official NeurIPS proceedings URL. Its current author order was already correct; no author change was required.
- Updated Xia et al. (2025) from its arXiv link to the verified accepted ICLR 2025 OpenReview record.
- Corrected SCN's pages to **6896–6905**, matching the cited IEEE DOI. Its accepted CVF paper is paginated **6897–6906**; using one version's pagination consistently avoids a hybrid record.

The root's preceding MediaPipe correction is verified: the 2019 workshop paper is *MediaPipe: A Framework for Perceiving and Processing Reality*. The Google Research record links the actual workshop PDF, whose author list includes **Ming Guang Yong**. The differently titled arXiv manuscript's DOI is not attached to this workshop record.

## Per-entry primary-source map

| Key | Primary source | Sentence-support boundary |
| --- | --- | --- |
| `barsoum2016ferplus` | [2016: Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution](https://dl.acm.org/doi/10.1145/2993148.2993165); DOI 10.1145/2993148.2993165 | Official author repository and paper support multiple annotations and distribution-based supervision of FER2013 images. The manuscript does not borrow its benchmark scores. |
| `geifman2017selective` | [2017: Selective Classification for Deep Neural Networks](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) | Supports risk-coverage selective classification and rejection. Its published risk-control method is not represented as implemented by the present paper. |
| `geifman2019selectivenet` | [2019: SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a.html) | Supports the statement that prediction and rejection are optimized jointly. No claim of comparison against its learned rejector is made. |
| `wang2020scn` | [2020: Suppressing Uncertainties for Large-Scale Facial Expression Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Suppressing_Uncertainties_for_Large-Scale_Facial_Expression_Recognition_CVPR_2020_paper.html); DOI 10.1109/CVPR42600.2020.00693 | Supports weighting uncertain samples and relabeling in Self-Cure Network, not a claim that present annotations were manually corrected. |
| `goodfellow2013fer` | [2013: Challenges in Representation Learning: A Report on Three Machine Learning Contests](https://link.springer.com/chapter/10.1007/978-3-642-42051-1_16); DOI 10.1007/978-3-642-42051-1_16 | Supports the FER2013 challenge origin and dataset context. Actual acquired row counts and partitions are additionally supported by this study's source manifests. |
| `deng2009imagenet` | [2009: ImageNet: A Large-Scale Hierarchical Image Database](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf); DOI 10.1109/CVPR.2009.5206848 | Supports ImageNet as the source pretraining dataset. Specific torchvision weight enum and software versions are implementation records, not claims established by this 2009 paper. |
| `liu2021swin` | [2021: Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html); DOI 10.1109/ICCV48922.2021.00986 | Supports hierarchical shifted-window backbone design. Present head placement and parameter counts remain descriptions of this implementation. |
| `hu2018senet` | [2018: Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html); DOI 10.1109/CVPR.2018.00745 | Supports established channel recalibration/gating. Manuscript makes no ungated-model superiority or new-SE-architecture claim. |
| `lugaresi2019mediapipe` | [2019: MediaPipe: A Framework for Perceiving and Processing Reality](https://research.google/pubs/mediapipe-a-framework-for-perceiving-and-processing-reality/) | Supports MediaPipe framework attribution. Detector model-selection and confidence settings are implementation-specific and are not attributed to the 2019 workshop paper. |
| `wang2016emotionpush` | [2016: Sensing Emotions in Text Messages: An Application and Deployment Study of EmotionPush](https://aclanthology.org/C16-2030/) | Supports text-based emotion notifications as application context, not facial inference or a communication benefit of the present prototype. |
| `barrett2019emotions` | [2019: Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements](https://journals.sagepub.com/doi/abs/10.1177/1529100619832930); DOI 10.1177/1529100619832930 | Supports caution against identifying internal emotion or intention from category predictions/facial movements. |
| `sokolova2009metrics` | [2009: A Systematic Analysis of Performance Measures for Classification Tasks](https://www.sciencedirect.com/science/article/pii/S0306457309000259); DOI 10.1016/j.ipm.2009.03.002 | Provides general classification-measure context. The manuscript explicitly defines its weighted-F1 formula, zero-division convention and supported-majority population rather than claiming those dataset-specific conventions come from this source. |
| `paszke2019pytorch` | [2019: PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) | Supports PyTorch framework attribution, not this study's exact software versions or deterministic execution assertions. |
| `chong2019emochat` | [2019: EmoChat: Bringing Multimodal Emotion Detection to Mobile Conversation](https://tns.thss.tsinghua.edu.cn/sun/publications/2019.EmoChat.pdf); DOI 10.1109/BIGCOM.2019.00037 | Author-hosted accepted paper supports combining facial and textual signals in messaging. Its system or participant results are not imported as present-paper evidence. |
| `liu2019selfawareness` | [2019: Integration of User Emotion and Self-Awareness in Text Messaging](https://commons.clarku.edu/faculty_computer_sciences/166/); DOI 10.1109/CIC48465.2019.00030 | Supports expression-informed emoji selection/self-awareness as application context, without asserting the same benefit for the present preview. |
| `loshchilov2019adamw` | [2019: Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7) | Supports decoupled-weight-decay AdamW attribution; learning rates, schedules and hyperparameters remain documented experimental choices. |
| `guo2017calibration` | [2017: On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Supports conventional confidence calibration, neural-model miscalibration and post-hoc temperature scaling; no population-risk guarantee is attributed to it. |
| `le2023uncertainty` | [2023: Uncertainty-aware Label Distribution Learning for Facial Expression Recognition](https://openaccess.thecvf.com/content/WACV2023/html/Le_Uncertainty-Aware_Label_Distribution_Learning_for_Facial_Expression_Recognition_WACV_2023_paper.html); DOI 10.1109/WACV56688.2023.00603 | Supports valence-arousal-neighborhood construction of training distributions, not a tested comparator in this study. |
| `kawamura2024midas` | [2024: MIDAS: Mixing Ambiguous Data with Soft Labels for Dynamic Facial Expression Recognition](https://openaccess.thecvf.com/content/WACV2024/html/Kawamura_MIDAS_Mixing_Ambiguous_Data_With_Soft_Labels_for_Dynamic_Facial_WACV_2024_paper.html); DOI 10.1109/WACV57701.2024.00642 | Supports soft-label video augmentation by mixup-style combinations, not a reproduced benchmark comparator. |
| `hoeffding1963` | [1963: Probability Inequalities for Sums of Bounded Random Variables](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830); DOI 10.1080/01621459.1963.10500830 | Supports the bounded-independent-loss concentration inequality; the manuscript separately states its fixed-model, unchanged-population and union-bound conditions rather than claiming verified deployment guarantees. |
| `khurana2024crowd` | [2024: Crowd-Calibrator: Can Annotator Disagreement Inform Calibration in Subjective Tasks?](https://openreview.net/forum?id=VWWzO3ewMS) | Supports estimated crowd-distribution distance-based rejection, optional entropy and distinct soft/categorical/selective results. Sections 2.3 and 5.1 explicitly report improved soft/calibration measures without corresponding F1 gains and worse soft-label MaxProb AUC/AUROC in its hate-speech comparison. |
| `he2016resnet` | [2016: Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html); DOI 10.1109/CVPR.2016.90 | Supports ResNet backbone identity; the gate and matched twelve-epoch comparison are the present implementation, not results from this source. |
| `washington2021crowdsourcing` | [2021: Training Affective Computer Vision Models by Crowdsourcing Soft-Target Labels](https://link.springer.com/article/10.1007/s12559-021-09936-4); DOI 10.1007/s12559-021-09936-4 | Directly supports the distinction between categorical F1 and full-distribution agreement in hard/soft CAFE ResNet-152 models. It is prior art, not evidence of novel general discovery here. |
| `livingstone2018ravdess` | [2018: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391); DOI 10.1371/journal.pone.0196391 | Supports original RAVDESS population, recordings, response semantics and neutral/calm convention. Actual retained clip/frame counts and annotation adapter are separately audited artifacts, not assumed from the paper title. |
| `xia2025smoothing` | [2025: Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It](https://openreview.net/forum?id=6oWFn6fY4A) | Supports selective-ranking degradation under conventional label smoothing and post-hoc logit normalization as relevant context. It does not establish the behavior of this paper's image-dependent targets or vote-disagreement endpoint. |
| `baan2022calibration` | [2022: Stop Measuring Calibration When Humans Disagree](https://aclanthology.org/2022.emnlp-main.124/); DOI 10.18653/v1/2022.emnlp-main.124 | Supports theoretical and empirical distinction between majority-label calibration and human-judgment distribution properties, including class frequency, ranking and entropy. |

## Version and metadata handling

Publisher-deposited IEEE pagination is retained consistently for Swin (**9992–10002**, versus CVF **10012–10022**), Le et al. (**6077–6086**, versus CVF **6088–6097**) and MIDAS (**6538–6548**, versus CVF **6552–6562**). The live Crossref snapshot confirms those IEEE ranges. Le and MIDAS are also confirmed by author-university publication records linked in the JSON audit.

Author identities are normalized rather than blindly copying malformed metadata fields: FER+'s official repository uses the compound family name *Canton Ferrer*; ImageNet's original citation/PDF supplies the author names where Crossref leaves two names undivided. Baan's accented author spelling and the official record refer to the same person. Hoeffding uses the original **1963** publication year, not its **2012** online digitization date.

NeurIPS official BibTeX downloads currently omit printed-volume pages. PyTorch's conventional retained range **8024–8035** is not newly established by that export; its author/title/year/venue identity and accepted PDF are verified. Geifman (2017) has no invented page range.

## Unpublished works remain text-only

- [Facial Expression Recognition using Squeeze and Excitation-powered Swin Transformers](https://arxiv.org/abs/2301.10906) — Arpita Vats; Aman Chadha. arXiv v7; no accepted venue verified; text-only attribution to architectural precedent, no score borrowed.
- [Confidence Calibration under Ambiguous Ground Truth](https://arxiv.org/abs/2603.22879) — Linwei Tao; Haoyang Luo; Minjing Dong; Chang Xu. arXiv v1, 24 March 2026; no accepted venue verified; text-only.
- [Predicting Human Disagreement for Calibrated Dynamic Facial Expression Recognition](https://arxiv.org/abs/2609.17130) — Yiming Wang; Frederick W. B. Li; Jingyun Wang. arXiv v1, 15 September 2026; comments say submitted to ICASSP 2027, not accepted; text-only.
- [Interpretable Uncertainty Routing Separating Emotion Ambiguity from Distribution Shift in Facial Expression Recognition](https://arxiv.org/abs/2606.22725) — Keito Inoshita; Takato Ueno. arXiv v1, 21 June 2026; no accepted venue verified; text-only.

No accepted status is inferred from submission comments. No performance score from these works is used as a directly comparable benchmark.

## Citation precision

The present manuscript's dataset-specific population rules, majority threshold, zero-division convention, torchvision weight enums, model-head placement, checkpoints and software versions are author-defined implementation facts. The cited background publications do not verify those artifacts. The manuscript gives their definitions explicitly.

Crowd-Calibrator's Sections 2.3 and 5.1 support the current soft/categorical/selective distinction. The related-work sentence now begins **“In its hate-speech comparison”**, restricting the soft/calibration, categorical and MaxProb AUC/AUROC findings to the supported experimental context. This factual narrowing changes `source/generated/related_work_revised.tex` only; it adds no method, analysis or experiment.

Direct publisher/OpenReview requests sometimes returned browser challenges, but indexed primary accepted-paper text, official proceedings pages and author repositories remained available. Live Crossref fetching succeeded for every DOI entry. No unavailable full text is represented as newly reviewed.

## Audit identity

- Bibliography SHA256 before this audit: `0afb686ae575dde3e920106c9a3bd080dc1cfbe9950445f1775bc0cd56f2b86a`.
- Bibliography SHA256 after this audit: `437550be594d1ed2fd63efcd874370720d693a0642dc4f60d49764491013e0dd`.
- Structured audit: `current_reference_audit.json`.
- Fetched DOI metadata: `current_crossref_metadata.json`.
