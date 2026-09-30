# Reference and claim audit for the final experiment edition

Checked 21 September 2026. The bibliography is a cited-only subset of the
previous primary-source-audited bibliography, plus two verified related works.
The generator refuses missing entries; the manuscript check rejects unused keys,
duplicate keys, duplicate DOIs, and incomplete rendered bibliography markers.

| Key | Primary source and supported use |
|---|---|
| barsoum2016ferplus | https://doi.org/10.1145/2993148.2993165 ; official https://github.com/microsoft/FERPlus ; multiple annotations and prior hard/distribution learning, not an independent image source |
| goodfellow2013fer | https://doi.org/10.1007/978-3-642-42051-1_16 ; FER2013 challenge provenance; exact current membership additionally established from retained acquisition/row manifests |
| geifman2017selective | https://arxiv.org/abs/1705.08500 ; published NeurIPS paper on selective classification/rejection, not a guarantee for our empirical threshold |
| geifman2019selectivenet | https://proceedings.mlr.press/v97/geifman19a.html ; learned reject option; not implemented here |
| wang2020scn | https://doi.org/10.1109/CVPR42600.2020.00693 ; uncertainty-aware weighting/relabeling prior art |
| liu2021swin | https://doi.org/10.1109/ICCV48922.2021.00986 ; shifted-window hierarchical backbone |
| hu2018senet | https://doi.org/10.1109/CVPR.2018.00745 ; channel recalibration principle, not proof that our pooled gate is novel |
| deng2009imagenet | https://doi.org/10.1109/CVPR.2009.5206848 ; pretraining data source; exact weight version established from code |
| paszke2019pytorch | https://arxiv.org/abs/1912.01703 ; published NeurIPS PyTorch framework paper |
| lugaresi2019mediapipe | https://arxiv.org/abs/1906.08172 ; MediaPipe framework; detector settings established from app code |
| wang2016emotionpush | https://aclanthology.org/C16-2030/ ; text-based emotion notification system, not facial inference |
| chong2019emochat | https://doi.org/10.1109/BIGCOM.2019.00037 ; author paper https://tns.thss.tsinghua.edu.cn/sun/publications/2019.EmoChat.pdf ; multimodal mobile-message cues predate this prototype |
| liu2019selfawareness | https://doi.org/10.1109/CIC48465.2019.00030 ; author institution https://commons.clarku.edu/faculty_computer_sciences/166/ ; facial-expression-informed messaging and self-awareness |
| barrett2019emotions | https://doi.org/10.1177/1529100619832930 ; limitations of internal-emotion inference from facial movement |
| loshchilov2019adamw | https://openreview.net/forum?id=Bkg6RiCqY7 ; decoupled weight decay/AdamW |
| sokolova2009metrics | https://doi.org/10.1016/j.ipm.2009.03.002 ; classification measures; actual formulas explicit in manuscript |
| guo2017calibration | https://proceedings.mlr.press/v70/guo17a.html ; distinction between confidence and calibration |
| le2023uncertainty | https://doi.org/10.1109/WACV56688.2023.00603 ; author institution https://livrepository.liverpool.ac.uk/3169130/ ; distributions from valence-arousal neighbors |
| kawamura2024midas | https://doi.org/10.1109/WACV57701.2024.00642 ; CVF WACV2024 accepted paper, soft-label mixing for dynamic FER |

## New metadata checks

Read IEEE-deposited Crossref API records for both new DOIs. Le et al. author
order: Nhat Le, Khanh Nguyen, Quang Tran, Erman Tjiputra, Bac Le, Anh Nguyen.
IEEE pages 6077-6086 differ from CVF open-access pagination 6088-6097; the
DOI-backed reference uses IEEE pages. MIDAS authors: Ryosuke Kawamura,
Hideaki Hayashi, Noriko Takemura, Hajime Nagahara. IEEE pages 6538-6548 differ
from CVF pagination 6552-6562; again the DOI-backed reference uses IEEE pages.
This difference is not silently treated as contradictory authorship or title.

Wang, Li, and Wang (https://arxiv.org/abs/2609.17130) is explicitly unpublished,
submitted 15 September 2026 to ICASSP 2027. It is acknowledged in text, not
listed as an accepted publication or used as a validated numerical comparator.
No result percentages are borrowed from it.

## Journal requirements checked live

https://link.springer.com/journal/42979/submission-guidelines
https://link.springer.com/journal/42979/how-to-publish-with-us

Structured abstract 150-250 words; 4-6 keywords; mathematical LaTeX accepted;
numbered references; author/corresponding details; declarations; explicit
AI-assistance disclosure beyond copyediting. Vector figures use embedded fonts,
lowercase panel labels, and patterns/line styles that do not depend on color.
No acceptance timetable or likelihood is guaranteed.
