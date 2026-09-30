# Released inference checkpoints

Crowd Supervision and Vocabulary-Mismatched Selective Facial-Expression Prediction
SN Computer Science
Gunjan Kumar Mishra; Bijaya Ghimire; Badri Raj Lamichhane; Alan Shah
Corresponding author: Badri Raj Lamichhane, School of Information, Computer and Communication Technology, SIIT, Thammasat University, Thailand; d6622300231@g.siit.tu.ac.th


24 checkpoints: four supervision conditions (Hard, Soft, Uniform, Tie-hard), seeds 17/42/89, selected or fixed epoch 12. Each has the same seven-output grayscale3/224/bilinear/mean0.5/std0.5 Swin-T plus one pooled-vector channel gate. Class order: angry, disgust, fear, happy, neutral, sad, surprise. The export removes optimizer/RNG states, not model tensors, and adds the release protocol hash and evaluation state. The inherited `protocol` field names the historical shared training implementation; `configuration.condition`, `release_protocol` and the frozen follow-up protocol identify the actual condition and provenance.

Selected/final inference on the supplied path uses BF16 CUDA autocast; the prototype uses FP32. Cross-hardware bitwise equality is not assumed. No checkpoint predicts internal feelings, intention or sincerity. No runtime demographic fairness, out-of-domain accuracy, formal privacy guarantee or communication benefit is established. Do not use for consequential decisions about people.

No source face images are distributed. Dataset provenance and rights limitations are stated in the manuscript. Use the weights with the packaged code and inspect SHA256_MANIFEST.json. The fixed-grid policy is an offline assumption-qualified comparator, not an installed safeguard.
