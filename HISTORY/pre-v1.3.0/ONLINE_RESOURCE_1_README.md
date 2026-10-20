# Reproducing the second-backbone sensitivity

Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, NVIDIA RTX 3070 Laptop GPU. See each run environment.txt for complete pins. This is an adaptive sensitivity experiment, not optimized ResNet accuracy or independent-source confirmation.

Acquire the same FER2013 source and official FER+ annotations using retained provenance records; images are not supplied. Restore data/fer2013 NPZ partitions and data/ferplus_annotations/fer2013new.csv using the retained conversion/acquisition code and hashes. Pretrained source is https://download.pytorch.org/models/resnet18-f37072fd.pth. Shared grayscale and normalization differ from ImageNet default inference preprocessing by design.

Extract OnlineResource1.zip into a working folder. Extract OnlineResource7.zip into that same folder for ResNet inference weights. Swin archives 2–6 remain unchanged at the previous versioned releases. The public reproduction helper reloads the 18 weights and produces all 54 role-specific prediction arrays; CPU reproduction is permitted but is not promised bitwise equal to matched CUDA/BF16 inference. Evaluation.py records original full training checkpoint hashes; those full optimizer states are retained locally, not distributed in inference archives. audit.py checks those original full-file hashes only when the source files are available, and always checks the supplied inference weights and prediction-file hashes. Do not run the original training driver over the archived configurations expecting it to overwrite evidence. Fresh retraining uses retrain.py with a new output folder inside the extracted study root; it retains the declared fixed budget.

From the extracted root, use:

    python backbone_followup/reproduce_inference.py --output-dir reproduced-resnet --device cuda
    python backbone_followup/audit.py
    python backbone_followup/retrain.py --output-dir fresh-resnet-soft17 --condition soft --seed 17

Inference and fresh retraining require acquired images; arithmetic auditing requires only the released arrays and inference weights. The train-only target audit also requires images/source data. Set PYTHONPATH to code/ when running the combined pytest suite. Current-result ledger: backbone_followup/reported_results.csv; earlier root ledgers remain historical provenance.

The merged reported_results.csv labels every retained row by backbone. audit.py recomputes reference F1, common-coverage risk, strata, unsupported mass and fixed-set comparisons through a separate arithmetic implementation. This is internal verification, not independent human replication. Freeze files and per-run logs retain a pre-epoch Windows worker-import repair; no calibration/test evaluation preceded the nine completed runs.

Unit tests cover the 512-dimensional pooled gate, targets and zero-mass gradients, as well as the retained study tests. The messaging app retains its Swin-only inference contract; do not supply a ResNet research checkpoint to that app. No original face images, credentials or participant data are supplied. Scope and ethics limitations remain in the manuscript.
