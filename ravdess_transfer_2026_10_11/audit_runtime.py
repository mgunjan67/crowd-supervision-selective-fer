"""Check the actual inference runtime and detector assets against the local freeze."""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import PIL
import torch
import torchvision

from common import HERE, require_frozen, sha


def main():
    frozen = require_frozen()
    actual = dict(python=sys.version, torch=torch.__version__,
                  torchvision=torchvision.__version__, opencv=cv2.__version__,
                  mediapipe=mp.__version__, numpy=np.__version__, pillow=PIL.__version__,
                  platform=platform.platform(),
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    assert actual == frozen['software'], (actual, frozen['software'])
    detector_dir = Path(mp.__file__).resolve().parent / 'modules/face_detection'
    assets = {name: sha(detector_dir / name) for name in frozen['detector_hashes']}
    assert assets == frozen['detector_hashes'], assets
    report = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
                  freeze_sha256=sha(HERE / 'freeze.json'), software=actual,
                  detector_assets_sha256=assets, shared_source_hashes_match=True,
                  scope='Current local environment and detector assets match the recorded freeze; not a cross-platform reproducibility claim.')
    (HERE / 'runtime_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
