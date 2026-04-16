---
license: other
license_name: mekanet-weights-review-license
license_link: LICENSE
library_name: ultralytics
tags:
  - pathology
  - object-detection
  - yolo
  - sahi
  - medical-imaging
---

# MekaNet Detection Weights

This repository hosts public weight artifacts for the MekaNet manuscript-aligned release.

## Included artifact

- `epoch60.pt`: YOLOv8 detection checkpoint for megakaryocyte detection

## Provenance

- Source code repository: `LeGenAI/mekanet-release`
- Manuscript source: private, not distributed in the public repository
- Intended public release consumer: reviewers and readers validating the detection pipeline

## License

This model artifact is distributed for research, manuscript review, and reproducibility verification only.

- no clinical deployment
- no commercial deployment
- no representation of regulatory approval
- preserve attribution and license notice on redistribution

See the repository `LICENSE` file in this Hugging Face model repo for the controlling terms.

## Integrity

- `epoch60.pt` SHA-256: `dc3b411530457815219a6549587cf18f57590e4d77c859597c9647d176568d53`

## Usage

```python
from mekanet.models.yolo_sahi import YoloSahiDetector

detector = YoloSahiDetector("epoch60.pt", confidence_threshold=0.20, device="cpu")
```

## Notes

- This weight artifact is distributed separately from the main code repository to keep the public code release lightweight and reviewer-friendly.
- The main repository uses a manifest-driven downloader and checksum verification.
