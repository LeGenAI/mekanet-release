<div align="center">

# MekaNet

**A manuscript-aligned public release for megakaryocyte detection and myeloproliferative neoplasm classification**

[![License: MIT](https://img.shields.io/badge/code-MIT-2ea44f)](./LICENSE)
[![Weights](https://img.shields.io/badge/weights-Hugging%20Face-f59e0b)](https://huggingface.co/LeBrony/mekanet-release-weights)
[![Citation](https://img.shields.io/badge/citation-CITATION.cff-6f42c1)](./CITATION.cff)

</div>

> [!IMPORTANT]
> This repository is intentionally trimmed for manuscript review and reproducibility. It distinguishes between:
> what the paper describes, what is publicly shipped here, and what remains private.

## At A Glance

| Item | Public status | Source |
| --- | --- | --- |
| Manuscript | Not public | Removed from the public release |
| Detection weight | Public | [HF model repo](https://huggingface.co/LeBrony/mekanet-release-weights) |
| Public tabular feature table | Not public | Removed from the public release |
| Patient-level master cohort | Not public | Not included in this release |
| Private image manifests and full clinical tables | Not public | Not included in this release |

## Quick Links

- Citation metadata: [CITATION.cff](./CITATION.cff)
- Alignment note: [docs/manuscript_alignment.md](./docs/manuscript_alignment.md)
- Weight policy: [weights/README.md](./weights/README.md)
- Data policy: [data/README.md](./data/README.md)

## Manuscript Alignment

- The private manuscript uses `Department of Laboratory Medicine`, not `Department of Pathology`.
- Public-facing docs in this repository were rewritten to match manuscript wording and remove stale release-summary material.
- The repository now prioritizes auditability over promotional presentation.

## Public Data Accounting

The current repository supports the following components with local evidence:

| Scope | Public evidence in repo | Notes |
| --- | --- | --- |
| Detection training cohort | Manuscript describes a partially labeled B-hospital MPN image subset | Image-level manifest is not public |
| Detection internal fully labeled evaluation | Manuscript describes fully labeled images used for internal testing | Image files are not public |
| External validation | Manuscript describes S-hospital test sets used for generalizability | Detailed in the manuscript evaluation |

The following items are **not** present in this public release to maintain anonymity and compliance:

- The complete patient-level raw dataset
- The full patient/control manifest used to reconstruct the exact classification cohort from raw IDs
- The control cases cited in the classification section of the manuscript
- Detailed ID mappings for hospital-specific subsets
- Historically uploaded public tabular feature tables
- Production model weights beyond the released detection checkpoint

For a detailed reconciliation note, see [docs/manuscript_alignment.md](./docs/manuscript_alignment.md).

## Repository Layout

```text
.
├── CITATION.cff
├── README.md
├── data/
│   └── README.md
├── experiments/
│   ├── classification/
│   └── detection/
├── mekanet/
├── weights/
└── run_paper_reproduction.py
```

### Module Roles

- `mekanet/`: reusable Python package components
- `experiments/detection/`: manuscript-facing detection pipeline
- `experiments/classification/`: manuscript-facing classification analysis
- `data/`: documentation about non-public data scope
- `weights/`: weight handling policy and verification helpers

## Canonical Entry Points

For this slimmed public release, the canonical reproducibility entry points are:

```bash
python run_paper_reproduction.py --quick --dry-run
python experiments/classification/run_all_experiments.py
```

Legacy deployment wrappers, benchmark helpers, auto-generated reports, unused config variants, and unreferenced figures were removed to keep the repository reviewer-friendly.

## Current Execution Status

The codebase has been bootstrapped so that package imports and dry-run entrypoints work in a clean local environment after installing dependencies.

```bash
pip install -r requirements.txt
python run_paper_reproduction.py --quick --dry-run
python experiments/classification/run_all_experiments.py
```

### Current limitations

- Detection execution still requires the released weight artifact.
- Classification execution still requires a user-supplied `data/demo_data/classification.csv`.
- No public tabular feature CSV is shipped in this repository.

## Public Assets

### Detection Weight

- repo: [LeBrony/mekanet-release-weights](https://huggingface.co/LeBrony/mekanet-release-weights)
- direct file: [epoch60.pt](https://huggingface.co/LeBrony/mekanet-release-weights/resolve/main/epoch60.pt)

The local downloader uses the Hugging Face source first and verifies the SHA-256 recorded in `weights/manifest.json`.

## Reviewer Notes

> [!NOTE]
> Earlier drafts of this repository contained release-summary, quickstart, benchmark, and deployment files with stale cohort counts or non-canonical execution paths. Those were removed to reduce ambiguity during review.

The remaining documentation favors manuscript-aligned descriptions over broad showcase material.

The manuscript source file itself is private and is not distributed in this repository.

## Clinical Affiliation

- `Sang Mee Hwang`: Department of Laboratory Medicine, Seoul National University Bundang Hospital
- `Young-eun Lee`: Department of Laboratory Medicine, Seoul National University College of Medicine

## License

License scope is intentionally split:

- code in this GitHub repository: MIT
- public weight artifacts: separate review/research-only terms

See:

- [LICENSE](./LICENSE)
- [weights/WEIGHTS_LICENSE.md](./weights/WEIGHTS_LICENSE.md)

Patient-level source data is not included in this public release.
