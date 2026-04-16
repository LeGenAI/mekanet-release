# Model Weights Policy

Production model weights are **not** bundled in this public repository.

## License Scope

Code in the main GitHub repository remains MIT-licensed, but the distributed model artifact is covered by separate review/research-only terms:

- [WEIGHTS_LICENSE.md](./WEIGHTS_LICENSE.md)

## Expected Filenames

- `epoch60.pt` for the detection model
- `classifier.pkl` for the classification model

## Current Public-Release Behavior

- this directory documents the expected filenames only
- `download_weights.py` is manifest-driven and never creates fake placeholder models
- experiment code will fail honestly if real weights are absent
- `manifest.json` is the canonical list of expected filenames and sources
- public model repo: [LeBrony/mekanet-release-weights](https://huggingface.co/LeBrony/mekanet-release-weights)

## Verification

```bash
cd weights
python download_weights.py --verify
```

## Download

```bash
cd weights
python download_weights.py
```

By default the downloader tries, in order:

1. the configured Hugging Face Hub entry from `manifest.json`
2. an environment-variable override such as `MEKANET_EPOCH60_PT_URL`
3. the configured GitHub release asset URL from `manifest.json`

The current canonical model namespace is intended to be `LeBrony/mekanet-release-weights`, with filenames and checksums tracked in `manifest.json`.

## Important Note

If real weights are released later, the download script can be updated with canonical URLs. Until then, do not treat this repository as a fully reproducible weight distribution.
