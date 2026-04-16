"""
Manifest-driven weight downloader for the public MekaNet release.

This script intentionally avoids generating placeholder artifacts. It downloads
only from explicitly configured public sources and verifies checksums when they
are available in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


WEIGHTS_DIR = Path(__file__).parent
MANIFEST_PATH = WEIGHTS_DIR / "manifest.json"
CHUNK_SIZE = 1024 * 1024


def load_manifest() -> Dict[str, dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    weights = manifest.get("weights", {})
    if not weights:
        raise ValueError(f"Manifest has no weights section: {MANIFEST_PATH}")
    return weights


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_url(source: dict, filename: str) -> Tuple[Optional[str], Optional[str]]:
    source_type = source.get("type")

    if source_type == "env_url":
        env_name = source["env"]
        value = os.environ.get(env_name)
        if not value:
            return None, f"env var {env_name} is not set"
        return value, None

    if source_type == "direct":
        return source["url"], None

    if source_type == "github_release":
        owner = source["owner"]
        repo = source["repo"]
        tag = source["tag"]
        asset = source.get("asset", filename)
        url = f"https://github.com/{owner}/{repo}/releases/download/{tag}/{asset}"
        return url, None

    if source_type == "huggingface":
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError:
            return None, "huggingface_hub is not installed"

        repo_id = source["repo_id"]
        revision = source.get("revision", "main")
        repo_type = source.get("repo_type", "model")
        asset = source.get("asset", filename)

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=asset,
                revision=revision,
                repo_type=repo_type,
                local_dir=str(WEIGHTS_DIR),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:  # pragma: no cover - passthrough reporting
            return None, f"huggingface download failed: {exc}"
        return downloaded, None

    return None, f"unsupported source type: {source_type}"


def download_to_path(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mekanet-weight-downloader/1.0"})
    with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out_file:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            out_file.write(chunk)
    tmp_path.replace(destination)


def verify_one(filename: str, spec: dict) -> bool:
    path = WEIGHTS_DIR / filename
    if not path.exists():
        print(f"❌ {filename}: missing")
        return False

    expected_sha = spec.get("sha256")
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha.lower() != expected_sha.lower():
            print(f"❌ {filename}: sha256 mismatch")
            print(f"   expected: {expected_sha}")
            print(f"   actual:   {actual_sha}")
            return False

    print(f"✅ {filename}: present")
    return True


def download_one(filename: str, spec: dict, force: bool = False) -> bool:
    target_path = WEIGHTS_DIR / filename
    expected_sha = spec.get("sha256")

    if target_path.exists() and not force:
        if expected_sha:
            actual_sha = sha256_file(target_path)
            if actual_sha.lower() == expected_sha.lower():
                print(f"✅ {filename}: already present and checksum verified")
                return True
            print(f"⚠️ {filename}: checksum mismatch, re-downloading")
        else:
            print(f"✅ {filename}: already present (no checksum in manifest)")
            return True

    failures = []
    for source in spec.get("sources", []):
        url_or_path, error = build_source_url(source, filename)
        if error:
            failures.append(error)
            continue

        try:
            if source.get("type") == "huggingface":
                downloaded_path = Path(url_or_path)
                if downloaded_path.resolve() != target_path.resolve():
                    downloaded_path.replace(target_path)
            else:
                print(f"⬇️ Downloading {filename} from {url_or_path}")
                download_to_path(url_or_path, target_path)

            if expected_sha:
                actual_sha = sha256_file(target_path)
                if actual_sha.lower() != expected_sha.lower():
                    failures.append(
                        f"downloaded from {source.get('type')} but sha256 mismatched "
                        f"(expected {expected_sha}, got {actual_sha})"
                    )
                    if target_path.exists():
                        target_path.unlink()
                    continue

            print(f"✅ Downloaded {filename}")
            return True
        except urllib.error.HTTPError as exc:
            failures.append(f"{source.get('type')} HTTP {exc.code}: {exc.reason}")
        except urllib.error.URLError as exc:
            failures.append(f"{source.get('type')} URL error: {exc.reason}")
        except Exception as exc:  # pragma: no cover - passthrough reporting
            failures.append(f"{source.get('type')} failed: {exc}")

    print(f"❌ Failed to download {filename}")
    for failure in failures:
        print(f"   - {failure}")
    return False


def list_weights(manifest: Dict[str, dict]) -> None:
    print("Available weight entries:")
    for filename, spec in manifest.items():
        print(f"- {filename}")
        print(f"  description: {spec.get('description', '-')}")
        print(f"  required: {spec.get('required', True)}")
        print(f"  sha256: {spec.get('sha256') or 'not provided'}")
        print(f"  sources: {len(spec.get('sources', []))}")


def verify_all(manifest: Dict[str, dict], names: Iterable[str]) -> bool:
    overall = True
    for name in names:
        overall = verify_one(name, manifest[name]) and overall
    return overall


def download_all(manifest: Dict[str, dict], names: Iterable[str], force: bool = False) -> bool:
    overall = True
    for name in names:
        overall = download_one(name, manifest[name], force=force) and overall
    return overall


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or verify MekaNet model weights")
    parser.add_argument("--verify", action="store_true", help="Verify locally available weights")
    parser.add_argument("--list", action="store_true", help="List manifest entries")
    parser.add_argument("--weight", action="append", help="Specific weight filename to target")
    parser.add_argument("--force", action="store_true", help="Re-download even if a file exists")
    args = parser.parse_args()

    manifest = load_manifest()
    selected = args.weight or list(manifest.keys())

    missing_entries = [name for name in selected if name not in manifest]
    if missing_entries:
        for name in missing_entries:
            print(f"Unknown weight entry: {name}")
        return 2

    if args.list:
        list_weights(manifest)
        return 0

    if args.verify:
        return 0 if verify_all(manifest, selected) else 1

    success = download_all(manifest, selected, force=args.force)
    if not success:
        print()
        print("No placeholder files were created.")
        print("If weights are hosted privately, export an override such as:")
        for name in selected:
            env_name = manifest[name].get("env_url")
            if env_name:
                print(f"  export {env_name}=https://example.com/{name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
