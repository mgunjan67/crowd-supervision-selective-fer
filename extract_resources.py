"""Verify and combine reviewer resources without overwriting archive manifests.

No photographs or videos are downloaded. This utility needs Python 3.10+ only.
The index describes both current resources and unchanged historical weight ZIPs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import re
import urllib.request
import urllib.parse
import zipfile


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return digest(stream)


def safe_name(name):
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe ZIP member: {name}")
    return value


def verify_archive(path, expected=None):
    """Independently hash every member and require a complete stored manifest."""
    if expected and file_sha(path) != expected:
        raise ValueError(f"Archive SHA-256 mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate members in {path}")
        for name in names:
            safe_name(name)
        candidates = [name for name in names if "/" not in name and
                      (name.startswith("SHA256") or name.endswith("-SHA256SUMS.json"))]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one root member manifest: {path}: {candidates}")
        manifest_name = candidates[0]
        manifest = json.loads(archive.read(manifest_name))
        if not isinstance(manifest, dict) or any(not isinstance(value, str) or
                not re.fullmatch(r"[0-9a-f]{64}", value) for value in manifest.values()):
            raise ValueError(f"Malformed SHA-256 manifest: {path}")
        if set(manifest) != set(names) - {manifest_name}:
            raise ValueError(f"Incomplete or extraneous member manifest: {path}")
        for name, expected_hash in manifest.items():
            with archive.open(name) as stream:
                if digest(stream) != expected_hash:
                    raise ValueError(f"Member SHA-256 mismatch: {path}: {name}")
    return manifest_name, manifest


def assemble(index_path, destination, archive_dir=None, include_weights=False,
             download=False, verify_only=False, allow_local_paths=False, purpose="unspecified"):
    index_path = Path(index_path).resolve()
    data = json.loads(index_path.read_text(encoding="utf-8"))
    ids = [row["id"] for row in data["resources"]]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids):
        raise ValueError("Resource IDs must be unique simple names")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if purpose not in {"unspecified", "replay", "immutable-staging"}:
        raise ValueError("Unknown extraction purpose")
    purpose_path = destination / "EXTRACTION_PURPOSE.json"
    if purpose_path.is_file():
        recorded = json.loads(purpose_path.read_text(encoding="utf-8"))["purpose"]
        if purpose != "unspecified" and recorded != purpose:
            raise ValueError("Do not change extraction purpose: use separate fresh staging and replay directories")
        purpose = recorded
    elif not verify_only:
        purpose_path.write_text(json.dumps({"purpose": purpose,
            "guidance": "Numerical replay may regenerate timestamps/ledgers. Never stage publication source from a replay working tree."}, indent=2) + "\n", encoding="utf-8")
    archive_dir = Path(archive_dir or index_path.parent).resolve()
    results = []
    for resource in data["resources"]:
        if resource.get("weights") and not include_weights:
            continue
        filename = resource["filename"]
        if safe_name(filename).name != filename:
            raise ValueError(f"Archive filename must be a basename: {filename}")
        path = archive_dir / filename
        if not path.exists() and allow_local_paths and resource.get("local_path"):
            path = Path(resource["local_path"])
        if not path.exists() and download:
            url = resource.get("verified_public_url")
            if not url:
                raise FileNotFoundError(f"No verified public URL for {resource['filename']}")
            if urllib.parse.urlparse(url).scheme != "https":
                raise ValueError("Resource download URL must use HTTPS")
            path = archive_dir / resource["filename"]
            archive_dir.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url) as response, path.open("wb") as out:
                shutil.copyfileobj(response, out)
        if not path.exists():
            raise FileNotFoundError(f"Place {resource['filename']} in {archive_dir}")
        manifest_name, manifest = verify_archive(path, resource["sha256"])
        prefix = safe_name(resource.get("extract_prefix", ""))
        mapped = {}
        collisions = 0
        if not verify_only:
            with zipfile.ZipFile(path) as archive:
                for name, expected_hash in manifest.items():
                    relative = prefix / safe_name(name)
                    target = destination.joinpath(*relative.parts)
                    if not target.resolve().is_relative_to(destination):
                        raise ValueError(f"Escaped destination: {target}")
                    if target.exists():
                        if file_sha(target) != expected_hash:
                            raise FileExistsError(f"Conflicting resource member: {relative}")
                        collisions += 1
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(name) as stream, target.open("wb") as out:
                            shutil.copyfileobj(stream, out)
                        if file_sha(target) != expected_hash:
                            raise ValueError(f"Extracted member mismatch: {relative}")
                    mapped[relative.as_posix()] = expected_hash
            # Historical checkpoint ZIPs deliberately remain byte-identical.
            # Their identical root manifest names never reach the shared root.
            manifest_target = destination / "resource-manifests" / f"{resource['id']}-SHA256SUMS.json"
            manifest_target.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(mapped, indent=2) + "\n"
            if manifest_target.exists() and manifest_target.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"Conflicting existing resource manifest: {manifest_target}")
            if not manifest_target.exists():
                manifest_target.write_text(content, encoding="utf-8")
        results.append({"id": resource["id"], "file": str(path),
                        "members_verified": len(manifest),
                        "archive_sha256": resource["sha256"],
                        "original_manifest": manifest_name,
                        "identical_existing_members": collisions,
                        "manifest_destination": f"resource-manifests/{resource['id']}-SHA256SUMS.json"})
        print(f"{resource['id']}: {len(manifest)} member hashes passed", flush=True)
    report = {"status": "passed", "mode": "verify_only" if verify_only else "assembled",
              "destination": str(destination), "resources": results, "purpose": purpose,
              "manifest_overwrites": 0,
              "existing_nonidentical_files_overwritten": 0}
    (destination / "RESOURCE_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(__file__).with_name("RESOURCE_INDEX.json"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--include-weights", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-local-paths", action="store_true", help="Author workspace only")
    parser.add_argument("--purpose", choices=("replay", "immutable-staging"), default="replay",
                        help="Use separate fresh roots for mutable replay and immutable source staging")
    args = parser.parse_args()
    assemble(args.index, args.destination, args.archive_dir, args.include_weights,
             args.download, args.verify_only, args.allow_local_paths, args.purpose)
