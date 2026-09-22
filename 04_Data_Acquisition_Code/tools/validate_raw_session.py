"""CLI: python tools/validate_raw_session.py <session_dir>

Walks a `session_YYYYMMDD_NNN/` directory and checks:
  - `manifest.json` is present and parses
  - every stream `manifest.json` lists in `enabled_streams` has at least
    one raw file under `raw/` (a sensor that legitimately never started --
    see `manifest.collector_status` -- is simply absent from
    `enabled_streams`, not a failure)
  - every `*.jsonl` file (not `*.jsonl.partial`) parses line-by-line as
    JSON

Prints a pass/fail summary; exit code 0 on pass, 1 on fail. This is the
acceptance-check tool referenced in master spec section 26's example
command list.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _check_manifest(session_dir: str) -> tuple[bool, dict | None, list[str]]:
    manifest_path = os.path.join(session_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False, None, [f"FAIL manifest.json missing at {manifest_path}"]
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError as exc:
        return False, None, [f"FAIL manifest.json does not parse: {exc}"]
    return True, manifest, ["PASS manifest.json present and parses"]


def _check_jsonl_files(raw_dir: str) -> tuple[bool, list[str]]:
    if not os.path.isdir(raw_dir):
        return False, [f"FAIL raw/ directory missing at {raw_dir}"]

    ok = True
    lines: list[str] = []
    jsonl_paths = sorted(glob.glob(os.path.join(raw_dir, "*.jsonl")))
    partial_paths = sorted(glob.glob(os.path.join(raw_dir, "*.jsonl.partial")))
    if not jsonl_paths and not partial_paths:
        lines.append("WARN raw/ contains no *.jsonl or *.jsonl.partial files")

    for path in jsonl_paths:
        name = os.path.basename(path)
        bad_line = None
        count = 0
        with open(path, encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                text = raw_line.strip()
                if not text:
                    continue
                try:
                    json.loads(text)
                    count += 1
                except json.JSONDecodeError as exc:
                    bad_line = (lineno, str(exc))
                    break
        if bad_line:
            ok = False
            lines.append(f"FAIL {name}: invalid JSON at line {bad_line[0]}: {bad_line[1]}")
        else:
            lines.append(f"PASS {name}: {count} valid JSONL record(s)")

    for path in partial_paths:
        lines.append(f"INFO {os.path.basename(path)}: left as .partial (session not clean-closed here; excluded from validity check)")

    return ok, lines


def _check_expected_streams(raw_dir: str, manifest: dict) -> tuple[bool, list[str]]:
    enabled = manifest.get("enabled_streams")
    if not enabled:
        return True, ["WARN manifest has no 'enabled_streams' list -- skipping expected-file-exists check"]

    ok = True
    lines: list[str] = []
    # A `.partial` file is an interrupted segment, never a complete one
    # (core/jsonl_writer.py: only a clean close renames it). It must not
    # satisfy the "this stream produced data" check -- otherwise a crashed
    # session validates identically to a clean one.
    final = {os.path.basename(p) for p in glob.glob(os.path.join(raw_dir, "*.jsonl"))}
    partial = {os.path.basename(p) for p in glob.glob(os.path.join(raw_dir, "*.jsonl.partial"))}
    for stream_id in enabled:
        prefix = f"{stream_id}_segment"
        matches = sorted(name for name in final if name.startswith(prefix))
        partial_matches = sorted(name for name in partial if name.startswith(prefix))
        if matches:
            lines.append(f"PASS stream '{stream_id}' has raw file(s): {matches}")
        elif partial_matches:
            ok = False
            lines.append(
                f"FAIL stream '{stream_id}' has only interrupted .partial file(s) "
                f"{partial_matches} -- not a completed stream; run "
                f"tools/recover_partial_jsonl.py on them"
            )
        else:
            ok = False
            lines.append(f"FAIL stream '{stream_id}' is enabled in manifest but has no raw file in {raw_dir}")
    return ok, lines


def validate(session_dir: str) -> tuple[bool, list[str]]:
    all_lines = [f"Validating session: {session_dir}"]

    manifest_ok, manifest, manifest_lines = _check_manifest(session_dir)
    all_lines += manifest_lines
    if not manifest_ok:
        all_lines.append("FAIL")
        return False, all_lines

    raw_dir = os.path.join(session_dir, "raw")
    jsonl_ok, jsonl_lines = _check_jsonl_files(raw_dir)
    all_lines += jsonl_lines

    streams_ok, streams_lines = _check_expected_streams(raw_dir, manifest)
    all_lines += streams_lines

    complete = bool(manifest.get("session_complete"))
    all_lines.append(f"{'PASS' if complete else 'WARN'} manifest.session_complete={complete} interrupted={manifest.get('interrupted')}")

    overall = manifest_ok and jsonl_ok and streams_ok
    all_lines.append("PASS" if overall else "FAIL")
    return overall, all_lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.session_dir):
        print(f"error: not a directory: {args.session_dir}", file=sys.stderr)
        return 1

    ok, lines = validate(args.session_dir)
    for line in lines:
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
