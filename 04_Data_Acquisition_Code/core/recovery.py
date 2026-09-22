"""Recovery for interrupted (`.partial`) JSONL sessions.

Master spec section 16: a crash/kill leaves `<stream>_segment{NNN}.jsonl.
partial` behind instead of the atomically-renamed final file (see
`core/jsonl_writer.py`). This module never touches the original `.partial`
file -- it only reads it and writes NEW files next to it:

- `<name>.recovered.jsonl`      -- the successfully-parsed complete records
- `<name>.recovery_report.json` -- what was recovered / dropped / gapped

A `.partial` file can end mid-line if the process was killed between a
`write()` and the next flush -- that trailing fragment (and, per spec,
everything after the first parse failure) is separated out rather than
silently discarded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

# core.schemas.base_record's field name -- imported by convention, not
# value, since schemas.py defines the dict shape, not a constant to import.
HOST_SEQ_FIELD = "host_seq"

_TRUNCATE_CHARS = 500
_PARTIAL_SUFFIX = ".partial"
_JSONL_SUFFIX = ".jsonl"


@dataclass
class RecoveryResult:
    partial_path: str
    recovered_path: str
    report_path: str
    total_lines: int
    valid_record_count: int
    dropped_lines: list[dict[str, Any]]
    seq_gaps: list[list[int]]

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "partial_path": self.partial_path,
            "recovered_path": self.recovered_path,
            "total_lines_scanned": self.total_lines,
            "valid_records_recovered": self.valid_record_count,
            "corrupt_or_incomplete_lines_dropped": self.dropped_lines,
            "host_seq_gaps": self.seq_gaps,
        }


def _output_paths(partial_path: str) -> tuple[str, str]:
    base = partial_path[: -len(_PARTIAL_SUFFIX)] if partial_path.endswith(_PARTIAL_SUFFIX) else partial_path
    stem = base[: -len(_JSONL_SUFFIX)] if base.endswith(_JSONL_SUFFIX) else base
    return stem + ".recovered.jsonl", stem + ".recovery_report.json"


def recover_partial_jsonl(partial_path: str) -> RecoveryResult:
    """Parse `partial_path` line by line. The first line that fails
    `json.loads` (and everything after it, to end of file) is treated as
    one incomplete trailing fragment and separated out -- per spec, we do
    not keep trying to parse lines past the first failure, since a crash
    mid-write only ever corrupts the tail. Sequence gaps (`host_seq`
    jumping by more than 1 between consecutive successfully-parsed
    records) are reported but not otherwise acted on.
    """
    with open(partial_path, "r", encoding="utf-8") as fh:
        raw_lines = fh.readlines()  # keeps '\n'; original file is never rewritten

    valid_records: list[dict[str, Any]] = []
    dropped_lines: list[dict[str, Any]] = []
    first_bad_index: Optional[int] = None

    for index, line in enumerate(raw_lines):
        stripped = line.rstrip("\n")
        if stripped == "":
            continue  # tolerate a trailing blank line -- not corruption
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            first_bad_index = index
            break
        valid_records.append(record)

    if first_bad_index is not None:
        fragment_text = "".join(raw_lines[first_bad_index:])
        dropped_lines.append(
            {
                "starting_line_number": first_bad_index + 1,
                "line_count": len(raw_lines) - first_bad_index,
                "raw_text_truncated": fragment_text[:_TRUNCATE_CHARS],
                "truncated": len(fragment_text) > _TRUNCATE_CHARS,
            }
        )

    seq_gaps: list[list[int]] = []
    prev_seq: Optional[int] = None
    for record in valid_records:
        seq = record.get(HOST_SEQ_FIELD) if isinstance(record, dict) else None
        if isinstance(seq, int):
            if prev_seq is not None and seq - prev_seq > 1:
                seq_gaps.append([prev_seq, seq])
            prev_seq = seq

    recovered_path, report_path = _output_paths(partial_path)

    with open(recovered_path, "w", encoding="utf-8") as out:
        for record in valid_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = RecoveryResult(
        partial_path=partial_path,
        recovered_path=recovered_path,
        report_path=report_path,
        total_lines=len(raw_lines),
        valid_record_count=len(valid_records),
        dropped_lines=dropped_lines,
        seq_gaps=seq_gaps,
    )

    with open(report_path, "w", encoding="utf-8") as out:
        json.dump(result.as_report_dict(), out, ensure_ascii=False, indent=2)

    return result
