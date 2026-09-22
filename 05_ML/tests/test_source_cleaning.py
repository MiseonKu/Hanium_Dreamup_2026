from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from imu_vive_learning.cleaning import CleaningContractError, clean_trials
from imu_vive_learning.source import (
    EXPECTED_TRIALS_SCHEMA,
    SourceContractError,
    load_trials,
    load_trials_from_config,
    sha256_file,
)


WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "static_endpoint_v1.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_frame(config: dict):
    return load_trials_from_config(config["source"], base_dir=WORKSPACE)


@pytest.fixture(scope="module")
def canonical(config: dict, source_frame):
    return clean_trials(
        source_frame,
        config["cleaning"],
        session_id=config["contract"]["session_id"],
        source_member_sha256=config["source"]["member_sha256"],
    )


def test_real_source_archive_matches_hash_schema_and_row_contract(
    config: dict, source_frame
) -> None:
    assert len(source_frame) == 10_055
    assert list(source_frame.columns) == EXPECTED_TRIALS_SCHEMA.names
    assert source_frame.attrs["source_archive_sha256"] == config["source"][
        "archive_sha256"
    ]
    assert source_frame.attrs["source_member_sha256"] == config["source"][
        "member_sha256"
    ]


def test_unmaterialized_lfs_pointer_fails_closed(tmp_path: Path) -> None:
    pointer = tmp_path / "source.zip"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'0' * 64}\nsize 123\n",
        encoding="ascii",
    )
    with pytest.raises(SourceContractError, match="unmaterialized Git LFS pointer"):
        load_trials(
            pointer,
            member="trials.parquet",
            expected_archive_sha256="0" * 64,
            expected_member_sha256="0" * 64,
            expected_rows=1,
        )


def test_member_hash_and_arrow_schema_fail_closed(tmp_path: Path) -> None:
    member = "trials.parquet"
    raw_buffer = BytesIO()
    pq.write_table(pa.table({"unexpected": [1]}), raw_buffer)
    raw_member = raw_buffer.getvalue()
    archive_path = tmp_path / "source.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member, raw_member)

    with pytest.raises(SourceContractError, match="member SHA-256 mismatch"):
        load_trials(
            archive_path,
            member=member,
            expected_archive_sha256=sha256_file(archive_path),
            expected_member_sha256="0" * 64,
            expected_rows=1,
        )

    with pytest.raises(SourceContractError, match="Arrow schema mismatch"):
        load_trials(
            archive_path,
            member=member,
            expected_archive_sha256=sha256_file(archive_path),
            expected_member_sha256=sha256(raw_member).hexdigest(),
            expected_rows=1,
        )


def test_cleaning_reproduces_frozen_counts_and_retains_qc_failures(
    canonical,
) -> None:
    assert len(canonical) == 7_707
    assert int(canonical["train_eligible"].sum()) == 7_702
    assert canonical.groupby("segment_id", sort=True).size().to_dict() == {
        36: 1_672,
        37: 1_597,
        38: 1_707,
        39: 1_659,
        40: 1_072,
    }
    eligible = canonical.loc[canonical["train_eligible"]]
    assert eligible.groupby("segment_id", sort=True).size().to_dict() == {
        36: 1_672,
        37: 1_597,
        38: 1_707,
        39: 1_659,
        40: 1_067,
    }
    excluded = canonical.loc[~canonical["train_eligible"]]
    assert excluded["segment_id"].tolist() == [40] * 5
    assert excluded["imu1_link_root_n"].tolist() == [3.0, 20.0, 23.0, 16.0, 22.0]
    assert excluded["train_exclusion_reasons"].unique().tolist() == [
        "imu1_link_root_samples_below_min"
    ]


def test_tick_delta_labels_and_trial_uid_are_deterministic(
    config: dict, source_frame, canonical
) -> None:
    expected = (
        canonical["encoder_tick_j0"]
        - float(config["cleaning"]["encoder_reference_ticks"][0])
    ) * (2.0 * np.pi / 4096.0)
    np.testing.assert_allclose(canonical["encoder_delta_rad_j0"], expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        canonical["vive_pos_raw_x_mm"], canonical["vive_pos_x_m"] * 1000.0
    )
    repeated = clean_trials(
        source_frame,
        config["cleaning"],
        session_id=config["contract"]["session_id"],
        source_member_sha256=config["source"]["member_sha256"],
    )
    assert canonical["trial_uid"].is_unique
    assert canonical["trial_uid"].tolist() == repeated["trial_uid"].tolist()


def test_quality_warnings_are_not_hard_exclusions(canonical) -> None:
    warned = canonical["qc_warning_flags"].ne("")
    assert int(warned.sum()) == 32
    assert int((warned & canonical["train_eligible"]).sum()) == 32
    assert int(canonical["strict_qc_eligible"].sum()) == 7_670


def test_expected_count_drift_fails_closed(config: dict, source_frame) -> None:
    changed = dict(config["cleaning"])
    changed["expected_canonical_rows"] = 7_706
    with pytest.raises(CleaningContractError, match="canonical row count changed"):
        clean_trials(
            source_frame,
            changed,
            session_id=config["contract"]["session_id"],
            source_member_sha256=config["source"]["member_sha256"],
        )
