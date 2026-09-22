from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

import imu_vive_learning.openaxes_external_native_v2_parent_receipt as parent_receipt_module

from imu_vive_learning.openaxes_external_native_v2_parent_receipt import (
    ACTIVE_SOURCE_AUDIT_RECEIPT_PATH,
    BUNDLE_KEYS,
    AUDITOR_MODULE_PATH,
    EXACT_TEST_ARGV,
    ERRATUM1_AUDITED_SOURCE_COMMIT,
    ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT,
    ERRATUM1_SOURCE_AUDIT_RECEIPT_PATH,
    PINNED_TEST_DISTRIBUTIONS,
    METHOD_ORDER,
    OUTAGE_VIEW_ORDER,
    PREPARATION_CONFIG_SHA256,
    PREPARATION_CONTRACT,
    PRE_ERRATUM_V1_MODEL_AUTHORITY_COMMIT,
    PRE_ERRATUM2_V1_MODEL_AUTHORITY_COMMIT,
    PROJECT_DIRECTORY,
    REQUIREMENTS_EVAL_PINS,
    RETIRED_AUDITED_SOURCE_COMMIT,
    RETIRED_SOURCE_AUDIT_EVIDENCE_COMMIT,
    RETIRED_SOURCE_AUDIT_RECEIPT_PATH,
    SEED_ORDER,
    SOURCE_CLOSURE,
    SOURCE_AUDIT_ERRATUM_KEYS,
    SOURCE_AUDIT_ERRATUM_ID,
    SOURCE_TEST_KEYS,
    TERMINAL_KEYS,
    SUCCESSFUL_V1_RUN_ID,
    V1_CONFIG_SHA256,
    V1_CONTRACT,
    V1_SOURCE_CLOSURE,
    V1_SOURCE_BLOB_OID_ORACLE,
    V1_SOURCE_REFERENCE_ORACLE,
    V1_SOURCE_SET_SHA256,
    V1_TERMINAL_STAGE,
    VIEW_ORDER,
    OpenAxesNativeV2ParentReceiptError,
    _absolute_root,
    _expected_tasks,
    _fixed_mean,
    _git_blob,
    _exact_test_environment,
    _read_nofollow,
    _reference,
    _repository,
    _write_exclusive,
    audit_python_source_safety,
    build_source_audit_receipt,
    canonical_json_bytes,
    canonical_sha256,
    issue_parent_receipts,
    strict_json_bytes,
    validate_comparator_bundle,
    validate_source_audit_receipt,
    validate_terminal_receipt,
)


MODULE_PATH = Path("src/imu_vive_learning/openaxes_external_native_v2_parent_receipt.py")
FIXED_TIME = "2026-08-24T00:00:00Z"
REAL_RUN_EXACT_SOURCE_TESTS = parent_receipt_module._run_exact_source_tests
REAL_VERIFY_V1_SOURCES = parent_receipt_module._verify_v1_sources_at_model_head


def _write_json(path: Path, value: object, *, canonical: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        canonical_json_bytes(value, final_newline=True)
        if canonical
        else (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )
    path.write_bytes(raw)
    return _reference(path.name, raw)


def _ref(root: Path, relative: str) -> dict:
    raw = (root / relative).read_bytes()
    return _reference(relative, raw)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
        },
    )
    return result.stdout.decode("ascii").strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Synthetic Auditor",
        "-c",
        "user.email=auditor@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(root, "rev-parse", "HEAD")


def _test_receipt(project: Path, commit: str, source_set: str, closure: dict) -> dict:
    launch = Path(sys.executable).absolute()
    executable = Path(os.path.realpath(launch))
    sys_prefix = executable.parents[1]
    site = launch.parents[1] / "lib/python3.12/site-packages"
    stdout = "............                                                             [100%]\n12 passed in 0.01s\n"
    stderr = ""
    report = canonical_json_bytes(
        {"exit_code": 0, "stderr_utf8": stderr, "stdout_utf8": stdout}
    )
    runner_files = {}
    for name in PINNED_TEST_DISTRIBUTIONS:
        path = site / name / "__init__.py"
        raw = path.read_bytes()
        runner_files[name] = {
            "path_absolute": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    pinned_distributions = []
    inventory = []
    for name, version in REQUIREMENTS_EVAL_PINS:
        normalized = parent_receipt_module._normalized_distribution_name(name)
        dist_info = (
            f"lib/python3.12/site-packages/{normalized.replace('-', '_')}-{version}.dist-info"
        )
        pinned_distributions.append(
            {
                "name": name,
                "normalized_name": normalized,
                "pinned_version": version,
                "installed_version": version,
                "dist_info_directory": dist_info,
                "metadata_reference": {
                    "path": f"{dist_info}/METADATA",
                    "bytes": 1,
                    "sha256": hashlib.sha256(b"m").hexdigest(),
                },
                "record_reference": {
                    "path": f"{dist_info}/RECORD",
                    "bytes": 1,
                    "sha256": hashlib.sha256(b"r").hexdigest(),
                },
                "installed_file_count": 1,
                "installed_total_bytes": 1,
                "installed_file_set_sha256": hashlib.sha256(
                    f"{normalized}:{version}".encode()
                ).hexdigest(),
            }
        )
        inventory.append(
            {"name": name, "normalized_name": normalized, "version": version}
        )
    inventory.sort(key=lambda item: item["normalized_name"])
    dependency_closure = {
        "file_count": len(pinned_distributions),
        "total_bytes": len(pinned_distributions),
        "manifest_sha256": hashlib.sha256(b"synthetic dependency closure").hexdigest(),
    }
    pyvenv_raw = (
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n"
        "executable = /usr/bin/python3.12\n"
        f"command = /usr/bin/python3.12 -m venv {project}/.venv312\n"
    ).encode()
    bootstrap_environment = _exact_test_environment(site)
    return {
        "python_launch_path_absolute": str(launch),
        "python_real_executable_absolute": str(executable),
        "python_real_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "python_version": "3.12.3",
        "sys_prefix_absolute": str(sys_prefix),
        "venv_site_packages_absolute": str(site),
        "working_directory_absolute": str(project),
        "argv_after_executable": list(EXACT_TEST_ARGV),
        "environment": bootstrap_environment,
        "source_audit_bootstrap": {
            "safe_path": True,
            "no_site": 1,
            "no_user_site": 1,
            "dont_write_bytecode": 1,
            "sys_executable_absolute": str(launch),
            "sys_real_executable_absolute": str(executable),
            "module_file_absolute": str(project / AUDITOR_MODULE_PATH),
            "sys_path_exact": [
                str(project / "src"),
                str(site),
                "/usr/lib/python312.zip",
                "/usr/lib/python3.12",
                "/usr/lib/python3.12/lib-dynload",
            ],
            "environment_exact": bootstrap_environment,
        },
        "pyvenv_cfg_reference": {
            "path": ".venv312/pyvenv.cfg",
            "bytes": len(pyvenv_raw),
            "sha256": hashlib.sha256(pyvenv_raw).hexdigest(),
        },
        "pyvenv_cfg_fields": {
            "home": "/usr/bin",
            "include-system-site-packages": "false",
            "version": "3.12.3",
            "executable": "/usr/bin/python3.12",
            "command": f"/usr/bin/python3.12 -m venv {project}/.venv312",
        },
        "requirements_eval_lock_reference": closure["requirements-eval.lock"],
        "dependency_resolution": "version_pinned_without_wheel_artifact_hashes;linux_installed_file_closure_verified;visible_extras_inventoried",
        "pinned_distributions": pinned_distributions,
        "installed_distribution_inventory": inventory,
        "dependency_file_closure_before_tests": dependency_closure,
        "dependency_file_closure_after_tests": deepcopy(dependency_closure),
        "dependency_file_closure_unchanged": True,
        "runner_module_files": runner_files,
        "source_import_boundary": {
            "direct_top_level_files": [],
            "top_level_directories": ["imu_vive_learning"],
            "imported_package_files": {
                path: closure[path]
                for path in (
                    "src/imu_vive_learning/__init__.py",
                    AUDITOR_MODULE_PATH,
                )
            },
        },
        "audited_git_commit": commit,
        "source_set_sha256": source_set,
        "exit_code": 0,
        "passed": 12,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "stdout_utf8": stdout,
        "stdout_bytes": len(stdout.encode()),
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_utf8": stderr,
        "stderr_bytes": 0,
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "report_bytes": len(report),
        "report_sha256": hashlib.sha256(report).hexdigest(),
    }


@pytest.fixture
def _synthetic_external_authorities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parent_receipt_module, "_now_utc", lambda: FIXED_TIME)
    monkeypatch.setattr(
        parent_receipt_module,
        "_run_exact_source_tests",
        lambda project, commit, source_set, closure: _test_receipt(
            project, commit, source_set, dict(closure)
        ),
    )
    monkeypatch.setattr(
        parent_receipt_module,
        "_verify_v1_sources_at_model_head",
        lambda _repo, _head: {
            path: "0" * 40 for path in V1_SOURCE_CLOSURE
        },
    )
    monkeypatch.setattr(
        parent_receipt_module,
        "_verify_live_source_audit_environment",
        lambda _project, _receipt: None,
    )


def _folds() -> list[dict]:
    scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    folds: list[dict] = []
    for scenario in scenarios:
        folds.append(
            {
                "fold_id": f"scenario_loso::{scenario}",
                "family": "scenario_loso",
                "role": "primary",
                "train_scenarios": [item for item in scenarios if item != scenario],
                "test_scenarios": [scenario],
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    for speed in (100, 150, 200, 250):
        test = [f"circle-{speed}", f"triangle-{speed}"]
        folds.append(
            {
                "fold_id": f"speed_pair_out::{speed}",
                "family": "speed_pair_out",
                "role": "stress",
                "train_scenarios": [item for item in scenarios if item not in test],
                "test_scenarios": test,
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    for shape in ("circle", "triangle"):
        test = [item for item in scenarios if item.startswith(f"{shape}-")]
        folds.append(
            {
                "fold_id": f"shape_out::{shape}",
                "family": "shape_out",
                "role": "stress",
                "train_scenarios": [item for item in scenarios if item not in test],
                "test_scenarios": test,
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    return folds


def _source_repo(tmp_path: Path) -> tuple[Path, Path, str, str, dict]:
    workspace = tmp_path / "workspace"
    project = workspace / PROJECT_DIRECTORY
    project.mkdir(parents=True)
    _git(workspace, "init", "-q")
    module_raw = MODULE_PATH.read_bytes()
    for relative in SOURCE_CLOSURE:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "src/imu_vive_learning/openaxes_external_native_v2_parent_receipt.py":
            target.write_bytes(module_raw)
        elif relative == "tests/test_openaxes_external_native_v2_parent_receipt.py":
            target.write_text("def test_synthetic_closure():\n    assert True\n", encoding="utf-8")
        elif relative.endswith("__init__.py"):
            target.write_text('"""synthetic package"""\n', encoding="utf-8")
        else:
            target.write_text(f"synthetic fixture for {relative}\n", encoding="utf-8")
    audited = _commit(workspace, "freeze v2 preregistration and auditor")
    source_audit = build_source_audit_receipt(workspace, audited)
    receipt_path = project / ACTIVE_SOURCE_AUDIT_RECEIPT_PATH
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(canonical_json_bytes(source_audit, final_newline=True))
    evidence = _commit(workspace, "commit source audit receipt")
    return workspace, project, audited, evidence, source_audit


def _context(fold_id: str, scenario: str, view_id: str, value: float) -> dict:
    identity = hashlib.sha256(f"{fold_id}|{scenario}|{view_id}".encode()).hexdigest()
    return {
        "scenario_id": scenario,
        "view_id": view_id,
        "rows": 7,
        "rmse3d_m": value,
        "p95_distance_m": value,
        "identity_sha256": identity,
    }


def _finished_v1_fixture(
    workspace: Path,
    project: Path,
    evidence_commit: str,
) -> tuple[Path, Path, Path, str]:
    marker = project / "synthetic_v1_lineage.txt"
    marker.write_text("model contract head\n", encoding="utf-8")
    model_head = _commit(workspace, "synthetic v1 immutable model head")

    preparation = project / (
        "artifacts/openaxes_external_v1/prepared_openaxes-v1-sealed-003"
    )
    v1 = project / (
        "artifacts/openaxes_external_model_v1/run_openaxes-v1-sealed-003"
    )
    v2 = project / "synthetic-v2"
    preparation.mkdir(parents=True)
    v1.mkdir(parents=True)
    v2.mkdir()
    (v1 / "quarantine").mkdir()
    table_raw = b"PAR1opaque synthetic table bytesPAR1"
    (preparation / "prepared.parquet").write_bytes(table_raw)
    table_ref = _reference("prepared.parquet", table_raw)
    preparation_manifest = {
        "contract": PREPARATION_CONTRACT,
        "config_sha256": PREPARATION_CONFIG_SHA256,
        "folds": {"items": _folds()},
        "files": {"prepared.parquet": {**table_ref, "rows": 56}},
    }
    _write_json(preparation / "manifest.json", preparation_manifest)
    preparation_manifest_ref = _ref(preparation, "manifest.json")
    (preparation / "source.zip").write_bytes(b"opaque preparation source")
    preparation_config = project / "configs/openaxes_external_v1.yaml"
    preparation_config.parent.mkdir(parents=True, exist_ok=True)
    preparation_config.write_text(
        f"config_sha256: {PREPARATION_CONFIG_SHA256}\n", encoding="utf-8"
    )
    _write_json(preparation / "COMPLETE.json", {"immutable": True})

    def external_preparation_ref(filename: str) -> dict:
        raw = (preparation / filename).read_bytes()
        return {
            "absolute_path": str(preparation / filename),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    tasks = _expected_tasks(preparation_manifest)
    task_index = {
        "contract": V1_CONTRACT,
        "task_count": 336,
        "primary_task_count": 192,
        "stress_task_count": 144,
        "tasks": tasks,
    }
    _write_json(v1 / "tasks/index.json", task_index)

    gpu_inventory = [
        {
            "physical_index": physical_index,
            "uuid": f"GPU-SYNTHETIC-{physical_index}",
            "pci_bus_id": f"00000000:0{physical_index + 1}:00.0",
            "name": "Synthetic GPU",
        }
        for physical_index in range(4)
    ]

    source_files = deepcopy(V1_SOURCE_REFERENCE_ORACLE)
    assert canonical_sha256(source_files) == V1_SOURCE_SET_SHA256
    prepared = {
        "contract": V1_CONTRACT,
        "config_sha256": V1_CONFIG_SHA256,
        "stage": "prepared_before_external_model_results",
        "synthetic_test_only": False,
        "preparation": {
            "artifacts": {
                "synthetic_test_only": False,
                "source": external_preparation_ref("source.zip"),
                "config": {
                    "absolute_path": str(preparation_config),
                    "bytes": preparation_config.stat().st_size,
                    "sha256": hashlib.sha256(preparation_config.read_bytes()).hexdigest(),
                },
                "manifest": {
                    "absolute_path": str(preparation / "manifest.json"),
                    "bytes": preparation_manifest_ref["bytes"],
                    "sha256": preparation_manifest_ref["sha256"],
                },
                "parquet": {
                    "absolute_path": str(preparation / "prepared.parquet"),
                    "bytes": table_ref["bytes"],
                    "sha256": table_ref["sha256"],
                },
                "complete": external_preparation_ref("COMPLETE.json"),
            },
            "parent_git_authority": {"model_contract_git_head": model_head},
            "validation": {"production_ready": True, "synthetic_test_only": False},
        },
        "task_grid": {"task_index": _ref(v1, "tasks/index.json")},
        "gpu_inventory": gpu_inventory,
        "execution": {
            "prelaunch_session_recovery": {
                "required_launch_mode": parent_receipt_module.PRELAUNCH_RECOVERY_MODE,
                "authorized_action_scope": parent_receipt_module.PRELAUNCH_AUTHORIZED_ACTION_SCOPE,
            }
        },
        "source_closure": {"files": source_files, "set_sha256": V1_SOURCE_SET_SHA256},
        "repository": {
            "model_contract_git_head": model_head,
            "model_contract_sources_committed_at_git_head": True,
            "downstream_results_blocked_by_uncommitted_contract": False,
        },
        "runtime": {},
        "results_started": False,
        "challenge_route_present": False,
    }
    _write_json(v1 / "prepared_manifest.json", prepared)
    prepared_marker = {
        "contract": V1_CONTRACT,
        "config_sha256": V1_CONFIG_SHA256,
        "prepared_manifest": _ref(v1, "prepared_manifest.json"),
        "task_index": _ref(v1, "tasks/index.json"),
        "prepared_runtime_sha256": canonical_sha256(prepared["runtime"]),
        "synthetic_test_only": False,
        "model_contract_sources_committed_at_git_head": True,
        "downstream_results_blocked_by_uncommitted_contract": False,
        "immutable": True,
    }
    _write_json(v1 / "PREPARED.json", prepared_marker)

    pre_fit_validation = {"validated": True}
    _write_json(v1 / "runtime/pre_fit_validation.json", pre_fit_validation)
    gpu_by_slot = [gpu_inventory[slot // 2] for slot in range(8)]
    slots = []
    for slot in range(8):
        slot_payload = {
            "slot_index": slot,
            "local_slot": slot % 2,
            "gpu": gpu_by_slot[slot],
            "task_ids": [task["task_id"] for task in tasks[slot::8]],
        }
        slots.append(
            {**slot_payload, "assignment_sha256": canonical_sha256(slot_payload)}
        )
    slot_assignment_sha256 = [slot["assignment_sha256"] for slot in slots]
    task_assignment_sha256 = canonical_sha256(slots)
    controller_source_sha256 = V1_SOURCE_REFERENCE_ORACLE[
        "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
    ]["sha256"]
    prepared_ref = _ref(v1, "prepared_manifest.json")
    task_index_ref = _ref(v1, "tasks/index.json")
    parent_validation_sha256 = canonical_sha256(
        prepared["preparation"]["validation"]
    )
    pre_fit_sha256 = canonical_sha256(pre_fit_validation)
    execution_session_id = canonical_sha256(
        {
            "contract": V1_CONTRACT,
            "config_sha256": V1_CONFIG_SHA256,
            "prepared_manifest": prepared_ref,
            "model_contract_git_head": model_head,
            "parent_git_authority_sha256": canonical_sha256(
                prepared["preparation"]["parent_git_authority"]
            ),
            "parent_public_validation_sha256": parent_validation_sha256,
            "model_source_set_sha256": V1_SOURCE_SET_SHA256,
            "prepared_runtime_sha256": canonical_sha256(prepared["runtime"]),
            "pre_fit_validation_receipt_sha256": pre_fit_sha256,
            "task_index_sha256": task_index_ref["sha256"],
            "gpu_inventory_sha256": canonical_sha256(gpu_inventory),
            "workers_per_gpu": 2,
            "task_assignment_sha256": task_assignment_sha256,
        }
    )
    execution_session = {
        "contract": V1_CONTRACT,
        "execution_session_id": execution_session_id,
        "prepared_manifest": prepared_ref,
        "model_contract_git_head": model_head,
        "controller_source_sha256": controller_source_sha256,
        "model_source_set_sha256": V1_SOURCE_SET_SHA256,
        "prepared_runtime_sha256": canonical_sha256(prepared["runtime"]),
        "parent_public_validation_sha256": parent_validation_sha256,
        "pre_fit_validation": _ref(v1, "runtime/pre_fit_validation.json"),
        "pre_fit_validation_receipt_sha256": pre_fit_sha256,
        "workers_per_gpu": 2,
        "gpu_inventory": gpu_inventory,
        "task_index_sha256": task_index_ref["sha256"],
        "task_assignment_sha256": task_assignment_sha256,
        "slots": slots,
        "immutable": True,
    }
    execution_session["session_payload_sha256"] = canonical_sha256(execution_session)
    _write_json(v1 / "runtime/execution_session.json", execution_session)

    recovery = {
        "contract": V1_CONTRACT,
        "receipt_kind": "authenticated_session_only_prelaunch_recovery",
        "config_sha256": V1_CONFIG_SHA256,
        "prelaunch_recovery_contract_sha256": canonical_sha256(
            prepared["execution"]["prelaunch_session_recovery"]
        ),
        "execution_session_id": execution_session_id,
        "execution_session": _ref(v1, "runtime/execution_session.json"),
        "session_payload_sha256": execution_session["session_payload_sha256"],
        "pre_fit_validation": _ref(v1, "runtime/pre_fit_validation.json"),
        "pre_fit_validation_receipt_sha256": pre_fit_sha256,
        "task_assignment_sha256": task_assignment_sha256,
        "empty_evidence_at_exclusive_creation": {
            "execution_launch_directories": 0,
            "worker_ready_directories": 0,
            "attempt_rosters": 0,
            "attempt_ledgers": 0,
            "task_phase_directories": 0,
            "result_directories": 0,
            "quarantine_events": 0,
        },
        "local_scientific_attempt_evidence_observed_at_exclusive_creation": False,
        "authorized_action_scope": parent_receipt_module.PRELAUNCH_AUTHORIZED_ACTION_SCOPE,
        "forbidden_action_scopes": list(
            parent_receipt_module.PRELAUNCH_FORBIDDEN_ACTION_SCOPES
        ),
        "recovery_receipt_is_attempt_authority": False,
        "authorizes_scientific_refit_under_declared_non_total_erasure_threat_model": False,
        "no_refit_guarantee_threat_model": parent_receipt_module.NO_REFIT_GUARANTEE_THREAT_MODEL,
        "external_append_only_or_signature_authority_present": False,
        "total_erasure_prior_execution_provable": False,
        "total_erasure_limitation": parent_receipt_module.TOTAL_ERASURE_LIMITATION,
    }
    recovery["receipt_payload_sha256"] = canonical_sha256(recovery)
    _write_json(v1 / "runtime/prelaunch_session_recovery.json", recovery)
    recovery_ref = _ref(v1, "runtime/prelaunch_session_recovery.json")
    recovery_sha256 = canonical_sha256(recovery)

    def create_launch(
        ordinal: int,
        previous_launch: dict | None,
        *,
        ready_slots: tuple[int, ...] = tuple(range(8)),
        with_controller: bool = True,
    ) -> dict:
        mode = (
            parent_receipt_module.PRELAUNCH_RECOVERY_MODE
            if ordinal == 0
            else "resume_after_prior_launch"
        )
        launch_id = canonical_sha256(
            {
                "contract": V1_CONTRACT,
                "execution_session_id": execution_session_id,
                "launch_ordinal": ordinal,
                "previous_launch": previous_launch,
                "launch_mode": mode,
            }
        )
        launch_path = f"runtime/execution_launches/launch_{launch_id}/launch.started.json"
        launch = {
            "contract": V1_CONTRACT,
            "execution_session_id": execution_session_id,
            "launch_id": launch_id,
            "launch_ordinal": ordinal,
            "launch_mode": mode,
            "previous_launch": previous_launch,
            "prelaunch_session_recovery": recovery_ref,
            "prelaunch_session_recovery_sha256": recovery_sha256,
            "resume": True,
            "session": _ref(v1, "runtime/execution_session.json"),
            "workers_per_gpu": 2,
            "slot_count": 8,
            "task_assignment_sha256": task_assignment_sha256,
            "controller_source_sha256": controller_source_sha256,
            "model_source_set_sha256": V1_SOURCE_SET_SHA256,
            "parent_public_validation_sha256": canonical_sha256(
                prepared["preparation"]["validation"]
            ),
            "pre_fit_validation_receipt_sha256": canonical_sha256(
                {"validated": True}
            ),
        }
        launch["launch_payload_sha256"] = canonical_sha256(launch)
        _write_json(v1 / launch_path, launch)
        launch_ref = _ref(v1, launch_path)
        ready_refs: list[dict] = []
        ready_receipts: list[dict] = []
        for slot in ready_slots:
            ready_path = (
                f"runtime/worker_ready/launch_{launch_id}/slot_{slot:04d}.json"
            )
            ready = {
                "contract": V1_CONTRACT,
                "execution_session_id": execution_session_id,
                "launch_id": launch_id,
                "launch_started": launch_ref,
                "launch_payload_sha256": launch["launch_payload_sha256"],
                "slot_index": slot,
                "local_slot": slot % 2,
                "gpu": gpu_by_slot[slot],
                "visible_child_device": "cuda:0",
                "cuda_visible_devices": gpu_by_slot[slot]["uuid"],
                "controller_source_sha256": controller_source_sha256,
                "task_assignment_sha256": task_assignment_sha256,
                "slot_assignment_sha256": slot_assignment_sha256[slot],
                "task_ids_sha256": canonical_sha256(slots[slot]["task_ids"]),
                "child_process_id": 1000 + ordinal * 100 + slot,
                "ready_after_exact_gpu_validation": True,
            }
            ready["ready_receipt_sha256"] = canonical_sha256(ready)
            _write_json(v1 / ready_path, ready)
            ready_refs.append(_ref(v1, ready_path))
            ready_receipts.append(ready)
        observations = []
        controller_ref = None
        controller = None
        controller_path = (
            f"runtime/execution_launches/launch_{launch_id}/"
            "controller_ready_observations.json"
        )
        if with_controller:
            assert ready_slots == tuple(range(8))
            observations = [
                {
                    "slot_index": slot,
                    "child_process_id": ready_receipts[slot]["child_process_id"],
                    "child_process_name": f"openaxes-model-slot-{slot}",
                    "gpu": gpu_by_slot[slot],
                    "slot_assignment_sha256": slot_assignment_sha256[slot],
                    "ready_receipt": ready_refs[slot],
                    "ready_receipt_sha256": ready_receipts[slot][
                        "ready_receipt_sha256"
                    ],
                }
                for slot in range(8)
            ]
            controller = {
                "contract": V1_CONTRACT,
                "execution_session_id": execution_session_id,
                "launch_id": launch_id,
                "launch_started": launch_ref,
                "launch_payload_sha256": launch["launch_payload_sha256"],
                "controller_source_sha256": controller_source_sha256,
                "task_assignment_sha256": task_assignment_sha256,
                "slot_count": 8,
                "observations": observations,
                "controller_observed_spawn_process_identity": True,
            }
            controller["controller_observations_sha256"] = canonical_sha256(controller)
            _write_json(v1 / controller_path, controller)
            controller_ref = _ref(v1, controller_path)
        return {
            "launch": launch,
            "launch_ref": launch_ref,
            "controller_ref": controller_ref,
            "controller": controller,
            "ready_refs": ready_refs,
        }

    partial_launch = create_launch(
        0, None, ready_slots=(0, 3), with_controller=False
    )
    launch_zero = create_launch(1, partial_launch["launch_ref"])
    launch_one = create_launch(2, launch_zero["launch_ref"])
    latest = launch_one
    latest_launch = latest["launch"]
    launch_completed = {
        "contract": V1_CONTRACT,
        "execution_session_id": execution_session_id,
        "launch_id": latest_launch["launch_id"],
        "launch_started": latest["launch_ref"],
        "launch_payload_sha256": latest_launch["launch_payload_sha256"],
        "launch_ordinal": latest_launch["launch_ordinal"],
        "resume": True,
        "slot_count": 8,
        "task_count": 336,
        "task_assignment_sha256": task_assignment_sha256,
        "controller_source_sha256": controller_source_sha256,
        "model_source_set_sha256": V1_SOURCE_SET_SHA256,
        "parent_public_validation_sha256": latest_launch[
            "parent_public_validation_sha256"
        ],
        "pre_fit_validation_receipt_sha256": latest_launch[
            "pre_fit_validation_receipt_sha256"
        ],
        "controller_ready_observations": latest["controller_ref"],
        "controller_observations_sha256": latest["controller"][
            "controller_observations_sha256"
        ],
        "ready_receipts": latest["ready_refs"],
        "all_tasks_authenticated_complete": True,
    }
    launch_completed["launch_completed_sha256"] = canonical_sha256(launch_completed)
    _write_json(
        v1
        / (
            f"runtime/execution_launches/launch_{latest_launch['launch_id']}/"
            "launch.completed.json"
        ),
        launch_completed,
    )

    def execution_binding(launch_authority: dict, slot: int) -> dict:
        launch = launch_authority["launch"]
        return {
            "execution_session_id": execution_session_id,
            "launch_id": launch["launch_id"],
            "launch_started": launch_authority["launch_ref"],
            "launch_payload_sha256": launch["launch_payload_sha256"],
            "transport_slot_index": slot,
            "local_slot": slot % 2,
            "gpu": gpu_by_slot[slot],
            "controller_source_sha256": controller_source_sha256,
            "task_assignment_sha256": task_assignment_sha256,
            "slot_assignment_sha256": slot_assignment_sha256[slot],
            "worker_ready": launch_authority["ready_refs"][slot],
            "controller_ready_observations": launch_authority["controller_ref"],
            "controller_observations_sha256": launch_authority["controller"][
                "controller_observations_sha256"
            ],
            "worker_identity_source": "fixed_process_transport_and_ready_receipt",
            "scientific_fit_attempt": 1,
        }

    result_index: dict[str, dict] = {}
    primary_tasks = [
        task
        for task in tasks
        if task["method_id"] == "masked_erm" and task["family"] == "scenario_loso"
    ]
    primary_fold_ids = [
        fold["fold_id"] for fold in _folds() if fold["family"] == "scenario_loso"
    ]
    for task in tasks:
        task_id = task["task_id"]
        directory = v1 / "results" / task_id
        directory.mkdir(parents=True)
        (directory / "checkpoint.pt").write_bytes(b"opaque checkpoint")
        _write_json(directory / "transform.json", {"sealed": True})
        _write_json(directory / "history.json", {"updates": 400})
        (directory / "outer_predictions.parquet").write_bytes(b"PAR1opaque predictionsPAR1")
        (directory / "outer_evaluation_rows.parquet").write_bytes(b"PAR1opaque rowsPAR1")
        training_plain = {
            "method_id": task["method_id"],
            "seed": task["seed"],
            "updates": 400,
            "checkpoint_update": 200,
            "checkpoint_score_m": 0.02,
            "checkpoint_selection_scope": "training_scenarios_only",
            "outer_eval_count": 0,
            "initial_state_sha256": hashlib.sha256(f"seed:{task['seed']}".encode()).hexdigest(),
            "checkpoint_state_sha256": "c" * 64,
            "view_sha256": "d" * 64,
        }
        _write_json(directory / "training_receipt.json", training_plain)
        training = {
            **training_plain,
            "artifacts": {
                key: _ref(v1, f"results/{task_id}/{filename}")
                for key, filename in {
                    "checkpoint": "checkpoint.pt",
                    "transform": "transform.json",
                    "history": "history.json",
                    "training_receipt": "training_receipt.json",
                }.items()
            },
        }
        fold_index = (
            primary_fold_ids.index(task["fold_id"])
            if task["fold_id"] in primary_fold_ids
            else 0
        )
        views = []
        for view_index, view_id in enumerate(VIEW_ORDER):
            value = float(
                0.01
                + fold_index * 0.001
                + view_index * 0.0001
                + SEED_ORDER.index(task["seed"]) * 0.00001
            )
            views.append(
                {
                    "view_id": view_id,
                    "rows": 7,
                    "rmse3d_m": value,
                    "p95_distance_m": value,
                    "mean_distance_m": value,
                }
            )
        outer = {
            "task_id": task_id,
            "method_id": task["method_id"],
            "fold_id": task["fold_id"],
            "family": task["family"],
            "role": task["role"],
            "seed": task["seed"],
            "checkpoint_selection_scope": "training_scenarios_only",
            "outer_eval_count": 1,
            "metrics": {"outer_eval_count": 1, "views": views, "finite": True},
        }
        _write_json(directory / "outer_evaluation.json", outer)
        prediction_ref = _ref(v1, f"results/{task_id}/outer_predictions.parquet")
        evaluation = {
            "outer_eval_count": 1,
            "checkpoint_selection_scope": "training_scenarios_only",
            "evaluation_rows": _ref(v1, f"results/{task_id}/outer_evaluation_rows.parquet"),
            "evaluation_metrics": _ref(v1, f"results/{task_id}/outer_evaluation.json"),
        }
        slot_index = task["ordinal"] % 8
        binding = execution_binding(launch_zero, slot_index)
        resumed_binding = execution_binding(launch_one, slot_index)
        attempt_id = canonical_sha256(
            {
                "contract": V1_CONTRACT,
                "task_id": task_id,
                "task_spec_sha256": task["spec_sha256"],
                "first_execution_binding": binding,
                "scientific_fit_attempt": 1,
            }
        )
        roster = {
            "contract": V1_CONTRACT,
            "task_id": task_id,
            "task_spec_sha256": task["spec_sha256"],
            "attempt_id": attempt_id,
            "origin": "production_execution",
            "binding": binding,
            "attempt_started_or_state_observed": True,
            "monotonic_first_attempt_roster": True,
            "survives_ledger_deletion_and_quarantine": True,
        }
        roster["attempt_roster_sha256"] = canonical_sha256(roster)
        _write_json(v1 / f"runtime/attempt_roster/{task_id}.json", roster)
        ledger = {
            "contract": V1_CONTRACT,
            "task_id": task_id,
            "task_spec_sha256": task["spec_sha256"],
            "attempt_id": attempt_id,
            "origin": "production_execution",
            "binding": binding,
            "attempt_started_or_state_observed": True,
            "survives_quarantine": True,
        }
        ledger["attempt_ledger_sha256"] = canonical_sha256(ledger)
        _write_json(v1 / f"runtime/attempt_ledger/{task_id}.json", ledger)
        phase_payloads = [
            binding,
            training,
            {
                "rows": 35,
                "base_rows": 7,
                "views": 5,
                "finite": True,
                "prediction": prediction_ref,
            },
            {
                "prediction": prediction_ref,
                "truth_read_count_before": 0,
                "evaluator_call_count_before": 0,
            },
            evaluation,
        ]
        phases: list[dict] = []
        phase_refs: list[dict] = []
        phase_names = parent_receipt_module.V1_PHASE_ORDER
        for index, payload in enumerate(phase_payloads):
            phase_binding = binding if index < 2 else resumed_binding
            phase = {
                "contract": V1_CONTRACT,
                "task_id": task_id,
                "task_spec_sha256": task["spec_sha256"],
                "phase": phase_names[index],
                "attempt_id": attempt_id,
                "previous_phase": None if index == 0 else phase_refs[-1],
                "execution_binding": phase_binding,
                "payload": payload,
            }
            phase_path = f"runtime/ledgers/{task_id}/{phase_names[index]}.json"
            _write_json(v1 / phase_path, phase)
            phases.append(phase)
            phase_refs.append(_ref(v1, phase_path))
        bundle = {
            "contract": V1_CONTRACT,
            "task": task,
            "training": training,
            "evaluation": evaluation,
            "phase_lineage_sha256": canonical_sha256(phases[:5]),
        }
        _write_json(directory / "bundle.json", bundle)
        bundle_ref = _ref(v1, f"results/{task_id}/bundle.json")
        bundle_phase = {
            "contract": V1_CONTRACT,
            "task_id": task_id,
            "task_spec_sha256": task["spec_sha256"],
            "phase": phase_names[5],
            "attempt_id": attempt_id,
            "previous_phase": phase_refs[-1],
            "execution_binding": resumed_binding,
            "payload": {"bundle": bundle_ref},
        }
        bundle_phase_path = f"runtime/ledgers/{task_id}/{phase_names[5]}.json"
        _write_json(v1 / bundle_phase_path, bundle_phase)
        phase_refs.append(_ref(v1, bundle_phase_path))
        result = {
            "contract": V1_CONTRACT,
            "task": task,
            "bundle": bundle_ref,
            "checkpoint_selection_scope": "training_scenarios_only",
            "outer_eval_count": 1,
            "scientific_fit_count": 1,
            "synthetic_test_only": False,
        }
        _write_json(directory / "result.json", result)
        result_ref = _ref(v1, f"results/{task_id}/result.json")
        result_phase = {
            "contract": V1_CONTRACT,
            "task_id": task_id,
            "task_spec_sha256": task["spec_sha256"],
            "phase": phase_names[6],
            "attempt_id": attempt_id,
            "previous_phase": phase_refs[-1],
            "execution_binding": resumed_binding,
            "payload": {"result": result_ref},
        }
        _write_json(v1 / f"runtime/ledgers/{task_id}/{phase_names[6]}.json", result_phase)
        result_index[task_id] = result_ref

    fold_records: list[dict] = []
    primary_folds = [fold for fold in _folds() if fold["family"] == "scenario_loso"]
    for method_id in METHOD_ORDER:
        for fold_index, fold in enumerate(_folds()):
            if method_id == "masked_erm" and fold["family"] == "scenario_loso":
                scenario = fold["test_scenarios"][0]
                outage_contexts = []
                for view_index, view_id in enumerate(OUTAGE_VIEW_ORDER, start=1):
                    seed_values = [
                        0.01 + fold_index * 0.001 + view_index * 0.0001 + seed_index * 0.00001
                        for seed_index in range(3)
                    ]
                    # Synthetic ensemble metrics are independent persisted inputs;
                    # their values need not be reconstructed from predictions here.
                    outage_contexts.append(
                        _context(fold["fold_id"], scenario, view_id, float(_fixed_mean(seed_values, name="fixture mean")))
                    )
                full_values = [0.01 + fold_index * 0.001 + seed_index * 0.00001 for seed_index in range(3)]
                full_context = _context(
                    fold["fold_id"], scenario, "full", float(_fixed_mean(full_values, name="fixture full mean"))
                )
                def aggregate(contexts: list[dict]) -> dict:
                    values = [item["rmse3d_m"] for item in contexts]
                    mean = float(_fixed_mean(values, name="fixture aggregate"))
                    return {
                        "context_count": len(contexts),
                        "A_mean_context_rmse3d_m": mean,
                        "WS_worst_scenario_mean_rmse3d_m": mean,
                        "WM_worst_mask_mean_rmse3d_m": max(values),
                        "robust_J_m": mean,
                        "contexts": contexts,
                    }
                fold_records.append(
                    {
                        "method_id": "masked_erm",
                        "fold_id": fold["fold_id"],
                        "family": "scenario_loso",
                        "role": "primary",
                        "test_scenarios": [scenario],
                        "seed_ensemble_size": 3,
                        "outage": aggregate(outage_contexts),
                        "full_diagnostic": aggregate([full_context]),
                    }
                )
            else:
                fold_records.append({"method_id": method_id, "fold_id": fold["fold_id"]})
    zero_shift = {
        "selection": {"selected_method_id": "masked_erm"},
        "method_scores": [],
        "directional_gate": {"passes": False},
        "fold_records": fold_records,
        "stress_report_only": [],
        "counts": {},
    }
    development = {
        "contract": V1_CONTRACT,
        "config_sha256": V1_CONFIG_SHA256,
        "stage": V1_TERMINAL_STAGE,
        "synthetic_test_only": False,
        "prepared_manifest": _ref(v1, "prepared_manifest.json"),
        "scientific_fit_count": 336,
        "outer_evaluation_count": 336,
        "result_index": result_index,
        "paired_controls": {},
        "zero_shift": zero_shift,
        "alignment_sensitivity_report_only": {"report_only": True},
        "claim_boundary": {"challenge_access": False},
    }
    _write_json(v1 / "development_manifest.json", development)
    support = [
        {
            "scenario_id": scenario,
            "shift_ms": shift_ms,
            "base_uid_rows": 7,
            "base_uid_sha256": hashlib.sha256(
                f"{scenario}|{shift_ms}|base-uids".encode()
            ).hexdigest(),
        }
        for scenario in sorted(
            f"{shape}-{speed}"
            for shape in ("circle", "triangle")
            for speed in (100, 150, 200, 250)
        )
        for shift_ms in (-50, -20, -10, 10, 20, 50)
    ]
    full_replay = {
        "contract": V1_CONTRACT,
        "receipt_kind": "immutable_full_signed_source_semantic_replay",
        "config_sha256": V1_CONFIG_SHA256,
        "prepared_manifest": _ref(v1, "prepared_manifest.json"),
        "development_manifest": _ref(v1, "development_manifest.json"),
        "preparation_artifacts": {
            key: prepared["preparation"]["artifacts"][key]
            for key in ("source", "config", "manifest", "parquet", "complete")
        },
        "parent_public_validation_sha256": canonical_sha256(
            prepared["preparation"]["validation"]
        ),
        "parent_git_authority_sha256": canonical_sha256(
            prepared["preparation"]["parent_git_authority"]
        ),
        "reconstruction_receipt_sha256": "6" * 64,
        "scenario_shift_support_identity": support,
        "scenario_shift_support_identity_sha256": canonical_sha256(support),
        "signed_report_sha256": canonical_sha256(
            development["alignment_sensitivity_report_only"]
        ),
        "validator_source_sha256": V1_SOURCE_REFERENCE_ORACLE[
            "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
        ]["sha256"],
        "validator_source_set_sha256": V1_SOURCE_SET_SHA256,
        "full_signed_source_replay": True,
        "semantic_replay": True,
    }
    full_replay["receipt_payload_sha256"] = canonical_sha256(full_replay)
    _write_json(v1 / "runtime/full_signed_source_replay.json", full_replay)
    complete = {
        "contract": V1_CONTRACT,
        "config_sha256": V1_CONFIG_SHA256,
        "prepared_manifest": _ref(v1, "prepared_manifest.json"),
        "development_manifest": _ref(v1, "development_manifest.json"),
        "scientific_fits": 336,
        "outer_evaluations": 336,
        "selected_method_id": "masked_erm",
        "selection_sha256": canonical_sha256(zero_shift["selection"]),
        "zero_shift_sha256": canonical_sha256(zero_shift),
        "directional_gate_passes": False,
        "directional_gate_sha256": canonical_sha256(zero_shift["directional_gate"]),
        "alignment_sensitivity_sha256": canonical_sha256(
            development["alignment_sensitivity_report_only"]
        ),
        "full_signed_source_replay": _ref(v1, "runtime/full_signed_source_replay.json"),
        "full_signed_source_replay_sha256": canonical_sha256(full_replay),
        "full_signed_source_replay_performed": True,
        "production_validation_requires_signed_source_replay": True,
        "stress_or_sensitivity_used_for_selection": False,
        "challenge_access_performed": False,
        "immutable": True,
    }
    _write_json(v1 / "COMPLETE.json", complete)
    assert evidence_commit != model_head
    return v1, preparation, v2, model_head


def _full_fixture(tmp_path: Path) -> dict:
    workspace, project, audited, evidence, source_audit = _source_repo(tmp_path)
    v1, preparation, v2, model_head = _finished_v1_fixture(workspace, project, evidence)
    return {
        "workspace": workspace,
        "project": project,
        "audited": audited,
        "evidence": evidence,
        "source_audit": source_audit,
        "v1": v1,
        "preparation": preparation,
        "v2": v2,
        "model_head": model_head,
    }


def _reseal_first_task_to_wrong_authenticated_slot(fixture: dict) -> str:
    v1 = fixture["v1"]
    task_index = json.loads((v1 / "tasks/index.json").read_text(encoding="utf-8"))
    task = task_index["tasks"][0]
    task_id = task["task_id"]
    session = json.loads(
        (v1 / "runtime/execution_session.json").read_text(encoding="utf-8")
    )
    wrong_slot = session["slots"][1]

    def wrong_slot_binding(original: dict) -> dict:
        attacked = deepcopy(original)
        launch_id = attacked["launch_id"]
        attacked.update(
            {
                "transport_slot_index": 1,
                "local_slot": wrong_slot["local_slot"],
                "gpu": wrong_slot["gpu"],
                "slot_assignment_sha256": wrong_slot["assignment_sha256"],
                "worker_ready": _ref(
                    v1,
                    f"runtime/worker_ready/launch_{launch_id}/slot_0001.json",
                ),
            }
        )
        return attacked

    roster_path = v1 / f"runtime/attempt_roster/{task_id}.json"
    ledger_path = v1 / f"runtime/attempt_ledger/{task_id}.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    first_binding = wrong_slot_binding(roster["binding"])
    attempt_id = canonical_sha256(
        {
            "contract": V1_CONTRACT,
            "task_id": task_id,
            "task_spec_sha256": task["spec_sha256"],
            "first_execution_binding": first_binding,
            "scientific_fit_attempt": 1,
        }
    )
    roster["binding"] = first_binding
    roster["attempt_id"] = attempt_id
    roster["attempt_roster_sha256"] = canonical_sha256(
        {key: value for key, value in roster.items() if key != "attempt_roster_sha256"}
    )
    _write_json(roster_path, roster)
    ledger["binding"] = deepcopy(first_binding)
    ledger["attempt_id"] = attempt_id
    ledger["attempt_ledger_sha256"] = canonical_sha256(
        {key: value for key, value in ledger.items() if key != "attempt_ledger_sha256"}
    )
    _write_json(ledger_path, ledger)

    previous: dict | None = None
    for index, phase_name in enumerate(parent_receipt_module.V1_PHASE_ORDER):
        phase_path = v1 / f"runtime/ledgers/{task_id}/{phase_name}.json"
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        phase["attempt_id"] = attempt_id
        phase["execution_binding"] = wrong_slot_binding(phase["execution_binding"])
        phase["previous_phase"] = previous
        if index == 0:
            phase["payload"] = deepcopy(first_binding)
        _write_json(phase_path, phase)
        previous = _ref(v1, f"runtime/ledgers/{task_id}/{phase_name}.json")
    return task_id


def test_source_audit_closes_exact_eight_git_blobs_and_digest(
    tmp_path: Path, _synthetic_external_authorities: None
) -> None:
    workspace, project, audited, _evidence, source_audit = _source_repo(tmp_path)
    validated = validate_source_audit_receipt(source_audit)
    assert tuple(validated["source_closure"]) == SOURCE_CLOSURE
    assert validated["source_set_sha256"] == canonical_sha256(validated["source_closure"])
    assert validated["erratum"]["erratum_id"] == SOURCE_AUDIT_ERRATUM_ID
    assert validated["erratum"]["active_receipt_path"] == ACTIVE_SOURCE_AUDIT_RECEIPT_PATH
    assert [
        item["receipt_path"]
        for item in validated["erratum"]["historical_source_audit_authorities"]
    ] == [RETIRED_SOURCE_AUDIT_RECEIPT_PATH, ERRATUM1_SOURCE_AUDIT_RECEIPT_PATH]
    assert validated["erratum"]["terminal_bound_successful_run_id"] == SUCCESSFUL_V1_RUN_ID
    assert validated["erratum"]["sealed_002_stage_counts"] == {
        "attempt_ledger": 0,
        "attempt_roster": 0,
        "evaluation": 0,
        "execution_session": 1,
        "launch_started": 1,
        "model_prepared": 1,
        "prediction": 0,
        "public_preparation": 1,
        "quarantine_member": 0,
        "result": 0,
        "scientific_fit": 0,
        "task_ledger": 0,
        "task_result_directory": 0,
        "worker_ready": 0,
    }
    sealed_002 = validated["erratum"]["sealed_002_evidence"]
    assert sealed_002["audited_git_head"] == ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT
    assert sealed_002["failure_status"] == "cleanup_blocked_nonterminal"
    assert sealed_002["public_preparation_valid_despite_supervisor_progress_flag"] is True
    assert sealed_002["failed_terminal_written"] is False
    assert sealed_002["marked_processes_after_verified_natural_exit"] == 0
    assert sealed_002["manual_signal_performed"] is False
    assert sealed_002["resume_scientifically_forbidden"] is True
    assert sealed_002["fresh_run_id_required"] == "openaxes-v1-sealed-003"
    assert validated["erratum"]["torch_uuid_observation"] == {
        "assigned_nvidia_uuid": "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "literal_gpu_prefix_is_only_representation_difference": True,
        "pytorch_property_string": "5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "pytorch_property_type_name": "_CUuuid",
        "pytorch_version": "2.12.1+cu132",
    }
    assert validated["erratum"][
        "no_v1_model_result_or_metric_artifact_existed_or_was_accessed"
    ] is True
    assert validated["test_receipt"]["passed"] == 12
    assert (
        validated["source_closure"][AUDITOR_MODULE_PATH]
        != validated["source_closure"]["tests/test_openaxes_external_native_v2_parent_receipt.py"]
    )
    for relative in SOURCE_CLOSURE:
        _oid, reference, _raw = _git_blob(workspace, audited, relative, name="test closure")
        assert reference == validated["source_closure"][relative]
    isolated_output = (
        project / ACTIVE_SOURCE_AUDIT_RECEIPT_PATH
    )
    isolated_output.unlink()
    isolated_output.parent.rmdir()
    assert parent_receipt_module.main(
        [
            "source-audit",
            "--workspace-root",
            str(workspace),
            "--audited-git-commit",
            audited,
            "--output",
            str(isolated_output),
        ]
    ) == 0
    isolated_cli_receipt = strict_json_bytes(
        isolated_output.read_bytes(), name="isolated source-audit CLI", require_canonical=True
    )
    assert set(isolated_cli_receipt["source_closure"]) == set(SOURCE_CLOSURE)
    fresh_project = tmp_path / "fresh-project"
    fresh_project.mkdir()
    fresh_output = fresh_project / (
        ACTIVE_SOURCE_AUDIT_RECEIPT_PATH
    )
    parent_receipt_module._write_new_path(fresh_output, b"sealed\n")
    assert fresh_output.read_bytes() == b"sealed\n"


def test_pre_v1_source_audit_api_has_no_v1_run_input_or_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _synthetic_external_authorities: None,
) -> None:
    workspace, project, audited, _evidence, _receipt = _source_repo(tmp_path)
    observed: list[tuple[Path, str]] = []
    original = parent_receipt_module._read_nofollow

    def guarded_read(root: Path, relative: str, *, name: str) -> bytes:
        observed.append((root, relative))
        assert root == project
        assert relative in SOURCE_CLOSURE
        assert not any(token in relative for token in ("result", "metric", "artifact"))
        return original(root, relative, name=name)

    monkeypatch.setattr(
        "imu_vive_learning.openaxes_external_native_v2_parent_receipt._read_nofollow",
        guarded_read,
    )
    receipt = build_source_audit_receipt(workspace, audited)
    assert receipt["assertions"]["source_written_without_v1_result_or_metric_access"] is True
    assert [relative for _root, relative in observed] == list(SOURCE_CLOSURE) * 2

    monkeypatch.setattr(parent_receipt_module, "_read_nofollow", original)
    real_project = Path.cwd()
    real_closure = {
        relative: _reference(relative, (real_project / relative).read_bytes())
        for relative in SOURCE_CLOSURE
    }

    class Completed:
        returncode = 0
        stdout = b"............ [100%]\n12 passed in 149.18s (0:02:29)\n"
        stderr = b""

    monkeypatch.setattr(parent_receipt_module.subprocess, "run", lambda *args, **kwargs: Completed())
    # pytest mutates both the environment and import path after process
    # bootstrap. Restore the exact production bootstrap state while directly
    # exercising the production guard; the guard itself remains fail-closed.
    exact_environment = _exact_test_environment(
        real_project / ".venv312/lib/python3.12/site-packages"
    )
    for key in tuple(os.environ):
        if key not in exact_environment:
            monkeypatch.delenv(key, raising=False)
    for key, value in exact_environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        sys,
        "path",
        [
            str(real_project / "src"),
            str(real_project / ".venv312/lib/python3.12/site-packages"),
            "/usr/lib/python312.zip",
            "/usr/lib/python3.12",
            "/usr/lib/python3.12/lib-dynload",
        ],
    )
    if sys.flags.no_site != 1:
        with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="env -i"):
            REAL_RUN_EXACT_SOURCE_TESTS(
                real_project,
                "a" * 40,
                canonical_sha256(real_closure),
                real_closure,
            )
        return
    direct = REAL_RUN_EXACT_SOURCE_TESTS(
        real_project,
        "a" * 40,
        canonical_sha256(real_closure),
        real_closure,
    )
    assert direct["passed"] == 12
    assert direct["argv_after_executable"] == list(EXACT_TEST_ARGV)
    assert direct["environment"]["GIT_NO_REPLACE_OBJECTS"] == "1"
    cached_dependencies = {
        "site_packages": Path(direct["venv_site_packages_absolute"]),
        "pyvenv_cfg_reference": direct["pyvenv_cfg_reference"],
        "pyvenv_cfg_fields": direct["pyvenv_cfg_fields"],
        "pinned_distributions": direct["pinned_distributions"],
        "installed_distribution_inventory": direct["installed_distribution_inventory"],
        "dependency_file_closure": direct["dependency_file_closure_after_tests"],
        "runner_module_files": direct["runner_module_files"],
    }
    monkeypatch.setattr(
        parent_receipt_module,
        "_dependency_evidence",
        lambda _project, _lock: deepcopy(cached_dependencies),
    )

    class Short(Completed):
        stdout = b"............ [100%]\n12 passed in 0.01s\n"

    monkeypatch.setattr(parent_receipt_module.subprocess, "run", lambda *args, **kwargs: Short())
    assert REAL_RUN_EXACT_SOURCE_TESTS(
        real_project,
        "a" * 40,
        canonical_sha256(real_closure),
        real_closure,
    )["passed"] == 12

    malformed_reports = (
        b"12 passed in 149.18s (00:02:29)\n",
        b"12 passed in 149.18s (0:60:00)\n",
        b"12 passed in 149.18s (0:00:60)\n",
        b"12 passed in 149.18s arbitrary\n",
        b"12 passed in 0.01s\n12 passed in 0.01s\n",
        b"12 passed in 0.01s\n1 warning in 0.01s\n",
        b"12 passed in 0.01s\n1 failed in 0.01s\n",
        b"12 passed in 0.01s\n1 skipped in 0.01s\n",
        b"12 passed in 0.01s\n1 xfailed in 0.01s\n",
    )
    for malformed in malformed_reports:
        result_type = type(
            "MalformedSourceTestResult",
            (),
            {"returncode": 0, "stdout": malformed, "stderr": b""},
        )
        monkeypatch.setattr(
            parent_receipt_module.subprocess,
            "run",
            lambda *args, _result_type=result_type, **kwargs: _result_type(),
        )
        with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="exactly 12"):
            REAL_RUN_EXACT_SOURCE_TESTS(
                real_project,
                "a" * 40,
                canonical_sha256(real_closure),
                real_closure,
            )

    class Failed(Completed):
        returncode = 1
        stdout = b"1 failed in 0.01s\n"

    monkeypatch.setattr(parent_receipt_module.subprocess, "run", lambda *args, **kwargs: Failed())
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="exactly 12"):
        REAL_RUN_EXACT_SOURCE_TESTS(
            real_project,
            "a" * 40,
            canonical_sha256(real_closure),
            real_closure,
        )


def test_strict_json_rejects_duplicate_nonfinite_and_noncanonical() -> None:
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="duplicate"):
        strict_json_bytes(b'{"x":1,"x":2}\n', name="duplicate")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="constant"):
        strict_json_bytes(b'{"x":NaN}\n', name="nonfinite")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="canonical"):
        strict_json_bytes(b'{"x": 1}\n', name="pretty", require_canonical=True)
    prereg = yaml.safe_load(Path("configs/openaxes_external_native_v2.yaml").read_text())
    auditor = prereg["parent_receipt_auditor"]
    assert set(auditor["source_audit_test_receipt_keys_exact"]) == set(SOURCE_TEST_KEYS)
    assert set(auditor["source_audit_erratum_keys_exact"]) == set(
        SOURCE_AUDIT_ERRATUM_KEYS
    )
    assert auditor["exact_source_closure_paths"] == list(SOURCE_CLOSURE)
    assert auditor["mandatory_pre_v1_source_test_invocation"][
        "argv_after_executable_exact"
    ] == list(EXACT_TEST_ARGV)
    for command in ("source-audit", "issue"):
        with pytest.raises(SystemExit):
            parent_receipt_module._parser().parse_args(
                [command, "--created-at-utc", FIXED_TIME]
            )
    with pytest.raises(TypeError, match="created_at_utc"):
        build_source_audit_receipt(".", "a" * 40, created_at_utc=FIXED_TIME)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="created_at_utc"):
        issue_parent_receipts(
            workspace_root=".",
            v1_run_dir="/v1",
            preparation_run_dir="/preparation",
            v2_run_dir="/v2",
            source_audit_evidence_git_commit="a" * 40,
            created_at_utc=FIXED_TIME,  # type: ignore[call-arg]
        )


def test_reference_resolution_rejects_escape_symlink_and_alternate_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (root / "safe.json").write_bytes(b"{}")
    (other / "outside.json").write_bytes(b"{}")
    (root / "link.json").symlink_to(other / "outside.json")
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(root, target_is_directory=True)
    assert _read_nofollow(_absolute_root(root, name="root"), "safe.json", name="safe") == b"{}"
    for path in ("../other/outside.json", "/tmp/absolute", "a//b", "./safe.json", "a\\b"):
        with pytest.raises(OpenAxesNativeV2ParentReceiptError):
            _read_nofollow(root, path, name="attack")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="no-follow"):
        _read_nofollow(root, "link.json", name="symlink")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="realpath"):
        _absolute_root(root_alias, name="symlink root")
    pinned = parent_receipt_module._pin_root(root, name="sibling-churn root")
    try:
        sibling = tmp_path / "unrelated-sibling"
        sibling.mkdir()
        (sibling / "unrelated.txt").write_text("churn", encoding="utf-8")
        pinned.revalidate(name="sibling-churn root")
        (sibling / "unrelated.txt").unlink()
        sibling.rmdir()
        pinned.revalidate(name="sibling-churn root")
    finally:
        pinned.close()

    venv_root = tmp_path / "record-venv"
    site_packages = venv_root / "lib/python3.12/site-packages"
    package_file = site_packages / "demo/__init__.py"
    dist_info = site_packages / "demo-1.0.dist-info"
    package_file.parent.mkdir(parents=True)
    dist_info.mkdir(parents=True)
    package_raw = b"VALUE = 1\n"
    metadata_raw = b"Metadata-Version: 2.1\nName: Demo\nVersion: 1.0\n"
    package_file.write_bytes(package_raw)
    (dist_info / "METADATA").write_bytes(metadata_raw)

    def record_hash(raw: bytes) -> str:
        encoded = parent_receipt_module.base64.urlsafe_b64encode(
            hashlib.sha256(raw).digest()
        ).rstrip(b"=").decode("ascii")
        return f"sha256={encoded}"

    valid_rows = [
        f"demo/__init__.py,{record_hash(package_raw)},{len(package_raw)}",
        f"demo-1.0.dist-info/METADATA,{record_hash(metadata_raw)},{len(metadata_raw)}",
        "demo-1.0.dist-info/RECORD,,",
    ]
    record_path = dist_info / "RECORD"

    def write_record(rows: list[str]) -> None:
        record_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def close_distribution(distribution: object | None = None) -> None:
        authority = (
            parent_receipt_module.importlib.metadata.Distribution.at(dist_info)
            if distribution is None
            else distribution
        )
        parent_receipt_module._distribution_file_closure(
            venv_root=venv_root,
            site_packages=site_packages,
            owner="demo",
            expected_version="1.0",
            distribution=authority,
        )

    write_record(valid_rows)
    close_distribution()
    package_file.write_bytes(b"VALUE = 2\n")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="hash differs"):
        close_distribution()
    package_file.write_bytes(package_raw)
    write_record([valid_rows[0].rsplit(",", 1)[0] + ",999", *valid_rows[1:]])
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="size differs"):
        close_distribution()
    write_record(
        [
            valid_rows[0].replace("demo/__init__.py", "demo/../demo/__init__.py"),
            *valid_rows[1:],
        ]
    )
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="parent traversal"):
        close_distribution()
    write_record(["demo/__init__.py,,", *valid_rows[1:]])
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="permits empty integrity"):
        close_distribution()
    write_record([*valid_rows, valid_rows[0]])
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="duplicate raw path"):
        close_distribution()
    write_record(valid_rows)
    real_distribution = parent_receipt_module.importlib.metadata.Distribution.at(dist_info)

    class MissingDistributionFile:
        _path = real_distribution._path
        version = real_distribution.version
        metadata = real_distribution.metadata
        files = list(real_distribution.files or ())[:-1]

        @staticmethod
        def read_text(filename: str) -> str | None:
            return real_distribution.read_text(filename)

    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="Distribution.files"):
        close_distribution(MissingDistributionFile())
    package_file.unlink()
    package_file.symlink_to(dist_info / "METADATA")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="no-follow"):
        close_distribution()


def test_static_import_and_call_closure_rejects_forbidden_surfaces() -> None:
    assert audit_python_source_safety(MODULE_PATH.read_bytes())["stdlib_only_imports"] is True
    fixed_git_wrapper = (
        b"\nimport subprocess\n"
        b"def _git_bytes():\n subprocess.run([])\n"
        b"def _run_exact_source_tests():\n subprocess.run([])\n"
    )
    attacks = (
        b"import torch\n" + fixed_git_wrapper,
        b"import cuda\n" + fixed_git_wrapper,
        b"import socket\n" + fixed_git_wrapper,
        b"import os as operating_system\n" + fixed_git_wrapper,
        fixed_git_wrapper + b"\ndef x():\n eval('1')\n",
        fixed_git_wrapper + b"\ndef x():\n __import__('socket')\n",
        fixed_git_wrapper + b"\ndef x():\n os.system('id')\n",
        fixed_git_wrapper + b"\ndef x():\n os.fork()\n",
        fixed_git_wrapper + b"\ndef x():\n subprocess.Popen([])\n",
        fixed_git_wrapper + b"\ndef x():\n train_model()\n",
        fixed_git_wrapper + b"\ndef x():\n predict_outer_views_once()\n",
        fixed_git_wrapper + b"\ndef x():\n load_source_truth()\n",
        fixed_git_wrapper + b"\ndef x():\n evaluate_outer_once()\n",
        fixed_git_wrapper + b"\ndef x():\n open_challenge()\n",
        fixed_git_wrapper + b"\ndef x():\n network_connect()\n",
        b"from pathlib import Path" + fixed_git_wrapper + b"\ndef x():\n Path('/v1').write_text('x')\n",
        fixed_git_wrapper + b"\ndef x():\n open('/v1/file', 'w')\n",
    )
    for raw in attacks:
        with pytest.raises(OpenAxesNativeV2ParentReceiptError):
            audit_python_source_safety(raw)
    for argv in (["status"], ["cat-file", "blob", "--batch"], ["rev-parse", "HEAD"]):
        with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="allowlist"):
            parent_receipt_module._validate_git_argv(argv, name="attack")


def test_source_audit_closed_schema_and_payload_reject_tamper(
    tmp_path: Path, _synthetic_external_authorities: None
) -> None:
    _workspace, _project, _audited, _evidence, receipt = _source_repo(tmp_path)
    for mutation in ("extra", "missing", "hash"):
        attacked = deepcopy(receipt)
        if mutation == "extra":
            attacked["unknown"] = True
        elif mutation == "missing":
            attacked.pop("assertions")
        else:
            attacked["source_set_sha256"] = "0" * 64
            attacked["payload_sha256"] = canonical_sha256(
                {key: value for key, value in attacked.items() if key != "payload_sha256"}
            )
        with pytest.raises(OpenAxesNativeV2ParentReceiptError):
            validate_source_audit_receipt(attacked)
    attacked = deepcopy(receipt)
    attacked["test_receipt"]["stdout_utf8"] += "fabricated\n"
    attacked["payload_sha256"] = canonical_sha256(
        {key: value for key, value in attacked.items() if key != "payload_sha256"}
    )
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="stdout"):
        validate_source_audit_receipt(attacked)

    def receipt_with_stdout(stdout: str) -> dict:
        candidate = deepcopy(receipt)
        test_receipt = candidate["test_receipt"]
        stderr = test_receipt["stderr_utf8"]
        stdout_raw = stdout.encode("utf-8")
        report_raw = canonical_json_bytes(
            {"exit_code": 0, "stderr_utf8": stderr, "stdout_utf8": stdout}
        )
        test_receipt["stdout_utf8"] = stdout
        test_receipt["stdout_bytes"] = len(stdout_raw)
        test_receipt["stdout_sha256"] = hashlib.sha256(stdout_raw).hexdigest()
        test_receipt["report_bytes"] = len(report_raw)
        test_receipt["report_sha256"] = hashlib.sha256(report_raw).hexdigest()
        candidate["payload_sha256"] = canonical_sha256(
            {key: value for key, value in candidate.items() if key != "payload_sha256"}
        )
        return candidate

    for accepted in (
        "............ [100%]\n12 passed in 0.01s\n",
        "............ [100%]\n12 passed in 149.18s (0:02:29)\n",
        "............ [100%]\n12 passed in 360000.00s (100:00:00)\n",
    ):
        validate_source_audit_receipt(receipt_with_stdout(accepted))
    for rejected in (
        "12 passed in 149.18s (00:02:29)\n",
        "12 passed in 149.18s (0:2:29)\n",
        "12 passed in 149.18s (0:02:2)\n",
        "12 passed in 149.18s (0:02:29) extra\n",
        "12 passed in 0.01s\n12 passed in 0.01s\n",
        "12 passed in 0.01s\n1 warning in 0.01s\n",
        "12 passed in 0.01s\n1 failed in 0.01s\n",
        "12 passed in 0.01s\n1 skipped in 0.01s\n",
        "12 passed in 0.01s\n1 xfailed in 0.01s\n",
    ):
        with pytest.raises(
            OpenAxesNativeV2ParentReceiptError,
            match="exact accepted outcome counts",
        ):
            validate_source_audit_receipt(receipt_with_stdout(rejected))


def test_git_blob_oid_bytes_and_dual_ancestry_fail_closed(
    tmp_path: Path, _synthetic_external_authorities: None
) -> None:
    fixture = _full_fixture(tmp_path)
    _oid, reference, raw = _git_blob(
        fixture["workspace"], fixture["audited"], SOURCE_CLOSURE[0], name="prereg blob"
    )
    assert reference["sha256"] == hashlib.sha256(raw).hexdigest()

    # The original and erratum1 audits remain immutable Git history, but no
    # historical path/commit can authorize the successful sealed-003 lineage.
    real_workspace = Path.cwd().parent
    _retired_oid, _retired_reference, retired_raw = _git_blob(
        real_workspace,
        RETIRED_SOURCE_AUDIT_EVIDENCE_COMMIT,
        RETIRED_SOURCE_AUDIT_RECEIPT_PATH,
        name="retired source-audit history",
    )
    retired_receipt = strict_json_bytes(
        retired_raw, name="retired source-audit history", require_canonical=True
    )
    assert retired_receipt["audited_git_commit"] == RETIRED_AUDITED_SOURCE_COMMIT
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="closed schema|contract|retired"):
        validate_source_audit_receipt(retired_receipt)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="historical"):
        parent_receipt_module._verify_source_audit_git_evidence(
            real_workspace,
            retired_raw,
            retired_receipt,
            RETIRED_SOURCE_AUDIT_EVIDENCE_COMMIT,
        )
    with pytest.raises(OpenAxesNativeV2ParentReceiptError):
        _git_blob(
            real_workspace,
            RETIRED_SOURCE_AUDIT_EVIDENCE_COMMIT,
            ACTIVE_SOURCE_AUDIT_RECEIPT_PATH,
            name="forbidden retired active-path lookup",
        )
    _erratum1_oid, _erratum1_reference, erratum1_raw = _git_blob(
        real_workspace,
        ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT,
        ERRATUM1_SOURCE_AUDIT_RECEIPT_PATH,
        name="erratum1 source-audit history",
    )
    erratum1_receipt = strict_json_bytes(
        erratum1_raw, name="erratum1 source-audit history", require_canonical=True
    )
    assert erratum1_receipt["audited_git_commit"] == ERRATUM1_AUDITED_SOURCE_COMMIT
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="contract|historical"):
        validate_source_audit_receipt(erratum1_receipt)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="historical"):
        parent_receipt_module._verify_source_audit_git_evidence(
            real_workspace,
            erratum1_raw,
            erratum1_receipt,
            ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT,
        )
    with pytest.raises(OpenAxesNativeV2ParentReceiptError):
        _git_blob(
            real_workspace,
            ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT,
            ACTIVE_SOURCE_AUDIT_RECEIPT_PATH,
            name="forbidden erratum1 active-path lookup",
        )

    # Both errata change only outcome-free representation consumers and
    # authority plumbing.  The exact old Git config blob and scientific values
    # (methods, hyperparameters, folds/seeds, and gates) remain identical.
    _old_config_oid, old_config_reference, old_config_raw = _git_blob(
        real_workspace,
        PRE_ERRATUM_V1_MODEL_AUTHORITY_COMMIT,
        "configs/openaxes_external_model_v1.yaml",
        name="pre-erratum v1 scientific config",
    )
    new_config_raw = Path("configs/openaxes_external_model_v1.yaml").read_bytes()
    assert old_config_reference == V1_SOURCE_REFERENCE_ORACLE[
        "configs/openaxes_external_model_v1.yaml"
    ]
    assert old_config_raw == new_config_raw
    old_config = yaml.safe_load(old_config_raw)
    new_config = yaml.safe_load(new_config_raw)
    assert canonical_sha256(old_config) == V1_CONFIG_SHA256
    assert canonical_sha256(new_config) == V1_CONFIG_SHA256
    for key in (
        "schema",
        "preprocessing",
        "model",
        "methods",
        "optimization",
        "folds",
        "selection",
        "alignment_sensitivity",
    ):
        assert new_config[key] == old_config[key]
    current_v1_references: dict[str, dict] = {}
    current_v1_blob_oids: dict[str, str] = {}
    for relative in V1_SOURCE_CLOSURE:
        current_raw = Path(relative).read_bytes()
        current_v1_references[relative] = _reference(relative, current_raw)
        current_v1_blob_oids[relative] = hashlib.sha1(
            f"blob {len(current_raw)}\0".encode("ascii") + current_raw
        ).hexdigest()
    assert current_v1_references == V1_SOURCE_REFERENCE_ORACLE
    assert canonical_sha256(current_v1_references) == V1_SOURCE_SET_SHA256
    assert current_v1_blob_oids == V1_SOURCE_BLOB_OID_ORACLE

    _old_v2_oid, _old_v2_reference, old_v2_raw = _git_blob(
        real_workspace,
        PRE_ERRATUM_V1_MODEL_AUTHORITY_COMMIT,
        "configs/openaxes_external_native_v2.yaml",
        name="pre-erratum native-v2 preregistration",
    )
    old_v2 = yaml.safe_load(old_v2_raw)
    new_v2 = yaml.safe_load(Path("configs/openaxes_external_native_v2.yaml").read_bytes())
    _erratum1_v2_oid, _erratum1_v2_reference, erratum1_v2_raw = _git_blob(
        real_workspace,
        PRE_ERRATUM2_V1_MODEL_AUTHORITY_COMMIT,
        "configs/openaxes_external_native_v2.yaml",
        name="erratum1 native-v2 preregistration",
    )
    erratum1_v2 = yaml.safe_load(erratum1_v2_raw)
    assert new_v2["authority_erratum1"] == erratum1_v2["authority_erratum1"]
    authority_only_changed_keys = {
        "contract",
        "lineage",
        "freeze_sequence",
        "git_preregistration",
        "parent_receipt_auditor",
        "v1_terminal_receipt",
    }
    assert set(new_v2) == set(old_v2) | {"authority_erratum1", "authority_erratum2"}
    assert {
        key for key in old_v2 if old_v2[key] != new_v2[key]
    } == authority_only_changed_keys
    frozen_native_v2_science_keys = (
        "schema",
        "preprocessing",
        "architectures",
        "initialization_and_determinism",
        "row_order_and_reductions",
        "environment_and_scheduler",
        "candidates",
        "training_risk",
        "cagrad",
        "optimization",
        "checkpoint",
        "folds_and_fits",
        "outer_boundary",
        "execution_state_machine",
        "runtime_closed_world",
        "outer_label_capability_acceptance",
        "aggregation",
        "primary_gates",
        "report_only_analyses",
        "exact_accounting",
        "claims_and_limitations",
    )
    for key in frozen_native_v2_science_keys:
        assert new_v2[key] == old_v2[key]
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    _git(unrelated, "init", "-q")
    (unrelated / "x").write_text("x", encoding="utf-8")
    unrelated_head = _commit(unrelated, "unrelated")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError):
        _git_blob(fixture["workspace"], unrelated_head, SOURCE_CLOSURE[0], name="missing object")

    replace_repo = tmp_path / "replace-repo"
    replace_project = replace_repo / PROJECT_DIRECTORY
    replace_project.mkdir(parents=True)
    _git(replace_repo, "init", "-q")
    (replace_project / "x").write_text("one", encoding="utf-8")
    first = _commit(replace_repo, "first")
    (replace_project / "x").write_text("two", encoding="utf-8")
    second = _commit(replace_repo, "second")
    _git(replace_repo, "replace", first, second)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="replace refs"):
        _repository(replace_repo)
    _git(replace_repo, "replace", "-d", first)
    (replace_repo / ".git/info/grafts").write_text(f"{second} {first}\n", encoding="ascii")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="grafts"):
        _repository(replace_repo)

    mode_repo = tmp_path / "mode-repo"
    mode_project = mode_repo / PROJECT_DIRECTORY
    mode_project.mkdir(parents=True)
    _git(mode_repo, "init", "-q")
    (mode_project / SOURCE_CLOSURE[0]).parent.mkdir(parents=True, exist_ok=True)
    (mode_project / "target").write_text("target", encoding="utf-8")
    (mode_project / SOURCE_CLOSURE[0]).symlink_to("../target")
    mode_head = _commit(mode_repo, "symlink mode")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="regular-file mode"):
        _git_blob(mode_repo, mode_head, SOURCE_CLOSURE[0], name="symlink blob")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError):
        REAL_VERIFY_V1_SOURCES(mode_repo, mode_head)


def test_synthetic_masked_erm_only_fixture_round_trip(
    tmp_path: Path, _synthetic_external_authorities: None
) -> None:
    fixture = _full_fixture(tmp_path)
    terminal = issue_parent_receipts(
        workspace_root=fixture["workspace"],
        v1_run_dir=fixture["v1"],
        preparation_run_dir=fixture["preparation"],
        v2_run_dir=fixture["v2"],
        source_audit_evidence_git_commit=fixture["evidence"],
    )
    parents = fixture["v2"] / "parents"
    assert sorted(path.name for path in parents.iterdir()) == [
        "parent_receipt_auditor_source_audit.json",
        "v1_masked_erm_comparator_bundle.json",
        "v1_terminal_for_native_v2.json",
    ]
    for path in parents.iterdir():
        raw = path.read_bytes()
        strict_json_bytes(raw, name=path.name, require_canonical=True)
    bundle = strict_json_bytes(
        (parents / "v1_masked_erm_comparator_bundle.json").read_bytes(),
        name="bundle",
        require_canonical=True,
    )
    rows = bundle["primary_seed_ensemble_outage_contexts"]
    arithmetic_mean = sum(row["rmse3d_m"] for row in rows) / len(rows)
    fold_means = [
        sum(row["rmse3d_m"] for row in rows if row["fold_id"] == fold) / 4
        for fold in bundle["primary_fold_order"]
    ]
    mask_means = [
        sum(row["rmse3d_m"] for row in rows if row["view_id"] == view) / 8
        for view in OUTAGE_VIEW_ORDER
    ]
    arithmetic_robust = (arithmetic_mean + max(fold_means) + max(mask_means)) / 3
    assert bundle["derived_gate_inputs"]["A_mean_context_rmse3d_m"] == pytest.approx(
        arithmetic_mean, rel=0.0, abs=1e-15
    )
    assert bundle["derived_gate_inputs"]["WS_worst_scenario_mean_rmse3d_m"] == pytest.approx(
        max(fold_means), rel=0.0, abs=1e-15
    )
    assert bundle["derived_gate_inputs"]["WM_worst_mask_mean_rmse3d_m"] == pytest.approx(
        max(mask_means), rel=0.0, abs=1e-15
    )
    assert bundle["derived_gate_inputs"]["robust_J_m"] == pytest.approx(
        arithmetic_robust, rel=0.0, abs=1e-15
    )
    copied = strict_json_bytes(
        (parents / "parent_receipt_auditor_source_audit.json").read_bytes(),
        name="copied audit",
        require_canonical=True,
    )
    assert set(terminal) == set(TERMINAL_KEYS)
    assert set(bundle) == set(BUNDLE_KEYS)
    assert len(bundle["primary_seed_ensemble_outage_contexts"]) == 32
    assert len(bundle["primary_per_seed_outage_contexts"]) == 96
    validate_terminal_receipt(
        terminal,
        workspace_root=fixture["workspace"],
        v1_run_dir=fixture["v1"],
        preparation_run_dir=fixture["preparation"],
        v2_run_dir=fixture["v2"],
        source_audit_evidence_git_commit=fixture["evidence"],
    )
    extra_file = parents / "unexpected.json"
    extra_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="member set"):
        validate_terminal_receipt(
            terminal,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )
    extra_file.unlink()
    extra_directory = parents / "unexpected-directory"
    extra_directory.mkdir()
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="member set"):
        validate_terminal_receipt(
            terminal,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )
    extra_directory.rmdir()

    launch_path = next(
        path
        for path in fixture["v1"].glob(
            "runtime/execution_launches/launch_*/launch.started.json"
        )
        if json.loads(path.read_text(encoding="utf-8"))["launch_ordinal"] == 0
    )
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch_relative = launch_path.relative_to(fixture["v1"]).as_posix()
    launch_ref = _ref(fixture["v1"], launch_relative)
    ready_relative = (
        f"runtime/worker_ready/launch_{launch['launch_id']}/slot_0000.json"
    )
    ready_path = fixture["v1"] / ready_relative
    ready_raw = ready_path.read_bytes()
    ready = json.loads(ready_raw)
    ready["launch_started"] = deepcopy(launch_ref)
    ready["launch_started"]["bytes"] += 1
    ready["ready_receipt_sha256"] = canonical_sha256(
        {key: value for key, value in ready.items() if key != "ready_receipt_sha256"}
    )
    _write_json(ready_path, ready)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="launch-start raw reference"):
        parent_receipt_module._validate_worker_ready_authority(
            fixture["v1"],
            _ref(fixture["v1"], ready_relative),
            launch=launch,
            launch_reference=launch_ref,
            expected_slot_index=0,
            name="tampered worker-ready",
            session=json.loads(
                (fixture["v1"] / "runtime/execution_session.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="launch-start raw reference"):
        validate_terminal_receipt(
            terminal,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )
    ready_path.write_bytes(ready_raw)

    recovery_path = fixture["v1"] / "runtime/prelaunch_session_recovery.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery["authorized_action_scope"] = "forged_scope"
    recovery["receipt_payload_sha256"] = canonical_sha256(
        {key: value for key, value in recovery.items() if key != "receipt_payload_sha256"}
    )
    _write_json(recovery_path, recovery)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="official semantics"):
        validate_terminal_receipt(
            terminal,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )


def test_comparator_rejects_nonmasked_method_and_forbidden_references(
    tmp_path: Path, _synthetic_external_authorities: None
) -> None:
    fixture = _full_fixture(tmp_path)
    terminal = issue_parent_receipts(
        workspace_root=fixture["workspace"],
        v1_run_dir=fixture["v1"],
        preparation_run_dir=fixture["preparation"],
        v2_run_dir=fixture["v2"],
        source_audit_evidence_git_commit=fixture["evidence"],
    )
    bundle = strict_json_bytes(
        (fixture["v2"] / "parents/v1_masked_erm_comparator_bundle.json").read_bytes(),
        name="bundle",
        require_canonical=True,
    )
    attacked = deepcopy(bundle)
    attacked["method_id"] = "paired_pord"
    attacked["payload_sha256"] = canonical_sha256(
        {key: value for key, value in attacked.items() if key != "payload_sha256"}
    )
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="method"):
        validate_comparator_bundle(attacked)
    for forbidden_path in (
        "results/task_0000_000000000000/outer_evaluation_rows.parquet",
        "results/task_0000_000000000000/outer_predictions.parquet",
        "results/task_0000_000000000000/source_truth.json",
        "results/task_0000_000000000000/challenge_outer_evaluation.json",
    ):
        attacked = deepcopy(bundle)
        attacked["source_references"]["masked_erm_primary_outer_metric_receipts"][0]["path"] = forbidden_path
        attacked["payload_sha256"] = canonical_sha256(
            {key: value for key, value in attacked.items() if key != "payload_sha256"}
        )
        with pytest.raises(OpenAxesNativeV2ParentReceiptError):
            validate_comparator_bundle(attacked)
    metric_relative = bundle["source_references"][
        "masked_erm_primary_outer_metric_receipts"
    ][0]["path"]
    metric_path = fixture["v1"] / metric_relative
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    metric["unsealed"] = True
    _write_json(metric_path, metric)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError):
        validate_terminal_receipt(
            terminal,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )
    wrong_task = _reseal_first_task_to_wrong_authenticated_slot(fixture)
    assert wrong_task not in json.loads(
        (fixture["v1"] / "runtime/execution_session.json").read_text(encoding="utf-8")
    )["slots"][1]["task_ids"]
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="not assigned"):
        validate_terminal_receipt(
            terminal,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )


def test_bundle_fixed_reductions_and_hash_reject_tamper(
    tmp_path: Path, _synthetic_external_authorities: None
) -> None:
    fixture = _full_fixture(tmp_path)
    issue_parent_receipts(
        workspace_root=fixture["workspace"],
        v1_run_dir=fixture["v1"],
        preparation_run_dir=fixture["preparation"],
        v2_run_dir=fixture["v2"],
        source_audit_evidence_git_commit=fixture["evidence"],
    )
    bundle = strict_json_bytes(
        (fixture["v2"] / "parents/v1_masked_erm_comparator_bundle.json").read_bytes(),
        name="bundle",
        require_canonical=True,
    )
    attacked = deepcopy(bundle)
    attacked["derived_gate_inputs"]["robust_J_m"] += 1e-6
    attacked["payload_sha256"] = canonical_sha256(
        {key: value for key, value in attacked.items() if key != "payload_sha256"}
    )
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="fixed reduction"):
        validate_comparator_bundle(attacked)
    attacked = deepcopy(bundle)
    attacked["payload_sha256"] = "0" * 64
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="payload hash"):
        validate_comparator_bundle(attacked)


def test_issue_is_exclusive_and_never_mutates_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _synthetic_external_authorities: None,
) -> None:
    fixture = _full_fixture(tmp_path)
    before = {
        path.relative_to(fixture["v1"]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in fixture["v1"].rglob("*")
        if path.is_file()
    }
    arguments = {
        "workspace_root": fixture["workspace"],
        "v1_run_dir": fixture["v1"],
        "preparation_run_dir": fixture["preparation"],
        "v2_run_dir": fixture["v2"],
        "source_audit_evidence_git_commit": fixture["evidence"],
    }
    wrong_v1 = fixture["project"] / (
        "artifacts/openaxes_external_model_v1/run_openaxes-v1-sealed-009"
    )
    wrong_v1.mkdir(parents=True)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="sealed-003"):
        issue_parent_receipts(**{**arguments, "v1_run_dir": wrong_v1})
    wrong_preparation = fixture["project"] / (
        "artifacts/openaxes_external_v1/prepared_openaxes-v1-sealed-009"
    )
    wrong_preparation.mkdir(parents=True)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="sealed-003"):
        issue_parent_receipts(
            **{**arguments, "preparation_run_dir": wrong_preparation}
        )
    issue_parent_receipts(**arguments)
    after = {
        path.relative_to(fixture["v1"]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in fixture["v1"].rglob("*")
        if path.is_file()
    }
    assert after == before
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="fresh"):
        issue_parent_receipts(**arguments)

    output = tmp_path / "partial-output"
    output.mkdir()
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    original_write = parent_receipt_module.os.write
    calls = 0

    def partial_write(fd: int, raw: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(fd, raw[:2])
        return 0

    monkeypatch.setattr(parent_receipt_module.os, "write", partial_write)
    try:
        with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="no progress"):
            _write_exclusive(descriptor, "receipt.json", b"abcdef")
        assert not (output / "receipt.json").exists()
    finally:
        os.close(descriptor)
        monkeypatch.setattr(parent_receipt_module.os, "write", original_write)

    fault_fixture = _full_fixture(tmp_path / "three-file-fault")
    original_exclusive = parent_receipt_module._write_exclusive
    persisted_calls = 0

    def fail_second_parent_file(directory_descriptor: int, name: str, raw: bytes) -> None:
        nonlocal persisted_calls
        persisted_calls += 1
        if persisted_calls == 2:
            raise OpenAxesNativeV2ParentReceiptError(
                "injected parent three-file persistence fault"
            )
        original_exclusive(directory_descriptor, name, raw)

    monkeypatch.setattr(
        parent_receipt_module, "_write_exclusive", fail_second_parent_file
    )
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="injected"):
        issue_parent_receipts(
            workspace_root=fault_fixture["workspace"],
            v1_run_dir=fault_fixture["v1"],
            preparation_run_dir=fault_fixture["preparation"],
            v2_run_dir=fault_fixture["v2"],
            source_audit_evidence_git_commit=fault_fixture["evidence"],
        )
    fault_parents = fault_fixture["v2"] / "parents"
    assert [path.name for path in fault_parents.iterdir()] == [
        "parent_receipt_auditor_source_audit.json"
    ]
    monkeypatch.setattr(parent_receipt_module, "_write_exclusive", original_exclusive)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="fresh"):
        issue_parent_receipts(
            workspace_root=fault_fixture["workspace"],
            v1_run_dir=fault_fixture["v1"],
            preparation_run_dir=fault_fixture["preparation"],
            v2_run_dir=fault_fixture["v2"],
            source_audit_evidence_git_commit=fault_fixture["evidence"],
        )

    chmod_fixture = _full_fixture(tmp_path / "chmod-race")
    original_v2_mode = os.stat(chmod_fixture["v2"]).st_mode & 0o7777
    chmod_injected = False

    def chmod_v2_then_write(
        directory_descriptor: int, name: str, raw: bytes
    ) -> None:
        nonlocal chmod_injected
        if not chmod_injected:
            os.chmod(chmod_fixture["v2"], original_v2_mode ^ 0o001)
            chmod_injected = True
        original_exclusive(directory_descriptor, name, raw)

    monkeypatch.setattr(parent_receipt_module, "_write_exclusive", chmod_v2_then_write)
    try:
        with pytest.raises(
            OpenAxesNativeV2ParentReceiptError,
            match="v2_run_root pinned directory identity changed",
        ):
            issue_parent_receipts(
                workspace_root=chmod_fixture["workspace"],
                v1_run_dir=chmod_fixture["v1"],
                preparation_run_dir=chmod_fixture["preparation"],
                v2_run_dir=chmod_fixture["v2"],
                source_audit_evidence_git_commit=chmod_fixture["evidence"],
            )
    finally:
        os.chmod(chmod_fixture["v2"], original_v2_mode)
        monkeypatch.setattr(
            parent_receipt_module, "_write_exclusive", original_exclusive
        )

    swap_fixture = _full_fixture(tmp_path / "swap-case")
    swapped = False

    def swap_then_write(directory_descriptor: int, name: str, raw: bytes) -> None:
        nonlocal swapped
        if not swapped:
            moved = swap_fixture["project"] / "synthetic-v1-moved"
            os.rename(swap_fixture["v1"], moved)
            swap_fixture["v1"].mkdir()
            swapped = True
        original_exclusive(directory_descriptor, name, raw)

    monkeypatch.setattr(parent_receipt_module, "_write_exclusive", swap_then_write)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="pinned|parent entry"):
        issue_parent_receipts(
            workspace_root=swap_fixture["workspace"],
            v1_run_dir=swap_fixture["v1"],
            preparation_run_dir=swap_fixture["preparation"],
            v2_run_dir=swap_fixture["v2"],
            source_audit_evidence_git_commit=swap_fixture["evidence"],
        )


def test_terminal_and_bundle_closed_schemas_reject_unknown_or_missing(
    tmp_path: Path, _synthetic_external_authorities: None
) -> None:
    fixture = _full_fixture(tmp_path)
    terminal = issue_parent_receipts(
        workspace_root=fixture["workspace"],
        v1_run_dir=fixture["v1"],
        preparation_run_dir=fixture["preparation"],
        v2_run_dir=fixture["v2"],
        source_audit_evidence_git_commit=fixture["evidence"],
    )
    parents = fixture["v2"] / "parents"
    bundle = strict_json_bytes(
        (parents / "v1_masked_erm_comparator_bundle.json").read_bytes(), name="bundle", require_canonical=True
    )
    copied = strict_json_bytes(
        (parents / "parent_receipt_auditor_source_audit.json").read_bytes(), name="audit", require_canonical=True
    )
    terminal_path = parents / "v1_terminal_for_native_v2.json"
    for historical_path, historical_evidence in (
        (RETIRED_SOURCE_AUDIT_RECEIPT_PATH, RETIRED_SOURCE_AUDIT_EVIDENCE_COMMIT),
        (ERRATUM1_SOURCE_AUDIT_RECEIPT_PATH, ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT),
    ):
        historical_authority = deepcopy(terminal)
        historical_authority["auditor_source_audit"][
            "source_audit_receipt_path"
        ] = historical_path
        historical_authority["auditor_source_audit"][
            "source_audit_git_commit"
        ] = historical_evidence
        historical_authority["payload_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in historical_authority.items()
                if key != "payload_sha256"
            }
        )
        terminal_path.write_bytes(
            canonical_json_bytes(historical_authority, final_newline=True)
        )
        with pytest.raises(
            OpenAxesNativeV2ParentReceiptError, match="active source-audit|authority"
        ):
            validate_terminal_receipt(
                historical_authority,
                workspace_root=fixture["workspace"],
                v1_run_dir=fixture["v1"],
                preparation_run_dir=fixture["preparation"],
                v2_run_dir=fixture["v2"],
                source_audit_evidence_git_commit=fixture["evidence"],
            )
    terminal_path.write_bytes(canonical_json_bytes(terminal, final_newline=True))
    attacked = deepcopy(terminal)
    attacked["unknown"] = True
    with pytest.raises(OpenAxesNativeV2ParentReceiptError):
        validate_terminal_receipt(
            attacked,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )
    attacked_bundle = deepcopy(bundle)
    attacked_bundle.pop("primary_context_identity")
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="closed schema"):
        validate_comparator_bundle(attacked_bundle)
    forged = deepcopy(terminal)
    forged["git_lineage"]["v2_yaml_git_blob_oid"] = "f" * 40
    forged["payload_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "payload_sha256"}
    )
    terminal_path.write_bytes(canonical_json_bytes(forged, final_newline=True))
    with pytest.raises(OpenAxesNativeV2ParentReceiptError, match="public recomputation"):
        validate_terminal_receipt(
            forged,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )
    terminal_path.write_bytes(canonical_json_bytes(terminal, final_newline=True))
    replay_path = fixture["v1"] / "runtime/full_signed_source_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["semantic_replay"] = False
    replay["receipt_payload_sha256"] = canonical_sha256(
        {key: value for key, value in replay.items() if key != "receipt_payload_sha256"}
    )
    _write_json(replay_path, replay)
    with pytest.raises(OpenAxesNativeV2ParentReceiptError):
        validate_terminal_receipt(
            terminal,
            workspace_root=fixture["workspace"],
            v1_run_dir=fixture["v1"],
            preparation_run_dir=fixture["preparation"],
            v2_run_dir=fixture["v2"],
            source_audit_evidence_git_commit=fixture["evidence"],
        )
