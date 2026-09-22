from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from imu_vive_learning.features import FUSION_COLUMNS
import imu_vive_learning.gpu_outage_gbdt_v4_training as gbdt_module
from imu_vive_learning.gpu_outage_gbdt_v4_training import (
    GBDTBundleSpec,
    GBDTV4TrainingError,
    GBDTReplicatedTrainingData,
    GBDT_V4_EXPECTED_LIBXGBOOST_SHA256,
    GBDT_V4_FEATURE_NAMES,
    GBDT_V4_FIXED_XGB_PARAMS,
    GBDT_V4_MASK_PROBABILITIES,
    GBDT_V4_MASK_STATES,
    GBDT_V4_NUM_BOOST_ROUND,
    GBDT_V4_PAIRED_SEEDS,
    build_mask_replicated_training_data,
    derive_xgb_axis_seed,
    development_bundle_specs,
    development_evaluation_contexts,
    fit_xgb_bundle,
    fixed_xgb_parameter_sha256,
    force_gbdt_outage_features,
    load_xgb_bundle,
    predict_xgb_bundle_forced_outage,
    save_xgb_bundle,
    transform_gbdt_outage_features,
    validate_development_bundle_coverage,
    validate_xgb_bundle_models,
    validate_xgb_bundle_replay,
    validate_xgb_training_replay,
    validate_xgboost_runtime_provenance,
)
from imu_vive_learning.gpu_v3_backbones import IMU_SENSOR_INDEX_MAP


def _problem(*, rows_per_segment: tuple[int, ...] = (6, 6, 6)) -> dict:
    generator = np.random.default_rng(20260824)
    segment_values = (36, 37, 38, 39)[: len(rows_per_segment)]
    segment_ids = np.concatenate(
        [np.full(rows, segment, dtype=np.int64) for segment, rows in zip(segment_values, rows_per_segment, strict=True)]
    )
    rows = len(segment_ids)
    features = generator.normal(size=(rows, 43)).astype(np.float32)
    targets = np.column_stack(
        (
            1.5 * features[:, 0] - 0.7 * features[:, 3] + 0.2,
            -0.9 * features[:, 1] + 0.4 * features[:, 15] - 0.1,
            0.6 * features[:, 2] + 0.3 * features[:, 27] + 0.5,
        )
    ).astype(np.float32)
    return {
        "X": features,
        "y": targets,
        "trial_uids": np.asarray([f"synthetic-{index:05d}" for index in range(rows)]),
        "segment_ids": segment_ids,
    }


def _rewrite_metadata(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def fitted_bundle() -> tuple[dict, object]:
    xgboost = pytest.importorskip("xgboost")
    if xgboost.__version__ != "3.3.0":
        pytest.skip("the frozen v4 serialization test requires xgboost==3.3.0")
    problem = _problem()
    bundle = fit_xgb_bundle(
        problem["X"],
        problem["y"],
        problem["trial_uids"],
        problem["segment_ids"],
        held_out_segment=39,
        paired_seed=GBDT_V4_PAIRED_SEEDS[0],
        device="cpu",
        allow_cpu_for_testing=True,
    )
    return problem, bundle


def test_frozen_recipe_seed_and_exact_twelve_bundle_specs() -> None:
    assert GBDT_V4_NUM_BOOST_ROUND == 1000
    assert dict(GBDT_V4_FIXED_XGB_PARAMS) == {
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eta": 0.03,
        "max_depth": 6,
        "min_child_weight": 5.0,
        "subsample": 1.0,
        "sampling_method": "uniform",
        "colsample_bytree": 0.8,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "max_bin": 256,
        "grow_policy": "depthwise",
        "num_parallel_tree": 1,
        "seed_per_iteration": False,
        "nthread": 1,
    }
    assert "early_stopping_rounds" not in GBDT_V4_FIXED_XGB_PARAMS
    assert len(fixed_xgb_parameter_sha256()) == 64

    assert derive_xgb_axis_seed(20260824, 0) == 947_044_288
    assert derive_xgb_axis_seed(20260824, "x") == 947_044_288
    assert derive_xgb_axis_seed(20260824, 1) == 454_706_684
    assert derive_xgb_axis_seed(20260824, 2) == 2_793_573_466
    all_axis_seeds = {
        derive_xgb_axis_seed(seed, axis)
        for seed in GBDT_V4_PAIRED_SEEDS
        for axis in range(3)
    }
    assert len(all_axis_seeds) == 9

    specs = development_bundle_specs()
    assert len(specs) == 12
    assert len(set(specs)) == 12
    assert {(spec.held_out_segment, spec.paired_seed) for spec in specs} == {
        (segment, seed)
        for segment in (36, 37, 38, 39)
        for seed in GBDT_V4_PAIRED_SEEDS
    }
    assert all("outage" not in spec.bundle_id for spec in specs)
    assert validate_development_bundle_coverage(specs) == {
        "bundle_count": 12,
        "fold_count": 4,
        "forced_outage_fit_multiplier": 1,
        "paired_seed_count": 3,
        "scalar_booster_count": 36,
    }
    contexts = development_evaluation_contexts()
    assert len(contexts) == 48
    assert len({spec for spec, _ in contexts}) == 12
    assert {outage for _, outage in contexts} == {0, 1, 2, 3}
    assert all(
        sum(context_spec == spec for context_spec, _ in contexts) == 4
        for spec in specs
    )
    with pytest.raises(GBDTV4TrainingError, match="4 folds by 3"):
        validate_development_bundle_coverage(specs[:-1])
    assert "forced_outage" not in inspect.signature(fit_xgb_bundle).parameters


def test_five_replicas_and_segment_balanced_probability_weights() -> None:
    problem = _problem(rows_per_segment=(2, 3, 4))
    training = build_mask_replicated_training_data(
        problem["X"],
        problem["y"],
        problem["trial_uids"],
        problem["segment_ids"],
    )

    source_rows = len(problem["X"])
    assert training.feature_names == GBDT_V4_FEATURE_NAMES
    assert len(GBDT_V4_FEATURE_NAMES) == 47
    assert training.features.shape == (5 * source_rows, 47)
    assert training.targets.shape == (5 * source_rows, 3)
    assert np.array_equal(
        training.source_row_indices, np.repeat(np.arange(source_rows), 5)
    )
    assert np.array_equal(
        training.mask_states,
        np.tile(np.asarray(GBDT_V4_MASK_STATES, dtype=np.int8), source_rows),
    )
    assert np.array_equal(
        training.targets, np.repeat(problem["y"], 5, axis=0)
    )
    assert not training.features.flags.writeable
    assert not training.sample_weight.flags.writeable

    q = np.asarray(GBDT_V4_MASK_PROBABILITIES)
    counts = {36: 2, 37: 3, 38: 4}
    for source_index, segment in enumerate(problem["segment_ids"]):
        expected = source_rows / (3 * counts[int(segment)]) * q
        assert np.array_equal(
            training.sample_weight[source_index * 5 : (source_index + 1) * 5],
            expected,
        )
    assert training.sample_weight.sum() == pytest.approx(source_rows, abs=1e-12)
    repeated_segments = np.repeat(problem["segment_ids"], 5)
    for segment in counts:
        assert training.sample_weight[repeated_segments == segment].sum() == pytest.approx(
            source_rows / 3, abs=1e-12
        )

    expected_base_score = np.mean(
        np.stack(
            [
                np.mean(problem["y"][problem["segment_ids"] == segment], axis=0)
                for segment in counts
            ]
        ),
        axis=0,
    )
    assert np.allclose(training.weighted_base_scores, expected_base_score, atol=1e-7)
    assert training.audit()["replicas_per_uid"] == 5
    assert training.audit()["data_sha256"] == training.data_sha256

    for source_index in range(source_rows):
        rows = slice(source_index * 5, source_index * 5 + 5)
        replica = training.features[rows]
        assert np.array_equal(replica[0, :43], problem["X"][source_index])
        assert np.array_equal(replica[0, 43:], np.ones(4, dtype=np.float32))
        for row, sensor_index in enumerate(range(4), start=1):
            sensor_columns = np.asarray(IMU_SENSOR_INDEX_MAP[sensor_index])
            expected_availability = np.ones(4, dtype=np.float32)
            expected_availability[sensor_index] = 0.0
            assert np.array_equal(replica[row, sensor_columns], np.zeros(10))
            assert np.array_equal(replica[row, 43:], expected_availability)

    tampered_replicas = training.features.copy()
    tampered_replicas[1, 0] += 1.0  # Encoder remains available in every state.
    with pytest.raises(GBDTV4TrainingError, match="every replica"):
        GBDTReplicatedTrainingData(
            feature_names=training.feature_names,
            features=tampered_replicas,
            targets=training.targets,
            sample_weight=training.sample_weight,
            source_row_indices=training.source_row_indices,
            mask_states=training.mask_states,
            source_trial_uids=training.source_trial_uids,
            source_segment_ids=training.source_segment_ids,
        )


def test_development_coverage_requires_exact_isolated_cuda_zero() -> None:
    bundles = []
    for spec in development_bundle_specs():
        bundle = object.__new__(gbdt_module.XGBBundle)
        object.__setattr__(bundle, "spec", spec)
        object.__setattr__(bundle, "training_device", "cuda:1")
        bundles.append(bundle)
    with pytest.raises(GBDTV4TrainingError, match="isolated cuda:0"):
        validate_development_bundle_coverage(bundles)
    for bundle in bundles:
        object.__setattr__(bundle, "training_device", "cuda:0")
    assert validate_development_bundle_coverage(bundles)["bundle_count"] == 12


def test_training_data_rejects_identity_leakage_and_invalid_rows() -> None:
    problem = _problem()
    with pytest.raises(GBDTV4TrainingError, match="exact ordered 43"):
        build_mask_replicated_training_data(
            problem["X"],
            problem["y"],
            problem["trial_uids"],
            problem["segment_ids"],
            feature_names=(*FUSION_COLUMNS[:-1], "vive_pos_raw_x_mm"),
        )
    duplicate_uids = problem["trial_uids"].copy()
    duplicate_uids[-1] = duplicate_uids[0]
    with pytest.raises(GBDTV4TrainingError, match="unique"):
        build_mask_replicated_training_data(
            problem["X"], problem["y"], duplicate_uids, problem["segment_ids"]
        )
    nonfinite = problem["X"].copy()
    nonfinite[0, 4] = np.nan
    with pytest.raises(GBDTV4TrainingError, match="finite"):
        build_mask_replicated_training_data(
            nonfinite,
            problem["y"],
            problem["trial_uids"],
            problem["segment_ids"],
        )
    challenge_segment = problem["segment_ids"].copy()
    challenge_segment[0] = 40
    with pytest.raises(GBDTV4TrainingError, match="segments 36--39"):
        build_mask_replicated_training_data(
            problem["X"],
            problem["y"],
            problem["trial_uids"],
            challenge_segment,
        )


def test_forced_outage_is_evaluation_only_and_zero_fill_is_fail_closed() -> None:
    problem = _problem()
    for forced_outage in range(4):
        transformed = force_gbdt_outage_features(problem["X"], forced_outage)
        assert transformed.shape == (len(problem["X"]), 47)
        expected_availability = np.ones((len(transformed), 4), dtype=np.float32)
        expected_availability[:, forced_outage] = 0.0
        assert np.array_equal(transformed[:, 43:], expected_availability)
        assert np.array_equal(
            transformed[:, np.asarray(IMU_SENSOR_INDEX_MAP[forced_outage])],
            np.zeros((len(transformed), 10), dtype=np.float32),
        )

    outage = force_gbdt_outage_features(problem["X"], 1)
    assert np.array_equal(
        transform_gbdt_outage_features(outage[:, :43], outage[:, 43:]), outage
    )
    bad_zero_fill = outage[:, :43].copy()
    bad_zero_fill[0, IMU_SENSOR_INDEX_MAP[1][0]] = 1.0
    with pytest.raises(GBDTV4TrainingError, match="exactly zero"):
        transform_gbdt_outage_features(bad_zero_fill, outage[:, 43:])
    multiple = outage[:, 43:].copy()
    multiple[:, 2] = 0.0
    with pytest.raises(GBDTV4TrainingError, match="one-IMU-outage"):
        transform_gbdt_outage_features(outage[:, :43], multiple)

    natural_features = problem["X"][:5].copy()
    natural_availability = np.ones((5, 4), dtype=np.float32)
    for row, sensor_index in enumerate(range(4), start=1):
        natural_availability[row, sensor_index] = 0.0
        natural_features[row, np.asarray(IMU_SENSOR_INDEX_MAP[sensor_index])] = 0.0
    natural = transform_gbdt_outage_features(
        natural_features, natural_availability
    )
    assert np.array_equal(natural[:, :43], natural_features)
    assert np.array_equal(natural[:, 43:], natural_availability)
    with pytest.raises(GBDTV4TrainingError, match="numeric or boolean"):
        transform_gbdt_outage_features(
            natural_features, natural_availability.astype(str)
        )


def test_fixed_scalar_fit_is_deterministic_and_replays_training(
    fitted_bundle: tuple[dict, object],
) -> None:
    problem, bundle = fitted_bundle
    assert bundle.num_boost_round == 1000
    assert len(bundle.axis_model_bytes) == 3
    assert bundle.axis_seeds == tuple(
        derive_xgb_axis_seed(GBDT_V4_PAIRED_SEEDS[0], axis) for axis in range(3)
    )
    expected_training = build_mask_replicated_training_data(
        problem["X"],
        problem["y"],
        problem["trial_uids"],
        problem["segment_ids"],
    )
    assert np.allclose(bundle.axis_base_scores, expected_training.weighted_base_scores)
    model_audit = validate_xgb_bundle_models(bundle)
    assert model_audit["scalar_booster_count"] == 3
    assert model_audit["early_stopping_attributes"] == 0
    replay = validate_xgb_training_replay(
        bundle,
        problem["X"],
        problem["y"],
        problem["trial_uids"],
        problem["segment_ids"],
        device="cpu",
    )
    assert replay["replay_exact"] is True
    assert replay["replicated_rows"] == 5 * len(problem["X"])

    repeated = fit_xgb_bundle(
        problem["X"],
        problem["y"],
        problem["trial_uids"],
        problem["segment_ids"],
        held_out_segment=39,
        paired_seed=GBDT_V4_PAIRED_SEEDS[0],
        device="cpu",
        allow_cpu_for_testing=True,
    )
    assert repeated.bundle_sha256 == bundle.bundle_sha256
    assert repeated.axis_model_bytes == bundle.axis_model_bytes
    assert repeated.training_prediction_sha256 == bundle.training_prediction_sha256


def test_cpu_fit_is_test_only_and_scope_validation_precedes_training() -> None:
    problem = _problem()
    with pytest.raises(GBDTV4TrainingError, match="CPU is test-only"):
        fit_xgb_bundle(
            problem["X"],
            problem["y"],
            problem["trial_uids"],
            problem["segment_ids"],
            held_out_segment=39,
            paired_seed=GBDT_V4_PAIRED_SEEDS[0],
            device="cpu",
        )
    with pytest.raises(GBDTV4TrainingError, match="training segments"):
        fit_xgb_bundle(
            problem["X"],
            problem["y"],
            problem["trial_uids"],
            problem["segment_ids"],
            held_out_segment=38,
            paired_seed=GBDT_V4_PAIRED_SEEDS[0],
            device="cpu",
            allow_cpu_for_testing=True,
        )


def test_training_replay_binds_ordered_uids_and_row_level_segments(
    fitted_bundle: tuple[dict, object],
) -> None:
    problem, bundle = fitted_bundle
    training = build_mask_replicated_training_data(
        problem["X"],
        problem["y"],
        problem["trial_uids"],
        problem["segment_ids"],
    )
    assert bundle.source_ordered_trial_uid_sha256 == (
        training.ordered_trial_uid_sha256
    )
    assert bundle.source_row_segment_sha256 == training.row_segment_sha256

    reversed_uids = problem["trial_uids"][::-1].copy()
    reversed_training = build_mask_replicated_training_data(
        problem["X"],
        problem["y"],
        reversed_uids,
        problem["segment_ids"],
    )
    assert reversed_training.sorted_trial_uid_sha256 == training.sorted_trial_uid_sha256
    assert reversed_training.ordered_trial_uid_sha256 != training.ordered_trial_uid_sha256
    assert reversed_training.data_sha256 != training.data_sha256
    with pytest.raises(GBDTV4TrainingError, match="ordered UID"):
        validate_xgb_training_replay(
            bundle,
            problem["X"],
            problem["y"],
            reversed_uids,
            problem["segment_ids"],
            device="cpu",
        )

    reassigned_segments = problem["segment_ids"].copy()
    reassigned_segments[0], reassigned_segments[6] = (
        reassigned_segments[6],
        reassigned_segments[0],
    )
    reassigned_training = build_mask_replicated_training_data(
        problem["X"],
        problem["y"],
        problem["trial_uids"],
        reassigned_segments,
    )
    assert reassigned_training.segment_counts == training.segment_counts
    assert reassigned_training.row_segment_sha256 != training.row_segment_sha256
    assert reassigned_training.data_sha256 != training.data_sha256
    with pytest.raises(GBDTV4TrainingError, match="row-level segment"):
        validate_xgb_training_replay(
            bundle,
            problem["X"],
            problem["y"],
            problem["trial_uids"],
            reassigned_segments,
            device="cpu",
        )


def test_checkpoint_serialization_hash_binding_and_exact_prediction_replay(
    fitted_bundle: tuple[dict, object], tmp_path: Path
) -> None:
    problem, bundle = fitted_bundle
    checkpoint = tmp_path / "checkpoint"
    saved = save_xgb_bundle(bundle, checkpoint)
    assert saved["scalar_booster_count"] == 3
    assert set(path.name for path in checkpoint.iterdir()) == {
        "metadata.json",
        "axis_x.ubj",
        "axis_y.ubj",
        "axis_z.ubj",
    }
    loaded = load_xgb_bundle(
        checkpoint,
        expected_held_out_segment=39,
        expected_paired_seed=GBDT_V4_PAIRED_SEEDS[0],
        expected_source_sorted_trial_uid_sha256=bundle.source_sorted_trial_uid_sha256,
        expected_source_ordered_trial_uid_sha256=bundle.source_ordered_trial_uid_sha256,
        expected_source_row_segment_sha256=bundle.source_row_segment_sha256,
    )
    assert loaded.bundle_sha256 == bundle.bundle_sha256
    expected = predict_xgb_bundle_forced_outage(
        bundle, problem["X"], 2, device="cpu"
    )
    replayed = predict_xgb_bundle_forced_outage(
        loaded, problem["X"], 2, device="cpu"
    )
    assert np.array_equal(replayed, expected)
    model_features = force_gbdt_outage_features(problem["X"], 2)
    assert validate_xgb_bundle_replay(
        loaded, model_features, expected, device="cpu"
    )["replay_exact"] is True
    with pytest.raises(GBDTV4TrainingError, match="already exists"):
        save_xgb_bundle(bundle, checkpoint)

    tampered = tmp_path / "tampered"
    save_xgb_bundle(bundle, tampered)
    axis_path = tampered / "axis_x.ubj"
    axis_path.write_bytes(axis_path.read_bytes() + b"tamper")
    with pytest.raises(GBDTV4TrainingError, match="size/hash"):
        load_xgb_bundle(tampered)

    extra = tmp_path / "extra"
    save_xgb_bundle(bundle, extra)
    (extra / "unregistered.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(GBDTV4TrainingError, match="members"):
        load_xgb_bundle(extra)


def test_seed_heldout_and_boolean_inputs_reject_coercion(tmp_path: Path) -> None:
    for invalid_seed in (True, 20_260_824.0, "20260824"):
        with pytest.raises(GBDTV4TrainingError, match="paired_seed must be an integer"):
            derive_xgb_axis_seed(invalid_seed, 0)
    for invalid_held_out in (None, True, 39.0, "39"):
        with pytest.raises(GBDTV4TrainingError, match="held_out_segment must be an integer"):
            GBDTBundleSpec(
                held_out_segment=invalid_held_out,
                paired_seed=GBDT_V4_PAIRED_SEEDS[0],
            )
    with pytest.raises(GBDTV4TrainingError, match="require_cuda must be a boolean"):
        validate_development_bundle_coverage(
            development_bundle_specs(), require_cuda="false"
        )

    problem = _problem()
    with pytest.raises(
        GBDTV4TrainingError, match="allow_cpu_for_testing must be a boolean"
    ):
        fit_xgb_bundle(
            problem["X"],
            problem["y"],
            problem["trial_uids"],
            problem["segment_ids"],
            held_out_segment=39,
            paired_seed=GBDT_V4_PAIRED_SEEDS[0],
            device="cpu",
            allow_cpu_for_testing="false",
        )

    missing = tmp_path / "not-needed-because-arguments-fail-first"
    with pytest.raises(GBDTV4TrainingError, match="validate_models must be a boolean"):
        load_xgb_bundle(missing, validate_models="false")
    with pytest.raises(
        GBDTV4TrainingError, match="expected_held_out_segment must be an integer"
    ):
        load_xgb_bundle(missing, expected_held_out_segment="39")
    with pytest.raises(
        GBDTV4TrainingError, match="expected_held_out_segment must be an integer"
    ):
        load_xgb_bundle(missing, expected_held_out_segment=None)
    with pytest.raises(GBDTV4TrainingError, match="expected_paired_seed must be an integer"):
        load_xgb_bundle(missing, expected_paired_seed=20_260_824.0)
    with pytest.raises(
        GBDTV4TrainingError,
        match="expected_source_ordered_trial_uid_sha256 must be a string",
    ):
        load_xgb_bundle(
            missing,
            expected_source_ordered_trial_uid_sha256=b"0" * 64,
        )


def test_raw_metadata_rejects_float_string_and_bool_count_coercions(
    fitted_bundle: tuple[dict, object], tmp_path: Path
) -> None:
    _, bundle = fitted_bundle

    cases = (
        (
            "schema_float",
            lambda payload: payload.__setitem__("schema_version", 1.0),
            "schema_version must be an integer",
        ),
        (
            "row_count_string",
            lambda payload: payload["source"].__setitem__("row_count", "18"),
            "source.row_count must be an integer",
        ),
        (
            "axis_index_bool",
            lambda payload: payload["axis_models"][0].__setitem__(
                "axis_index", False
            ),
            "axis model index must be an integer",
        ),
        (
            "segment_rows_float",
            lambda payload: payload["source"]["segment_counts"][0].__setitem__(
                "rows", 6.0
            ),
            "source segment rows must be an integer",
        ),
    )
    for name, mutate, message in cases:
        checkpoint = tmp_path / name
        save_xgb_bundle(bundle, checkpoint)
        metadata_path = checkpoint / "metadata.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        mutate(payload)
        _rewrite_metadata(metadata_path, payload)
        with pytest.raises(GBDTV4TrainingError, match=message):
            load_xgb_bundle(checkpoint)


def test_xgboost_runtime_library_and_requested_cuda_ordinal_are_bound(
    fitted_bundle: tuple[dict, object],
) -> None:
    _, bundle = fitted_bundle
    provenance = bundle.runtime_provenance
    assert provenance.libxgboost_sha256 == GBDT_V4_EXPECTED_LIBXGBOOST_SHA256
    assert Path(provenance.libxgboost_resolved_path).is_absolute()
    assert Path(provenance.libxgboost_resolved_path).name == "libxgboost.so"
    assert provenance.canonical_build_info["libxgboost"] == "libxgboost.so"
    assert provenance.canonical_build_info["USE_CUDA"] is True
    runtime_audit = validate_xgboost_runtime_provenance(provenance)
    assert runtime_audit["libxgboost_sha256"] == (
        GBDT_V4_EXPECTED_LIBXGBOOST_SHA256
    )
    with pytest.raises(GBDTV4TrainingError, match="differs from bundle provenance"):
        validate_xgboost_runtime_provenance(
            replace(provenance, libxgboost_sha256="0" * 64)
        )

    class _FakeLiveBooster:
        def save_config(self) -> str:
            return json.dumps(
                {
                    "learner": {
                        "generic_param": {
                            "device": "cuda:1",
                            "nthread": "1",
                            "seed": str(bundle.axis_seeds[0]),
                            "seed_per_iteration": "0",
                        },
                        "gradient_booster": {
                            "gbtree_model_param": {"num_parallel_tree": "1"},
                            "gbtree_train_param": {"tree_method": "hist"},
                            "name": "gbtree",
                            "tree_train_param": {},
                        },
                        "learner_model_param": {
                            "base_score": f"[{bundle.axis_base_scores[0]}]",
                            "num_feature": "47",
                            "num_target": "1",
                        },
                        "learner_train_param": {
                            "booster": "gbtree",
                            "objective": "reg:squarederror",
                        },
                        "objective": {"name": "reg:squarederror"},
                    }
                }
            )

    with pytest.raises(GBDTV4TrainingError, match="ordinal"):
        gbdt_module._validate_live_booster_fit_config(
            _FakeLiveBooster(),
            axis_seed=bundle.axis_seeds[0],
            base_score=bundle.axis_base_scores[0],
            device="cuda:0",
        )
    with pytest.raises(GBDTV4TrainingError, match=r"cuda\[:index\]"):
        gbdt_module._resolve_fit_device("cuda", allow_cpu_for_testing=False)
