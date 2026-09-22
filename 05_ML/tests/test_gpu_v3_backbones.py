from __future__ import annotations

from copy import deepcopy

import pytest

from imu_vive_learning.gpu_v3_backbones import (
    IMU_SENSOR_INDEX_MAP,
    V3_BACKBONE_NAMES,
    build_chain_sensor_message_passing_backbone,
    build_gpu_v3_backbone,
    build_masked_mlp_backbone,
    build_residual_mlp_backbone,
    build_tabm_batchensemble_backbone,
    missing_imu_reconstruction_mse,
    prepare_gpu_v3_input,
    split_fusion_sensor_features,
)


torch = pytest.importorskip("torch")


SMALL_BUILDERS = {
    "masked_mlp": lambda: build_masked_mlp_backbone(
        torch, hidden_layer_sizes=(24, 16)
    ),
    "residual_mlp": lambda: build_residual_mlp_backbone(
        torch, width=24, depth=2
    ),
    "tabm_batchensemble": lambda: build_tabm_batchensemble_backbone(
        torch, width=20, depth=2, ensemble_size=4
    ),
    "chain_sensor_message_passing": lambda: (
        build_chain_sensor_message_passing_backbone(torch, width=16, depth=2)
    ),
}


def _features() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260824)
    return torch.randn(6, 43, generator=generator)


def _masked_features(
    sensor_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    complete = _features()
    _, reconstruction_target = split_fusion_sensor_features(torch, complete)
    masked = complete.clone()
    masked[:, 3 + 6 * sensor_index : 3 + 6 * (sensor_index + 1)] = 0.0
    masked[:, 27 + 3 * sensor_index : 27 + 3 * (sensor_index + 1)] = 0.0
    masked[:, 39 + sensor_index] = 0.0
    availability = torch.ones(len(masked), 4, dtype=masked.dtype)
    availability[:, sensor_index] = 0.0
    return torch.cat((masked, availability), dim=1), reconstruction_target, availability


@pytest.mark.parametrize("name", V3_BACKBONE_NAMES)
def test_backbone_forward_backward_is_finite_and_seed_deterministic(name: str) -> None:
    features = _features()

    torch.manual_seed(913)
    first = SMALL_BUILDERS[name]()
    first.train()
    first_prediction = first(features)
    first_loss = first_prediction.square().mean()
    first_loss.backward()

    assert first_prediction.shape == (6, 3)
    assert torch.isfinite(first_prediction).all()
    assert torch.isfinite(first_loss)
    first_gradients = {
        key: parameter.grad.detach().clone()
        for key, parameter in first.named_parameters()
        if parameter.grad is not None
    }
    assert first_gradients
    assert all(torch.isfinite(value).all() for value in first_gradients.values())

    torch.manual_seed(913)
    second = SMALL_BUILDERS[name]()
    second.train()
    second_prediction = second(features)
    second_prediction.square().mean().backward()
    second_gradients = {
        key: parameter.grad.detach().clone()
        for key, parameter in second.named_parameters()
        if parameter.grad is not None
    }

    assert torch.equal(first_prediction, second_prediction)
    assert tuple(first_gradients) == tuple(second_gradients)
    assert all(
        torch.equal(first_gradients[key], second_gradients[key])
        for key in first_gradients
    )


@pytest.mark.parametrize("name", V3_BACKBONE_NAMES)
def test_state_dict_roundtrip_preserves_exact_cpu_predictions(name: str) -> None:
    features = _features()
    torch.manual_seed(17)
    original = SMALL_BUILDERS[name]().eval()
    with torch.inference_mode():
        expected = original(features)

    state = deepcopy(original.state_dict())
    torch.manual_seed(99)
    restored = SMALL_BUILDERS[name]().eval()
    restored.load_state_dict(state, strict=True)
    with torch.inference_mode():
        actual = restored(features)

    assert tuple(state) == tuple(restored.state_dict())
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("name", V3_BACKBONE_NAMES)
def test_public_prediction_is_exact_member_mean(name: str) -> None:
    torch.manual_seed(71)
    model = SMALL_BUILDERS[name]().eval()
    features = _features()
    with torch.inference_mode():
        members = model.forward_members(features)
        prediction = model(features)

    expected_members = 4 if name == "tabm_batchensemble" else 1
    assert members.shape == (6, expected_members, 3)
    assert torch.isfinite(members).all()
    assert torch.equal(prediction, members.mean(dim=1))


def test_tabm_uses_shared_dense_weights_but_members_are_distinct() -> None:
    torch.manual_seed(113)
    model = SMALL_BUILDERS["tabm_batchensemble"]().eval()
    with torch.inference_mode():
        members = model.forward_members(_features())

    assert model.stem.shared_weight.shape == (20, 47)
    assert model.stem.member_input_scale.shape == (4, 47)
    assert model.stem.member_output_scale.shape == (4, 20)
    assert model.stem.member_bias.shape == (4, 20)
    assert not torch.equal(members[:, 0, :], members[:, 1, :])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    dense_kernels = [model.stem, *(
        layer
        for block in model.blocks
        for layer in (block.expand, block.contract)
    ), model.head]
    independent_dense_count = 4 * sum(
        layer.input_count * layer.output_count + layer.output_count
        for layer in dense_kernels
    )
    # LayerNorm is additional shared state, yet the complete TabM backbone is
    # still smaller than four independent copies of its dense transforms.
    assert parameter_count < independent_dense_count


def test_chain_layout_regroups_signal_families_into_four_ten_value_sensors() -> None:
    features = torch.arange(43, dtype=torch.float32).reshape(1, 43)
    encoders, sensors = split_fusion_sensor_features(torch, features)

    assert torch.equal(encoders, torch.tensor([[0.0, 1.0, 2.0]]))
    assert sensors.shape == (1, 4, 10)
    assert IMU_SENSOR_INDEX_MAP[0] == (3, 4, 5, 6, 7, 8, 27, 28, 29, 39)
    assert IMU_SENSOR_INDEX_MAP[3] == (21, 22, 23, 24, 25, 26, 36, 37, 38, 42)
    assert torch.equal(
        sensors[0, 0],
        torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 27.0, 28.0, 29.0, 39.0]),
    )
    assert torch.equal(
        sensors[0, 3],
        torch.tensor(
            [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 36.0, 37.0, 38.0, 42.0]
        ),
    )


def test_backbones_accept_only_finite_float_batch_by_43_or_47_tensors() -> None:
    model = SMALL_BUILDERS["residual_mlp"]()
    with pytest.raises(ValueError, match=r"\[batch, 47\]"):
        model(torch.zeros(43))
    with pytest.raises(ValueError, match="47 ordered"):
        model(torch.zeros(2, 44))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 47, dtype=torch.int64))
    nonfinite = torch.zeros(2, 47)
    nonfinite[0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        model(nonfinite)


def test_named_factory_is_closed_to_four_prospective_backbones() -> None:
    for name in V3_BACKBONE_NAMES:
        settings = (
            {"hidden_layer_sizes": (12, 12)}
            if name == "masked_mlp"
            else {"width": 12, "depth": 1}
        )
        if name == "tabm_batchensemble":
            settings["ensemble_size"] = 3
        model = build_gpu_v3_backbone(torch, name, **settings)
        assert model.backbone_name == name
        assert model.input_feature_count == 47
        assert model.sensor_feature_count == 43
        assert model.output_count == 3

    with pytest.raises(ValueError, match="unknown GPU-v3 backbone"):
        build_gpu_v3_backbone(torch, "uses_segment_id")


@pytest.mark.parametrize("name", V3_BACKBONE_NAMES)
def test_legacy_43_input_is_exactly_all_available_47_input(name: str) -> None:
    torch.manual_seed(81)
    model = SMALL_BUILDERS[name]().eval()
    features = _features()
    explicit = torch.cat((features, torch.ones(len(features), 4)), dim=1)
    with torch.inference_mode():
        legacy_prediction = model(features)
        explicit_prediction = model(explicit)
    assert torch.equal(legacy_prediction, explicit_prediction)


@pytest.mark.parametrize("name", V3_BACKBONE_NAMES)
@pytest.mark.parametrize("missing_sensor", range(4))
def test_every_backbone_accepts_zero_imputed_imu_availability_mask(
    name: str, missing_sensor: int
) -> None:
    torch.manual_seed(191)
    model = SMALL_BUILDERS[name]().eval()
    masked, _, _ = _masked_features(missing_sensor)
    with torch.inference_mode():
        prediction = model(masked)
        members = model.forward_members(masked)
    assert prediction.shape == (6, 3)
    assert members.shape[0] == 6
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(members).all()


def test_explicit_mask_requires_binary_bits_and_zero_standardized_values() -> None:
    features = _features()
    nonbinary = torch.cat((features, torch.ones(len(features), 4)), dim=1)
    nonbinary[:, 0 + 43] = 0.5
    with pytest.raises(ValueError, match="only zero or one"):
        prepare_gpu_v3_input(torch, nonbinary)

    nonzero_missing = torch.cat((features, torch.ones(len(features), 4)), dim=1)
    nonzero_missing[:, 43] = 0.0
    with pytest.raises(ValueError, match="must be exactly zero"):
        prepare_gpu_v3_input(torch, nonzero_missing)


def test_prepare_input_preserves_signal_family_permutation_and_appends_mask() -> None:
    features = torch.arange(43, dtype=torch.float32).reshape(1, 43)
    dense, encoders, sensors, availability = prepare_gpu_v3_input(torch, features)
    assert dense.shape == (1, 47)
    assert torch.equal(dense[:, :43], features)
    assert torch.equal(dense[:, 43:], torch.ones(1, 4))
    assert torch.equal(encoders, features[:, :3])
    assert torch.equal(
        sensors[0, 2],
        torch.tensor(
            [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 33.0, 34.0, 35.0, 41.0]
        ),
    )
    assert torch.equal(availability, torch.ones(1, 4))


def test_chain_masks_prediction_node_and_returns_missing_node_reconstruction() -> None:
    masked, target, availability = _masked_features(sensor_index=0)
    torch.manual_seed(211)
    model = SMALL_BUILDERS["chain_sensor_message_passing"]().train()
    prediction, reconstruction = model.forward_with_aux(masked)

    assert prediction.shape == (6, 3)
    assert reconstruction.shape == (6, 4, 10)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(reconstruction).all()
    loss = prediction.square().mean() + missing_imu_reconstruction_mse(
        torch, reconstruction, target, availability
    )
    loss.backward()
    assert model.reconstruction_head[1].weight.grad is not None
    assert torch.isfinite(model.reconstruction_head[1].weight.grad).all()

    model.eval()
    with torch.inference_mode():
        tokens, encoders, mask = model._encode(masked)
        baseline = model._predict_members(tokens, encoders, mask)
        perturbed = tokens.clone()
        perturbed[:, 0, :] = 1e6
        changed_missing_latent = model._predict_members(perturbed, encoders, mask)
    assert torch.equal(baseline, changed_missing_latent)


def test_missing_reconstruction_loss_rejects_no_missing_node() -> None:
    prediction = torch.zeros(2, 4, 10)
    target = torch.ones(2, 4, 10)
    with pytest.raises(ValueError, match="requires a masked IMU"):
        missing_imu_reconstruction_mse(
            torch, prediction, target, torch.ones(2, 4)
        )
