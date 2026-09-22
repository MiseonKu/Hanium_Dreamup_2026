"""Prospective GPU-v3 backbones for the 43-feature endpoint proxy.

This module is deliberately independent of the frozen GPU-v1 and robust-v2
training code.  It imports no optional runtime dependency at module import
time: callers inject the already imported :mod:`torch` module into one of the
builder functions.

Every model consumes the ordered standardized ``fusion_v1`` tensor followed
by four binary IMU-availability bits and emits three scaled XYZ values.  For
backward-compatible research calls, a 43-column tensor is accepted and an
all-available mask is appended internally.  The final 47-column layout is::

    3 encoder deltas
    4 IMUs * 6 Euler sine/cosine values
    4 IMUs * 3 accelerometer values
    4 IMUs * 1 gyroscope-RMS value
    4 IMU availability bits

Identifiers, segment/target labels, timestamps, planned coordinates, and Vive
labels are consequently absent from every forward signature.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


V3_SENSOR_FEATURE_COUNT = 43
V3_INPUT_FEATURE_COUNT = 47
V3_OUTPUT_COUNT = 3
V3_ENCODER_COUNT = 3
V3_IMU_COUNT = 4
V3_IMU_FEATURE_COUNT = 10

ENCODER_SLICE = slice(0, 3)
EULER_TRIG_SLICE = slice(3, 27)
ACCEL_SLICE = slice(27, 39)
GYRO_RMS_SLICE = slice(39, 43)
AVAILABILITY_SLICE = slice(43, 47)

# Exact ``fusion_v1`` column positions for each physical IMU.  Keeping this
# map literal prevents an accidental 43 -> 3 + 4 * 10 reshape, which would mix
# Euler, acceleration, and gyroscope families across sensors.
IMU_SENSOR_INDEX_MAP = (
    (3, 4, 5, 6, 7, 8, 27, 28, 29, 39),
    (9, 10, 11, 12, 13, 14, 30, 31, 32, 40),
    (15, 16, 17, 18, 19, 20, 33, 34, 35, 41),
    (21, 22, 23, 24, 25, 26, 36, 37, 38, 42),
)

V3_BACKBONE_NAMES = (
    "masked_mlp",
    "residual_mlp",
    "tabm_batchensemble",
    "chain_sensor_message_passing",
)


def _validate_builder_settings(
    torch: Any,
    *,
    width: int,
    depth: int,
    dropout: float,
) -> tuple[int, int, float]:
    if not hasattr(torch, "nn") or not hasattr(torch, "Tensor"):
        raise TypeError("torch must be an imported PyTorch module")
    resolved_width = int(width)
    resolved_depth = int(depth)
    resolved_dropout = float(dropout)
    if resolved_width <= 0:
        raise ValueError("width must be positive")
    if resolved_depth <= 0:
        raise ValueError("depth must be positive")
    if not math.isfinite(resolved_dropout) or not 0.0 <= resolved_dropout < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return resolved_width, resolved_depth, resolved_dropout


def _validate_feature_tensor(features: Any, *, allow_legacy_43: bool = True) -> None:
    if (
        not hasattr(features, "ndim")
        or not hasattr(features, "shape")
        or not hasattr(features, "is_floating_point")
    ):
        raise TypeError("features must be a torch.Tensor")
    if int(features.ndim) != 2:
        raise ValueError("features must have shape [batch, 47] (or legacy [batch, 43])")
    valid_counts = (
        (V3_SENSOR_FEATURE_COUNT, V3_INPUT_FEATURE_COUNT)
        if allow_legacy_43
        else (V3_SENSOR_FEATURE_COUNT,)
    )
    if int(features.shape[-1]) not in valid_counts:
        raise ValueError(
            "features must contain 47 ordered values "
            "(or 43 sensor values for the all-available compatibility path)"
        )
    if not bool(features.is_floating_point()):
        raise TypeError("features must use a floating-point dtype")
    if not bool(features.isfinite().all().item()):
        raise ValueError("features contain non-finite values")


def split_fusion_sensor_features(
    torch: Any, features: Any
) -> tuple[Any, Any]:
    """Return ``[B, 3]`` encoders and repository-ordered ``[B, 4, 10]`` IMUs.

    ``fusion_v1`` groups features by signal family, not by physical sensor.
    This helper performs the explicit regrouping used by the chain-aware
    backbone and is kept public so the layout can be audited independently.
    """

    if not hasattr(torch, "cat"):
        raise TypeError("torch must be an imported PyTorch module")
    _validate_feature_tensor(features, allow_legacy_43=False)
    batch_size = int(features.shape[0])
    encoders = features[:, ENCODER_SLICE]
    imu_features = torch.stack(
        [features[:, list(indices)] for indices in IMU_SENSOR_INDEX_MAP], dim=1
    )
    if tuple(imu_features.shape) != (
        batch_size,
        V3_IMU_COUNT,
        V3_IMU_FEATURE_COUNT,
    ):
        raise AssertionError("internal IMU feature index map changed shape")
    return encoders, imu_features


def _mask_signal_families(torch: Any, sensor_features: Any, mask: Any) -> Any:
    """Apply sensor availability without assuming sensor-major input order."""

    batch_size = int(sensor_features.shape[0])
    encoders = sensor_features[:, ENCODER_SLICE]
    euler = sensor_features[:, EULER_TRIG_SLICE].reshape(
        batch_size, V3_IMU_COUNT, 6
    ) * mask.unsqueeze(-1)
    accel = sensor_features[:, ACCEL_SLICE].reshape(
        batch_size, V3_IMU_COUNT, 3
    ) * mask.unsqueeze(-1)
    gyro_rms = sensor_features[:, GYRO_RMS_SLICE].reshape(
        batch_size, V3_IMU_COUNT, 1
    ) * mask.unsqueeze(-1)
    return torch.cat(
        (
            encoders,
            euler.reshape(batch_size, -1),
            accel.reshape(batch_size, -1),
            gyro_rms.reshape(batch_size, -1),
        ),
        dim=-1,
    )


def prepare_gpu_v3_input(torch: Any, features: Any) -> tuple[Any, Any, Any, Any]:
    """Resolve 43/47 input into masked dense features and physical groups.

    Returns ``(dense_47, encoders_3, imu_features_4x10, availability_4)``.
    For an explicit 47-column input, unavailable IMU values must already be
    exactly zero after standardization.  This fail-closed check prevents a
    nonzero missing-sensor placeholder from becoming a hidden shortcut.
    """

    _validate_feature_tensor(features)
    sensor_features = features[:, :V3_SENSOR_FEATURE_COUNT]
    if int(features.shape[-1]) == V3_SENSOR_FEATURE_COUNT:
        mask = torch.ones(
            (int(features.shape[0]), V3_IMU_COUNT),
            dtype=features.dtype,
            device=features.device,
        )
    else:
        mask = features[:, AVAILABILITY_SLICE]
        if not bool(torch.isfinite(mask).all().item()):
            raise ValueError("IMU availability mask contains non-finite values")
        if not bool(torch.logical_or(mask == 0, mask == 1).all().item()):
            raise ValueError("IMU availability mask must contain only zero or one")

    encoders, imu_features = split_fusion_sensor_features(torch, sensor_features)
    unavailable = (mask == 0).unsqueeze(-1).expand_as(imu_features)
    if bool(unavailable.any().item()):
        unavailable_values = imu_features.masked_select(unavailable)
        if not bool((unavailable_values == 0).all().item()):
            raise ValueError(
                "standardized features for unavailable IMUs must be exactly zero"
            )
    masked_sensor_features = _mask_signal_families(torch, sensor_features, mask)
    dense_input = torch.cat((masked_sensor_features, mask), dim=-1)
    masked_imus = imu_features * mask.unsqueeze(-1)
    return dense_input, encoders, masked_imus, mask


def missing_imu_reconstruction_mse(
    torch: Any,
    reconstruction: Any,
    target_standardized_imu_features: Any,
    availability_mask: Any,
) -> Any:
    """Mean squared reconstruction error over deliberately missing nodes only.

    This helper is intended for sensor-dropout research where the unmasked
    standardized 10-D target remains known during training.  A truly absent
    challenge sensor has no reconstruction label and must not be invented.
    """

    expected = tuple(target_standardized_imu_features.shape)
    if tuple(reconstruction.shape) != expected or expected[-2:] != (
        V3_IMU_COUNT,
        V3_IMU_FEATURE_COUNT,
    ):
        raise ValueError("reconstruction and target must both have shape [batch, 4, 10]")
    if tuple(availability_mask.shape) != expected[:-1]:
        raise ValueError("availability_mask must have shape [batch, 4]")
    missing = (availability_mask == 0).unsqueeze(-1).expand_as(reconstruction)
    missing_count = int(missing.sum().item())
    if missing_count == 0:
        raise ValueError("missing-node reconstruction loss requires a masked IMU")
    residual = reconstruction - target_standardized_imu_features
    return torch.square(residual).masked_select(missing).mean()


def build_masked_mlp_backbone(
    torch: Any,
    *,
    hidden_layer_sizes: Sequence[int] = (64, 64),
) -> Any:
    """Build the frozen two-layer ReLU control with explicit availability.

    This is the GPU-v3 control counterpart of the legacy 43-input MLP.  It
    keeps the frozen ``64 -> 64`` ReLU topology while accepting the v3
    47-column masked input, including the four unscaled availability bits.
    """

    if not hasattr(torch, "nn") or not hasattr(torch, "Tensor"):
        raise TypeError("torch must be an imported PyTorch module")
    hidden = tuple(int(value) for value in hidden_layer_sizes)
    if len(hidden) != 2 or any(value <= 0 for value in hidden):
        raise ValueError("hidden_layer_sizes must contain two positive widths")

    class _MaskedMLPBackbone(torch.nn.Module):
        backbone_name = "masked_mlp"
        ensemble_size = 1
        input_feature_count = V3_INPUT_FEATURE_COUNT
        sensor_feature_count = V3_SENSOR_FEATURE_COUNT
        output_count = V3_OUTPUT_COUNT

        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(V3_INPUT_FEATURE_COUNT, hidden[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden[0], hidden[1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden[1], V3_OUTPUT_COUNT),
            )

        def forward_members(self, features: Any) -> Any:
            dense_input, _, _, _ = prepare_gpu_v3_input(torch, features)
            return self.layers(dense_input).unsqueeze(1)

        def forward(self, features: Any) -> Any:
            return self.forward_members(features).mean(dim=1)

    return _MaskedMLPBackbone()


def build_residual_mlp_backbone(
    torch: Any,
    *,
    width: int = 192,
    depth: int = 4,
    expansion: int = 2,
    dropout: float = 0.0,
) -> Any:
    """Build a pre-normalized gated residual MLP baseline.

    The baseline is intentionally stronger than the frozen two-layer ReLU MLP:
    it uses a learned input projection, LayerNorm, SwiGLU-like gated residual
    blocks, and a normalized regression head.
    """

    width, depth, dropout = _validate_builder_settings(
        torch, width=width, depth=depth, dropout=dropout
    )
    expansion = int(expansion)
    if expansion <= 0:
        raise ValueError("expansion must be positive")
    hidden_width = width * expansion

    class _GatedResidualBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(width)
            self.gate_value = torch.nn.Linear(width, 2 * hidden_width)
            self.output = torch.nn.Linear(hidden_width, width)
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, hidden: Any) -> Any:
            gate, value = self.gate_value(self.norm(hidden)).chunk(2, dim=-1)
            update = torch.nn.functional.silu(gate) * value
            return hidden + self.dropout(self.output(update))

    class _ResidualMLPBackbone(torch.nn.Module):
        backbone_name = "residual_mlp"
        ensemble_size = 1
        input_feature_count = V3_INPUT_FEATURE_COUNT
        sensor_feature_count = V3_SENSOR_FEATURE_COUNT
        output_count = V3_OUTPUT_COUNT

        def __init__(self) -> None:
            super().__init__()
            self.input_norm = torch.nn.LayerNorm(V3_INPUT_FEATURE_COUNT)
            self.stem = torch.nn.Linear(V3_INPUT_FEATURE_COUNT, width)
            self.blocks = torch.nn.ModuleList(
                [_GatedResidualBlock() for _ in range(depth)]
            )
            self.head_norm = torch.nn.LayerNorm(width)
            self.head = torch.nn.Linear(width, V3_OUTPUT_COUNT)

        def forward_members(self, features: Any) -> Any:
            dense_input, _, _, _ = prepare_gpu_v3_input(torch, features)
            hidden = self.stem(self.input_norm(dense_input))
            for block in self.blocks:
                hidden = block(hidden)
            return self.head(self.head_norm(hidden)).unsqueeze(1)

        def forward(self, features: Any) -> Any:
            return self.forward_members(features).mean(dim=1)

    return _ResidualMLPBackbone()


def build_tabm_batchensemble_backbone(
    torch: Any,
    *,
    width: int = 128,
    depth: int = 3,
    expansion: int = 2,
    ensemble_size: int = 8,
    dropout: float = 0.0,
    rank1_init_std: float = 0.1,
) -> Any:
    """Build a parameter-efficient TabM-style BatchEnsemble MLP.

    Each linear transform shares one dense kernel across members while learned
    member-specific input/output rank-one factors modulate that kernel.  This
    is the BatchEnsemble mechanism used by the TabM method family; it avoids
    materializing ``ensemble_size`` independent dense MLPs while preserving
    distinct member predictions.
    """

    width, depth, dropout = _validate_builder_settings(
        torch, width=width, depth=depth, dropout=dropout
    )
    expansion = int(expansion)
    ensemble_size = int(ensemble_size)
    rank1_init_std = float(rank1_init_std)
    if expansion <= 0:
        raise ValueError("expansion must be positive")
    if ensemble_size < 2:
        raise ValueError("ensemble_size must be at least two")
    if not math.isfinite(rank1_init_std) or rank1_init_std <= 0.0:
        raise ValueError("rank1_init_std must be finite and positive")
    hidden_width = width * expansion

    class _BatchEnsembleLinear(torch.nn.Module):
        def __init__(self, input_count: int, output_count: int) -> None:
            super().__init__()
            self.input_count = int(input_count)
            self.output_count = int(output_count)
            self.shared_weight = torch.nn.Parameter(
                torch.empty(self.output_count, self.input_count)
            )
            self.member_input_scale = torch.nn.Parameter(
                torch.empty(ensemble_size, self.input_count)
            )
            self.member_output_scale = torch.nn.Parameter(
                torch.empty(ensemble_size, self.output_count)
            )
            self.member_bias = torch.nn.Parameter(
                torch.empty(ensemble_size, self.output_count)
            )
            self.reset_parameters()

        def reset_parameters(self) -> None:
            torch.nn.init.kaiming_uniform_(self.shared_weight, a=math.sqrt(5.0))
            torch.nn.init.normal_(
                self.member_input_scale, mean=1.0, std=rank1_init_std
            )
            torch.nn.init.normal_(
                self.member_output_scale, mean=1.0, std=rank1_init_std
            )
            bound = 1.0 / math.sqrt(self.input_count)
            torch.nn.init.uniform_(self.member_bias, -bound, bound)

        def forward(self, member_features: Any) -> Any:
            if int(member_features.ndim) != 3:
                raise ValueError(
                    "BatchEnsemble features must have shape [batch, members, width]"
                )
            if int(member_features.shape[1]) != ensemble_size:
                raise ValueError("BatchEnsemble member dimension changed")
            if int(member_features.shape[2]) != self.input_count:
                raise ValueError("BatchEnsemble feature width changed")
            scaled = member_features * self.member_input_scale.unsqueeze(0)
            projected = torch.einsum(
                "bki,oi->bko", scaled, self.shared_weight
            )
            return (
                projected * self.member_output_scale.unsqueeze(0)
                + self.member_bias.unsqueeze(0)
            )

    class _BatchEnsembleResidualBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(width)
            self.expand = _BatchEnsembleLinear(width, hidden_width)
            self.contract = _BatchEnsembleLinear(hidden_width, width)
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, hidden: Any) -> Any:
            update = self.expand(self.norm(hidden))
            update = torch.nn.functional.silu(update)
            return hidden + self.dropout(self.contract(update))

    class _TabMBatchEnsembleBackbone(torch.nn.Module):
        backbone_name = "tabm_batchensemble"
        input_feature_count = V3_INPUT_FEATURE_COUNT
        sensor_feature_count = V3_SENSOR_FEATURE_COUNT
        output_count = V3_OUTPUT_COUNT

        def __init__(self) -> None:
            super().__init__()
            self.ensemble_size = ensemble_size
            self.input_norm = torch.nn.LayerNorm(V3_INPUT_FEATURE_COUNT)
            self.stem = _BatchEnsembleLinear(V3_INPUT_FEATURE_COUNT, width)
            self.blocks = torch.nn.ModuleList(
                [_BatchEnsembleResidualBlock() for _ in range(depth)]
            )
            self.head_norm = torch.nn.LayerNorm(width)
            self.head = _BatchEnsembleLinear(width, V3_OUTPUT_COUNT)

        def forward_members(self, features: Any) -> Any:
            dense_input, _, _, _ = prepare_gpu_v3_input(torch, features)
            normalized = self.input_norm(dense_input)
            member_features = normalized.unsqueeze(1).expand(
                -1, self.ensemble_size, -1
            )
            hidden = torch.nn.functional.silu(self.stem(member_features))
            for block in self.blocks:
                hidden = block(hidden)
            return self.head(self.head_norm(hidden))

        def forward(self, features: Any) -> Any:
            return self.forward_members(features).mean(dim=1)

    return _TabMBatchEnsembleBackbone()


def build_chain_sensor_message_passing_backbone(
    torch: Any,
    *,
    width: int = 96,
    depth: int = 4,
    expansion: int = 2,
    dropout: float = 0.0,
) -> Any:
    """Build a four-sensor chain with three encoder-conditioned edges.

    The four IMUs become ordered sensor tokens.  Encoder deltas condition the
    three physical neighbor edges (0-1, 1-2, 2-3), and every message-passing
    round propagates distinct proximal-to-distal and distal-to-proximal
    messages before a residual node update.  The readout retains all four
    ordered tokens instead of collapsing the chain to an orderless mean.
    """

    width, depth, dropout = _validate_builder_settings(
        torch, width=width, depth=depth, dropout=dropout
    )
    expansion = int(expansion)
    if expansion <= 0:
        raise ValueError("expansion must be positive")
    hidden_width = width * expansion

    class _MessagePassingRound(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_norm = torch.nn.LayerNorm(width)
            self.forward_message = torch.nn.Sequential(
                torch.nn.Linear(2 * width, hidden_width),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_width, width),
            )
            self.backward_message = torch.nn.Sequential(
                torch.nn.Linear(2 * width, hidden_width),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_width, width),
            )
            self.update_norm = torch.nn.LayerNorm(2 * width)
            self.update = torch.nn.Sequential(
                torch.nn.Linear(2 * width, hidden_width),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_width, width),
            )
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, tokens: Any, edge_tokens: Any, node_mask: Any) -> Any:
            normalized = self.node_norm(tokens)
            messages = torch.zeros_like(normalized)
            incoming_count = torch.zeros(
                (int(tokens.shape[0]), V3_IMU_COUNT, 1),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            for edge_index in range(V3_ENCODER_COUNT):
                edge = edge_tokens[:, edge_index, :]
                proximal = normalized[:, edge_index, :]
                distal = normalized[:, edge_index + 1, :]
                proximal_available = node_mask[:, edge_index].unsqueeze(-1)
                distal_available = node_mask[:, edge_index + 1].unsqueeze(-1)
                to_distal = self.forward_message(
                    torch.cat((proximal, edge), dim=-1)
                ) * proximal_available
                to_proximal = self.backward_message(
                    torch.cat((distal, edge), dim=-1)
                ) * distal_available
                messages[:, edge_index + 1, :] = (
                    messages[:, edge_index + 1, :] + to_distal
                )
                messages[:, edge_index, :] = (
                    messages[:, edge_index, :] + to_proximal
                )
                incoming_count[:, edge_index + 1, :] = (
                    incoming_count[:, edge_index + 1, :] + proximal_available
                )
                incoming_count[:, edge_index, :] = (
                    incoming_count[:, edge_index, :] + distal_available
                )
            messages = messages / incoming_count.clamp_min(1.0)
            update_input = self.update_norm(torch.cat((normalized, messages), dim=-1))
            return tokens + self.dropout(self.update(update_input))

    class _ChainSensorMessagePassingBackbone(torch.nn.Module):
        backbone_name = "chain_sensor_message_passing"
        ensemble_size = 1
        input_feature_count = V3_INPUT_FEATURE_COUNT
        sensor_feature_count = V3_SENSOR_FEATURE_COUNT
        output_count = V3_OUTPUT_COUNT
        sensor_count = V3_IMU_COUNT
        edge_count = V3_ENCODER_COUNT

        def __init__(self) -> None:
            super().__init__()
            self.imu_norm = torch.nn.LayerNorm(V3_IMU_FEATURE_COUNT)
            self.imu_stem = torch.nn.Linear(V3_IMU_FEATURE_COUNT, width)
            self.sensor_position = torch.nn.Parameter(
                torch.empty(V3_IMU_COUNT, width)
            )
            self.edge_stem = torch.nn.Sequential(
                torch.nn.Linear(1, width),
                torch.nn.SiLU(),
                torch.nn.Linear(width, width),
            )
            self.rounds = torch.nn.ModuleList(
                [_MessagePassingRound() for _ in range(depth)]
            )
            self.encoder_summary = torch.nn.Sequential(
                torch.nn.LayerNorm(V3_ENCODER_COUNT),
                torch.nn.Linear(V3_ENCODER_COUNT, width),
                torch.nn.SiLU(),
            )
            self.availability_summary = torch.nn.Sequential(
                torch.nn.Linear(V3_IMU_COUNT, width),
                torch.nn.SiLU(),
            )
            self.readout = torch.nn.Sequential(
                torch.nn.LayerNorm(V3_IMU_COUNT * width + 2 * width),
                torch.nn.Linear(V3_IMU_COUNT * width + 2 * width, 2 * width),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(2 * width, V3_OUTPUT_COUNT),
            )
            self.reconstruction_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width),
                torch.nn.Linear(width, V3_IMU_FEATURE_COUNT),
            )
            torch.nn.init.normal_(self.sensor_position, mean=0.0, std=0.02)

        def _encode(self, features: Any) -> tuple[Any, Any, Any]:
            _, encoders, imu_features, mask = prepare_gpu_v3_input(torch, features)
            tokens = self.imu_stem(self.imu_norm(imu_features))
            tokens = tokens + self.sensor_position.unsqueeze(0)
            edge_tokens = self.edge_stem(encoders.unsqueeze(-1))
            for message_round in self.rounds:
                tokens = message_round(tokens, edge_tokens, mask)
            return tokens, encoders, mask

        def _predict_members(self, tokens: Any, encoders: Any, mask: Any) -> Any:
            prediction_tokens = tokens * mask.unsqueeze(-1)
            ordered_tokens = prediction_tokens.reshape(int(tokens.shape[0]), -1)
            encoder_summary = self.encoder_summary(encoders)
            availability_summary = self.availability_summary(mask)
            prediction = self.readout(
                torch.cat(
                    (ordered_tokens, encoder_summary, availability_summary), dim=-1
                )
            )
            return prediction.unsqueeze(1)

        def forward_members(self, features: Any) -> Any:
            tokens, encoders, mask = self._encode(features)
            return self._predict_members(tokens, encoders, mask)

        def forward_with_aux(self, features: Any) -> tuple[Any, Any]:
            """Return XYZ and per-IMU standardized 10-D reconstructions."""

            tokens, encoders, mask = self._encode(features)
            members = self._predict_members(tokens, encoders, mask)
            reconstruction = self.reconstruction_head(tokens)
            return members.mean(dim=1), reconstruction

        def forward(self, features: Any) -> Any:
            return self.forward_members(features).mean(dim=1)

    return _ChainSensorMessagePassingBackbone()


def build_gpu_v3_backbone(torch: Any, name: str, **settings: Any) -> Any:
    """Build one named prospective backbone without touching frozen runners."""

    builders = {
        "masked_mlp": build_masked_mlp_backbone,
        "residual_mlp": build_residual_mlp_backbone,
        "tabm_batchensemble": build_tabm_batchensemble_backbone,
        "chain_sensor_message_passing": build_chain_sensor_message_passing_backbone,
    }
    try:
        builder = builders[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown GPU-v3 backbone {name!r}; choose from {V3_BACKBONE_NAMES}"
        ) from exc
    return builder(torch, **settings)
