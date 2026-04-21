"""Persisted serving-profile resolution, workload classification, and request-time application helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import CapabilityName, GenerateAttachment, GenerateMessage, ModelManifest, ModelModality, RuntimeContract
from lewlm.storage import MetadataStore


SERVING_PROFILE_SETTING_KEYS = (
    "runtime_policy",
    "continuous_batch_window_milliseconds",
    "continuous_batch_max_batch_size",
    "kv_cache_page_size",
    "kv_cache_max_pages",
    "kv_cache_quantization_bits",
    "prefill_token_batch_size",
    "mlx_graph_compile_enabled",
    "mlx_attention_kernel_mode",
)

_ServingProfileValue = int | float | str | bool | None
_TEXT_ONLY_WORKLOAD_CLASS = "text_only"
_TEXT_ONLY_MULTIMODAL_WORKLOAD_CLASS = "text_only_multimodal"
_SINGLE_IMAGE_WORKLOAD_CLASS = "single_image"
_REPEATED_IMAGE_WORKLOAD_CLASS = "repeated_image"
_FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS = "frame_bundle_video"
_AUDIO_CONDITIONED_WORKLOAD_CLASS = "audio_conditioned"
SERVING_PROFILE_WORKLOAD_CLASS_CHOICES = (
    _TEXT_ONLY_WORKLOAD_CLASS,
    _TEXT_ONLY_MULTIMODAL_WORKLOAD_CLASS,
    _SINGLE_IMAGE_WORKLOAD_CLASS,
    _REPEATED_IMAGE_WORKLOAD_CLASS,
    _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
    _AUDIO_CONDITIONED_WORKLOAD_CLASS,
)
_TEXT_LIKE_WORKLOAD_CLASSES = frozenset(
    {
        _TEXT_ONLY_WORKLOAD_CLASS,
        _TEXT_ONLY_MULTIMODAL_WORKLOAD_CLASS,
    },
)
_IMAGE_WORKLOAD_CLASSES = frozenset(
    {
        _SINGLE_IMAGE_WORKLOAD_CLASS,
        _REPEATED_IMAGE_WORKLOAD_CLASS,
        _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
    },
)
_ATTACHMENT_WORKLOAD_CLASSES = frozenset(
    {
        _SINGLE_IMAGE_WORKLOAD_CLASS,
        _REPEATED_IMAGE_WORKLOAD_CLASS,
        _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
        _AUDIO_CONDITIONED_WORKLOAD_CLASS,
    },
)
_WORKLOAD_CLASS_ALIASES = {
    "text": _TEXT_ONLY_WORKLOAD_CLASS,
    "text_only": _TEXT_ONLY_WORKLOAD_CLASS,
    "text-only": _TEXT_ONLY_WORKLOAD_CLASS,
    "text_only_multimodal": _TEXT_ONLY_MULTIMODAL_WORKLOAD_CLASS,
    "text-only-multimodal": _TEXT_ONLY_MULTIMODAL_WORKLOAD_CLASS,
    "single_image": _SINGLE_IMAGE_WORKLOAD_CLASS,
    "single-image": _SINGLE_IMAGE_WORKLOAD_CLASS,
    "repeated_image": _REPEATED_IMAGE_WORKLOAD_CLASS,
    "repeated-image": _REPEATED_IMAGE_WORKLOAD_CLASS,
    "frame_bundle_video": _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
    "frame-bundle-video": _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
    "frame_bundle": _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
    "frame-bundle": _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
    "video": _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
    "audio_conditioned": _AUDIO_CONDITIONED_WORKLOAD_CLASS,
    "audio-conditioned": _AUDIO_CONDITIONED_WORKLOAD_CLASS,
    "audio": _AUDIO_CONDITIONED_WORKLOAD_CLASS,
}


class ServingProfileRejectedSetting(BaseModel):
    requested_value: _ServingProfileValue = None
    reason: str


class ServingProfileApplication(BaseModel):
    status: Literal["selected", "disabled", "not_found", "runtime_mismatch", "unavailable"]
    source: str = "persisted_autotune"
    capability: str = CapabilityName.CHAT.value
    workload_class: str = _TEXT_ONLY_WORKLOAD_CLASS
    profile_id: str | None = None
    runtime: str | None = None
    reason: str
    recommendation_reason: str | None = None
    recommended_at: datetime | None = None
    artifact_id: str | None = None
    accepted_settings: dict[str, _ServingProfileValue] = Field(default_factory=dict)
    rejected_settings: dict[str, ServingProfileRejectedSetting] = Field(default_factory=dict)
    effective_settings: dict[str, _ServingProfileValue] = Field(default_factory=dict)


def serving_profile_effective_settings(settings: LewLMSettings) -> dict[str, _ServingProfileValue]:
    return {key: getattr(settings, key) for key in SERVING_PROFILE_SETTING_KEYS}


def normalize_serving_profile_workload_class(workload_class: str | None) -> str | None:
    if workload_class is None:
        return None
    normalized = workload_class.strip().casefold().replace(" ", "_")
    if not normalized:
        return None
    return _WORKLOAD_CLASS_ALIASES.get(normalized, normalized)


def default_serving_profile_workload_class(*, manifest: ModelManifest | None = None) -> str:
    if manifest is None:
        return _TEXT_ONLY_WORKLOAD_CLASS
    if any(modality in {ModelModality.VISION, ModelModality.AUDIO, ModelModality.MULTIMODAL} for modality in manifest.modality):
        return _TEXT_ONLY_MULTIMODAL_WORKLOAD_CLASS
    return _TEXT_ONLY_WORKLOAD_CLASS


def supported_serving_profile_workload_classes(*, manifest: ModelManifest) -> tuple[str, ...]:
    supported_classes: list[str] = []
    for workload_class in (
        default_serving_profile_workload_class(manifest=manifest),
        _SINGLE_IMAGE_WORKLOAD_CLASS,
        _REPEATED_IMAGE_WORKLOAD_CLASS,
        _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS,
        _AUDIO_CONDITIONED_WORKLOAD_CLASS,
    ):
        if workload_class in supported_classes:
            continue
        if serving_profile_supports_workload(manifest=manifest, workload_class=workload_class):
            supported_classes.append(workload_class)
    return tuple(supported_classes)


def serving_profile_workload_class(
    *,
    messages: list[GenerateMessage] | tuple[GenerateMessage, ...],
    manifest: ModelManifest | None = None,
    workload_class_hint: str | None = None,
) -> str:
    normalized_hint = normalize_serving_profile_workload_class(workload_class_hint)
    if normalized_hint is not None:
        return normalized_hint
    attachments = [
        attachment
        for message in messages
        for attachment in message.attachments
    ]
    attachment_hint = _attachment_workload_class_hint(attachments)
    if attachment_hint is not None:
        return attachment_hint
    if not attachments:
        return default_serving_profile_workload_class(manifest=manifest)
    if any(_is_frame_bundle_or_video_attachment(attachment) for attachment in attachments):
        return _FRAME_BUNDLE_VIDEO_WORKLOAD_CLASS
    if any(_is_audio_attachment(attachment) for attachment in attachments):
        return _AUDIO_CONDITIONED_WORKLOAD_CLASS
    image_attachments = [attachment for attachment in attachments if _is_image_attachment(attachment)]
    if image_attachments:
        if len(image_attachments) > 1 or _contains_repeated_image_hint(image_attachments):
            return _REPEATED_IMAGE_WORKLOAD_CLASS
        return _SINGLE_IMAGE_WORKLOAD_CLASS
    return default_serving_profile_workload_class(manifest=manifest)


def serving_profile_supports_workload(*, manifest: ModelManifest, workload_class: str | None) -> bool:
    normalized = normalize_serving_profile_workload_class(workload_class)
    if normalized is None:
        return True
    modalities = set(manifest.modality)
    if normalized == _TEXT_ONLY_MULTIMODAL_WORKLOAD_CLASS:
        return bool(modalities & {ModelModality.VISION, ModelModality.AUDIO, ModelModality.MULTIMODAL})
    if normalized in _IMAGE_WORKLOAD_CLASSES:
        return bool(modalities & {ModelModality.VISION, ModelModality.MULTIMODAL})
    if normalized == _AUDIO_CONDITIONED_WORKLOAD_CLASS:
        return bool(modalities & {ModelModality.AUDIO, ModelModality.MULTIMODAL})
    return True


def serving_profile_requires_materialization(
    *,
    profile: ServingProfileApplication | None,
    settings: LewLMSettings,
) -> bool:
    if profile is None or profile.status != "selected" or not profile.accepted_settings:
        return False
    return any(getattr(settings, key) != value for key, value in profile.accepted_settings.items())


def resolve_serving_profile_application(
    *,
    settings: LewLMSettings,
    metadata_store: MetadataStore | None,
    host_platform: dict[str, object] | None,
    runtime: RuntimeContract,
    model_id: str,
    request_capability: CapabilityName,
    apply_serving_profile: bool,
    stored_capability: str = CapabilityName.CHAT.value,
    workload_class: str | None = None,
) -> ServingProfileApplication:
    current_effective_settings = serving_profile_effective_settings(settings)
    normalized_workload_class = normalize_serving_profile_workload_class(workload_class) or _TEXT_ONLY_WORKLOAD_CLASS
    if not apply_serving_profile:
        return ServingProfileApplication(
            status="disabled",
            capability=stored_capability,
            workload_class=normalized_workload_class,
            runtime=runtime.name,
            reason="Request-level serving-profile adoption was disabled explicitly.",
            effective_settings=current_effective_settings,
        )
    if metadata_store is None or host_platform is None:
        return ServingProfileApplication(
            status="unavailable",
            capability=stored_capability,
            workload_class=normalized_workload_class,
            runtime=runtime.name,
            reason="Persisted serving profiles are unavailable in this service context.",
            effective_settings=current_effective_settings,
        )
    profile_payload = metadata_store.get_serving_profile(
        model_id=model_id,
        capability=stored_capability,
        host_platform=host_platform,
        runtime_name=runtime.name,
        workload_class=normalized_workload_class,
    )
    if profile_payload is None:
        return ServingProfileApplication(
            status="not_found",
            capability=stored_capability,
            workload_class=normalized_workload_class,
            runtime=runtime.name,
            reason=(
                "No persisted serving profile is available for this host/model/runtime/workload tuple."
            ),
            effective_settings=current_effective_settings,
        )

    stored_runtime = profile_payload.get("runtime")
    stored_workload_class = normalize_serving_profile_workload_class(_string_or_none(profile_payload.get("workload_class")))
    resolved_workload_class = stored_workload_class or normalized_workload_class
    settings_overrides = profile_payload.get("settings_overrides")
    if not isinstance(settings_overrides, dict):
        settings_overrides = {}
    sanitized_overrides = {
        key: value
        for key, value in settings_overrides.items()
        if key in SERVING_PROFILE_SETTING_KEYS
    }
    rejected_settings: dict[str, ServingProfileRejectedSetting] = {}
    if isinstance(stored_runtime, str) and stored_runtime and stored_runtime != runtime.name:
        for key, value in sanitized_overrides.items():
            rejected_settings[key] = ServingProfileRejectedSetting(
                requested_value=value,
                reason=f"Profile was benchmarked against runtime `{stored_runtime}`, but this request routed to `{runtime.name}`.",
            )
        return ServingProfileApplication(
            status="runtime_mismatch",
            capability=stored_capability,
            workload_class=resolved_workload_class,
            profile_id=_string_or_none(profile_payload.get("profile_id")),
            runtime=runtime.name,
            reason="Persisted profile runtime does not match the active routed runtime.",
            recommendation_reason=_string_or_none(profile_payload.get("reason")),
            recommended_at=profile_payload.get("recommended_at"),
            artifact_id=_artifact_id(profile_payload),
            rejected_settings=rejected_settings,
            effective_settings=current_effective_settings,
        )

    runtime_features = runtime.performance_feature_snapshot()
    accepted_settings: dict[str, _ServingProfileValue] = {}
    for key, value in sanitized_overrides.items():
        rejection_reason = _setting_rejection_reason(
            key=key,
            request_capability=request_capability,
            runtime=runtime,
            runtime_features=runtime_features,
        )
        if rejection_reason is None:
            accepted_settings[key] = value
            continue
        rejected_settings[key] = ServingProfileRejectedSetting(
            requested_value=value,
            reason=rejection_reason,
        )

    effective_settings = current_effective_settings
    if accepted_settings:
        effective_settings = serving_profile_effective_settings(settings.with_updates(**accepted_settings))
    return ServingProfileApplication(
        status="selected",
        capability=stored_capability,
        workload_class=resolved_workload_class,
        profile_id=_string_or_none(profile_payload.get("profile_id")),
        runtime=runtime.name,
        reason=_selection_reason(accepted_settings=accepted_settings, rejected_settings=rejected_settings),
        recommendation_reason=_string_or_none(profile_payload.get("reason")),
        recommended_at=profile_payload.get("recommended_at"),
        artifact_id=_artifact_id(profile_payload),
        accepted_settings=accepted_settings,
        rejected_settings=rejected_settings,
        effective_settings=effective_settings,
    )


def _setting_rejection_reason(
    *,
    key: str,
    request_capability: CapabilityName,
    runtime: RuntimeContract,
    runtime_features: dict[str, object],
) -> str | None:
    if key == "runtime_policy":
        return None
    if key in {"continuous_batch_window_milliseconds", "continuous_batch_max_batch_size"}:
        if runtime.supports_continuous_batching(request_capability):
            return None
        return f"Runtime `{runtime.name}` does not advertise continuous batching for `{request_capability.value}` requests."
    if key in {"kv_cache_page_size", "kv_cache_max_pages"}:
        if _feature_supported(runtime_features, "paged_kv_cache"):
            return None
        return f"Runtime `{runtime.name}` does not advertise paged KV-cache support."
    if key == "kv_cache_quantization_bits":
        if _feature_supported(runtime_features, "kv_cache_quantization"):
            return None
        return f"Runtime `{runtime.name}` does not advertise KV-cache quantization support."
    if key == "prefill_token_batch_size":
        if _feature_supported(runtime_features, "prefill_optimization"):
            return None
        return f"Runtime `{runtime.name}` does not advertise prefill optimization support."
    if key == "mlx_graph_compile_enabled":
        if _feature_supported(runtime_features, "graph_compilation"):
            return None
        return f"Runtime `{runtime.name}` does not advertise graph-compilation support."
    if key == "mlx_attention_kernel_mode":
        if _feature_supported(runtime_features, "attention_kernel_acceleration"):
            return None
        return f"Runtime `{runtime.name}` does not advertise accelerated attention-kernel support."
    return f"Setting `{key}` is not a recognized serving-profile override."


def _selection_reason(
    *,
    accepted_settings: dict[str, _ServingProfileValue],
    rejected_settings: dict[str, ServingProfileRejectedSetting],
) -> str:
    if accepted_settings and rejected_settings:
        return "Applied the persisted profile and rejected runtime-unsupported settings explicitly."
    if accepted_settings:
        return "Applied the persisted serving-profile overrides selected by autotuning."
    if rejected_settings:
        return "Persisted profile was found, but every recommended override was rejected for the active runtime."
    return "Persisted profile was selected, but it does not override any request-time serving settings."


def _feature_supported(runtime_features: dict[str, object], feature_name: str) -> bool:
    payload = runtime_features.get(feature_name)
    return isinstance(payload, dict) and bool(payload.get("supported"))


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _artifact_id(payload: dict[str, object]) -> str | None:
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        return None
    artifact_id = artifact.get("artifact_id")
    return artifact_id if isinstance(artifact_id, str) and artifact_id else None


def is_text_like_workload_class(workload_class: str | None) -> bool:
    normalized = normalize_serving_profile_workload_class(workload_class)
    return normalized in _TEXT_LIKE_WORKLOAD_CLASSES or normalized is None


def is_attachment_workload_class(workload_class: str | None) -> bool:
    normalized = normalize_serving_profile_workload_class(workload_class)
    return normalized in _ATTACHMENT_WORKLOAD_CLASSES


def _attachment_workload_class_hint(attachments: list[GenerateAttachment]) -> str | None:
    for attachment in attachments:
        metadata = attachment.metadata
        if not isinstance(metadata, dict):
            continue
        hint = normalize_serving_profile_workload_class(_string_or_none(metadata.get("serving_profile_workload_class")))
        if hint is not None:
            return hint
    return None


def _contains_repeated_image_hint(attachments: list[GenerateAttachment]) -> bool:
    for attachment in attachments:
        metadata = attachment.metadata
        if isinstance(metadata, dict) and bool(metadata.get("repeated_image")):
            return True
    identity_keys: list[str] = []
    for attachment in attachments:
        metadata = attachment.metadata
        if isinstance(metadata, dict):
            for key in ("content_sha256", "cache_key", "source_locator"):
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    identity_keys.append(value)
                    break
            else:
                if attachment.source_path:
                    identity_keys.append(attachment.source_path)
                else:
                    identity_keys.append(attachment.name)
            continue
        identity_keys.append(attachment.source_path or attachment.name)
    return len(identity_keys) > len(set(identity_keys))


def _is_image_attachment(attachment: GenerateAttachment) -> bool:
    if attachment.attachment_type == "image":
        return True
    return isinstance(attachment.media_type, str) and attachment.media_type.startswith("image/")


def _is_audio_attachment(attachment: GenerateAttachment) -> bool:
    if attachment.attachment_type == "audio":
        return True
    return isinstance(attachment.media_type, str) and attachment.media_type.startswith("audio/")


def _is_frame_bundle_or_video_attachment(attachment: GenerateAttachment) -> bool:
    if isinstance(attachment.media_type, str) and attachment.media_type.startswith("video/"):
        return True
    metadata = attachment.metadata
    if isinstance(metadata, dict):
        source_kind = metadata.get("source_kind")
        if source_kind in {"frame_bundle", "video"}:
            return True
        frame_count = metadata.get("frame_count")
        if isinstance(frame_count, int) and frame_count > 1:
            return True
    if not attachment.source_path:
        return False
    return Path(attachment.source_path).expanduser().is_dir()
