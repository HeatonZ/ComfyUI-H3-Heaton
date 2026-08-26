"""FeiHou Easy H3 bundle bridge."""

from .registry import AdapterBase, register_adapter
from ..core.bundle import HTBundle

try:
    import nodes as _comfy_nodes
    import importlib
    _feihou = importlib.import_module("ComfyUI-FeiHou-Easy-H3.nodes")
except ImportError:
    _comfy_nodes = None
    _feihou = None


@register_adapter
class FeiHouAdapter(AdapterBase):
    name = "FeiHou MINIMAX_H3_BUNDLE"
    target_package = "ComfyUI-FeiHou-Easy-H3"
    external_type = "MINIMAX_H3_BUNDLE"

    @classmethod
    def available(cls):
        return _feihou is not None

    @classmethod
    def bundle_to_external(cls, ht_bundle):
        if _feihou is None:
            raise ImportError("ComfyUI-FeiHou-Easy-H3 is unavailable")
        bundle_type = getattr(_feihou, "MiniMaxH3Bundle", None)
        if bundle_type is None:
            raise RuntimeError("FeiHou nodes.py exposes no MiniMaxH3Bundle")
        info = ht_bundle.base_info
        none_model = _feihou.NONE_MODEL
        return bundle_type(
            fl2va_model_name=info.get("fl2va_model_name", none_model),
            ref2va_model_name=info.get("ref2va_model_name", none_model),
            clip_name=info.get("clip_name", ""),
            video_vae_name=info.get("video_vae_name", ""),
            audio_vae_name=info.get("audio_vae_name", ""),
            clip=ht_bundle.clip, video_vae=ht_bundle.video_vae,
            audio_vae=ht_bundle.audio_vae,
            lora_stack=tuple(ht_bundle.lora_stack),
            fl2va_model_obj=ht_bundle.unet_model,
            ref2va_model_obj=ht_bundle.unet_model,
            second_sampling_enabled=info.get("second_sampling_enabled", False),
            second_fl2va_model_name=info.get("second_fl2va_model_name", none_model),
            second_ref2va_model_name=info.get("second_ref2va_model_name", none_model),
            second_sampling_use_lora=info.get("second_sampling_use_lora", True),
            second_lora_stack=tuple(info.get("second_lora_stack", ())),
        )

    @classmethod
    def bundle_from_external(cls, ext_bundle):
        if not isinstance(ext_bundle, getattr(_feihou, "MiniMaxH3Bundle")):
            raise ValueError("expected FeiHou MINIMAX_H3_BUNDLE")
        return HTBundle(
            unet_model=ext_bundle.fl2va_model_obj or ext_bundle.ref2va_model_obj,
            clip=ext_bundle.clip, video_vae=ext_bundle.video_vae,
            audio_vae=ext_bundle.audio_vae,
            lora_stack=tuple(ext_bundle.lora_stack),
            second_model=None,
            loaded_report="Imported from ComfyUI-FeiHou-Easy-H3",
            base_info={
                "fl2va_model_name": ext_bundle.fl2va_model_name,
                "ref2va_model_name": ext_bundle.ref2va_model_name,
                "clip_name": ext_bundle.clip_name,
                "video_vae_name": ext_bundle.video_vae_name,
                "audio_vae_name": ext_bundle.audio_vae_name,
                "second_sampling_enabled": ext_bundle.second_sampling_enabled,
                "second_fl2va_model_name": ext_bundle.second_fl2va_model_name,
                "second_ref2va_model_name": ext_bundle.second_ref2va_model_name,
                "second_sampling_use_lora": ext_bundle.second_sampling_use_lora,
                "second_lora_stack": tuple(ext_bundle.second_lora_stack),
            },
        )
