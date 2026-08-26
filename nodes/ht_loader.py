"""Unified H3 loader and LoRA stack."""
import logging
import os
from ..core.bundle import HTBundle

log = logging.getLogger("HT.loader")
try:
    import folder_paths
    import nodes
except ImportError:
    folder_paths = nodes = None


def _need_comfy():
    if folder_paths is None or nodes is None:
        raise RuntimeError("HT H3 Loader requires a running ComfyUI installation")


class HTH3UnifiedLoader:
    """Load native ComfyUI UNET, CLIP and both H3 VAEs into HT_H3_BUNDLE."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models") if folder_paths else [],),
            "clip_name": (folder_paths.get_filename_list("text_encoders") if folder_paths else [],),
            "video_vae": (folder_paths.get_filename_list("vae") if folder_paths else [],),
            "audio_vae": (folder_paths.get_filename_list("vae") if folder_paths else [],),
        }}
    RETURN_TYPES = ("HT_H3_BUNDLE",)
    RETURN_NAMES = ("bundle",)
    FUNCTION = "load"
    CATEGORY = "H3-Heaton/loader"
    DESCRIPTION = "Load the native H3 transformer, minimax CLIP and paired video/audio VAEs."

    def load(self, unet_name, clip_name, video_vae, audio_vae):
        _need_comfy()
        model = nodes.UNETLoader().load_unet(unet_name, "default")[0]
        clip = nodes.CLIPLoader().load_clip(clip_name, "minimax_h3")[0]
        video = nodes.VAELoader().load_vae(video_vae)[0]
        audio = nodes.VAELoader().load_vae(audio_vae)[0]
        report = "unet=%s\nclip=%s\nvideo_vae=%s\naudio_vae=%s" % (unet_name, clip_name, video_vae, audio_vae)
        return (HTBundle(model, clip, video, audio, loaded_report=report,
                         base_info={"unet_name": unet_name, "clip_name": clip_name,
                                    "video_vae_name": video_vae, "audio_vae_name": audio_vae}),)


class HTH3LoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        opts = folder_paths.get_filename_list("loras") if folder_paths else []
        required = {"bundle": ("HT_H3_BUNDLE",)}
        for i in range(1, 4):
            required["lora_%d" % i] = (["None"] + opts,)
            required["strength_%d" % i] = ("FLOAT", {"default": 0.0, "min": -10, "max": 10, "step": .01})
        return {"required": required}
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "loaded_report")
    FUNCTION = "apply"
    CATEGORY = "H3-Heaton/loader"

    def apply(self, bundle, lora_1, strength_1, lora_2, strength_2, lora_3, strength_3):
        _need_comfy()
        model = bundle.unet_model
        applied = []
        for name, strength in ((lora_1, strength_1), (lora_2, strength_2), (lora_3, strength_3)):
            if name == "None" or float(strength) == 0:
                continue
            path = folder_paths.get_full_path_or_raise("loras", name)
            data = nodes.comfy.utils.load_torch_file(path, safe_load=True)
            model, _ = nodes.comfy.sd.load_lora_for_models(model, None, data, float(strength), 0.0)
            applied.append("%s (%d tensors)" % (name, len(data)))
        if not applied:
            log.warning("HT H3 LoRA Stack: 0 patches applied")
        report = "%s\nLoRA patches: %d\n%s" % (bundle.loaded_report, len(applied), ", ".join(applied) or "NONE")
        return (model, report)

NODE_CLASS_MAPPINGS = {"HT_H3_UnifiedLoader": HTH3UnifiedLoader, "HT_H3_LoRAStack": HTH3LoRAStack}
NODE_DISPLAY_NAME_MAPPINGS = {"HT_H3_UnifiedLoader": "HT · H3 Unified Loader", "HT_H3_LoRAStack": "HT · H3 LoRA Stack"}
