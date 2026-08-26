"""Audio-only refinement for sampled H3 AV latents (freeze video, regen audio).

Core principle absorbed from Adudeguyman/ComfyUI-H3-AudioRefine (MIT): attach a
per-stream denoise mask to an already-sampled AV latent -- video rows marked
preserve (0.0), audio rows generate (1.0) -- and let ComfyUI's native masked
inpaint path do the rest. MiniMaxH3._denoise_mask_conds pools the mask onto the
token grid and scale_latent_inpaint injects the finished video at the visual
cond timestep every step, so a follow-up sampler pass at denoise < 1.0 refines
the audio *in the context of* the frozen video and returns the video slice
bit-identical. No custom sampler, no monkey-patching: stock KSampler /
SamplerCustomAdvanced with denoise < 1.0 is the whole pass.

Compute note: H3 is one packed token sequence, so each refinement step still
costs near a full forward pass; the saving is step arithmetic (e.g. 4-step
Turbo pass-1 + 4-6 audio-only tail steps instead of a 20-step joint run).

The frozen-video KV-cache accelerator from the upstream pack is intentionally
NOT absorbed: it approximates attention and patches model internals, which
conflicts with this package's no-monkey-patch principle.
"""
import torch

try:
    from comfy.nested_tensor import NestedTensor
except ImportError:  # standalone unit tests
    NestedTensor = None

from ..core.latent import split_av


def build_audio_refine_mask(video, audio, video_denoise=0.0):
    """Per-stream masks at full stream shape, 1 channel (core broadcasts).

    0.0 = preserve, 1.0 = generate. Returns the raw (video_mask, audio_mask)
    pair so tests can run without comfy installed.
    """
    v = torch.full(
        (1, 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4])),
        float(video_denoise), dtype=torch.float32,
    )
    a = torch.ones((1, 1, int(audio.shape[2]), int(audio.shape[3])), dtype=torch.float32)
    return v, a


class HTAudioRefineMask:
    """Mark a sampled AV latent: video preserved, audio regenerated."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "已采样的 MiniMax H3 AV latent（一采输出）"}),
            "video_denoise": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": .01,
                "tooltip": "0.0=视频完全冻结（位级不变）；>0 允许精修时部分重绘视频"}),
        }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply"
    CATEGORY = "H3-Heaton/sampling"

    def apply(self, latent, video_denoise=0.):
        if NestedTensor is None:
            raise RuntimeError("HT Audio Refine: comfy.nested_tensor unavailable")
        video, audio = split_av(latent["samples"])
        v_mask, a_mask = build_audio_refine_mask(video, audio, video_denoise)
        out = dict(latent)
        out["noise_mask"] = NestedTensor((v_mask, a_mask))
        return (out,)

NODE_CLASS_MAPPINGS = {"HT_H3_AudioRefineMask": HTAudioRefineMask}
NODE_DISPLAY_NAME_MAPPINGS = {"HT_H3_AudioRefineMask": "HT · H3 Audio Refine Mask"}
