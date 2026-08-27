"""Motion-context chain nodes for HT-H3.

References ComfyUI-H3-Motion-Context (NikoDemon80, GPLv3).
Patch_layout is imported at runtime from the installed community package.
Constants/helpers replicate the same latent-step grid math."""

import importlib.util, logging, math, os, pathlib
import torch
import node_helpers

try:
    import folder_paths
    from safetensors.torch import load_file, save_file
except ImportError:
    folder_paths = None; load_file = save_file = None

from ..core.latent import split_av, join_av

# ── Constants (same values as MiniMax H3 / community MC) ──────────────
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)
ENCODE_MODE = "video"
ANCHOR_MODE = "head"
AUDIO_MODE = "timeline"
CROP = "disabled"
MC_KEY = "motion_context_index"
MC_AUDIO_KEY = "motion_context_audio_end_frame"

_LOG = logging.getLogger("ht_mcchain")

# ── Layout patch (one-time install from community package) ─────────────
_layout_module = None

def _ensure_layout_patch():
    global _layout_module
    if _layout_module is not None and _layout_module.is_applied():
        return
    comfy_root = pathlib.Path(__file__).resolve().parents[3]
    candidates = list((comfy_root / "custom_nodes").glob("*Motion*Context*"))
    for d in candidates:
        pf = d / "patch_layout.py"
        if pf.exists():
            spec = importlib.util.spec_from_file_location("ht_mc_layout_patch", str(pf))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not mod.is_applied():
                if not mod.apply_patch():
                    raise RuntimeError("HT MC Context: layout patch self-test failed")
            _layout_module = mod
            return
    raise RuntimeError("HT MC Context: ComfyUI-H3-Motion-Context not found in custom_nodes")


# ── Helpers ───────────────────────────────────────────────────────────
def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _steps_for_frames(n):
    """Latent steps covering exactly n pixel frames from cycle position 0."""
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def _streams_from_latent(latent):
    """Unpack H3 AV latent into (video, audio) tensors."""
    samples = latent["samples"]
    if isinstance(samples, (list, tuple)):
        return list(samples)
    if hasattr(samples, "unbind"):
        return list(samples.unbind())
    raise ValueError("HT MC Context: expected H3 AV latent (nested pair)")


def _video_from_latent(latent):
    """Pull the video stream [B,C,T,H,W] from an H3 AV latent."""
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("HT MC Context: expected video latent [B,C,T,H,W], got %s" % (tuple(video.shape),))
    return video


def _video_tail_from_latent(latent, n):
    """Slice the last n pixel frames of video from a generated H3 latent.

    Returns (blocks, offsets, covered) where blocks are per-step tensor
    slices (cloned, bitwise-equal to the source), offsets are the pixel
    frame index of each step, and covered is the total pixel frames.
    The tail window always starts at cycle position 0 because clip lengths
    are 17g+5 and the grid windows are 2,7,12,17 latent steps.
    """
    video = _video_from_latent(latent)
    total = int(video.shape[2])
    steps = _steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "HT MC Context: %d frames is not a whole number of latent steps. "
            "Use 5, 22, 39 or 56." % n)
    if steps > total:
        raise ValueError(
            "HT MC Context: asked for %d latent steps, context_latent has %d." % (steps, total))
    start = total - steps
    if start % 5 != 0:
        raise RuntimeError(
            "HT MC Context: the %d step tail of a %d step latent starts at "
            "cycle position %d, not 0; refusing to render a shifted join."
            % (steps, total, start % 5))
    covered = _pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "HT MC Context: %d steps cover %d frames, expected %d." % (steps, covered, n))
    blocks = [video[:1, :, start + k:start + k + 1].clone()
              for k in range(steps)]
    return blocks, _step_offsets(steps), covered


def _audio_tail_from_latent(latent, a_frames):
    """Slice last a_frames worth of audio steps from H3 AV latent.

    Returns (tail_latent, rt, overhang) where rt is the number of 40 Hz
    latent steps and overhang is the signed fraction of a step by which
    the clip's audio grid overshoots its last pixel frame.
    """
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError("HT MC Context: context_latent has no audio stream.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("HT MC Context: expected audio latent [B,C,2,T], got %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (-0.5 < overhang < 0.5):
        _LOG.warning("HT MC Context: audio grid unexpected (%d steps for %d frames); assuming no overhang.",
                     total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning("HT MC Context: asked for %d audio steps, latent has %d. Pinning all.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("HT MC Context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


def _resize(image, width, height, crop):
    """Resize image [B, H, W, C] to [B, height, width, 3]."""
    import comfy.utils
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


# ── Core node: HT_MC_Context ──────────────────────────────────────────
class HTMCContext:
    """Pin previous clip's tail as never-denoised conditioning blocks.

    The previous clip's sampler output latent is sliced (last N pixel frames,
    respecting the 1,4,4,4,4 latent-step grid). Each latent step becomes its
    own conditioning keyframe block, tagged with motion_context_index so the
    layout patch rewrites PackedLayout position_ids to place them at B's
    frames 0..N-1. Stock sampling re-injects these blocks every step and
    never denoises them, so the overlap region is pixel-identical to A's tail.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "context_length": (["22", "5", "39", "56"], {
                    "default": "22",
                    "tooltip": "Frames of the previous clip's tail to carry over. "
                               "Only these lengths are whole numbers of latent steps. "
                               "22 is nearly seamless."}),
                "audio_context_length": ("INT", {
                    "default": 24, "min": 0, "max": 240,
                    "tooltip": "Frames of tail audio to pin, independent of the picture window. "
                               "0 follows the video span. Multiples of 24 pin whole seconds."}),
            },
            "optional": {
                "context_latent": ("LATENT", {
                    "tooltip": "Previous clip's SAMPLER OUTPUT latent (wire from the decode "
                               "path). Supplies both picture and sound, sliced straight out "
                               "with no h264 decode or VAE round-trip."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "H3-Heaton/motion-context"
    DESCRIPTION = ("Pin a run of consecutive tail frames from a previous clip as "
                   "never-denoised conditioning rows. The sampler reads real motion "
                   "instead of guessing from a single still.")

    def apply(self, conditioning, latent, context_length, audio_context_length,
              context_latent=None):
        context_length = int(context_length)
        audio_context_length = int(audio_context_length)
        _ensure_layout_patch()

        # --- resolve target geometry ---
        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        # --- resolve source from context_latent (preferred: no decode round-trip) ---
        if context_latent is None:
            raise ValueError("HT MC Context: connect context_latent from the previous segment")
        src_video = _video_from_latent(context_latent)
        src_w, src_h = int(src_video.shape[4]) * 16, int(src_video.shape[3]) * 16
        if src_w != width or src_h != height:
            raise ValueError(
                "HT MC Context: context_latent is %dx%d but this clip is %dx%d. "
                "Resolution must match." % (src_w, src_h, width, height))
        available = _pixel_frames(int(src_video.shape[2]))
        video_src = "latent"

        # --- snap to valid grid ---
        n = min(context_length, available)
        if n < 1:
            raise ValueError("HT MC Context: no frames available to pin")
        if n < context_length:
            _LOG.warning("HT MC Context: only %d frames available, pinning %d", available, n)

        run = next(g for g in VIDEO_RUN_GRID if g <= n)
        if run != n:
            _LOG.warning("HT MC Context: %d frames off VAE grid; pinning last %d (valid: 5,22,39,56)", n, run)
            n = run

        if n >= frame_count:
            raise ValueError("HT MC Context: asked to pin %d frames into a %d frame clip." % (n, frame_count))

        steps = _steps_for_frames(n)
        if steps is None:
            raise RuntimeError("HT MC Context: %d frames is not a whole number of latent steps." % n)

        # --- slice tail blocks from context_latent (bitwise identical to source) ---
        blocks, offsets, covered = _video_tail_from_latent(context_latent, n)
        span = covered

        # --- build per-step keyframe entries at target positions 0..N-1 ---
        keyframes = []
        for blk, px_offset in zip(blocks, offsets):
            keyframes.append({
                "resolved_frame_index": 0,       # stock sees 0 (always valid)
                MC_KEY: px_offset,                # real position for the layout patch
                "latent": blk,
            })

        # --- audio context (timeline-aligned) ---
        audio_ref = None
        a_frames = audio_context_length if audio_context_length > 0 else span
        audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(context_latent, a_frames)
        end_frame = float(span if ANCHOR_MODE == "head" else 0) + overhang / FRAME_RESCALE
        end_coord = round(FRAME_RESCALE * end_frame)
        end_frame = end_coord / FRAME_RESCALE
        ref = {
            "kind": "audio",
            MC_AUDIO_KEY: end_frame,
            "ref_audio_t": ref_audio_t,
            "audio_latent": audio_latent,
        }
        audio_ref = ref

        # --- merge with existing conditioning keyframes ---
        head_end = span if ANCHOR_MODE == "head" else 0
        out = []
        dropped = []
        for emb, extra in conditioning:
            d = extra.copy()
            prior = d.get("minimax_keyframes") or []
            pfc = d.get("minimax_frame_count")
            if prior and pfc is not None and int(pfc) != frame_count:
                raise ValueError(
                    "HT MC Context: conditioning resolved for %d frames but latent is %d."
                    % (int(pfc), frame_count))
            kept = []
            for kf in prior:
                p = int(kf.get(MC_KEY, kf.get("resolved_frame_index", 0)))
                if p < head_end:
                    dropped.append(p)
                    continue
                kf = dict(kf)
                kf[MC_KEY] = p
                kept.append(kf)
            d["minimax_keyframes"] = kept + keyframes
            d["minimax_frame_count"] = frame_count
            out.append([emb, d])
        if dropped:
            _LOG.warning("HT MC Context: dropped %d keyframe anchor(s) at frame(s) %s: "
                         "pinned head already decides frames 0..%d.",
                         len(dropped), sorted(set(dropped)), head_end - 1)

        if audio_ref is not None:
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [audio_ref]}, append=True)

        trim = span if ANCHOR_MODE == "head" else 0
        _LOG.info("HT MC Context: video from latent, %d/%s, %d frames -> %d cond blocks "
                  "at indices %d..%d, %d frame clip %dx%d, trim %d, audio %d steps",
                  n, ENCODE_MODE, n, len(blocks), offsets[0], offsets[-1],
                  frame_count, width, height, trim, ref_audio_t)
        return (out, trim)


# ── HT_MC_Trim: drop the pinned head from images + audio ──────────────
class HTMCTrim:
    """Drop the pinned head from decoded images and matching audio duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "H3-Heaton/motion-context"
    DESCRIPTION = ("Drop the pinned head from images and audio together. "
                   "Wires trim_frames from HT MC Context.")

    def trim(self, images, trim_frames, audio=None, fps=24.0):
        n = int(trim_frames)
        out = images[n:]
        if audio is None:
            return (out, None)
        cut = round(n / fps * audio["sample_rate"])
        return (out, {**audio, "waveform": audio["waveform"][..., cut:]})


# ── SaveLatent / LoadLatent (unchanged) ───────────────────────────────
class HTMCSaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",),
                             "slot_name": ("STRING", {"default": "h3_context"})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    CATEGORY = "H3-Heaton/motion-context"

    def save(self, latent, slot_name):
        if save_file is None:
            raise RuntimeError("HT MC SaveLatent requires safetensors")
        v, a = split_av(latent["samples"])
        out = folder_paths.get_output_directory()
        path = os.path.join(out, slot_name + ".safetensors")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file({"video": v.cpu(), "audio": a.cpu()}, path,
                  metadata={"format": "HT_H3_AV_v1"})
        return (path,)


class HTMCLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"slot_name": ("STRING", {"default": "h3_context"})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "H3-Heaton/motion-context"

    def load(self, slot_name):
        if load_file is None:
            raise RuntimeError("HT MC LoadLatent requires safetensors")
        path = (slot_name if os.path.isfile(slot_name)
                else os.path.join(folder_paths.get_output_directory(), slot_name + ".safetensors"))
        data = load_file(path)
        return ({"samples": [data["video"], data["audio"]]},)


# ── Node registration ─────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "HT_MC_Context": HTMCContext,
    "HT_MC_Trim": HTMCTrim,
    "HT_MC_SaveLatent": HTMCSaveLatent,
    "HT_MC_LoadLatent": HTMCLoadLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HT_MC_Context": "HT · MC Context",
    "HT_MC_Trim": "HT · MC Trim",
    "HT_MC_SaveLatent": "HT · MC SaveLatent",
    "HT_MC_LoadLatent": "HT · MC LoadLatent",
}
