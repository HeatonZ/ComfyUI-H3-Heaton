"""H3 AV latent structure tools.

An H3 latent packs a video stream [B, 24, T, H/16, W/16] and an audio stream
[B, 32, 2, T40] side by side. The packed form is a comfy.nested_tensor.NestedTensor;
the loose form is a plain list pair. Every helper here validates pairing and
raises instead of silently dropping a stream.
"""

import torch

from comfy.nested_tensor import NestedTensor

VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32


def split_av(samples):
    """Split a packed AV latent into (video, audio) tensors.

    Accepts a NestedTensor (sampler output) or a plain list/tuple pair.
    Raises when the payload is not a paired AV structure.
    """
    if isinstance(samples, NestedTensor):
        parts = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        parts = list(samples)
    else:
        raise ValueError(
            "HT: expected an H3 AV latent (NestedTensor or [video, audio] pair), got %s"
            % type(samples).__name__)
    if len(parts) != 2:
        raise ValueError(
            "HT: H3 AV latent must hold exactly 2 streams (video+audio), got %d" % len(parts))
    video, audio = parts[0], parts[1]
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise ValueError("HT: both AV streams must be torch.Tensor")
    if video.dim() == 4:
        video = video.unsqueeze(0)
    if audio.dim() == 3:
        audio = audio.unsqueeze(0)
    if video.dim() != 5:
        raise ValueError("HT: video latent must be [B,C,T,H,W], got %d dims" % video.dim())
    if audio.dim() != 4:
        raise ValueError("HT: audio latent must be [B,C,2,T], got %d dims" % audio.dim())
    if video.shape[1] != VIDEO_CHANNELS:
        raise ValueError(
            "HT: video latent expects %d channels, got %d — not an H3 video latent"
            % (VIDEO_CHANNELS, video.shape[1]))
    if audio.shape[1] != AUDIO_CHANNELS:
        raise ValueError(
            "HT: audio latent expects %d channels, got %d — not an H3 audio latent"
            % (AUDIO_CHANNELS, audio.shape[1]))
    return video, audio


def join_av(video, audio):
    """Pack a validated (video, audio) pair back into a LATENT dict."""
    if video.dim() == 4:
        video = video.unsqueeze(0)
    if audio.dim() == 3:
        audio = audio.unsqueeze(0)
    _check_pair(video, audio)
    return {"samples": NestedTensor((video.contiguous(), audio.contiguous()))}


def _check_pair(video, audio):
    """Frame-count pairing rule: the audio grid is round(5/3 * pixel frames)."""
    frames = pixel_frames(int(video.shape[2]))
    want_audio = int(round(frames * AUDIO_PER_FRAME))
    have_audio = int(audio.shape[-1])
    if abs(have_audio - want_audio) > 1:
        raise ValueError(
            "HT: AV streams out of pair — %d pixel frames want ~%d audio steps, got %d"
            % (frames, want_audio, have_audio))


def require_paired(samples):
    """Validate and return (video, audio); raises on any mismatch."""
    return split_av(samples)


def pixel_frames(latent_t):
    """Pixel frames covered by latent_t steps under H3's 1,4,4,4,4 cycle."""
    cycle = (1, 4, 4, 4, 4)
    return sum(cycle[k % 5] for k in range(latent_t))


def align_frame_count(n):
    """Snap a pixel frame count up onto H3's 17k+5 grid."""
    while n % 17 != 5:
        n += 1
    return max(5, n)


def align_size(n, multiple=16):
    """Round a pixel width/height to the VAE's spatial multiple."""
    return max(multiple, int(round(n / multiple)) * multiple)


AUDIO_PER_FRAME = 5.0 / 3.0  # 40 Hz audio steps per 24 fps frame


def probe(samples):
    """One-line structural summary used by QA and debug output."""
    try:
        video, audio = split_av(samples)
    except ValueError as exc:
        return "invalid AV latent: %s" % exc
    frames = pixel_frames(int(video.shape[2]))
    on_grid = int(video.shape[2]) == 0 or (frames - 5) % 17 == 0
    return (
        "video%s=%s audio%s=%s | pixel_frames=%d (%s) audio_steps=%d"
        % (tuple(video.shape), str(video.dtype),
           tuple(audio.shape), str(audio.dtype),
           frames, "on 17k+5 grid" if on_grid else "OFF grid",
           int(audio.shape[-1])))
