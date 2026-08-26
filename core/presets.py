"""Shared H3 presets: resolution tiers, frame tables, scheduler defaults."""

# Resolution tiers (width x height, already on the 16px VAE grid)
RESOLUTIONS = {
    "960x544": (960, 544),
    "1344x768": (1344, 768),
    "1920x1088": (1920, 1088),
}
DEFAULT_RESOLUTION = "1344x768"

# Frame table: pixel frame count -> seconds at 24 fps.
FRAME_TABLE = {
    121: 5,
    243: 10,
}

FPS = 24.0

# Latent upscale factors offered by HT AV Latent Upscale 3D
UPSCALE_RATIOS = (1.25, 1.5, 2.0)

# Second-pass sigma refinement defaults (see nodes/ht_sigma.py for the math)
SIGMA_PRESET_DEFAULTS = {
    "total_steps": 8,
    "split_step": 4,
    "refiner_extra_steps": 1,
    "start_at_sigma": 0.7,
    "end_at_sigma": 0.0,
    "spacing": "cosine",
}
SPACINGS = ("cosine", "beta", "karras")


def resolution_size(name):
    if name not in RESOLUTIONS:
        raise ValueError("HT: unknown resolution preset %r (have %s)" % (name, sorted(RESOLUTIONS)))
    return RESOLUTIONS[name]


def frames_to_seconds(frames):
    return FRAME_TABLE.get(int(frames)) if int(frames) in FRAME_TABLE else frames / FPS


def default_frames(seconds):
    best = min(FRAME_TABLE, key=lambda f: abs(FRAME_TABLE[f] - float(seconds)))
    return best
