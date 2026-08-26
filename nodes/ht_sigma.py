"""H3 two-pass sigma presets.

The schedule curve is selected independently from the YCNodes H3SigmaRefiner
curve: schedule_spacing controls beta/karras/linear sampling, while
refine_spacing uses YCNodes' cosine, linear, or alpha=3 exponential formula.
"""
import math

import torch
try:
    import numpy
    import scipy.stats
except ImportError:
    numpy = scipy = None

try:
    import comfy.samplers
except ImportError:
    comfy = None


def _schedule(total, spacing):
    if comfy is not None:
        try:
            class MS:
                sigmas = torch.linspace(1, 0, total + 1)
                sigma_min, sigma_max = 0.0, 1.0
            if spacing == "beta":
                return comfy.samplers.calculate_sigmas(MS(), "beta", total)
            if spacing == "karras":
                return comfy.samplers.calculate_sigmas(MS(), "karras", total)
        except Exception:
            pass
    if spacing == "beta" and numpy is not None:
        ts = 1 - numpy.linspace(0, 1, total, endpoint=False)
        indices = numpy.rint(scipy.stats.beta.ppf(ts, 0.6, 0.6) * total).astype(int)
        linear = torch.linspace(1, 0, total + 1)
        return torch.tensor([float(linear[index]) for index in indices] + [0.0], dtype=torch.float32)
    if spacing == "karras":
        ramp = torch.linspace(0, 1, total + 1)
        return (1.0 ** (1 / 7) + ramp * (0.0 ** (1 / 7) - 1.0 ** (1 / 7))) ** 7
    if spacing == "linear":
        return torch.linspace(1.0, 0.0, total + 1)
    raise ValueError("HT Sigma: unsupported schedule spacing %r" % spacing)


def _refine(sigmas, extra, start, end, spacing):
    if extra <= 0:
        return sigmas
    idx = next((i for i, value in enumerate(sigmas) if float(value) <= start), -1)
    if idx < 0 or idx >= len(sigmas) - 1:
        return sigmas
    head, a = sigmas[:idx], float(sigmas[idx])
    b = max(float(end), float(sigmas[-1]))
    terminal_zero = float(sigmas[-1]) == 0
    count = len(sigmas) - idx + extra - (1 if terminal_zero and b > 0 else 0)
    t = torch.linspace(0, 1, count, device=sigmas.device, dtype=sigmas.dtype)
    if spacing == "cosine":
        factor = (1 - torch.cos(t * math.pi)) / 2
    elif spacing == "linear":
        factor = t
    elif spacing == "exponential":
        alpha = 3.0
        factor = (torch.exp(t * alpha) - 1.0) / (math.exp(alpha) - 1.0)
    else:
        raise ValueError("HT Sigma: unsupported refine spacing %r" % spacing)
    tail = a + (b - a) * factor
    if terminal_zero and b > 0:
        tail = torch.cat((tail, sigmas.new_zeros(1)))
    return torch.cat((head, tail))


class HTH3SigmaPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "total_steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "split_step": ("INT", {"default": 4, "min": 0, "max": 100}),
            "refiner_extra_steps": ("INT", {"default": 1, "min": 0, "max": 15}),
            "start_at_sigma": ("FLOAT", {"default": .7, "min": 0, "max": 20, "step": .01}),
            "end_at_sigma": ("FLOAT", {"default": 0., "min": 0, "max": 5, "step": .01}),
            "schedule_spacing": (["beta", "karras", "linear"], {"default": "beta"}),
            "refine_spacing": (["cosine", "linear", "exponential"], {"default": "cosine"}),
        }}
    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    RETURN_NAMES = ("sigmas_high", "sigmas_low")
    FUNCTION = "make"
    CATEGORY = "H3-Heaton/sampling"

    def make(self, total_steps, split_step, refiner_extra_steps, start_at_sigma, end_at_sigma, schedule_spacing, refine_spacing):
        sigmas = _schedule(int(total_steps), schedule_spacing)
        high = sigmas[:int(split_step) + 1]
        low = sigmas[int(split_step):]
        low = _refine(low, int(refiner_extra_steps), float(start_at_sigma), float(end_at_sigma), refine_spacing)
        return high, low

NODE_CLASS_MAPPINGS = {"HT_H3_SigmaPreset": HTH3SigmaPreset}
NODE_DISPLAY_NAME_MAPPINGS = {"HT_H3_SigmaPreset": "HT · H3 Sigma Preset"}
