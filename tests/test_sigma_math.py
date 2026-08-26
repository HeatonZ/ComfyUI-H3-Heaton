"""Acceptance checks for the independent H3 sigma schedule/refiner curves."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from nodes.ht_sigma import HTH3SigmaPreset, _refine, _schedule


node = HTH3SigmaPreset()
high, low = node.make(8, 4, 1, 0.7, 0.0, "beta", "cosine")
assert len(high) == 5
assert len(low) == 6
assert float(low[-1]) == 0.0

base = _schedule(8, "beta")
idx = next(i for i, value in enumerate(base[4:]) if float(value) <= 0.7)
tail = base[4:][idx:]
a = float(tail[0])
b = max(0.0, float(base[-1]))
t = torch.linspace(0, 1, len(tail) + 1, dtype=tail.dtype)
expected = a + (b - a) * ((1 - torch.cos(t * math.pi)) / 2)
expected = torch.cat((base[4:][:idx], expected))
assert torch.allclose(_refine(base[4:], 1, 0.7, 0.0, "cosine"), expected)

try:
    import comfy.samplers
except ImportError:
    pass
else:
    class MS:
        sigmas = torch.linspace(1, 0, 9)
        sigma_min, sigma_max = 0.0, 1.0
    assert torch.allclose(_schedule(8, "beta"), comfy.samplers.calculate_sigmas(MS(), "beta", 8))

print("test_sigma_math: PASS")
