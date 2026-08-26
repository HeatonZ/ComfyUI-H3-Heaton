"""Mask-shape/value tests for HT_H3_AudioRefineMask (no comfy needed)."""
import os
import sys
import unittest

import torch

sys.modules.setdefault("comfy", type(sys)("comfy"))
nested_mod = type(sys)("comfy.nested_tensor")


class _NTStub:
    def __init__(self, parts): self.parts = parts
    @property
    def is_nested(self): return True
    def unbind(self): return list(self.parts)


nested_mod.NestedTensor = _NTStub
sys.modules["comfy.nested_tensor"] = nested_mod

import importlib  # noqa: E402
import importlib.machinery  # noqa: E402

# Register a synthetic parent package so the node module's relative imports work
_pkg_root = os.path.dirname(os.path.dirname(__file__))
_parent = "ht_heaton_testpkg"
_mod = importlib.util.module_from_spec(
    importlib.machinery.ModuleSpec(_parent, None, is_package=True))
_mod.__path__ = [_pkg_root]
sys.modules[_parent] = _mod

_spec = importlib.util.spec_from_file_location(
    _parent + ".nodes.ht_audiorefine",
    os.path.join(_pkg_root, "nodes", "ht_audiorefine.py"))
_ar = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ar
_spec.loader.exec_module(_ar)
build_audio_refine_mask = _ar.build_audio_refine_mask


class TestAudioRefineMask(unittest.TestCase):
    def test_mask_shapes_and_values(self):
        video = torch.zeros(1, 24, 8, 48, 84)
        audio = torch.zeros(1, 32, 2, 14)
        v, a = build_audio_refine_mask(video, audio, video_denoise=0.0)
        self.assertEqual(v.shape, (1, 1, 8, 48, 84))
        self.assertEqual(a.shape, (1, 1, 2, 14))  # [B,1,2,T] — core amaxes the 2-axis
        self.assertEqual(float(v.min()), 0.0)
        self.assertEqual(float(v.max()), 0.0)   # video fully frozen
        self.assertEqual(float(a.min()), 1.0)   # audio fully regenerated

    def test_partial_video_denoise(self):
        video = torch.zeros(1, 24, 4, 16, 16)
        audio = torch.zeros(1, 32, 2, 7)
        v, _ = build_audio_refine_mask(video, audio, video_denoise=0.25)
        self.assertAlmostEqual(float(v.mean()), 0.25)

    def test_sampler_node_attaches_mask(self):
        latent = {"samples": _NTStub([torch.zeros(1, 24, 8, 48, 84), torch.zeros(1, 32, 2, 14)])}
        out, = _ar.HTAudioRefineMask().apply(latent, video_denoise=0.0)
        mask = out["noise_mask"]
        self.assertIsInstance(mask, _NTStub)
        vm, am = mask.unbind()
        self.assertEqual(float(vm.max()), 0.0)
        self.assertEqual(float(am.min()), 1.0)


if __name__ == "__main__":
    unittest.main()
