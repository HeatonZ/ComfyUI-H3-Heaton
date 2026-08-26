"""H3 turbo LoRA with the MiniMax-H3 pruned-base silu_temb patch.

The e-grid interpolation and runtime adaln injection are copied in principle
from comfyui-minimax-h3-turbo/__init__.py: the bundled grid is cached, sampled
at the model's video/audio times, and B @ A @ silu(t_emb) is added by an object
patch.  This node intentionally keeps only the LoRA node (not that package's
sampler or debug logging), because H3-Heaton owns sampling separately.  Full
bases use the normal ComfyUI LoRA patch path; pruned bases use bypass for the
backbone and the e-grid path for adaln, matching the reference's two modes.
"""
import math
import os

import torch
import torch.nn.functional as F

try:
    import folder_paths
    import comfy.utils
    import comfy.lora
    import comfy.patcher_extension
except ImportError:
    folder_paths = None

SHIFT_V, SHIFT_A = 12.0, 3.0
_EGRID = None


def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _egrid():
    global _EGRID
    if _EGRID is None:
        here = os.path.dirname(os.path.dirname(__file__))
        candidates = [
            os.path.join(here, "h3_silu_temb_grid.safetensors"),
            os.path.join(os.path.dirname(here), "comfyui-minimax-h3-turbo",
                         "h3_silu_temb_grid.safetensors"),
        ]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            raise RuntimeError(
                "HT Turbo LoRA: h3_silu_temb_grid.safetensors not found in %s "
                "or comfyui-minimax-h3-turbo" % here)
        _EGRID = comfy.utils.load_torch_file(path)["silu_t_emb_grid"]
    return _EGRID


def _interp_egrid(times, grid, device, dtype):
    grid = grid.to(device)
    rows = []
    for value in times:
        pos = min(max(value, 0.0), 1.0) * (grid.shape[0] - 1)
        low = min(int(math.floor(pos)), grid.shape[0] - 2)
        rows.append(torch.lerp(grid[low].float(), grid[low + 1].float(), pos - low))
    return torch.stack(rows).to(dtype)


def _adaln_forward(base, a, b, shared):
    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        e = shared.get("silu_temb")
        if e is not None:
            x = x + (b.to(x.device, x.dtype) @ (a.to(x.device, x.dtype) @ e.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)
    return forward


def _apply_lora(clone, lora, modules, strength, merge):
    key_map = {name: "diffusion_model.%s.weight" % name for name in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    if merge:
        return len(clone.add_patches(loaded, strength))
    manager = comfy.weight_adapter.BypassInjectionManager()
    known = set(clone.model.state_dict())
    count = 0
    for key, adapter in loaded.items():
        if key in known:
            manager.add_adapter(key, adapter, strength=strength)
            count += 1
    injections = manager.create_injections(clone.model)
    if manager.get_hook_count():
        clone.set_injections("h3turbo_lora", injections)
    return count


class HTH3TurboLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras") if folder_paths else []
        defaults = [n for n in names if "larryvrh" in n.lower() and "v4" in n.lower()]
        return {"required": {
            "model": ("MODEL",),
            "lora_path": (names, {"default": defaults[0] if defaults else (names[0] if names else "")}),
            "strength": ("FLOAT", {"default": 0.75, "min": -10, "max": 10, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "H3-Heaton/turbo"
    DESCRIPTION = "Apply H3 turbo LoRA to the first sampling model."

    def apply(self, model, lora_path, strength):
        if folder_paths is None:
            raise RuntimeError("HT Turbo LoRA requires ComfyUI")
        path = folder_paths.get_full_path_or_raise("loras", lora_path)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        clone = model.clone()
        dm = clone.model.diffusion_model
        modules = sorted({key.rsplit(".lora_", 1)[0] for key in lora if ".lora_" in key})
        pruned = bool(getattr(dm, "use_adaln_curves", False))
        backbone = [name for name in modules if "adaln_proj" not in name] if pruned else modules
        adaln = [name for name in modules if "adaln_proj" in name] if pruned else []
        # Full bases retain the reference package's merged weight-patch path;
        # pruned bases need runtime bypass because their curve-reduced adaln
        # cannot represent the original LoRA input width.
        count = _apply_lora(clone, lora, backbone, float(strength), not pruned)
        if pruned and adaln:
            grid = _egrid()
            shared = {"silu_temb": None}
            def wrapper(executor, *args, **kwargs):
                timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
                context = args[2] if len(args) > 2 else kwargs.get("context")
                sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
                times = sorted({1.0 - sv, 1.0 - _time_shift_sigma(sv, SHIFT_V, SHIFT_A)})
                shared["silu_temb"] = _interp_egrid(times, grid, context.device, context.dtype)
                return executor(*args, **kwargs)
            clone.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo", wrapper)
            for name in adaln:
                base_name = name.rsplit(".linear", 1)[0]
                clone.add_object_patch("diffusion_model.%s.forward" % base_name,
                    _adaln_forward(clone.get_model_object("diffusion_model.%s" % base_name),
                                   lora[name + ".lora_A.weight"],
                                   lora[name + ".lora_B.weight"] * float(strength), shared))
            count += len(adaln)
        if count == 0:
            raise RuntimeError("HT Turbo LoRA: 0 patches applied from %s" % lora_path)
        return (clone,)


NODE_CLASS_MAPPINGS = {"HT_H3_TurboLoRA": HTH3TurboLoRA}
NODE_DISPLAY_NAME_MAPPINGS = {"HT_H3_TurboLoRA": "HT · H3 Turbo LoRA"}
