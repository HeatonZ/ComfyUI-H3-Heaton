"""Import smoke test with only the small ComfyUI surfaces needed by core.latent."""
import importlib.util
import pathlib
import sys
import types

nested = types.ModuleType("comfy.nested_tensor")
class NestedTensor:
    def __init__(self, tensors): self.tensors = tuple(tensors)
    def unbind(self): return self.tensors
nested.NestedTensor = NestedTensor
comfy = types.ModuleType("comfy")
comfy.__path__ = []
comfy.nested_tensor = nested
sys.modules["comfy"] = comfy
sys.modules["comfy.nested_tensor"] = nested

# The project venv used for this import-only check may not carry ComfyUI's
# heavyweight torch dependency. Node modules only touch torch when executed.
torch = types.ModuleType("torch")
torch.Tensor = type("Tensor", (), {})
torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional

root = pathlib.Path(__file__).parents[1]
name = "h3_heaton_smoke"
spec = importlib.util.spec_from_file_location(
    name, root / "__init__.py", submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)

assert module.NODE_CLASS_MAPPINGS, "no nodes registered"
keys = list(module.NODE_CLASS_MAPPINGS)
assert len(keys) == len(set(keys)), "duplicate node keys"
print("smoke_import: PASS")
print("registered_nodes=%d" % len(keys))
print("keys=" + ",".join(sorted(keys)))
