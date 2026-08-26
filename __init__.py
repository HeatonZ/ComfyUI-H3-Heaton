"""ComfyUI-H3-Heaton: MiniMax H3 nodes with a canonical bundle and adapters."""
import importlib
import logging

log=logging.getLogger("HT")
NODE_CLASS_MAPPINGS={}
NODE_DISPLAY_NAME_MAPPINGS={}

for _name in ("ht_loader","ht_sigma","ht_turbo","ht_avlatent","ht_ref2v","ht_mcchain","ht_qa"):
    try:
        _module=importlib.import_module("."+_name, __name__+".nodes")
        NODE_CLASS_MAPPINGS.update(getattr(_module,"NODE_CLASS_MAPPINGS",{}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module,"NODE_DISPLAY_NAME_MAPPINGS",{}))
    except Exception as _exc:
        log.warning("HT: skipped node module %s: %s",_name,_exc)

try:
    from .compat.registry import _discover
    _discover()
except Exception as _exc:
    log.warning("HT: compat discovery failed: %s",_exc)

try:
    from .compat.registry import available_adapters, to_external, from_external
    class HTCompatOut:
        @classmethod
        def INPUT_TYPES(cls): return {"required":{"bundle":("HT_H3_BUNDLE",),"target_format":(list(available_adapters()),)}}
        RETURN_TYPES=("ANY",); FUNCTION="convert"; CATEGORY="H3-Heaton/compat"
        def convert(self,bundle,target_format): return (to_external(target_format,bundle),)
    class HTCompatIn:
        @classmethod
        def INPUT_TYPES(cls): return {"required":{"external_bundle":("ANY",),"target_format":(list(available_adapters()),)}}
        RETURN_TYPES=("HT_H3_BUNDLE",); FUNCTION="convert"; CATEGORY="H3-Heaton/compat"
        def convert(self,external_bundle,target_format): return (from_external(target_format,external_bundle),)
    NODE_CLASS_MAPPINGS.update({"HT_Compat_Out":HTCompatOut,"HT_Compat_In":HTCompatIn})
    NODE_DISPLAY_NAME_MAPPINGS.update({"HT_Compat_Out":"HT · Compat Out","HT_Compat_In":"HT · Compat In"})
except Exception as _exc:
    log.warning("HT: compat nodes unavailable: %s",_exc)

