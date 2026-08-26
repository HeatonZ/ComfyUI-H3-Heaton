"""Motion-context chain nodes. Continuation must not connect first_frame."""
import os, logging
import torch
try:
 import folder_paths
 from safetensors.torch import load_file, save_file
except ImportError:
 folder_paths=None; load_file=save_file=None
from ..core.latent import split_av, join_av

class HTMCContext:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"conditioning":("CONDITIONING",),"latent":("LATENT",),"context_length":("INT",{"default":22,"min":5,"max":56}),"audio_context_length":("INT",{"default":24,"min":0,"max":240}),"clip_index":("INT",{"default":0,"min":0,"max":9999})},"optional":{"context_latent":("LATENT",)}}
 RETURN_TYPES=("CONDITIONING","INT"); RETURN_NAMES=("conditioning","trim_frames"); FUNCTION="apply"; CATEGORY="H3-Heaton/motion-context"
 DESCRIPTION="Pin prior AV context. Continuation segments must not connect first_frame."
 def apply(self,conditioning,latent,context_length,audio_context_length,clip_index,context_latent=None):
  if context_latent is None: raise ValueError("HT MC Context: connect context_latent from the previous segment")
  old_v,old_a=split_av(context_latent["samples"]); cur_v,cur_a=split_av(latent["samples"]); n=min(int(context_length),old_v.shape[2]);
  if n>=cur_v.shape[2]: raise ValueError("HT MC Context: context must be shorter than current clip")
  key={"resolved_frame_index":0,"latent":old_v[:,:,:n].clone(),"ht_mc_index":0}
  out=[]
  for emb,extra in conditioning:
   d=extra.copy(); d["minimax_keyframes"]=list(d.get("minimax_keyframes",[]))+[key]; out.append([emb,d])
  return (out,n)

class HTMCTrim:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"images":("IMAGE",),"trim_frames":("INT",{"default":0,"min":0})},"optional":{"audio":("AUDIO",),"fps":("FLOAT",{"default":24.})}}
 RETURN_TYPES=("IMAGE","AUDIO"); RETURN_NAMES=("images","audio"); FUNCTION="trim"; CATEGORY="H3-Heaton/motion-context"
 def trim(self,images,trim_frames,audio=None,fps=24.):
  n=int(trim_frames); out=images[n:]
  if audio is None:return out,None
  cut=round(n/fps*audio["sample_rate"]); return out,{**audio,"waveform":audio["waveform"][...,cut:]}

class HTMCSaveLatent:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"latent":("LATENT",),"slot_name":("STRING",{"default":"h3_context"})}}
 RETURN_TYPES=("STRING",); FUNCTION="save"; CATEGORY="H3-Heaton/motion-context"
 def save(self,latent,slot_name):
  if save_file is None: raise RuntimeError("HT MC SaveLatent requires safetensors")
  v,a=split_av(latent["samples"]); out=folder_paths.get_output_directory(); path=os.path.join(out,slot_name+".safetensors"); os.makedirs(os.path.dirname(path),exist_ok=True); save_file({"video":v.cpu(),"audio":a.cpu()},path,metadata={"format":"HT_H3_AV_v1"}); return (path,)

class HTMCLoadLatent:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"slot_name":("STRING",{"default":"h3_context"})}}
 RETURN_TYPES=("LATENT",); FUNCTION="load"; CATEGORY="H3-Heaton/motion-context"
 def load(self,slot_name):
  if load_file is None: raise RuntimeError("HT MC LoadLatent requires safetensors")
  path=slot_name if os.path.isfile(slot_name) else os.path.join(folder_paths.get_output_directory(),slot_name+".safetensors"); data=load_file(path); return ({"samples":[data["video"],data["audio"]]},)
NODE_CLASS_MAPPINGS={"HT_MC_Context":HTMCContext,"HT_MC_Trim":HTMCTrim,"HT_MC_SaveLatent":HTMCSaveLatent,"HT_MC_LoadLatent":HTMCLoadLatent}
NODE_DISPLAY_NAME_MAPPINGS={"HT_MC_Context":"HT · MC Context","HT_MC_Trim":"HT · MC Trim","HT_MC_SaveLatent":"HT · MC SaveLatent","HT_MC_LoadLatent":"HT · MC LoadLatent"}
