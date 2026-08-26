"""Reference image/audio boards and native MiniMax H3 ref conditioning."""
import logging
import math
import torch
try:
 import comfy.utils, node_helpers
except ImportError:
 comfy = node_helpers = None

class HTRefImageBoard:
 @classmethod
 def INPUT_TYPES(cls):
  return {"optional":{"image_1":("IMAGE",),"image_2":("IMAGE",),"image_3":("IMAGE",),"image_4":("IMAGE",)}}
 RETURN_TYPES=("HT_REF_IMAGES",); RETURN_NAMES=("ref_images",); FUNCTION="pack"; CATEGORY="H3-Heaton/reference"
 def pack(self,**kwargs):
  out={}
  for i in range(1,5):
   if kwargs.get("image_%d"%i) is not None: out["ref_image_%d"%(i-1)]=kwargs["image_%d"%i]
   else: logging.info("HT Ref Image Board: empty slot %d skipped",i)
  return (out,)

class HTRefAudioBoard:
 @classmethod
 def INPUT_TYPES(cls): return {"optional":{"audio_1":("AUDIO",),"audio_2":("AUDIO",),"audio_3":("AUDIO",),"audio_4":("AUDIO",)}}
 RETURN_TYPES=("HT_REF_AUDIOS",); RETURN_NAMES=("ref_audios",); FUNCTION="pack"; CATEGORY="H3-Heaton/reference"
 def pack(self,**kwargs): return ({"ref_audio_%d"%(i-1):kwargs["audio_%d"%i] for i in range(1,5) if kwargs.get("audio_%d"%i) is not None},)

class HTR2VConditioning:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"bundle":("HT_H3_BUNDLE",),"prompt":("STRING",{"multiline":True}),"width":("INT",{"default":1344,"min":32,"step":32}),"height":("INT",{"default":768,"min":32,"step":32}),"length":("INT",{"default":121,"min":5,"step":17}),"ref_image_size":(["match","max"],{"default":"match"}),"ref_images":("HT_REF_IMAGES",),"ref_audios":("HT_REF_AUDIOS",)}}
 RETURN_TYPES=("CONDITIONING","LATENT"); RETURN_NAMES=("conditioning","latent"); FUNCTION="condition"; CATEGORY="H3-Heaton/reference"
 def condition(self,bundle,prompt,width,height,length,ref_image_size,ref_images,ref_audios):
  if node_helpers is None: raise RuntimeError("HT R2V Conditioning requires ComfyUI")
  import comfy.nested_tensor
  frame_count=max(5,int(length));
  while frame_count%17!=5: frame_count+=1
  video=torch.zeros((1,24,2+(frame_count-5)//17*5,height//16,width//16),device="cpu")
  audio=torch.zeros((1,32,2,round(frame_count/24*40)),device="cpu")
  items=[]; blocks=[]
  for name,img in ref_images.items():
   h,w=img.shape[1:3]; scale=min(1.,math.sqrt((width*height)/(w*h))) if ref_image_size=="match" else min(1.,2048/min(w,h)); tw=max(32,round(w*scale/32)*32); th=max(32,round(h*scale/32)*32); resized=img[:1]
   resized=comfy.utils.common_upscale(resized.movedim(-1,1),tw,th,"lanczos","disabled").movedim(1,-1)
   items.append({"type":"image","data":resized}); blocks.append({"kind":"image","latent_h":th//16,"latent_w":tw//16,"latent":bundle.video_vae.encode(resized)})
  for name,audio in ref_audios.items():
   z=bundle.audio_vae.encode(audio["waveform"][:1].movedim(1,-1)); items.append({"type":"audio"}); blocks.append({"kind":"audio","ref_audio_t":z.shape[-1],"audio_latent":z})
  tokens=bundle.clip.tokenize(prompt,minimax_ref_items=items); cond=bundle.clip.encode_from_tokens_scheduled(tokens)
  if blocks: cond=node_helpers.conditioning_set_values(cond,{"minimax_refs":blocks})
  return (cond,{"samples":comfy.nested_tensor.NestedTensor((video,audio))})
NODE_CLASS_MAPPINGS={"HT_Ref_ImageBoard":HTRefImageBoard,"HT_Ref_AudioBoard":HTRefAudioBoard,"HT_R2V_Conditioning":HTR2VConditioning}
NODE_DISPLAY_NAME_MAPPINGS={"HT_Ref_ImageBoard":"HT · Ref Image Board","HT_Ref_AudioBoard":"HT · Ref Audio Board","HT_R2V_Conditioning":"HT · R2V Conditioning"}
