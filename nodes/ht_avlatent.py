"""AV latent separation, concatenation and geometric 3D upscale."""
import torch
import torch.nn.functional as F
from ..core.latent import split_av, join_av, pixel_frames, align_size, AUDIO_PER_FRAME

class HTAVSeparate:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"latent":("LATENT",)}}
 RETURN_TYPES=("LATENT","LATENT","INT"); RETURN_NAMES=("video_latent","audio_latent","sample_rate"); FUNCTION="separate"; CATEGORY="H3-Heaton/latent"
 def separate(self,latent):
  v,a=split_av(latent["samples"])
  return ({"samples":v},{"samples":a},40000)

class HTAVConcat:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"video_latent":("LATENT",),"audio_latent":("LATENT",),"sample_rate":("INT",{"default":40000,"min":1})}}
 RETURN_TYPES=("LATENT",); RETURN_NAMES=("latent",); FUNCTION="concat"; CATEGORY="H3-Heaton/latent"
 def concat(self,video_latent,audio_latent,sample_rate): return (join_av(video_latent["samples"],audio_latent["samples"]),)

class HTAVUpscale3D:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"latent":("LATENT",),"scale":("FLOAT",{"default":2.,"min":1.,"max":4.,"step":.05}),"device":(["cuda","cpu"],{"default":"cuda"}),"precision":(["fp16","fp32","bf16"],{"default":"fp16"}),"enable_chunking":("BOOLEAN",{"default":True}),"chunk":("INT",{"default":16,"min":1,"max":256})}}
 RETURN_TYPES=("LATENT",); RETURN_NAMES=("latent",); FUNCTION="upscale"; CATEGORY="H3-Heaton/latent"
 def upscale(self,latent,scale,device="cuda",precision="fp16",enable_chunking=True,chunk=16):
  v,a=split_av(latent["samples"]); dev=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu")
  dtype={"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}[precision]
  x=v.to(dev,dtype)
  target_h=align_size(round(x.shape[-2]*scale),16)//16; target_w=align_size(round(x.shape[-1]*scale),16)//16
  pieces=[]
  ranges=range(0,x.shape[2],chunk) if enable_chunking else [0]
  for start in ranges:
   end=min(x.shape[2],start+chunk) if enable_chunking else x.shape[2]
   pieces.append(F.interpolate(x[:,:,start:end],size=(end-start,target_h,target_w),mode="trilinear",align_corners=False))
  out=torch.cat(pieces,2).to(v.device,v.dtype)
  return (join_av(out,a),)

NODE_CLASS_MAPPINGS={"HT_AV_Separate":HTAVSeparate,"HT_AV_Concat":HTAVConcat,"HT_AV_Upscale3D":HTAVUpscale3D}
NODE_DISPLAY_NAME_MAPPINGS={"HT_AV_Separate":"HT · AV Latent Separate","HT_AV_Concat":"HT · AV Latent Concat","HT_AV_Upscale3D":"HT · AV Latent Upscale 3D"}
